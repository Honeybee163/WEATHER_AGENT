import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from scipy.stats import ks_2samp

from feature_store.hopswork_feature_store import (
    FEATURE_COLUMNS,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    TIME_COLUMN
)
from feature_store.model_registry import (
    MODEL_NAMES,
    LEDGER_NAME,
    LEDGER_VERSION,
    connect_to_hopsworks,
    get_champion
)
from models.baseline_model import METRICS, TIMEZONE, TRAIN_PATH
from models.random_forest import load_split
from agents.shared_state import get_logger, load_state, update_state

load_dotenv()

logger = get_logger("drift_agent")


# ---------------------------------------
# Configuration
# ---------------------------------------

# Data drift: training data against the latest days of live data
LIVE_WINDOW_DAYS = 7
MIN_LIVE_ROWS = 24
PSI_BINS = 10

# PSI: below 0.1 no change, 0.1 to 0.25 moderate, above 0.25 large
PSI_THRESHOLD = 0.25
KS_P_VALUE = 0.05

# Data drift is reported when at least this many features drifted
MIN_DRIFTED_FEATURES = 2

# Model drift: forecasts of the last days against what really happened
RECENT_DAYS = 7
MIN_RECENT_FORECASTS = 5

# Observation must be this close to the target time of the forecast
MATCH_TOLERANCE = pd.Timedelta("30min")

# Model drift when recent MAE is above validation MAE x ratio and
# also above a minimum error that matters for the metric
MODEL_DRIFT_RATIO = 2.0

MIN_ERROR = {
    "temperature": 1.0,
    "humidity": 5.0,
    "precipitation": 0.5,
    "uv_index": 1.0,
    "AQI": 10.0,
    "wind_speed": 1.0,
}

DRIFT_FEATURE_GROUP = "drift_events"
DRIFT_FEATURE_GROUP_VERSION = 1

DRIFT_EVENTS_PATH = "data/processed/drift_events.csv"
TEAM_ALERTS_PATH = "data/processed/team_alerts.log"
RETRAINING_REQUESTS_PATH = "data/processed/retraining_requests.csv"

# Optional Slack / Discord webhook for the team alert
TEAM_WEBHOOK_URL = os.getenv("TEAM_WEBHOOK_URL")


# ---------------------------------------
# Helpers
# ---------------------------------------

def to_float(value):

    return None if pd.isna(value) else float(value)


def to_utc(series):

    return pd.to_datetime(
        series,
        utc=True
    ).astype("datetime64[ns, UTC]")


def load_observations(feature_store):

    feature_group = feature_store.get_feature_group(
        FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    observations = feature_group.read().rename(
        columns={new: old for old, new in FEATURE_COLUMNS.items()}
    )

    observations[TIME_COLUMN] = to_utc(observations[TIME_COLUMN])

    return observations.sort_values(TIME_COLUMN).reset_index(drop=True)


# ---------------------------------------
# Part A: Data drift (PSI and KS test)
# ---------------------------------------

def calculate_psi(training, live):

    # Bins are decided by the training data
    edges = np.unique(
        np.quantile(training, np.linspace(0, 1, PSI_BINS + 1))
    )

    if len(edges) < 2:
        return np.nan

    edges[0], edges[-1] = -np.inf, np.inf

    expected = np.histogram(training, bins=edges)[0] / len(training)
    actual = np.histogram(live, bins=edges)[0] / len(live)

    # Empty bins would give log(0)
    expected = np.clip(expected, 1e-4, None)
    actual = np.clip(actual, 1e-4, None)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def detect_data_drift(train_df, observations):

    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LIVE_WINDOW_DAYS)

    live_df = observations[observations[TIME_COLUMN] >= start]

    features = []

    for metric in METRICS:

        if metric not in train_df.columns or metric not in live_df.columns:
            continue

        training = train_df[metric].dropna().to_numpy()
        live = live_df[metric].dropna().to_numpy()

        row = {
            "feature": metric,
            "train_mean": None,
            "live_mean": None,
            "psi": None,
            "ks_statistic": None,
            "ks_p_value": None,
            "drifted": False,
            "note": None,
        }

        if len(live) < MIN_LIVE_ROWS:
            row["note"] = f"only {len(live)} live rows"

        elif len(training) == 0:
            row["note"] = "no training data"

        else:
            psi = calculate_psi(training, live)

            if np.isnan(psi):
                row["note"] = "no variation in training data"

            else:
                ks = ks_2samp(training, live)

                row.update({
                    "train_mean": to_float(training.mean()),
                    "live_mean": to_float(live.mean()),
                    "psi": psi,
                    "ks_statistic": float(ks.statistic),
                    "ks_p_value": float(ks.pvalue),
                    # Both tests must agree, to avoid false alarms
                    "drifted": bool(
                        psi >= PSI_THRESHOLD and ks.pvalue < KS_P_VALUE
                    ),
                })

        features.append(row)

    drifted = [row["feature"] for row in features if row["drifted"]]

    return {
        "features": features,
        "drifted_features": drifted,
        "detected": len(drifted) >= MIN_DRIFTED_FEATURES,
    }


# ---------------------------------------
# Part B: Model drift (recent error)
# ---------------------------------------

def detect_model_drift(registry, observations, ledger):

    now = pd.Timestamp.now(tz="UTC")

    ledger = ledger.copy()

    ledger["target_time"] = to_utc(ledger["target_time"])

    # Only forecasts whose target time has already passed in the
    # last days. Newer forecasts have no actual value yet.
    ledger = ledger[
        (ledger["target_time"] <= now)
        & (ledger["target_time"] >= now - pd.Timedelta(days=RECENT_DAYS))
    ]

    metrics = []

    for metric in METRICS:

        row = {
            "metric": metric,
            "validation_mae": None,
            "recent_mae": None,
            "forecasts": 0,
            "matched": 0,
            "drifted": False,
            "note": None,
        }

        metrics.append(row)

        champion = get_champion(registry, MODEL_NAMES[metric])

        if champion is None:
            row["note"] = "no registered model"
            continue

        row["validation_mae"] = to_float(
            champion.training_metrics.get("MAE")
        )

        # Forecasts of the current champion only (not the naive fallback)
        forecasts = ledger[
            (ledger["metric"] == metric)
            & (ledger["model_name"] == MODEL_NAMES[metric])
            & (ledger["model_version"] == champion.version)
        ].sort_values("target_time")

        row["forecasts"] = len(forecasts)

        if forecasts.empty or metric not in observations.columns:
            row["note"] = "no recent forecasts"
            continue

        actual = observations[[TIME_COLUMN, metric]].dropna().rename(
            columns={metric: "actual"}
        )

        # Actual value measured at the target time of the forecast
        matched = pd.merge_asof(
            forecasts[["target_time", "forecast"]],
            actual,
            left_on="target_time",
            right_on=TIME_COLUMN,
            direction="nearest",
            tolerance=MATCH_TOLERANCE
        ).dropna(subset=["actual"])

        row["matched"] = len(matched)

        # A forecast without an actual value has no error yet
        if len(matched) < MIN_RECENT_FORECASTS:
            row["note"] = (
                f"only {len(matched)} forecasts have an actual value"
            )
            continue

        row["recent_mae"] = float(
            (matched["forecast"] - matched["actual"]).abs().mean()
        )

        limit = max(
            row["validation_mae"] * MODEL_DRIFT_RATIO,
            MIN_ERROR[metric]
        )

        row["drifted"] = bool(row["recent_mae"] > limit)

    drifted = [row["metric"] for row in metrics if row["drifted"]]

    return {
        "metrics": metrics,
        "drifted_metrics": drifted,
        "detected": len(drifted) > 0,
    }


# ---------------------------------------
# Part C: Drift actions
# ---------------------------------------

def describe_drift(data_drift, model_drift):

    lines = []

    # A single drifted feature is not reported as data drift
    for row in data_drift["features"] if data_drift["detected"] else []:

        if row["drifted"]:
            lines.append(
                f"Data drift - {row['feature']}: PSI {row['psi']:.2f}, "
                f"mean {row['train_mean']:.1f} (training) -> "
                f"{row['live_mean']:.1f} (live)"
            )

    for row in model_drift["metrics"]:

        if row["drifted"]:
            lines.append(
                f"Model drift - {row['metric']}: recent MAE "
                f"{row['recent_mae']:.2f} vs validation MAE "
                f"{row['validation_mae']:.2f}"
            )

    return lines


def record_event(event):

    os.makedirs("data/processed", exist_ok=True)

    record = pd.DataFrame([event])

    # Local copy: the event is kept even when Hopsworks is down
    record.to_csv(
        DRIFT_EVENTS_PATH,
        mode="a",
        header=not os.path.exists(DRIFT_EVENTS_PATH),
        index=False
    )

    try:
        feature_store = connect_to_hopsworks().get_feature_store()

        feature_group = feature_store.get_or_create_feature_group(
            name=DRIFT_FEATURE_GROUP,
            version=DRIFT_FEATURE_GROUP_VERSION,
            description="Data drift and model drift events",
            primary_key=["event_time"],
            event_time="event_time"
        )

        feature_group.insert(
            record.assign(event_time=to_utc(record["event_time"])),
            write_options={"wait_for_job": True}
        )

        return {"local": True, "feature_store": True}

    except Exception as error:
        logger.error(f"Drift event save to Feature Store failed - {error}")

        return {"local": True, "feature_store": False}


def alert_team(event, retraining):

    message = "\n".join(
        [f"DRIFT ALERT - AgentWeather ({event['event_time']})"]
        + event["summary"].split(" | ")
        + [
            "Retraining: "
            + (
                f"request {retraining['request_id']} created"
                if retraining["created"]
                else "a request is already pending"
            )
        ]
    )

    logger.warning(message)

    with open(TEAM_ALERTS_PATH, "a", encoding="utf-8") as file:
        file.write(message + "\n\n")

    sent = False

    if TEAM_WEBHOOK_URL:

        try:
            # "text" for Slack, "content" for Discord
            response = requests.post(
                TEAM_WEBHOOK_URL,
                json={"text": message, "content": message},
                timeout=10
            )

            response.raise_for_status()

            sent = True

        except requests.exceptions.RequestException as error:
            logger.error(f"Team webhook failed - {error}")

    return {"message": message, "webhook_sent": sent}


def request_retraining(event):

    os.makedirs("data/processed", exist_ok=True)

    # One open request is enough, no new request every day
    if os.path.exists(RETRAINING_REQUESTS_PATH):

        requests_df = pd.read_csv(RETRAINING_REQUESTS_PATH)

        pending = requests_df[requests_df["status"] == "pending"]

        if not pending.empty:
            return {
                "created": False,
                "request_id": pending.iloc[0]["request_id"],
            }

    request_id = datetime.now().strftime("retrain-%Y%m%d-%H%M%S")

    pd.DataFrame([{
        "request_id": request_id,
        "requested_at": event["event_time"],
        "drift_type": event["drift_type"],
        "reason": event["summary"],
        "status": "pending",
    }]).to_csv(
        RETRAINING_REQUESTS_PATH,
        mode="a",
        header=not os.path.exists(RETRAINING_REQUESTS_PATH),
        index=False
    )

    return {"created": True, "request_id": request_id}


def handle_drift(data_drift, model_drift):

    drift_type = " + ".join(
        name for name, result in (("data", data_drift), ("model", model_drift))
        if result["detected"]
    )

    event = {
        "event_time": pd.Timestamp.now(tz="UTC").floor("s").isoformat(),
        "drift_type": drift_type,
        "drifted_features": ", ".join(
            data_drift["drifted_features"] if data_drift["detected"] else []
        ),
        "drifted_metrics": ", ".join(model_drift["drifted_metrics"]),
        "summary": " | ".join(describe_drift(data_drift, model_drift)),
    }

    # 1. Record the drift event
    saved = record_event(event)

    # 3. Retraining request
    retraining = request_retraining(event)

    # 2. Team alert (also tells about the retraining request)
    team = alert_team(event, retraining)

    return {
        "event": event,
        "event_saved": saved,
        "retraining": retraining,
        "team_alert": team,
    }


# ---------------------------------------
# Main Drift Agent
# ---------------------------------------

def run_drift_agent(state=None):

    print("\nStarting Drift Agent...")

    state = load_state() if state is None else state

    try:
        train_df = load_split(TRAIN_PATH)

        project = connect_to_hopsworks()

        feature_store = project.get_feature_store()

        observations = load_observations(feature_store)

        ledger = feature_store.get_feature_group(
            LEDGER_NAME,
            version=LEDGER_VERSION
        ).read()

        data_drift = detect_data_drift(train_df, observations)

        model_drift = detect_model_drift(
            project.get_model_registry(),
            observations,
            ledger
        )

    except Exception as error:

        # The agents must never crash the whole system
        logger.error(f"Drift agent failed - {error}")

        return update_state(
            state,
            drift_status="failed",
            drift_error=str(error)
        )

    detected = data_drift["detected"] or model_drift["detected"]

    actions = handle_drift(data_drift, model_drift) if detected else None

    print("\nData drift:", data_drift["drifted_features"] or "none")
    print("Model drift:", model_drift["drifted_metrics"] or "none")

    return update_state(
        state,
        data_drift=data_drift,
        model_drift=model_drift,
        drift_detected=detected,
        drift_actions=actions,
        drift_status="ok",
        drift_error=None
    )


if __name__ == "__main__":

    run_drift_agent()