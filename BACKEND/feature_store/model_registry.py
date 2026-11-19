import os
import json
import shutil
import tempfile
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

import hopsworks

load_dotenv()


# ---------------------------------------
# Configuration
# ---------------------------------------

# Output of models/model_comparison.py
BEST_MODELS_PATH = "data/processed/best_models.csv"
BEST_MODEL_DIR = "models/best_models"

RF_PREDICTIONS_PATH = "data/processed/rf_validation_predictions.csv"
SARIMAX_PREDICTIONS_PATH = "data/processed/sarimax_validation_predictions.csv"
LSTM_PREDICTIONS_PATH = "data/processed/lstm_validation_predictions.csv"

LOG_PATH = "data/processed/model_registry_log.csv"

# Metric -> name in the Model Registry
MODEL_NAMES = {
    "temperature": "temperature_model",
    "AQI": "aqi_model",
    "humidity": "humidity_model",
    "precipitation": "rain_model",
    "uv_index": "uv_model",
    "wind_speed": "wind_model",
}

UNITS = {
    "temperature": "°C",
    "AQI": "AQI",
    "humidity": "%",
    "precipitation": "mm",
    "uv_index": "UV",
    "wind_speed": "m/s",
}

# Predicted values of each model type: (file, is the time the
# forecast is made). SARIMAX saves the forecasted hour instead.
PREDICTION_FILES = {
    "random_forest": (RF_PREDICTIONS_PATH, True),
    "sarimax": (SARIMAX_PREDICTIONS_PATH, False),
    "lstm": (LSTM_PREDICTIONS_PATH, True),
}

# ---------------------------------------
# Champion rules (fixed in advance)
# ---------------------------------------
# A new model replaces the champion only when:
#   1. it beats the naive baseline, and
#   2. its validation MAE is at least 1% lower than the
#      validation MAE of the champion.
PRIMARY_METRIC = "MAE"
MIN_IMPROVEMENT_PERCENT = 1.0

LEDGER_NAME = "forecast_ledger"
# Version 2: version 1 was created without stream=True
LEDGER_VERSION = 2

FORECAST_HOURS = 24


# ---------------------------------------
# Step 1: Connect and load best models
# ---------------------------------------

def connect_to_hopsworks():

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    if not api_key or not project_name:
        raise ValueError(
            "HOPSWORKS_API_KEY and HOPSWORKS_PROJECT "
            "must be set in the .env file."
        )

    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key
    )

    print(f"Connected to Hopsworks project: {project.name}")

    return project


def load_best_models():

    if not os.path.exists(BEST_MODELS_PATH):
        raise FileNotFoundError(
            f"File not found: {BEST_MODELS_PATH} "
            "(run models.model_comparison first)"
        )

    best = pd.read_csv(BEST_MODELS_PATH)

    best["model_name"] = best["metric"].map(MODEL_NAMES)

    # Models without a saved file cannot be registered
    missing = best["model_file"].isna() | ~best["model_file"].apply(
        lambda path: isinstance(path, str) and os.path.exists(path)
    )

    for metric in best.loc[missing, "metric"]:
        print(f"{metric}: model file not found, skipped")

    best = best[~missing].reset_index(drop=True)

    # Date when the model file was created
    best["training_date"] = best["model_file"].apply(
        lambda path: datetime.fromtimestamp(
            os.path.getmtime(path)
        ).strftime("%Y-%m-%d")
    )

    print(f"\nBest models loaded: {len(best)}")

    return best


# ---------------------------------------
# Step 2: Champion model and replacement rule
# ---------------------------------------

def get_champion(registry, model_name):

    # Only models that passed the rules are registered, so the
    # latest version is always the current champion.
    try:
        versions = registry.get_models(model_name)

    except Exception:
        # Model name does not exist yet
        return None

    if not versions:
        return None

    return max(versions, key=lambda model: model.version)


def should_replace(candidate, champion):

    if champion is None:
        return True, "first model"

    champion_score = champion.training_metrics.get(PRIMARY_METRIC)

    if champion_score is None:
        return True, "champion has no validation score"

    if not candidate["beats_naive"]:
        return False, "does not beat the naive baseline"

    improvement = (
        (champion_score - candidate[PRIMARY_METRIC])
        / champion_score * 100
        if champion_score > 0 else 0
    )

    if improvement < MIN_IMPROVEMENT_PERCENT:
        return False, (
            f"improvement {improvement:.2f}% is below "
            f"{MIN_IMPROVEMENT_PERCENT}%"
        )

    return True, f"validation {PRIMARY_METRIC} improved {improvement:.2f}%"


# ---------------------------------------
# Step 3: Register model with its information
# ---------------------------------------

def register_model(registry, row, version):

    metrics = {
        "MAE": float(row["MAE"]),
        "RMSE": float(row["RMSE"]),
        "R2": float(row["R2"]),
        "validation_score": float(row[PRIMARY_METRIC]),
    }

    info = {
        "model_name": row["model_name"],
        "model_version": version,
        "metric": row["metric"],
        "model_type": row["model"],
        "training_date": row["training_date"],
        "validation_score": metrics["validation_score"],
        "validation_score_metric": PRIMARY_METRIC,
        **{key: metrics[key] for key in ("MAE", "RMSE", "R2")},
    }

    model = registry.python.create_model(
        name=row["model_name"],
        version=version,
        metrics=metrics,
        description=(
            f"{row['model']} model for 24-hour-ahead {row['metric']} "
            f"forecast. Training date: {row['training_date']}. "
            f"Validation {PRIMARY_METRIC}: {metrics['validation_score']:.3f}"
        )
    )

    with tempfile.TemporaryDirectory() as folder:

        shutil.copy2(row["model_file"], folder)

        # The LSTM needs its scaler to work
        if row["model"] == "lstm":
            shutil.copy2(f"{BEST_MODEL_DIR}/lstm_scaler.joblib", folder)

        with open(f"{folder}/model_info.json", "w") as file:
            json.dump(info, file, indent=2)

        model.save(folder)

    print(
        f"Registered: {row['model_name']} version {version} "
        f"({row['model']})"
    )


def register_best_models(registry, best):

    decisions = []

    for _, row in best.iterrows():

        champion = get_champion(registry, row["model_name"])

        replace, reason = should_replace(row, champion)

        champion_version = champion.version if champion else None

        champion_score = (
            champion.training_metrics.get(PRIMARY_METRIC)
            if champion else None
        )

        registered_version = None

        if replace:

            registered_version = (
                champion_version + 1 if champion else 1
            )

            register_model(registry, row, registered_version)

        else:
            print(
                f"Kept: {row['model_name']} version {champion_version} "
                f"({reason})"
            )

        decisions.append({
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric": row["metric"],
            "model_name": row["model_name"],
            "model_type": row["model"],
            f"candidate_{PRIMARY_METRIC}": row[PRIMARY_METRIC],
            "champion_version": champion_version,
            f"champion_{PRIMARY_METRIC}": champion_score,
            "decision": "registered" if replace else "kept champion",
            "new_version": registered_version,
            "reason": reason,
        })

    return pd.DataFrame(decisions)


def show_registry(registry, best):

    print("\nModel Registry:")

    for model_name in best["model_name"]:

        for model in sorted(
            registry.get_models(model_name),
            key=lambda item: item.version
        ):

            scores = {
                key: round(value, 3)
                for key, value in model.training_metrics.items()
            }

            print(f"{model_name} v{model.version}: {scores}")


# ---------------------------------------
# Step 4: Forecast Ledger
# ---------------------------------------

def create_forecast_ledger(feature_store):

    # One row for every forecast: when it was made, which time
    # it is for, and which model version made it
    return feature_store.get_or_create_feature_group(
        name=LEDGER_NAME,
        version=LEDGER_VERSION,
        description="All forecasts with the model version that made them",
        primary_key=["forecast_time", "metric"],
        event_time="forecast_time",
        # Write through Kafka instead of writing to HDFS/Delta
        # directly from this machine (fixes the RPC/HDFS error)
        stream=True,
    )


def forecast_version(decision):

    # Version of the model that made the predictions:
    # - newly registered in this run, or
    # - the current champion, when the candidate is the same model
    #   (identical validation score, e.g. when the script is re-run)
    if pd.notna(decision["new_version"]):
        return int(decision["new_version"])

    candidate = decision[f"candidate_{PRIMARY_METRIC}"]
    champion = decision[f"champion_{PRIMARY_METRIC}"]

    if (
        pd.notna(decision["champion_version"])
        and pd.notna(champion)
        and abs(candidate - champion) < 1e-9
    ):
        return int(decision["champion_version"])

    return None


def latest_forecasts(decisions):

    rows = []

    for _, row in decisions.iterrows():

        version = forecast_version(row)

        # Predictions in the files belong to a model that is not
        # in the registry, so they are not added to the ledger
        if version is None:
            continue

        path, is_forecast_time = PREDICTION_FILES[row["model_type"]]

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        column = f"pred_{row['metric']}"

        if column not in df.columns:
            continue

        df = df.dropna(subset=[column])

        if df.empty:
            continue

        last = df.iloc[-1]

        time = pd.to_datetime(last["timestamp"], utc=True)

        offset = pd.Timedelta(hours=FORECAST_HOURS)

        rows.append({
            "forecast_time": time if is_forecast_time else time - offset,
            "target_time": time + offset if is_forecast_time else time,
            "metric": row["metric"],
            "forecast": float(last[column]),
            "unit": UNITS[row["metric"]],
            "model_name": row["model_name"],
            "model_version": version,
        })

    if not rows:
        return pd.DataFrame()

    forecasts = pd.DataFrame(rows)

    # Microsecond precision (avoids the ns -> us precision warning)
    for col in ("forecast_time", "target_time"):
        forecasts[col] = pd.to_datetime(
            forecasts[col], utc=True
        ).astype("datetime64[us, UTC]")

    return forecasts


def add_forecasts(ledger, forecasts):

    if forecasts.empty:
        print("\nNo new forecasts for the ledger.")
        return

    # Same key (forecast time + metric) is updated, not duplicated
    ledger.insert(
        forecasts,
        write_options={"wait_for_job": True}
    )

    print(f"\nForecasts added to ledger: {len(forecasts)}")


def show_ledger(ledger):

    try:
        entries = ledger.read()
    except Exception as error:
        print(f"Default read failed ({type(error).__name__}), retrying with Hive...")
        entries = ledger.read(read_options={"use_hive": True})

    entries = entries.sort_values(
        ["forecast_time", "metric"]
    ).reset_index(drop=True)

    print(f"\nForecast Ledger ({len(entries)} rows):")
    print(entries.tail(10).to_string(index=False))


# ---------------------------------------
# Step 5: Save decision log
# ---------------------------------------

def save_log(decisions):

    os.makedirs("data/processed", exist_ok=True)

    # Every run is added, so version changes can be tracked
    decisions.to_csv(
        LOG_PATH,
        mode="a",
        header=not os.path.exists(LOG_PATH),
        index=False
    )

    print(f"Saved: {LOG_PATH}")


# ---------------------------------------
# Main Model Registry pipeline
# ---------------------------------------

def run_model_registry():

    print("\nStarting Model Registry...")

    project = connect_to_hopsworks()

    registry = project.get_model_registry()

    feature_store = project.get_feature_store()

    best = load_best_models()

    decisions = register_best_models(registry, best)

    print("\nChampion Decisions:")
    print(decisions.drop(columns="checked_at").round(3).to_string(index=False))

    show_registry(registry, best)

    ledger = create_forecast_ledger(feature_store)

    add_forecasts(ledger, latest_forecasts(decisions))

    show_ledger(ledger)

    save_log(decisions)

    print("\nModel Registry Completed!")

    return decisions


if __name__ == "__main__":

    run_model_registry()