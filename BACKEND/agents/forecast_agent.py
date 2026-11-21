import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from feature_engineering.f_engineering import (
    create_time_features,
    create_rolling_features,
    create_change_features
)
from feature_store.hopswork_feature_store import (
    FEATURE_COLUMNS,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    TIME_COLUMN
)
from feature_store.model_registry import (
    MODEL_NAMES,
    UNITS,
    FORECAST_HOURS,
    connect_to_hopsworks,
    create_forecast_ledger,
    add_forecasts,
    get_champion
)
from models.baseline_model import METRICS, TIMEZONE
from agents.shared_state import get_logger, load_state, update_state

logger = get_logger("forecast_agent")


# ---------------------------------------
# Configuration
# ---------------------------------------

# Forecast is not reliable if the newest features are older
MAX_FEATURE_AGE_HOURS = 6

# LSTM input length and the minimum recent data for SARIMAX
LSTM_INPUT_HOURS = 48
SARIMAX_MIN_RECENT_HOURS = 24

# Name of the forecast in the shared state (alert agent keys)
STATE_KEYS = {metric: metric.lower() for metric in METRICS}


# ---------------------------------------
# Step 1: Latest features from Hopsworks
# ---------------------------------------

def load_latest_features(feature_store):

    feature_group = feature_store.get_feature_group(
        FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    history = feature_group.read()

    # Back to the column names used for training
    history = history.rename(
        columns={new: old for old, new in FEATURE_COLUMNS.items()}
    )

    history[TIME_COLUMN] = pd.to_datetime(
        history[TIME_COLUMN],
        utc=True
    ).dt.tz_convert(TIMEZONE)

    history = history.sort_values(
        TIME_COLUMN
    ).drop_duplicates(
        subset=[TIME_COLUMN],
        keep="last"
    ).reset_index(drop=True)

    # Same feature engineering as in training
    features = create_time_features(history.copy())
    features = create_rolling_features(features)
    features = create_change_features(features)

    latest = features.iloc[-1]

    age = (
        pd.Timestamp.now(tz=TIMEZONE) - latest[TIME_COLUMN]
    ) / pd.Timedelta(hours=1)

    if age > MAX_FEATURE_AGE_HOURS:
        logger.warning(
            f"Latest features are {age:.1f} hours old "
            f"(limit {MAX_FEATURE_AGE_HOURS}). Run the data pipeline first."
        )

    print(f"\nLatest features: {latest[TIME_COLUMN]} ({len(history)} rows)")

    return history, latest, age


def build_hourly(history):

    # Same hourly series as in the LSTM and SARIMAX training
    data = history.set_index(TIME_COLUMN)

    data.index = data.index.tz_localize(None)

    metrics = [col for col in METRICS if col in data.columns]

    hourly = data[metrics].resample("1h").mean()

    hourly["hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24)

    return hourly


# ---------------------------------------
# Step 2: Best (champion) models from Model Registry
# ---------------------------------------

def load_champion(registry, metric):

    champion = get_champion(registry, MODEL_NAMES[metric])

    if champion is None:
        raise LookupError(f"no model registered for {metric}")

    folder = champion.download()

    files = [
        os.path.join(root, name)
        for root, _, names in os.walk(folder)
        for name in names
    ]

    def find(condition):
        return next((path for path in files if condition(path)), None)

    return {
        "name": MODEL_NAMES[metric],
        "version": champion.version,
        "random_forest": find(
            lambda p: p.endswith(".joblib") and "scaler" not in p
        ),
        "sarimax": find(lambda p: p.endswith(".pkl")),
        "lstm": find(lambda p: p.endswith(".keras")),
        "scaler": find(lambda p: p.endswith("lstm_scaler.joblib")),
    }


def model_type(files):

    # The registered model contains one model file
    for kind in ("random_forest", "sarimax", "lstm"):

        if files[kind]:
            return kind

    raise LookupError("no model file found")


# ---------------------------------------
# Step 3: Forecast of each model type
# ---------------------------------------

def predict_random_forest(model_file, latest):

    pipeline = joblib.load(model_file)

    names = list(pipeline.feature_names_in_)

    missing = [name for name in names if name not in latest.index]

    if missing:
        logger.warning(f"Missing features (filled by imputer): {missing}")

    X = pd.DataFrame(
        [latest.reindex(names).astype(float)]
    )

    return float(pipeline.predict(X)[0])


def predict_lstm(model_file, scaler_file, hourly):

    from tensorflow import keras

    window = hourly.iloc[-LSTM_INPUT_HOURS:]

    if len(window) < LSTM_INPUT_HOURS or window.isna().any().any():
        raise ValueError(
            f"LSTM needs {LSTM_INPUT_HOURS} continuous hourly rows"
        )

    scaler = joblib.load(scaler_file)

    model = keras.models.load_model(model_file)

    scaled = scaler.transform(window)[np.newaxis, :, :]

    predicted = model.predict(scaled, verbose=0)[0]

    # The metrics are the first columns of the scaler
    count = len(predicted)

    values = predicted * scaler.scale_[:count] + scaler.mean_[:count]

    return dict(zip(hourly.columns[:count], values))


def predict_sarimax(model_file, hourly, metric):

    from statsmodels.tsa.statespace.sarimax import SARIMAXResults

    results = SARIMAXResults.load(model_file)

    train_end = results.model.data.row_labels[-1]

    latest_hour = hourly.index[-1]

    if latest_hour <= train_end:
        raise ValueError("newest data is not after the training period")

    # Hours since the end of training (missing hours stay NaN)
    recent = hourly[metric].reindex(
        pd.date_range(
            train_end + pd.Timedelta(hours=1),
            latest_hour,
            freq="1h"
        )
    )

    if recent.iloc[-48:].notna().sum() < SARIMAX_MIN_RECENT_HOURS:
        raise ValueError(
            f"SARIMAX needs {SARIMAX_MIN_RECENT_HOURS} real hours "
            "in the last 48 hours"
        )

    results = results.extend(recent)

    return float(
        results.get_forecast(steps=FORECAST_HOURS).predicted_mean.iloc[-1]
    )


def predict_naive(latest, history, metric):

    # Fallback: tomorrow will be the same as the latest value
    value = latest.get(metric)

    if pd.isna(value):
        recent = history[
            history[TIME_COLUMN] >= latest[TIME_COLUMN]
            - pd.Timedelta(hours=MAX_FEATURE_AGE_HOURS)
        ][metric].dropna()

        value = recent.iloc[-1] if len(recent) else None

    return None if value is None else float(value)


def limit_value(metric, value):

    # Physical limits of each metric
    upper = 100 if metric == "humidity" else None

    return float(np.clip(value, 0, upper))


def forecast_metric(registry, metric, latest, history, hourly):

    try:
        files = load_champion(registry, metric)

        kind = model_type(files)

        if kind == "random_forest":
            value = predict_random_forest(files["random_forest"], latest)

        elif kind == "lstm":
            value = predict_lstm(
                files["lstm"],
                files["scaler"],
                hourly
            )[metric]

        else:
            value = predict_sarimax(files["sarimax"], hourly, metric)

        return {
            "value": limit_value(metric, value),
            "model_name": files["name"],
            "model_version": int(files["version"]),
            "model_type": kind,
            "note": None,
        }

    except Exception as error:

        # One failed model must not stop the other metrics
        logger.error(f"{metric}: model forecast failed - {error}")

        value = predict_naive(latest, history, metric)

        return {
            "value": None if value is None else limit_value(metric, value),
            "model_name": "naive_baseline",
            "model_version": 0,
            "model_type": "naive_baseline",
            "note": f"fallback: {error}",
        }


# ---------------------------------------
# Step 4: Forecast Ledger
# ---------------------------------------

def save_to_ledger(feature_store, forecasts, forecast_time):

    rows = [
        {
            "forecast_time": forecast_time.tz_convert("UTC"),
            "target_time": (
                forecast_time + pd.Timedelta(hours=FORECAST_HOURS)
            ).tz_convert("UTC"),
            "metric": metric,
            "forecast": result["value"],
            "unit": UNITS[metric],
            "model_name": result["model_name"],
            "model_version": result["model_version"],
        }
        for metric, result in forecasts.items()
        if result["value"] is not None
    ]

    ledger = create_forecast_ledger(feature_store)

    add_forecasts(ledger, pd.DataFrame(rows))


# ---------------------------------------
# Main Forecast Agent
# ---------------------------------------

def run_forecast_agent(state=None):

    print("\nStarting Forecast Agent...")

    state = load_state() if state is None else state

    try:
        project = connect_to_hopsworks()

        feature_store = project.get_feature_store()

        registry = project.get_model_registry()

        history, latest, age = load_latest_features(feature_store)

        hourly = build_hourly(history)

        forecasts = {
            metric: forecast_metric(registry, metric, latest, history, hourly)
            for metric in METRICS
        }

        forecast_time = latest[TIME_COLUMN]

        try:
            save_to_ledger(feature_store, forecasts, forecast_time)
            ledger_saved = True

        except Exception as error:
            logger.error(f"Forecast ledger save failed - {error}")
            ledger_saved = False

    except Exception as error:

        # The agents must never crash the whole system
        logger.error(f"Forecast agent failed - {error}")

        return update_state(
            state,
            forecast_status="failed",
            forecast_error=str(error)
        )

    available = {
        STATE_KEYS[metric]: round(result["value"], 2)
        for metric, result in forecasts.items()
        if result["value"] is not None
    }

    fallbacks = [
        metric for metric, result in forecasts.items()
        if result["model_type"] == "naive_baseline"
    ]

    status = "ok" if len(available) == len(METRICS) else "partial"

    print("\nForecast (24 hours ahead):")

    for metric, result in forecasts.items():

        print(
            f"{metric:<14}{result['value']}  "
            f"({result['model_name']} v{result['model_version']})"
        )

    return update_state(
        state,
        forecast=available,
        forecast_time=forecast_time.tz_convert("UTC").isoformat(),
        target_time=(
            forecast_time + pd.Timedelta(hours=FORECAST_HOURS)
        ).tz_convert("UTC").isoformat(),
        forecast_models={
            metric: {
                "model_name": result["model_name"],
                "model_version": result["model_version"],
                "model_type": result["model_type"],
                "note": result["note"],
            }
            for metric, result in forecasts.items()
        },
        fallback_metrics=fallbacks,
        features_age_hours=round(float(age), 1),
        ledger_saved=ledger_saved,
        forecast_status=status,
        forecast_error=None
    )


if __name__ == "__main__":

    run_forecast_agent()