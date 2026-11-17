import os
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from models.baseline_model import (
    METRICS,
    TIMEZONE,
    TRAIN_PATH,
    VALIDATION_PATH
)
from models.random_forest import load_split


# ---------------------------------------
# Configuration
# ---------------------------------------

MODEL_DIR = "models/saved_models"

PREDICTIONS_PATH = "data/processed/sarimax_validation_predictions.csv"
RESULTS_PATH = "data/processed/sarimax_results.csv"

# Forecast horizon: next 24 hours
FORECAST_HOURS = 24

# order = (p, d, q), seasonal_order = (P, D, Q, 24)
# Seasonal period 24 = daily cycle of hourly data.
# Kept small so that training stays fast on ~2 months of data.
ORDER = (1, 0, 1)
SEASONAL_ORDER = (1, 1, 0, 24)

# At least one week of hourly data is needed to learn a daily cycle
MIN_TRAIN_HOURS = 24 * 7


# ---------------------------------------
# Step 1: Prepare hourly time series
# ---------------------------------------

def prepare_series(train_df, validation_df, metric):

    # Train and validation rows are already in time order.
    # The test set is not used here.
    data = pd.concat([train_df, validation_df])

    data = data.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    ).set_index("timestamp")

    # Same local time without timezone (statsmodels needs a
    # simple hourly index)
    data.index = data.index.tz_localize(None)

    # Regular hourly series. Missing hours stay NaN, they are
    # not filled here.
    series = data[metric].resample("1h").mean().asfreq("1h")

    train_end = train_df["timestamp"].max().tz_localize(None).floor("1h")

    train_series = series[:train_end]
    validation_series = series[train_end + pd.Timedelta(hours=1):]

    return train_series, validation_series, series


# ---------------------------------------
# Step 2: Train SARIMAX
# ---------------------------------------

def train_sarimax(train_series):

    model = SARIMAX(
        train_series,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        # Avoids failures when the series is short or noisy
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        return model.fit(disp=False, maxiter=50)


# ---------------------------------------
# Step 3: 24-hour-ahead forecasts
# ---------------------------------------

def forecast_validation(fitted, validation_series, metric):

    # Walk-forward: forecast the next 24 hours, then add the
    # real values of those hours (parameters are not refitted)
    # and forecast the next day.
    results = fitted

    forecasts = []

    for start in range(0, len(validation_series), FORECAST_HOURS):

        block = validation_series.iloc[start:start + FORECAST_HOURS]

        predicted = results.get_forecast(
            steps=len(block)
        ).predicted_mean

        predicted.index = block.index

        forecasts.append(predicted)

        results = results.extend(block)

    forecast = pd.concat(forecasts)

    # Physical limits of each metric
    upper = 100 if metric == "humidity" else None

    return forecast.clip(lower=0, upper=upper)


# ---------------------------------------
# Step 4: Evaluate against the naive baseline
# ---------------------------------------

def evaluate_forecast(metric, actual, predicted, naive):

    data = pd.DataFrame({
        "actual": actual,
        "predicted": predicted,
        "naive": naive
    }).dropna()

    if len(data) < 2:
        print(f"{metric}: not enough validation data")
        return None

    mae = mean_absolute_error(data["actual"], data["predicted"])

    naive_mae = mean_absolute_error(data["actual"], data["naive"])

    return {
        "dataset": "validation",
        "metric": metric,
        "rows": len(data),
        "MAE": mae,
        "RMSE": np.sqrt(
            mean_squared_error(data["actual"], data["predicted"])
        ),
        "R2": r2_score(data["actual"], data["predicted"]),
        "naive_MAE": naive_mae,
        # Positive value means SARIMAX beats the baseline
        "MAE_improvement_%": (
            (naive_mae - mae) / naive_mae * 100
            if naive_mae > 0 else np.nan
        ),
    }


# ---------------------------------------
# Step 5: Train and forecast every metric
# ---------------------------------------

def run_metric(metric, train_df, validation_df):

    train_series, validation_series, series = prepare_series(
        train_df,
        validation_df,
        metric
    )

    available = train_series.notna().sum()

    if available < MIN_TRAIN_HOURS:
        print(f"{metric}: not enough training data ({available} hours)")
        return None

    if train_series.dropna().nunique() < 2:
        print(f"{metric}: no variation in training data, skipped")
        return None

    fitted = train_sarimax(train_series)

    forecast = forecast_validation(fitted, validation_series, metric)

    # Baseline: the value of the same hour on the previous day
    naive = series.shift(FORECAST_HOURS).reindex(validation_series.index)

    result = evaluate_forecast(
        metric,
        validation_series,
        forecast,
        naive
    )

    predictions = pd.DataFrame({
        f"actual_{metric}": validation_series,
        f"pred_{metric}": forecast
    })

    print(f"Trained model: {metric} ({available} hours)")

    return fitted, predictions, result


# ---------------------------------------
# Step 6: Save models, predictions and results
# ---------------------------------------

def save_outputs(models, predictions, results):

    os.makedirs(MODEL_DIR, exist_ok=True)

    os.makedirs("data/processed", exist_ok=True)

    for metric, fitted in models.items():

        fitted.save(f"{MODEL_DIR}/sarimax_{metric}.pkl")

    # timestamp = the hour that is being forecasted
    predictions.index = predictions.index.tz_localize(TIMEZONE)

    predictions.to_csv(
        PREDICTIONS_PATH,
        index_label="timestamp"
    )

    results.to_csv(RESULTS_PATH, index=False)

    print(f"\nModels saved: {MODEL_DIR}")
    print(f"Saved: {PREDICTIONS_PATH}")
    print(f"Saved: {RESULTS_PATH}")


# ---------------------------------------
# Main SARIMAX pipeline
# ---------------------------------------

def run_sarimax():

    print("\nStarting SARIMAX Models...")

    train_df = load_split(TRAIN_PATH)
    validation_df = load_split(VALIDATION_PATH)

    models = {}
    predictions = []
    results = []

    # Temperature and AQI first, then the other metrics
    ordered_metrics = ["temperature", "AQI"] + [
        metric for metric in METRICS
        if metric not in ("temperature", "AQI")
    ]

    print("\nTraining Models:")

    for metric in ordered_metrics:

        try:
            output = run_metric(metric, train_df, validation_df)

        except Exception as e:
            print(f"{metric}: SARIMAX failed - {e}")
            continue

        if output is None:
            continue

        fitted, metric_predictions, result = output

        models[metric] = fitted
        predictions.append(metric_predictions)

        if result is not None:
            results.append(result)

    if not models:
        print("\nNo SARIMAX model could be trained.")
        return None

    predictions = pd.concat(predictions, axis=1)

    results = pd.DataFrame(results)

    print("\nSARIMAX Validation Results:")
    print(results.round(3).to_string(index=False))

    save_outputs(models, predictions, results)

    print("\nSARIMAX Models Completed!")

    return models


if __name__ == "__main__":

    run_sarimax()