import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

MODEL_PATH = f"{MODEL_DIR}/lstm_model.keras"
SCALER_PATH = f"{MODEL_DIR}/lstm_scaler.joblib"

PREDICTIONS_PATH = "data/processed/lstm_validation_predictions.csv"
RESULTS_PATH = "data/processed/lstm_results.csv"

# Input: previous 48 hours, output: exactly 24 hours ahead
INPUT_HOURS = 48
FORECAST_HOURS = 24

EPOCHS = 50
BATCH_SIZE = 32

# Minimum training sequences needed for a useful prototype
MIN_SEQUENCES = 100


# ---------------------------------------
# Step 1: Prepare hourly data
# ---------------------------------------

def prepare_hourly_data(train_df, validation_df):

    # Only train and validation data is used here.
    # The test set stays untouched.
    data = pd.concat([train_df, validation_df])

    data = data.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    ).set_index("timestamp")

    # Same local time without timezone
    data.index = data.index.tz_localize(None)

    metrics = [col for col in METRICS if col in data.columns]

    # Regular hourly rows. Missing hours stay NaN.
    hourly = data[metrics].resample("1h").mean()

    # Time of day as a circle (23:00 is close to 00:00)
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24)

    train_end = train_df["timestamp"].max().tz_localize(None).floor("1h")

    return hourly, metrics, train_end


# ---------------------------------------
# Step 2: Scale numerical features
# ---------------------------------------

def scale_features(hourly, train_end):

    # Scaler learns mean and standard deviation from the
    # training period only, so no validation information leaks in.
    scaler = StandardScaler()

    scaler.fit(hourly[:train_end])

    scaled = pd.DataFrame(
        scaler.transform(hourly),
        index=hourly.index,
        columns=hourly.columns
    )

    return scaled, scaler


# ---------------------------------------
# Step 3: Create sequences
# ---------------------------------------

def create_sequences(scaled, metrics, train_end):

    values = scaled.to_numpy()
    times = scaled.index

    X, y, origins, target_times = [], [], [], []

    # Sequence ending at hour t:
    #   input  = hours t-47 ... t
    #   target = metrics at exactly t + 24 hours
    for i in range(INPUT_HOURS - 1, len(values) - FORECAST_HOURS):

        window = values[i - INPUT_HOURS + 1:i + 1]

        target = values[i + FORECAST_HOURS, :len(metrics)]

        # LSTM cannot use missing values, such sequences are skipped
        if np.isnan(window).any() or np.isnan(target).any():
            continue

        X.append(window)
        y.append(target)
        origins.append(times[i])
        target_times.append(times[i + FORECAST_HOURS])

    X = np.array(X)
    y = np.array(y)
    origins = pd.DatetimeIndex(origins)
    target_times = pd.DatetimeIndex(target_times)

    # Train: the target must also be inside the training period.
    # Otherwise the last training sequences would already contain
    # validation values as their answer.
    train_mask = (origins <= train_end) & (target_times <= train_end)

    # Validation: forecasts are made after the training period.
    # Inputs may use the past hours of the training period.
    validation_mask = origins > train_end

    return {
        "train": (X[train_mask], y[train_mask], origins[train_mask]),
        "validation": (
            X[validation_mask],
            y[validation_mask],
            origins[validation_mask]
        ),
    }


# ---------------------------------------
# Step 4: LSTM architecture
# ---------------------------------------

def build_model(n_features, n_outputs):

    model = keras.Sequential([
        layers.Input(shape=(INPUT_HOURS, n_features)),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        # One output for each metric, 24 hours ahead
        layers.Dense(n_outputs)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    return model


# ---------------------------------------
# Step 5: Train model
# ---------------------------------------

def train_model(model, train_data, validation_data):

    X_train, y_train, _ = train_data
    X_validation, y_validation, _ = validation_data

    # Stops when validation loss stops improving
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        shuffle=False,
        verbose=2
    )

    return history


# ---------------------------------------
# Step 6: Predictions on validation data
# ---------------------------------------

def inverse_scale(values, scaler, metrics):

    # The metrics are the first columns of the scaler
    count = len(metrics)

    return values * scaler.scale_[:count] + scaler.mean_[:count]


def predict_validation(model, validation_data, scaler, metrics):

    X, y, origins = validation_data

    predicted = inverse_scale(
        model.predict(X, verbose=0),
        scaler,
        metrics
    )

    actual = inverse_scale(y, scaler, metrics)

    # Baseline: value of the metric at the forecast time
    naive = inverse_scale(X[:, -1, :len(metrics)], scaler, metrics)

    predictions = pd.DataFrame(index=origins)

    for k, metric in enumerate(metrics):

        # Physical limits of each metric
        upper = 100 if metric == "humidity" else None

        predictions[f"target_{metric}"] = actual[:, k]

        predictions[f"pred_{metric}"] = pd.Series(
            predicted[:, k],
            index=origins
        ).clip(lower=0, upper=upper)

        predictions[f"naive_{metric}"] = naive[:, k]

    return predictions


# ---------------------------------------
# Step 7: Evaluate against the naive baseline
# ---------------------------------------

def evaluate_model(predictions, metrics):

    results = []

    for metric in metrics:

        actual = predictions[f"target_{metric}"]
        predicted = predictions[f"pred_{metric}"]
        naive = predictions[f"naive_{metric}"]

        mae = mean_absolute_error(actual, predicted)

        naive_mae = mean_absolute_error(actual, naive)

        results.append({
            "dataset": "validation",
            "metric": metric,
            "rows": len(predictions),
            "MAE": mae,
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted),
            "naive_MAE": naive_mae,
            # Positive value means LSTM beats the baseline
            "MAE_improvement_%": (
                (naive_mae - mae) / naive_mae * 100
                if naive_mae > 0 else np.nan
            ),
        })

    return pd.DataFrame(results)


# ---------------------------------------
# Step 8: Save model, predictions and results
# ---------------------------------------

def save_outputs(model, scaler, predictions, results):

    os.makedirs(MODEL_DIR, exist_ok=True)

    os.makedirs("data/processed", exist_ok=True)

    model.save(MODEL_PATH)

    # Needed to scale new data the same way later
    joblib.dump(scaler, SCALER_PATH)

    # timestamp = last input hour (forecast is for 24 hours later)
    predictions.index = predictions.index.tz_localize(TIMEZONE)

    predictions.to_csv(
        PREDICTIONS_PATH,
        index_label="timestamp"
    )

    results.to_csv(RESULTS_PATH, index=False)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print(f"Saved: {PREDICTIONS_PATH}")
    print(f"Saved: {RESULTS_PATH}")


# ---------------------------------------
# Main LSTM pipeline
# ---------------------------------------

def run_lstm():

    print("\nStarting LSTM Model...")

    keras.utils.set_random_seed(42)

    train_df = load_split(TRAIN_PATH)
    validation_df = load_split(VALIDATION_PATH)

    hourly, metrics, train_end = prepare_hourly_data(
        train_df,
        validation_df
    )

    scaled, scaler = scale_features(hourly, train_end)

    sequences = create_sequences(scaled, metrics, train_end)

    train_data = sequences["train"]
    validation_data = sequences["validation"]

    print("\nSequences (48 hours input -> 24 hours ahead):")
    print("Train:", train_data[0].shape)
    print("Validation:", validation_data[0].shape)

    if len(train_data[0]) < MIN_SEQUENCES or len(validation_data[0]) == 0:
        print("\nNot enough data to train the LSTM.")
        return None

    model = build_model(
        n_features=train_data[0].shape[2],
        n_outputs=len(metrics)
    )

    model.summary()

    print("\nTraining LSTM:")

    train_model(model, train_data, validation_data)

    predictions = predict_validation(
        model,
        validation_data,
        scaler,
        metrics
    )

    results = evaluate_model(predictions, metrics)

    print("\nLSTM Validation Results:")
    print(results.round(3).to_string(index=False))

    save_outputs(model, scaler, predictions, results)

    print("\nLSTM Model Completed!")

    return model


if __name__ == "__main__":

    run_lstm()