import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from models.baseline_model import (
    METRICS,
    TIMEZONE,
    TRAIN_PATH,
    VALIDATION_PATH,
    get_feature_and_target_columns
)


# ---------------------------------------
# Configuration
# ---------------------------------------

MODEL_DIR = "models/saved_models"

PREDICTIONS_PATH = "data/processed/rf_validation_predictions.csv"
RESULTS_PATH = "data/processed/rf_results.csv"

# Minimum rows needed to train one model
MIN_TRAIN_ROWS = 10


# ---------------------------------------
# Step 1: Load train and validation data
# ---------------------------------------

def load_split(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path} "
            "(run models.baseline_model first)"
        )

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    ).dt.tz_convert(TIMEZONE)

    return df


# ---------------------------------------
# Step 2: Create Random Forest model
# ---------------------------------------

def create_model():

    # Imputer fills missing feature values (for example the first
    # rolling averages) with the median of the training data.
    # Random Forest itself cannot handle missing values in all
    # scikit-learn versions.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )


# ---------------------------------------
# Step 3: Train one model per metric
# ---------------------------------------

def train_models(train_df, feature_columns):

    models = {}

    for metric in METRICS:

        target = f"target_{metric}"

        if target not in train_df.columns:
            continue

        # Only rows where the 24-hour target is known
        rows = train_df.dropna(subset=[target])

        if len(rows) < MIN_TRAIN_ROWS:
            print(f"{metric}: not enough training data")
            continue

        model = create_model()

        model.fit(
            rows[feature_columns],
            rows[target]
        )

        models[metric] = model

        print(f"Trained model: {metric} ({len(rows)} rows)")

    return models


# ---------------------------------------
# Step 4: Predictions on validation data
# ---------------------------------------

def predict_validation(models, validation_df, feature_columns):

    predictions = validation_df[["timestamp"]].copy()

    for metric, model in models.items():

        predictions[f"target_{metric}"] = (
            validation_df[f"target_{metric}"]
        )

        predictions[f"pred_{metric}"] = model.predict(
            validation_df[feature_columns]
        )

    return predictions


# ---------------------------------------
# Step 5: Evaluate against the naive baseline
# ---------------------------------------

def evaluate_models(predictions, validation_df):

    results = []

    for metric in METRICS:

        if f"pred_{metric}" not in predictions.columns:
            continue

        data = pd.DataFrame({
            "actual": predictions[f"target_{metric}"],
            "predicted": predictions[f"pred_{metric}"],
            "naive": validation_df[f"naive_{metric}"]
        }).dropna()

        if len(data) < 2:
            print(f"{metric}: not enough validation data")
            continue

        mae = mean_absolute_error(data["actual"], data["predicted"])

        naive_mae = mean_absolute_error(data["actual"], data["naive"])

        results.append({
            "dataset": "validation",
            "metric": metric,
            "rows": len(data),
            "MAE": mae,
            "RMSE": np.sqrt(
                mean_squared_error(data["actual"], data["predicted"])
            ),
            "R2": r2_score(data["actual"], data["predicted"]),
            "naive_MAE": naive_mae,
            # Positive value means Random Forest beats the baseline
            "MAE_improvement_%": (
                (naive_mae - mae) / naive_mae * 100
                if naive_mae > 0 else np.nan
            ),
        })

    return pd.DataFrame(results)


# ---------------------------------------
# Step 6: Save models, predictions and results
# ---------------------------------------

def save_outputs(models, predictions, results):

    os.makedirs(MODEL_DIR, exist_ok=True)

    os.makedirs("data/processed", exist_ok=True)

    for metric, model in models.items():

        joblib.dump(
            model,
            f"{MODEL_DIR}/rf_{metric}.joblib"
        )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    results.to_csv(RESULTS_PATH, index=False)

    print(f"\nModels saved: {MODEL_DIR}")
    print(f"Saved: {PREDICTIONS_PATH}")
    print(f"Saved: {RESULTS_PATH}")


# ---------------------------------------
# Main Random Forest pipeline
# ---------------------------------------

def run_random_forest():

    print("\nStarting Random Forest Models...")

    train_df = load_split(TRAIN_PATH)
    validation_df = load_split(VALIDATION_PATH)

    feature_columns, _ = get_feature_and_target_columns(train_df)

    print("\nTraining Models:")

    models = train_models(train_df, feature_columns)

    predictions = predict_validation(
        models,
        validation_df,
        feature_columns
    )

    results = evaluate_models(predictions, validation_df)

    print("\nRandom Forest Validation Results:")
    print(results.round(3).to_string(index=False))

    save_outputs(models, predictions, results)

    print("\nRandom Forest Models Completed!")

    return models


if __name__ == "__main__":

    run_random_forest()