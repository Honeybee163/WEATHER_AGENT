import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ---------------------------------------
# Configuration
# ---------------------------------------

INPUT_PATH = "data/processed/training_weather.csv"

TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"
RESULTS_PATH = "data/processed/baseline_results.csv"

TIMEZONE = "Asia/Karachi"

# Chronological split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

# Metrics predicted 24 hours ahead
METRICS = [
    "temperature",
    "humidity",
    "precipitation",
    "uv_index",
    "AQI",
    "wind_speed"
]


# ---------------------------------------
# Step 1: Load dataset
# ---------------------------------------

def load_data():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert(TIMEZONE)

    df = df.dropna(
        subset=["timestamp"]
    ).sort_values("timestamp").reset_index(drop=True)

    print("\nTraining Dataset Loaded:", df.shape)

    return df


# ---------------------------------------
# Step 2: Naive baseline
# ---------------------------------------

def add_naive_baseline(df):

    # Naive rule: tomorrow will be the same as now.
    # target_<metric> is the value 24 hours ahead, so the
    # prediction for it is the current value of the metric.
    # This is the same idea as df["temperature"].shift(24),
    # but matched on the exact timestamp of each row.
    for metric in METRICS:

        if metric in df.columns:
            df[f"naive_{metric}"] = df[metric]

    return df


# ---------------------------------------
# Step 3: Separate features and targets
# ---------------------------------------

def get_feature_and_target_columns(df):

    target_columns = [
        col for col in df.columns
        if col.startswith("target_")
    ]

    # Only numeric columns can be model features.
    # Targets, naive predictions and empty columns are excluded.
    feature_columns = [
        col for col in df.select_dtypes(include="number").columns
        if not col.startswith("target_")
        and not col.startswith("naive_")
        and df[col].notna().any()
    ]

    print("\nFeatures:", len(feature_columns))
    print("Targets:", target_columns)

    return feature_columns, target_columns


# ---------------------------------------
# Step 4: Chronological split
# ---------------------------------------

def chronological_split(df, target_columns):

    # Rows without any target (latest 24 hours and live
    # records) cannot be used for training or testing.
    df = df.dropna(
        subset=target_columns,
        how="all"
    ).reset_index(drop=True)

    total = len(df)

    train_end = int(total * TRAIN_RATIO)
    validation_end = int(total * (TRAIN_RATIO + VALIDATION_RATIO))

    # No shuffling: past data trains, future data tests
    train_df = df.iloc[:train_end]
    validation_df = df.iloc[train_end:validation_end]
    test_df = df.iloc[validation_end:]

    for name, part in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df)
    ]:
        print(
            f"{name:<11}{len(part):>6} rows  "
            f"{part['timestamp'].min():%Y-%m-%d %H:%M} -> "
            f"{part['timestamp'].max():%Y-%m-%d %H:%M}"
        )

    return train_df, validation_df, test_df


# ---------------------------------------
# Step 5: Evaluate baseline (MAE, RMSE, R2)
# ---------------------------------------

def evaluate_baseline(df, dataset_name):

    results = []

    for metric in METRICS:

        target = f"target_{metric}"
        naive = f"naive_{metric}"

        if target not in df.columns or naive not in df.columns:
            continue

        # Use only rows where both values are available
        valid = df[[target, naive]].dropna()

        if len(valid) < 2:
            print(f"{dataset_name} - {metric}: not enough data")
            continue

        actual = valid[target]
        predicted = valid[naive]

        results.append({
            "dataset": dataset_name,
            "metric": metric,
            "rows": len(valid),
            # Average size of the error
            "MAE": mean_absolute_error(actual, predicted),
            # Like MAE, but big errors are punished more
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            # 1 = perfect, 0 = same as predicting the average,
            # below 0 = worse than predicting the average
            "R2": r2_score(actual, predicted),
        })

    return pd.DataFrame(results)


# ---------------------------------------
# Step 6: Save datasets and results
# ---------------------------------------

def save_outputs(train_df, validation_df, test_df, results):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    validation_df.to_csv(VALIDATION_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    results.to_csv(RESULTS_PATH, index=False)

    print(f"\nSaved: {TRAIN_PATH}")
    print(f"Saved: {VALIDATION_PATH}")
    print(f"Saved: {TEST_PATH}")
    print(f"Saved: {RESULTS_PATH}")


# ---------------------------------------
# Main baseline pipeline
# ---------------------------------------

def run_baseline():

    print("\nStarting Baseline Model...")

    df = load_data()

    df = add_naive_baseline(df)

    feature_columns, target_columns = get_feature_and_target_columns(df)

    print("\nChronological Split:")

    train_df, validation_df, test_df = chronological_split(
        df,
        target_columns
    )

    # Baseline needs no training, so it is checked on
    # validation and test data
    results = pd.concat([
        evaluate_baseline(validation_df, "validation"),
        evaluate_baseline(test_df, "test")
    ], ignore_index=True)

    print("\nBaseline Results:")
    print(results.round(3).to_string(index=False))

    save_outputs(train_df, validation_df, test_df, results)

    print("\nBaseline Model Completed!")

    return results


if __name__ == "__main__":

    run_baseline()