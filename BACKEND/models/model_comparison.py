import os
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

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
from models.random_forest import load_split


# ---------------------------------------
# Configuration
# ---------------------------------------

MODEL_DIR = "models/saved_models"
BEST_MODEL_DIR = "models/best_models"

SHAP_DIR = "data/processed/shap_plots"

# Prediction files of every model
RF_PREDICTIONS_PATH = "data/processed/rf_validation_predictions.csv"
SARIMAX_PREDICTIONS_PATH = "data/processed/sarimax_validation_predictions.csv"
LSTM_PREDICTIONS_PATH = "data/processed/lstm_validation_predictions.csv"

COMPARISON_PATH = "data/processed/model_comparison.csv"
BEST_MODELS_PATH = "data/processed/best_models.csv"
IMPORTANCE_PATH = "data/processed/shap_feature_importance.csv"
REPORT_PATH = "data/processed/model_comparison_report.md"

FORECAST_HOURS = 24

# Error used to choose the best model of each metric
SELECT_BY = "MAE"

# Rows used for SHAP (keeps it fast)
SHAP_SAMPLE_ROWS = 300


# ---------------------------------------
# Step 1: Load predictions of all models
# ---------------------------------------

def read_predictions(path, is_origin_time):

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    forecast_time = pd.to_datetime(
        df["timestamp"],
        utc=True
    ).dt.tz_convert(TIMEZONE)

    # Baseline, Random Forest and LSTM save the hour when the
    # forecast is made. The forecast is for 24 hours later.
    if is_origin_time:
        forecast_time = forecast_time + pd.Timedelta(hours=FORECAST_HOURS)

    df.index = forecast_time

    return df.drop(columns="timestamp")


def load_model_predictions():

    # Validation file has the real values and the naive baseline
    validation = read_predictions(VALIDATION_PATH, True)

    models = {
        "random_forest": read_predictions(RF_PREDICTIONS_PATH, True),
        "sarimax": read_predictions(SARIMAX_PREDICTIONS_PATH, False),
        "lstm": read_predictions(LSTM_PREDICTIONS_PATH, True),
    }

    for name, df in models.items():

        if df is None:
            print(f"{name}: predictions not found, skipped")

    return validation, {
        name: df for name, df in models.items() if df is not None
    }


# ---------------------------------------
# Step 2: Comparison table with naive baseline
# ---------------------------------------

def compare_models(validation, models):

    results = []

    for metric in METRICS:

        columns = {
            "actual": validation[f"target_{metric}"],
            "naive_baseline": validation[f"naive_{metric}"],
        }

        for name, df in models.items():

            if f"pred_{metric}" in df.columns:
                columns[name] = df[f"pred_{metric}"]

        # Only hours that every model forecasted, so that all
        # models are compared on exactly the same rows
        table = pd.concat(columns, axis=1, sort=True).dropna()

        if len(table) < 2:
            print(f"{metric}: not enough common rows")
            continue

        naive_mae = mean_absolute_error(
            table["actual"],
            table["naive_baseline"]
        )

        for name in table.columns.drop("actual"):

            mae = mean_absolute_error(table["actual"], table[name])

            results.append({
                "metric": metric,
                "model": name,
                "rows": len(table),
                "MAE": mae,
                "RMSE": np.sqrt(
                    mean_squared_error(table["actual"], table[name])
                ),
                "R2": r2_score(table["actual"], table[name]),
                # Positive value means better than the baseline
                "MAE_vs_naive_%": (
                    (naive_mae - mae) / naive_mae * 100
                    if naive_mae > 0 else np.nan
                ),
            })

    return pd.DataFrame(results)


# ---------------------------------------
# Step 3: Best validation model of each metric
# ---------------------------------------

def select_best_models(comparison):

    # The naive baseline is a reference, not a model to deploy
    candidates = comparison[comparison["model"] != "naive_baseline"]

    best = candidates.loc[
        candidates.groupby("metric")[SELECT_BY].idxmin()
    ].copy()

    best["beats_naive"] = best["MAE_vs_naive_%"] > 0

    # Keep the order of METRICS
    best["order"] = best["metric"].map(METRICS.index)

    return best.sort_values("order").drop(
        columns="order"
    ).reset_index(drop=True)


# ---------------------------------------
# Step 4 and 5: SHAP for Random Forest
# ---------------------------------------

def run_shap(train_df, validation_df):

    os.makedirs(SHAP_DIR, exist_ok=True)

    feature_columns, _ = get_feature_and_target_columns(train_df)

    sample = validation_df[feature_columns].sample(
        n=min(SHAP_SAMPLE_ROWS, len(validation_df)),
        random_state=42
    )

    importance = []

    for metric in METRICS:

        path = f"{MODEL_DIR}/rf_{metric}.joblib"

        if not os.path.exists(path):
            print(f"{metric}: Random Forest model not found, SHAP skipped")
            continue

        pipeline = joblib.load(path)

        # Same missing value handling as in training
        X = pd.DataFrame(
            pipeline[:-1].transform(sample),
            columns=feature_columns,
            index=sample.index
        )

        shap_values = shap.TreeExplainer(
            pipeline[-1]
        ).shap_values(X)

        # Summary plot
        shap.summary_plot(shap_values, X, show=False, max_display=15)

        plt.title(f"SHAP Summary - {metric}")

        plt.savefig(
            f"{SHAP_DIR}/shap_summary_{metric}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        # Feature importance plot
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            show=False,
            max_display=15
        )

        plt.title(f"SHAP Feature Importance - {metric}")

        plt.savefig(
            f"{SHAP_DIR}/shap_importance_{metric}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        # Average effect of each feature on the forecast
        metric_importance = pd.DataFrame({
            "metric": metric,
            "feature": feature_columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        importance.append(metric_importance)

        print(
            f"SHAP done: {metric} "
            f"(top feature: {metric_importance.iloc[0]['feature']})"
        )

    if not importance:
        return pd.DataFrame()

    return pd.concat(importance, ignore_index=True)


# ---------------------------------------
# Step 6: Save best models locally
# ---------------------------------------

def save_best_models(best):

    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    # Model file of each model type
    sources = {
        "random_forest": lambda m: (f"{MODEL_DIR}/rf_{m}.joblib", "joblib"),
        "sarimax": lambda m: (f"{MODEL_DIR}/sarimax_{m}.pkl", "pkl"),
        "lstm": lambda m: (f"{MODEL_DIR}/lstm_model.keras", "keras"),
    }

    saved_paths = []

    for _, row in best.iterrows():

        source, extension = sources[row["model"]](row["metric"])

        if not os.path.exists(source):
            print(f"{row['metric']}: model file not found: {source}")
            saved_paths.append(None)
            continue

        target = f"{BEST_MODEL_DIR}/{row['metric']}_{row['model']}.{extension}"

        shutil.copy(source, target)

        saved_paths.append(target)

    best["model_file"] = saved_paths

    # The LSTM needs its scaler to work
    if (best["model"] == "lstm").any():

        scaler = f"{MODEL_DIR}/lstm_scaler.joblib"

        if os.path.exists(scaler):
            shutil.copy(scaler, f"{BEST_MODEL_DIR}/lstm_scaler.joblib")

    return best


# ---------------------------------------
# Step 7: Comparison report
# ---------------------------------------

def markdown_table(df):

    header = "| " + " | ".join(df.columns) + " |"

    divider = "| " + " | ".join("---" for _ in df.columns) + " |"

    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.round(3).itertuples(index=False)
    ]

    return "\n".join([header, divider] + rows)


def write_report(comparison, best, importance):

    lines = [
        "# Model Comparison Report",
        "",
        "All models are compared on the same validation hours, "
        "24 hours ahead. The test set is not used.",
        "",
        "## Best model of each metric "
        f"(lowest validation {SELECT_BY})",
        "",
        markdown_table(
            best.drop(columns=["model_file"], errors="ignore")
        ),
        "",
    ]

    for metric in METRICS:

        part = comparison[comparison["metric"] == metric]

        if part.empty:
            continue

        lines += [
            f"## {metric}",
            "",
            markdown_table(part.drop(columns="metric")),
            "",
        ]

    if not importance.empty:

        lines += ["## Top 3 SHAP features (Random Forest)", ""]

        for metric, part in importance.groupby("metric", sort=False):

            top = ", ".join(part["feature"].head(3))

            lines.append(f"- {metric}: {top}")

        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Positive MAE_vs_naive_% means the model is better "
        "than the naive baseline.",
        "- SARIMAX forecasts the next 24 hours in blocks, so its "
        "forecast horizon is 1 to 24 hours. This is slightly "
        "easier than the exact 24 hours ahead of the other models.",
        "- precipitation is mostly zero, so R2 can be misleading. "
        "Use MAE for it.",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ---------------------------------------
# Main comparison pipeline
# ---------------------------------------

def run_comparison():

    print("\nStarting Model Comparison...")

    train_df = load_split(TRAIN_PATH)
    validation_df = load_split(VALIDATION_PATH)

    validation, models = load_model_predictions()

    comparison = compare_models(validation, models)

    print("\nModel Comparison:")
    print(comparison.round(3).to_string(index=False))

    best = select_best_models(comparison)

    print("\nBest Model of Each Metric:")
    print(best.round(3).to_string(index=False))

    print("\nSHAP for Random Forest:")

    importance = run_shap(train_df, validation_df)

    best = save_best_models(best)

    os.makedirs("data/processed", exist_ok=True)

    comparison.to_csv(COMPARISON_PATH, index=False)
    best.to_csv(BEST_MODELS_PATH, index=False)

    if not importance.empty:
        importance.to_csv(IMPORTANCE_PATH, index=False)

    write_report(comparison, best, importance)

    print(f"\nSaved: {COMPARISON_PATH}")
    print(f"Saved: {BEST_MODELS_PATH}")
    print(f"Saved: {IMPORTANCE_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"SHAP plots: {SHAP_DIR}")
    print(f"Best models: {BEST_MODEL_DIR}")

    print("\nModel Comparison Completed!")

    return best


if __name__ == "__main__":

    run_comparison()