import pandas as pd
import numpy as np
import os


# ---------------------------------------
# Configuration
# ---------------------------------------

INPUT_PATH = "data/processed/clean_weather.csv"

OUTPUT_PATH = "data/processed/training_weather.csv"

TIMEZONE = "Asia/Karachi"

TARGET_HOURS = 24

# Rate of change is only valid between close records
MAX_CHANGE_GAP = pd.Timedelta("3h")


# ---------------------------------------
# Step 1: Load dataset
# ---------------------------------------

def load_data():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("\nOriginal Dataset:")
    print(df.head())

    return df


# ---------------------------------------
# Step 2: Prepare timestamps
# ---------------------------------------

def prepare_data(df):

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert(TIMEZONE)

    df = df.dropna(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).drop_duplicates(
        subset=["timestamp"],
        keep="last"
    ).reset_index(drop=True)

    return df


# ---------------------------------------
# Step 3: Time-based features
# ---------------------------------------

def create_time_features(df):

    df["hour"] = df["timestamp"].dt.hour

    df["day"] = df["timestamp"].dt.day

    df["month"] = df["timestamp"].dt.month

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    # Cyclic time features
    # These help ML understand repeating patterns.

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df


# ---------------------------------------
# Step 4: Rolling averages
# ---------------------------------------

def create_rolling_features(df):

    columns = [
        "temperature",
        "humidity",
        "AQI",
        "wind_speed"
    ]

    df = df.sort_values(
        "timestamp"
    ).set_index("timestamp")

    for col in columns:

        if col in df.columns:

            df[f"{col}_rolling_3h"] = (
                df[col].rolling(
                    "3h",
                    min_periods=3
                ).mean()
            )

    return df.reset_index()


# ---------------------------------------
# Step 5: Rate of change
# ---------------------------------------

def create_change_features(df):

    columns = [
        "temperature",
        "humidity",
        "AQI",
        "wind_speed"
    ]

    df = df.sort_values(
        "timestamp"
    ).set_index("timestamp")

    # Time between each record and the previous one
    gap = df.index.to_series().diff()

    for col in columns:

        if col in df.columns:

            # Change across a long gap would not be a real rate of change
            df[f"{col}_change"] = df[col].diff().where(
                gap <= MAX_CHANGE_GAP
            )

    return df.reset_index()


# ---------------------------------------
# Step 6: Create 24-hour targets
# ---------------------------------------

def create_targets(df):

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    target_columns = [
        "temperature",
        "humidity",
        "precipitation",
        "uv_index",
        "AQI",
        "wind_speed"
    ]

    # Match the actual timestamp exactly 24 hours ahead.
    # This is safer than shift(-24) when timestamps
    # may have missing or irregular hourly records.

    future = df.set_index(
        "timestamp"
    )

    for col in target_columns:

        if col in df.columns:

            future_values = future[col].reindex(
                df["timestamp"] +
                pd.Timedelta(hours=TARGET_HOURS)
            )

            df[f"target_{col}"] = (
                future_values.to_numpy()
            )

    return df


# ---------------------------------------
# Step 7: Check hourly data frequency
# ---------------------------------------

def check_frequency(df):

    differences = (
        df["timestamp"].sort_values().diff()
    )

    hourly_records = (
        differences == pd.Timedelta(hours=1)
    ).sum()

    print("\nHourly Data Check:")
    print("Total records:", len(df))
    print("Exact 1-hour intervals:", hourly_records)

    if hourly_records == 0:
        print(
            "WARNING: No consecutive hourly records found."
        )

    return df


# ---------------------------------------
# Step 8: Save training dataset
# ---------------------------------------

def save_dataset(df):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nTraining Dataset Preview:")
    print(df.tail())

    print("\nDataset Shape:", df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print(
        f"\nTraining dataset saved: {OUTPUT_PATH}"
    )


# ---------------------------------------
# Main Feature Engineering Pipeline
# ---------------------------------------

def create_features():

    print("\nStarting Feature Engineering...")

    # Load
    df = load_data()

    # Prepare timestamps
    df = prepare_data(df)

    # Check data frequency
    df = check_frequency(df)

    # Create features
    df = create_time_features(df)

    df = create_rolling_features(df)

    df = create_change_features(df)

    # Create targets
    df = create_targets(df)

    # Save
    save_dataset(df)

    print("\nFeature Engineering Completed!")

    return df


if __name__ == "__main__":

    create_features()