import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------
# Configuration
# ---------------------------------------

INPUT_PATH = "data/processed/training_weather.csv"

FEATURE_GROUP_NAME = "weather_features"

# Version 2 already exists with the old schema (without pollutant columns).
# Use a NEW version so Hopsworks creates the updated schema.
FEATURE_GROUP_VERSION = 3

# Processed CSV column -> Hopsworks feature name
FEATURE_COLUMNS = {
    "timestamp": "timestamp",
    "temperature": "temperature",
    "humidity": "humidity",
    "precipitation": "precipitation",
    "rain_probability": "rain_probability",
    "AQI": "aqi",
    "uv_index": "uv_index",
    "wind_speed": "wind_speed",
    "wind_direction": "wind_direction",
    "PM2.5": "pm25",
    "PM10": "pm10",
    "O3": "o3",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "hour": "hour",
    "month": "month",
    "temperature_rolling_3h": "temp_rolling_3h",
}

TIME_COLUMN = "timestamp"
INTEGER_COLUMNS = ["hour", "month"]


# ---------------------------------------
# Step 1: Connect to Hopsworks
# ---------------------------------------

def connect_to_hopsworks():
    import hopsworks

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    if not api_key or not project_name:
        raise ValueError(
            "HOPSWORKS_API_KEY and HOPSWORKS_PROJECT "
            "must be set in the .env file."
        )

    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key,
    )

    print(f"Connected to Hopsworks project: {project.name}")
    return project.get_feature_store()


# ---------------------------------------
# Step 2: Load and prepare processed data
# ---------------------------------------

def load_features():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    raw_df = pd.read_csv(INPUT_PATH)

    # Check that all expected CSV columns are present before selecting them.
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in raw_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "These required columns are missing from the CSV: "
            + ", ".join(missing_columns)
        )

    df = raw_df[list(FEATURE_COLUMNS)].rename(columns=FEATURE_COLUMNS)

    # Hopsworks stores event time in UTC.
    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(subset=[TIME_COLUMN])
        .drop_duplicates(subset=[TIME_COLUMN], keep="last")
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )

    # Convert feature columns to consistent numeric types.
    for column in df.columns.drop(TIME_COLUMN):
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if column in INTEGER_COLUMNS:
            if df[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing/non-numeric values."
                )
            df[column] = df[column].astype("int64")
        else:
            df[column] = df[column].astype("float64")

    # Use microsecond precision for timestamp compatibility.
    df[TIME_COLUMN] = df[TIME_COLUMN].astype("datetime64[us, UTC]")

    print(f"\nFeatures loaded: {df.shape}")
    print("Feature columns:", df.columns.tolist())

    return df


# ---------------------------------------
# Step 3: Create/Get Feature Group
# ---------------------------------------

def create_feature_group(feature_store):
    return feature_store.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly weather and air quality features of Lahore",
        primary_key=[TIME_COLUMN],
        event_time=TIME_COLUMN,
        # Keep the streaming write mode used by the previously working code.
        stream=True,
    )


# ---------------------------------------
# Step 4: Insert data
# ---------------------------------------

def insert_features(feature_group, df):
    feature_group.insert(
        df,
        write_options={"wait_for_job": True},
    )

    print(
        f"Inserted {len(df)} rows into "
        f"'{FEATURE_GROUP_NAME}' (version {FEATURE_GROUP_VERSION})"
    )


# ---------------------------------------
# Step 5: Retrieve data
# ---------------------------------------

def retrieve_features(feature_group, expected_rows):
    try:
        retrieved = feature_group.read()
    except Exception as exc:
        print(
            f"Default read failed ({type(exc).__name__}), "
            "retrying with Hive..."
        )
        retrieved = feature_group.read(read_options={"use_hive": True})

    retrieved = retrieved.sort_values(TIME_COLUMN).reset_index(drop=True)

    print(
        f"\nRetrieved rows: {len(retrieved)} "
        f"(inserted: {expected_rows})"
    )

    if len(retrieved) != expected_rows:
        print("WARNING: Retrieved row count is different.")

    print(retrieved.tail())
    return retrieved


# ---------------------------------------
# Main Feature Store pipeline
# ---------------------------------------

def run_feature_store():
    print("\nStarting Hopsworks Feature Store...")

    feature_store = connect_to_hopsworks()
    df = load_features()

    feature_group = create_feature_group(feature_store)

    insert_features(feature_group, df)
    retrieved = retrieve_features(feature_group, len(df))

    print("\nHopsworks Feature Store Completed!")
    return retrieved


if __name__ == "__main__":
    run_feature_store()
