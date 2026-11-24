# 


import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------
# Configuration
# ---------------------------------------

INPUT_PATH = "data/processed/training_weather.csv"

FEATURE_GROUP_NAME = "weather_features"
# Version 2: pollutants added, the Random Forest models need them
FEATURE_GROUP_VERSION = 2

# Processed column -> Hopsworks feature name
# (Hopsworks feature names must be lowercase)
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

# Each row is one hour, so the time is the unique key
TIME_COLUMN = "timestamp"

INTEGER_COLUMNS = ["hour", "month"]


# ---------------------------------------
# Step 1: Connect to Hopsworks
# ---------------------------------------

def connect_to_hopsworks():

    # Imported here so that the local feature store can use
    # this file without installing hopsworks
    import hopsworks

    # strip(): a pasted secret often carries a space, quotes or a newline,
    # which breaks the Hopsworks login
    api_key = os.getenv("HOPSWORKS_API_KEY", "").strip().strip("\"'").strip()
    project_name = os.getenv("HOPSWORKS_PROJECT", "").strip().strip("\"'").strip()

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

    return project.get_feature_store()


# ---------------------------------------
# Step 2: Load and prepare processed data
# ---------------------------------------

def prepare_features(df):

    df = df[list(FEATURE_COLUMNS)].rename(columns=FEATURE_COLUMNS)

    # Hopsworks stores time in UTC
    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=[TIME_COLUMN]
    ).drop_duplicates(
        subset=[TIME_COLUMN],
        keep="last"
    ).sort_values(TIME_COLUMN).reset_index(drop=True)

    # One fixed type for every feature
    for col in df.columns.drop(TIME_COLUMN):

        df[col] = (
            df[col].astype("int64")
            if col in INTEGER_COLUMNS
            else df[col].astype("float64")
        )

    return df


def load_features():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = prepare_features(pd.read_csv(INPUT_PATH))

    print(f"\nFeatures loaded: {df.shape}")

    return df


# ---------------------------------------
# Step 3: Create Feature Group
# ---------------------------------------

def create_feature_group(feature_store):

    return feature_store.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly weather and air quality features of Lahore",
        primary_key=[TIME_COLUMN],
        event_time=TIME_COLUMN
    )


# ---------------------------------------
# Step 4: Insert data
# ---------------------------------------

def insert_features(feature_group, df):

    # Waits until Hopsworks has finished writing the data
    feature_group.insert(
        df,
        write_options={"wait_for_job": True}
    )

    print(
        f"Inserted {len(df)} rows into "
        f"'{FEATURE_GROUP_NAME}' (version {FEATURE_GROUP_VERSION})"
    )


# ---------------------------------------
# Step 5: Retrieve data
# ---------------------------------------

def retrieve_features(feature_group, expected_rows):

    retrieved = feature_group.read()

    retrieved = retrieved.sort_values(TIME_COLUMN).reset_index(drop=True)

    print(f"\nRetrieved rows: {len(retrieved)} (inserted: {expected_rows})")

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