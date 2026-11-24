import sys
import numpy as np
import pandas as pd

from data_processing.merge_data import merge_datasets
from feature_store.hopsworks_feature_store import (
    connect_to_hopsworks,
    create_feature_group,
    insert_features,
    prepare_features
)


# ---------------------------------------
# Hourly ingestion
# ---------------------------------------
# A GitHub Actions runner starts empty every hour, so the new
# observation is written to the Hopsworks Feature Store, which
# keeps the history.

def run_ingestion():

    print("\nStarting hourly ingestion...")

    # Fetch current weather and air quality and merge them
    combined_df = merge_datasets()

    if combined_df is None or combined_df.empty:
        raise RuntimeError("Weather or air quality data could not be fetched.")

    # The newest record is the observation of this run
    latest = combined_df.iloc[-1].to_dict()

    timestamp = pd.Timestamp(latest["timestamp"])

    latest["hour"] = timestamp.hour
    latest["month"] = timestamp.month

    # Needs the previous hours, the Forecast Agent calculates it again
    latest["temperature_rolling_3h"] = np.nan

    features = prepare_features(pd.DataFrame([latest]))

    feature_store = connect_to_hopsworks()

    feature_group = create_feature_group(feature_store)

    insert_features(feature_group, features)

    print(f"\nIngestion completed: {features['timestamp'].iloc[0]}")

    return features


if __name__ == "__main__":

    try:
        run_ingestion()

    except Exception as error:

        # GitHub Actions marks the step as failed
        print(f"::error::Ingestion failed - {error}")
        sys.exit(1)