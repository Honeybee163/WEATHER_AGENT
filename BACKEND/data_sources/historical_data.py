import os
import pandas as pd

from data_sources.weather_data import fetch_historical as fetch_weather_history
from data_sources.air_quality_data import fetch_historical as fetch_aqi_history

TIMEZONE = "Asia/Karachi"

OUTPUT_PATH = "data/processed/combined_weather.csv"

# Number of past days to download
HISTORY_DAYS = 90

# The weather archive is published with a delay of about 5 days
ARCHIVE_DELAY_DAYS = 5


def has_historical_data():

    # True when the backfill is already part of the saved dataset
    if not os.path.exists(OUTPUT_PATH):
        return False

    weather_type = pd.read_csv(
        OUTPUT_PATH,
        usecols=["weather_type"]
    )["weather_type"]

    return bool((weather_type == "historical_reanalysis").any())


def fetch_historical(days=HISTORY_DAYS):

    end_date = (
        pd.Timestamp.now(tz="UTC").normalize()
        - pd.Timedelta(days=ARCHIVE_DELAY_DAYS)
    )

    start_date = end_date - pd.Timedelta(days=days - 1)

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    weather_df = fetch_weather_history(start_date, end_date)
    aqi_df = fetch_aqi_history(start_date, end_date)

    if weather_df is None or aqi_df is None:
        print("Error: Unable to fetch historical data.")
        return None

    historical_df = weather_df.merge(
        aqi_df,
        on="timestamp",
        how="left"
    )

    historical_df["timestamp"] = (
        historical_df["timestamp"].dt.tz_convert(TIMEZONE)
    )

    # Archive can return empty rows for the latest hours.
    # Only that empty tail is removed, gaps in between are
    # left for the cleaning step.
    historical_df = historical_df.loc[
        :historical_df["temperature"].last_valid_index()
    ]

    # Source information (same columns as merge_data.py)
    historical_df["weather_type"] = "historical_reanalysis"
    historical_df["aqi_type"] = "historical_model"

    print(f"Historical records downloaded: {len(historical_df)}")

    return historical_df


def save_historical(historical_df):

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    combined_df = historical_df

    if os.path.exists(OUTPUT_PATH):

        old_df = pd.read_csv(OUTPUT_PATH)

        # Existing live records come last so they are kept
        combined_df = pd.concat(
            [historical_df, old_df],
            ignore_index=True
        )

    combined_df["timestamp"] = pd.to_datetime(
        combined_df["timestamp"],
        utc=True
    ).dt.tz_convert(TIMEZONE)

    combined_df = combined_df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    ).sort_values("timestamp").reset_index(drop=True)

    combined_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Historical data saved: {OUTPUT_PATH}")
    print(f"Total records: {len(combined_df)}")

    return combined_df


if __name__ == "__main__":

    historical_df = fetch_historical()

    if historical_df is not None:
        save_historical(historical_df)