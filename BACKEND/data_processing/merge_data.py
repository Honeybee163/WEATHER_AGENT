import pandas as pd
import os

from data_sources.weather_data import fetch_current as fetch_weather
from data_sources.air_quality_data import fetch_current as fetch_aqi


TIMEZONE = "Asia/Karachi"

OUTPUT_PATH = "data/processed/combined_weather.csv"


def merge_datasets():

    try:
        # ---------------------------------------
        # 1. Fetch data from existing modules
        # ---------------------------------------

        weather_data = fetch_weather()
        aqi_data = fetch_aqi()

        if weather_data is None or aqi_data is None:
            print("Error: Unable to fetch source data.")
            return None

        # ---------------------------------------
        # 2. Convert dictionaries into DataFrames
        # ---------------------------------------

        weather_df = pd.DataFrame([weather_data])
        aqi_df = pd.DataFrame([aqi_data])

        # ---------------------------------------
        # 3. Preserve actual API timestamps
        # ---------------------------------------
        # Convert both timestamps to UTC first

        weather_df["timestamp"] = pd.to_datetime(
            weather_df["timestamp"],
            utc=True
        )

        aqi_df["timestamp"] = pd.to_datetime(
            aqi_df["timestamp"],
            utc=True
        )

        # Convert both into Pakistan Standard Time

        weather_df["timestamp"] = (
            weather_df["timestamp"].dt.tz_convert(TIMEZONE)
        )

        aqi_df["timestamp"] = (
            aqi_df["timestamp"].dt.tz_convert(TIMEZONE)
        )

        # ---------------------------------------
        # 4. Sort both datasets
        # ---------------------------------------

        # Same time resolution is required by merge_asof
        weather_df["timestamp"] = weather_df["timestamp"].astype(
            f"datetime64[ns, {TIMEZONE}]"
        )

        aqi_df["timestamp"] = aqi_df["timestamp"].astype(
            f"datetime64[ns, {TIMEZONE}]"
        )

        weather_df = weather_df.sort_values("timestamp")
        aqi_df = aqi_df.sort_values("timestamp")

        # ---------------------------------------
        # 5. Merge datasets by nearest timestamp
        # ---------------------------------------

        combined_df = pd.merge_asof(
            weather_df,
            aqi_df,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("5h"),
            suffixes=("_weather", "_aqi")
        )

        # ---------------------------------------
        # 6. Add source information
        # ---------------------------------------

        combined_df["weather_type"] = "current"
        combined_df["aqi_type"] = "observation"

        # ---------------------------------------
        # 7. Create output directory
        # ---------------------------------------

        os.makedirs(
            os.path.dirname(OUTPUT_PATH),
            exist_ok=True
        )

        # ---------------------------------------
        # 8. Load old CSV and combine new records
        # ---------------------------------------

        if os.path.exists(OUTPUT_PATH):

            old_df = pd.read_csv(OUTPUT_PATH)

            # Same timestamp type as new data, so duplicates match
            old_df["timestamp"] = pd.to_datetime(
                old_df["timestamp"],
                utc=True
            ).dt.tz_convert(TIMEZONE)

            # Combine historical and new data
            combined_df = pd.concat(
                [old_df, combined_df],
                ignore_index=True
            )

            # Remove duplicate timestamps
            combined_df = combined_df.drop_duplicates(
                subset=["timestamp"],
                keep="last"
            )

        # ---------------------------------------
        # 9. Sort historical records
        # ---------------------------------------

        combined_df["timestamp"] = pd.to_datetime(
            combined_df["timestamp"],
            utc=True
        ).dt.tz_convert(TIMEZONE)

        combined_df = combined_df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # ---------------------------------------
        # 10. Save updated CSV
        # ---------------------------------------

        combined_df.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print("\nCombined Dataset:")
        print(combined_df.tail())

        print("\nMissing Values:")
        print(combined_df.isnull().sum())

        print(
            f"\nHistorical dataset updated: {OUTPUT_PATH}"
        )

        return combined_df

    except Exception as e:

        print(f"Data Merging Error: {e}")
        return None


if __name__ == "__main__":
    merge_datasets()