import sys
from datetime import datetime

from data_sources.historical_data import (
    has_historical_data,
    fetch_historical,
    save_historical
)
from data_processing.merge_data import merge_datasets
from data_cleaning.cleaning import clean_dataset

def run_pipeline():

    print("\n" + "=" * 50)
    print("       WEATHER DATA PROCESSING PIPELINE")
    print("=" * 50)

    start_time = datetime.now()

    try:

        # ---------------------------------
        # STEP 1: HISTORICAL BACKFILL
        # ---------------------------------

        print("\n[STEP 1/3] Checking historical data...")
        print("-" * 50)

        if has_historical_data():
            print("Historical data already available.")

        else:
            historical_df = fetch_historical()

            # Live data collection can continue without history
            if historical_df is None:
                print("WARNING: Historical backfill failed.")
            else:
                save_historical(historical_df)

        # ---------------------------------
        # STEP 2: FETCH AND MERGE DATA
        # ---------------------------------

        print("\n[STEP 2/3] Fetching and merging data...")
        print("-" * 50)

        combined_df = merge_datasets()

        if combined_df is None or combined_df.empty:
            raise RuntimeError(
                "Data fetching or merging failed."
            )

        print("\nSUCCESS: Data merging completed!")

        print(
            f"Combined records: {len(combined_df)}"
        )

        # ---------------------------------
        # STEP 3: CLEAN DATA
        # ---------------------------------

        print("\n[STEP 3/3] Cleaning dataset...")
        print("-" * 50)

        clean_df = clean_dataset()

        if clean_df is None or clean_df.empty:
            raise RuntimeError(
                "Data cleaning failed."
            )

        print("\nSUCCESS: Data cleaning completed!")

        print(
            f"Clean records: {len(clean_df)}"
        )

        # ---------------------------------
        # PIPELINE COMPLETED
        # ---------------------------------

        end_time = datetime.now()

        duration = end_time - start_time

        print("\n" + "=" * 50)
        print("       PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 50)

        print("\nFinal Output:")
        print("data/processed/combined_weather.csv")
        print("data/processed/clean_weather.csv")

        print(f"\nTotal execution time: {duration}")

        print("\nFinal Dataset Preview:")
        print(clean_df.tail())

    except Exception as e:

        print("\n" + "=" * 50)
        print("       PIPELINE FAILED")
        print("=" * 50)

        print(f"\nError: {e}")

        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()