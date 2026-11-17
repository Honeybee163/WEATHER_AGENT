import os
import numpy as np
import pandas as pd


# -----------------------------------------
# Configuration
# -----------------------------------------

INPUT_PATH = "data/processed/combined_weather.csv"

OUTPUT_PATH = "data/processed/clean_weather.csv"

TIMEZONE = "Asia/Karachi"

# Longest gap (between two real values) that may be interpolated
MAX_INTERPOLATION_GAP = pd.Timedelta("3h")

# Numeric columns
NUMERIC_COLUMNS = [
    "temperature",
    "humidity",
    "precipitation",
    "rain_probability",
    "uv_index",
    "wind_speed",
    "wind_direction",
    "AQI",
    "PM2.5",
    "PM10",
    "O3",
    "NO2",
    "SO2",
    "CO"
]

# Columns suitable for time interpolation
INTERPOLATION_COLUMNS = [
    "temperature",
    "humidity",
    "wind_speed"
]

# AQI and pollutant columns
AQI_COLUMNS = [
    "AQI",
    "PM2.5",
    "PM10",
    "O3",
    "NO2",
    "SO2",
    "CO"
]


# -----------------------------------------
# Step 1: Load dataset
# -----------------------------------------

def load_data():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("\nOriginal Dataset:")
    print(df.head())

    return df


# -----------------------------------------
# Step 2: Data type conversion
# -----------------------------------------

def convert_data_types(df):

    # Convert timestamp to Pakistan timezone
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert(TIMEZONE)

    # Convert numeric columns
    for col in NUMERIC_COLUMNS:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# -----------------------------------------
# Step 3: Missing values report
# -----------------------------------------

def missing_values_report(df, filename):

    report = pd.DataFrame({
        "column": df.columns,
        "missing_values": df.isnull().sum().values,
        "missing_percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    print(f"\nMissing Values Report - {filename}:")
    print(report)

    report.to_csv(
        f"data/processed/{filename}",
        index=False
    )

    return report


# -----------------------------------------
# Step 4: Remove duplicate timestamps
# -----------------------------------------

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    after = len(df)

    print(
        f"\nDuplicate records removed: {before - after}"
    )

    return df


# -----------------------------------------
# Step 5: Handle missing values
# -----------------------------------------

def interpolate_short_gaps(series):

    # Time interpolation, but only when the gap between the
    # previous and next real value is within MAX_INTERPOLATION_GAP.
    # Longer gaps stay missing.
    filled = series.interpolate(
        method="time",
        limit_area="inside"
    )

    times = series.index.to_series()
    valid_times = times.where(series.notna())

    gap = valid_times.bfill() - valid_times.ffill()

    return filled.where(
        series.notna() | (gap <= MAX_INTERPOLATION_GAP)
    )


def handle_missing_values(df):

    df = df.sort_values(
        "timestamp"
    ).set_index("timestamp")

    # Interpolate temperature, humidity and wind speed.
    # Only short gaps are filled (see MAX_INTERPOLATION_GAP).
    for col in INTERPOLATION_COLUMNS:

        if col in df.columns:

            df[col] = interpolate_short_gaps(df[col])

    # AQI and pollutants:
    # Forward fill only if previous observation
    # is no more than 1 hour old.
    for col in AQI_COLUMNS:

        if col not in df.columns:
            continue

        last_valid_time = None
        last_valid_value = None

        for timestamp in df.index:

            value = df.at[timestamp, col]

            if pd.notna(value):

                last_valid_time = timestamp
                last_valid_value = value

            elif (
                last_valid_time is not None
                and timestamp - last_valid_time
                <= pd.Timedelta("1h")
            ):

                df.at[timestamp, col] = last_valid_value

    # Rain, rain probability and UV:
    # Do not invent values when the source
    # does not provide them.

    # Wind direction:
    # Circular interpolation to handle
    # directions such as 359 and 1 degrees.
    if "wind_direction" in df.columns:

        direction = df["wind_direction"]

        radians = np.deg2rad(direction)

        sin_values = pd.Series(
            np.sin(radians),
            index=df.index
        )

        cos_values = pd.Series(
            np.cos(radians),
            index=df.index
        )

        sin_filled = interpolate_short_gaps(sin_values)

        cos_filled = interpolate_short_gaps(cos_values)

        interpolated_direction = (
            np.rad2deg(
                np.arctan2(sin_filled, cos_filled)
            ) + 360
        ) % 360

        df["wind_direction"] = direction.fillna(
            interpolated_direction
        )

    return df.reset_index()


# -----------------------------------------
# Step 6: Identify invalid values
# -----------------------------------------

def identify_invalid_values(df):

    valid_ranges = {
        "temperature": (-80, 60),
        "humidity": (0, 100),
        "precipitation": (0, None),
        "rain_probability": (0, 100),
        "uv_index": (0, 20),
        "wind_speed": (0, None),
        "wind_direction": (0, 360),
        "AQI": (0, None),
        "PM2.5": (0, None),
        "PM10": (0, None),
        "O3": (0, None),
        "NO2": (0, None),
        "SO2": (0, None),
        "CO": (0, None)
    }

    invalid_report = []

    for col, (minimum, maximum) in valid_ranges.items():

        if col not in df.columns:
            continue

        invalid = pd.Series(False, index=df.index)

        if minimum is not None:
            invalid |= df[col] < minimum

        if maximum is not None:
            invalid |= df[col] > maximum

        if invalid.any():

            invalid_report.append({
                "column": col,
                "invalid_count": int(invalid.sum())
            })

            # Mark invalid values as missing
            df.loc[invalid, col] = np.nan

    return df, pd.DataFrame(
        invalid_report,
        columns=["column", "invalid_count"]
    )


# -----------------------------------------
# Step 7: Identify statistical outliers
# -----------------------------------------

def identify_outliers(df):

    outlier_records = []

    for col in NUMERIC_COLUMNS:

        if col not in df.columns:
            continue

        values = df[col].dropna()

        # Need enough records for IQR analysis
        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            (df[col] < lower_bound)
            | (df[col] > upper_bound)
        )

        for index in df.index[mask]:

            outlier_records.append({
                "timestamp": df.loc[index, "timestamp"],
                "column": col,
                "value": df.loc[index, col],
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            })

    return pd.DataFrame(
        outlier_records,
        columns=[
            "timestamp",
            "column",
            "value",
            "lower_bound",
            "upper_bound"
        ]
    )


# -----------------------------------------
# Step 8: Save clean dataset and reports
# -----------------------------------------

def save_clean_data(df, outliers, invalid):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # Save clean dataset
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Save outlier report
    outliers.to_csv(
        "data/processed/outlier_report.csv",
        index=False
    )

    # Save invalid values report
    invalid.to_csv(
        "data/processed/invalid_values_report.csv",
        index=False
    )

    # Save missing values report after cleaning
    missing_values_report(
        df,
        "missing_values_after_cleaning.csv"
    )

    print("\nClean Dataset:")
    print(df.tail())

    print("\nCleaning completed successfully!")

    print(f"Clean file: {OUTPUT_PATH}")


# -----------------------------------------
# Main pipeline
# -----------------------------------------

def clean_dataset():

    # Load
    df = load_data()

    # Initial missing report
    missing_values_report(
        df,
        "missing_values_before_cleaning.csv"
    )

    # Convert types
    df = convert_data_types(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Identify physically invalid values
    df, invalid = identify_invalid_values(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Identify statistical outliers
    outliers = identify_outliers(df)

    # Save results
    save_clean_data(
        df,
        outliers,
        invalid
    )

    return df


if __name__ == "__main__":

    clean_dataset()