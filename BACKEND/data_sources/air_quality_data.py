import requests
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

lat = os.getenv("LATITUDE")
lon = os.getenv("LONGITUDE")

AIR_QUALITY_URL = f"https://api.waqi.info/feed/geo:{lat};{lon}/"

# AQICN token does not give past data, so history comes from here
AIR_QUALITY_HISTORY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_current():

    params = {
        "token": os.getenv("AQICN_API_TOKEN"),
    }

    try:
        response = requests.get(
            AIR_QUALITY_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        # Convert JSON response into dictionary
        response_data = response.json()

        # Check API response status
        if response_data.get("status") != "ok":
            print("WAQI API Error:", response_data.get("data"))
            return None

        # Access nested data
        data = response_data["data"]

        # Extract air quality information
        iaqi = data.get("iaqi", {})

        air_data = {
            "timestamp": pd.to_datetime(
                data["time"]["iso"], utc=True
            ),
            "AQI": data.get("aqi"),

            "PM2.5": iaqi.get("pm25", {}).get("v"),
            "PM10": iaqi.get("pm10", {}).get("v"),

            "O3": iaqi.get("o3", {}).get("v"),
            "NO2": iaqi.get("no2", {}).get("v"),

            "SO2": iaqi.get("so2", {}).get("v"),
            "CO": iaqi.get("co", {}).get("v"),
        }

        print("Required Air Quality Information:")
        print(air_data)

        return air_data

    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None

    except (KeyError, TypeError) as e:
        print(f"Data Error: {e}")
        return None


def fetch_historical(start_date, end_date):

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        # US AQI sub-indexes, same scale as AQICN iaqi values
        "hourly": ",".join([
            "us_aqi",
            "us_aqi_pm2_5",
            "us_aqi_pm10",
            "us_aqi_ozone",
            "us_aqi_nitrogen_dioxide",
            "us_aqi_sulphur_dioxide",
            "us_aqi_carbon_monoxide",
        ]),
    }

    try:
        response = requests.get(
            AIR_QUALITY_HISTORY_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = pd.DataFrame(response.json()["hourly"])

        # Same columns as fetch_current()
        return pd.DataFrame({
            "timestamp": pd.to_datetime(data["time"], utc=True),
            "AQI": data["us_aqi"],
            "PM2.5": data["us_aqi_pm2_5"],
            "PM10": data["us_aqi_pm10"],
            "O3": data["us_aqi_ozone"],
            "NO2": data["us_aqi_nitrogen_dioxide"],
            "SO2": data["us_aqi_sulphur_dioxide"],
            "CO": data["us_aqi_carbon_monoxide"],
        })

    except requests.exceptions.RequestException as e:
        print(f"Historical API Error: {e}")
        return None

    except (KeyError, TypeError) as e:
        print(f"Historical Data Error: {e}")
        return None


if __name__ == "__main__":
    fetch_current()