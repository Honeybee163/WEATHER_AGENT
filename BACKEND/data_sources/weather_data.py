import requests
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


# Free sources for the metrics OpenWeather free plan does not give
WEATHER_HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"
UV_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_rain_probability():

    # Probability of precipitation (pop) of the next forecast slot
    params = {
        "lat": os.getenv("LATITUDE"),
        "lon": os.getenv("LONGITUDE"),
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "cnt": 1,
    }

    try:
        response = requests.get(
            FORECAST_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        # pop is 0-1, converted to percentage
        return response.json()["list"][0]["pop"] * 100

    except requests.exceptions.RequestException as e:
        print(f"Rain Probability API Error: {e}")
        return None

    except (KeyError, IndexError, TypeError) as e:
        print(f"Rain Probability Data Error: {e}")
        return None


def fetch_uv_index():

    params = {
        "latitude": os.getenv("LATITUDE"),
        "longitude": os.getenv("LONGITUDE"),
        "current": "uv_index",
    }

    try:
        response = requests.get(
            UV_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        return response.json()["current"]["uv_index"]

    except requests.exceptions.RequestException as e:
        print(f"UV API Error: {e}")
        return None

    except (KeyError, TypeError) as e:
        print(f"UV Data Error: {e}")
        return None


def fetch_current():
    params = {
        "lat": os.getenv("LATITUDE"),
        "lon": os.getenv("LONGITUDE"),
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
    }

    try:
        response = requests.get(
            WEATHER_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        # convert to dict
        data = response.json()

        # Extract required fields
        weather_data = {
        "timestamp": pd.to_datetime(
            data["dt"], unit="s", utc=True
        ),
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "precipitation": data.get("rain", {}).get("1h", 0),
        "rain_probability": fetch_rain_probability(),
        "uv_index": fetch_uv_index(),
        "wind_speed": data["wind"]["speed"],
        "wind_direction": data["wind"].get("deg"),
}

        print("Required Weather Information:")
        print(weather_data)

        return weather_data

    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None

    except (KeyError, TypeError) as e:
        print(f"Data Error: {e}")
        return None


def fetch_historical(start_date, end_date):

    params = {
        "latitude": os.getenv("LATITUDE"),
        "longitude": os.getenv("LONGITUDE"),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }

    try:
        response = requests.get(
            WEATHER_HISTORY_URL,
            params={
                **params,
                "hourly": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "wind_speed_10m",
                    "wind_direction_10m",
                ]),
                # Same unit as OpenWeather (m/s)
                "wind_speed_unit": "ms",
            },
            timeout=30
        )

        response.raise_for_status()

        # UV index is not in the weather archive
        uv_response = requests.get(
            UV_URL,
            params={**params, "hourly": "uv_index"},
            timeout=30
        )

        uv_response.raise_for_status()

        weather = pd.DataFrame(response.json()["hourly"])
        uv = pd.DataFrame(uv_response.json()["hourly"])

        data = weather.merge(uv, on="time", how="left")

        # Same columns as fetch_current()
        return pd.DataFrame({
            "timestamp": pd.to_datetime(data["time"], utc=True),
            "temperature": data["temperature_2m"],
            "humidity": data["relative_humidity_2m"],
            "precipitation": data["precipitation"],
            # Not available in historical data
            "rain_probability": None,
            "uv_index": data["uv_index"],
            "wind_speed": data["wind_speed_10m"],
            "wind_direction": data["wind_direction_10m"],
        })

    except requests.exceptions.RequestException as e:
        print(f"Historical API Error: {e}")
        return None

    except (KeyError, TypeError) as e:
        print(f"Historical Data Error: {e}")
        return None


if __name__ == "__main__":
    fetch_current()