import os
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from feature_store.model_registry import connect_to_hopsworks
from models.baseline_model import TIMEZONE
from agents.alert_agent import GROQ_MODEL, LLM_TIMEOUT_SECONDS, THRESHOLDS
from agents.shared_state import get_logger, load_state, update_state

load_dotenv()

logger = get_logger("weather_report_agent")


# ---------------------------------------
# Configuration
# ---------------------------------------

CITY = "Lahore"

REPORT_FEATURE_GROUP = "weather_reports"
REPORT_FEATURE_GROUP_VERSION = 1

HISTORY_PATH = "data/processed/report_history.csv"

SECTIONS = [
    "Weather overview",
    "Air quality conditions",
    "Rain expectations",
    "Public safety recommendations",
]

# Report is rejected when it contains numbers that are not in
# the forecast (the LLM must not invent weather information)
CHECK_NUMBERS = True

# Numbers that may appear without being a forecast value:
# the 24 hour horizon and the AQI / UV category limits
REFERENCE_NUMBERS = {24, 50, 100, 150, 200, 300, 3, 6, 8, 11}

MAX_DATA_AGE_HOURS = 6


# ---------------------------------------
# Step 1 and 2: Forecast and alerts from the shared state
# ---------------------------------------

def get_forecast_values(forecast):

    values = {
        key: forecast[key]
        for key in (
            "temperature", "humidity", "precipitation", "aqi", "uv_index"
        )
        if forecast.get(key) is not None
    }

    # The models forecast wind in m/s, reports use km/h
    if forecast.get("wind_speed") is not None:
        values["wind_kmh"] = forecast["wind_speed"] * 3.6

    return values


# ---------------------------------------
# Categories (fixed rules, no guessing)
# ---------------------------------------

def aqi_category(aqi):

    for limit, name in [
        (50, "Good"),
        (100, "Moderate"),
        (150, "Unhealthy for Sensitive Groups"),
        (200, "Unhealthy"),
        (300, "Very Unhealthy"),
    ]:
        if aqi <= limit:
            return name

    return "Hazardous"


def uv_category(uv):

    for limit, name in [
        (3, "low"), (6, "moderate"), (8, "high"), (11, "very high")
    ]:
        if uv < limit:
            return name

    return "extreme"


def temperature_description(temperature):

    for limit, name in [
        (10, "very cold"), (18, "cool"), (25, "mild"),
        (32, "warm"), (40, "hot")
    ]:
        if temperature < limit:
            return name

    return "extremely hot"


def humidity_description(humidity):

    for limit, name in [(30, "dry"), (60, "comfortable"), (80, "humid")]:
        if humidity < limit:
            return name

    return "very humid"


def wind_description(kmh):

    for limit, name in [
        (12, "light"), (29, "gentle to moderate"),
        (39, "fresh"), (50, "strong")
    ]:
        if kmh < limit:
            return name

    return "very strong"


AQI_ADVICE = {
    "Good": "Air quality is satisfactory for everyone.",
    "Moderate": "Air quality is acceptable for most people.",
    "Unhealthy for Sensitive Groups": (
        "Children, the elderly and people with asthma or heart conditions "
        "should limit long outdoor activity."
    ),
    "Unhealthy": (
        "Everyone may begin to feel health effects, so outdoor exercise "
        "should be limited."
    ),
    "Very Unhealthy": (
        "Health risks are serious for everyone. Avoid outdoor activity."
    ),
    "Hazardous": (
        "This is an emergency level of pollution. Everyone should stay "
        "indoors."
    ),
}


# ---------------------------------------
# Step 3: Report from the LLM (Groq)
# ---------------------------------------

def build_messages(values, alerts, warnings):

    lines = []

    if "temperature" in values:
        lines.append(f"Temperature: {values['temperature']:.1f} °C")
    if "aqi" in values:
        lines.append(f"Air Quality Index (US AQI): {values['aqi']:.0f}")
    if "humidity" in values:
        lines.append(f"Humidity: {values['humidity']:.0f} %")
    if "precipitation" in values:
        lines.append(
            f"Rainfall: {values['precipitation']:.1f} mm per hour"
        )
    if "uv_index" in values:
        lines.append(f"UV Index: {values['uv_index']:.0f}")
    if "wind_kmh" in values:
        lines.append(f"Wind speed: {values['wind_kmh']:.1f} km/h")

    alert_lines = (
        "\n".join(f"- {name}: {text}" for name, text in warnings.items())
        if alerts else "None"
    )

    return [
        {
            "role": "system",
            "content": (
                f"You are a professional weather journalist writing for a "
                f"{CITY} newspaper. Do not invent weather information: "
                "use only the data you are given, with the exact numbers, "
                "and never add numbers, dates, clock times or conditions "
                "that are not in the data."
            )
        },
        {
            "role": "user",
            "content": (
                f"Write a weather report for {CITY} using this forecast. "
                "The values are the conditions expected about 24 hours from "
                "now (one point in time, not a daily total or average):\n\n"
                + "\n".join(lines)
                + f"\n\nSafety warnings already issued:\n{alert_lines}\n\n"
                "Rain probability is not available, so do not mention it.\n\n"
                "Write exactly these four sections, each with its title on "
                "its own line followed by a short paragraph:\n"
                + "\n".join(SECTIONS)
                + "\n\nUse plain text without markdown or numbering. Keep "
                "the public safety recommendations consistent with the "
                "warnings above. Do not write dates or clock times. "
                "Use 180 to 250 words in total."
            )
        },
    ]


def allowed_numbers(values):

    allowed = {round(float(number), 2) for number in REFERENCE_NUMBERS}

    for value in values.values():

        for digits in (0, 1, 2):
            allowed.add(round(float(f"{value:.{digits}f}"), 2))

    for limit in THRESHOLDS.values():
        allowed.add(round(float(limit), 2))

    return allowed


def validate_report(text, values):

    if not text or not text.strip():
        raise ValueError("LLM returned an empty report")

    lower = text.lower()

    missing = [
        section for section in SECTIONS
        if section.lower() not in lower
    ]

    if missing:
        raise ValueError(f"report is missing sections: {missing}")

    if CHECK_NUMBERS:

        allowed = allowed_numbers(values)

        invented = sorted({
            number for number in re.findall(r"\d+(?:\.\d+)?", text)
            if round(float(number), 2) not in allowed
        })

        if invented:
            raise ValueError(
                f"report contains numbers that are not in the forecast: "
                f"{invented}"
            )

    return text.strip()


def generate_llm_report(values, alerts, warnings):

    # Imported here so that a missing package also uses the fallback
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.3,
        max_completion_tokens=2000,
        messages=build_messages(values, alerts, warnings)
    )

    return validate_report(response.choices[0].message.content, values)


# ---------------------------------------
# Step 5: Python template (fallback)
# ---------------------------------------

def template_report(values, alerts, warnings):

    # Weather overview
    overview = []

    if "temperature" in values:
        temperature = values["temperature"]
        overview.append(
            f"Temperatures in {CITY} are expected to be around "
            f"{temperature:.0f}°C, which is "
            f"{temperature_description(temperature)}."
        )

    if "humidity" in values:
        humidity = values["humidity"]
        overview.append(
            f"Humidity is forecast at {humidity:.0f}%, "
            f"which is {humidity_description(humidity)}."
        )

    if "wind_kmh" in values:
        wind = values["wind_kmh"]
        overview.append(
            f"Winds are expected to be {wind_description(wind)}, "
            f"at about {wind:.0f} km/h."
        )

    if "uv_index" in values:
        uv = values["uv_index"]
        overview.append(
            f"The UV index is forecast at {uv:.0f}, "
            f"which is {uv_category(uv)}."
        )

    # Air quality
    if "aqi" in values:
        category = aqi_category(values["aqi"])
        air = (
            f"The Air Quality Index is forecast at {values['aqi']:.0f}, "
            f"which is rated {category}. {AQI_ADVICE[category]}"
        )
    else:
        air = "Air quality information is not available."

    # Rain
    if "precipitation" not in values:
        rain = "Rainfall information is not available."

    elif values["precipitation"] < THRESHOLDS["rain_light"]:
        rain = "No rain is expected at that time."

    else:
        rain = (
            f"Rain is expected at a rate of about "
            f"{values['precipitation']:.1f} mm per hour."
        )

    # Safety recommendations come from the Alert Agent
    if alerts:
        safety = "\n".join(f"- {warnings[name]}" for name in alerts)
    else:
        safety = (
            "No weather or air quality alerts are in effect. Normal "
            "precautions are enough: drink water and follow local updates."
        )

    body = [
        SECTIONS[0],
        " ".join(overview) or "Weather information is not available.",
        "",
        SECTIONS[1],
        air,
        "",
        SECTIONS[2],
        rain,
        "",
        SECTIONS[3],
        safety,
    ]

    return "\n".join(body)


def generate_report(values, alerts, warnings):

    try:
        return generate_llm_report(values, alerts, warnings), "llm"

    except Exception as error:

        # Groq can fail for many reasons, the system must continue
        logger.error(f"LLM report failed, using template - {error}")

        return template_report(values, alerts, warnings), "template"


# ---------------------------------------
# Header and notes (always written by Python)
# ---------------------------------------

def local_time(value):

    return pd.to_datetime(value, utc=True).tz_convert(TIMEZONE)


def build_header(state, issued):

    lines = [
        f"{CITY.upper()} WEATHER REPORT",
        f"Issued: {issued:%d %b %Y, %I:%M %p} PKT",
    ]

    if state.get("target_time"):
        target = local_time(state["target_time"])
        lines.append(
            f"Forecast for: {target:%d %b %Y, %I:%M %p} PKT "
            "(about 24 hours ahead)"
        )

    return "\n".join(lines)


def build_notes(state):

    notes = []

    if state.get("fallback_metrics"):
        notes.append(
            "Note: the forecast of "
            + ", ".join(state["fallback_metrics"])
            + " is based on the latest observed values (backup method)."
        )

    age = state.get("features_age_hours")

    if age is not None and age > MAX_DATA_AGE_HOURS:
        notes.append(
            f"Note: the latest weather data used is {age:.0f} hours old."
        )

    return notes


# ---------------------------------------
# Step 6: Report history (Feature Store and local file)
# ---------------------------------------

def save_report_history(record):

    os.makedirs("data/processed", exist_ok=True)

    csv_record = pd.DataFrame([record])

    # Local copy: the history exists even when Hopsworks is down
    csv_record.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not os.path.exists(HISTORY_PATH),
        index=False
    )

    saved = {"local": True, "feature_store": False}

    try:
        project = connect_to_hopsworks()

        feature_group = project.get_feature_store().get_or_create_feature_group(
            name=REPORT_FEATURE_GROUP,
            version=REPORT_FEATURE_GROUP_VERSION,
            description="Weather reports written by the report agent",
            primary_key=["report_time"],
            event_time="report_time"
        )

        feature_group.insert(
            csv_record.assign(**{
                column: pd.to_datetime(csv_record[column], utc=True)
                for column in ("report_time", "forecast_time", "target_time")
            }),
            write_options={"wait_for_job": True}
        )

        saved["feature_store"] = True

        history = feature_group.read().sort_values("report_time")

        print(f"\nReport history: {len(history)} reports")

    except Exception as error:
        logger.error(f"Report history save to Feature Store failed - {error}")

    return saved


# ---------------------------------------
# Main Weather Report Agent
# ---------------------------------------

def run_weather_report_agent(state=None):

    print("\nStarting Weather Report Agent...")

    state = load_state() if state is None else state

    forecast = state.get("forecast")

    if not forecast:

        logger.error("No forecast in the shared state, report skipped")

        return update_state(state, report_status="skipped")

    values = get_forecast_values(forecast)

    alerts = state.get("alerts", [])

    warnings = state.get("warnings", {})

    body, source = generate_report(values, alerts, warnings)

    issued = pd.Timestamp.now(tz=TIMEZONE)

    report = "\n\n".join(
        [build_header(state, issued), body] + build_notes(state)
    )

    print(f"\n{report}")

    saved = save_report_history({
        "report_time": issued.tz_convert("UTC").floor("s").isoformat(),
        "forecast_time": state.get("forecast_time"),
        "target_time": state.get("target_time"),
        "report": report,
        "source": source,
        "model": GROQ_MODEL if source == "llm" else "template",
        "alerts": ", ".join(alerts),
    })

    return update_state(
        state,
        weather_report=report,
        report_source=source,
        report_generated_at=issued.strftime("%Y-%m-%d %H:%M:%S"),
        report_saved=saved,
        report_status="ok"
    )


if __name__ == "__main__":

    run_weather_report_agent()