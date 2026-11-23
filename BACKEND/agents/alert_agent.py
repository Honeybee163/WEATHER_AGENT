import os
import json
from datetime import datetime
from dotenv import load_dotenv

from agents.shared_state import get_logger, load_state, update_state

load_dotenv()

logger = get_logger("alert_agent")


# ---------------------------------------
# Configuration
# ---------------------------------------

# Groq models change often, so the name can be set in .env
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

LLM_TIMEOUT_SECONDS = 30

# Fixed thresholds
# precipitation: mm per hour, wind_speed: m/s (10.8 m/s = 39 km/h)
THRESHOLDS = {
    "aqi_unhealthy": 150,
    "aqi_hazardous": 300,
    "heat": 40,
    "uv_high": 8,
    "rain_light": 0.5,
    "rain_heavy": 7.6,
    "humidity_high": 85,
    "humidity_low": 20,
    "wind_strong": 10.8,
    "wind_gale": 17.2,
}

# (alert, forecast key, condition, template used when the LLM fails)
# Conditions of one metric never overlap.
RULES = [
    (
        "Hazardous AQI", "aqi",
        lambda v: v > THRESHOLDS["aqi_hazardous"],
        "Air quality is forecast to be hazardous (AQI {value:.0f}). "
        "Stay indoors, keep windows closed and avoid outdoor activity."
    ),
    (
        "Unhealthy AQI", "aqi",
        lambda v: THRESHOLDS["aqi_unhealthy"] < v <= THRESHOLDS["aqi_hazardous"],
        "Air quality is forecast to be unhealthy (AQI {value:.0f}). "
        "Limit outdoor activity and wear a mask outside. Children, elderly "
        "people and people with asthma should stay indoors."
    ),
    (
        "Extreme Heat", "temperature",
        lambda v: v > THRESHOLDS["heat"],
        "Extreme heat is forecast (about {value:.0f}°C). Drink plenty of "
        "water, avoid the afternoon sun and check on elderly neighbours."
    ),
    (
        "High UV Risk", "uv_index",
        lambda v: v >= THRESHOLDS["uv_high"],
        "The UV index is forecast to be high ({value:.0f}). Use sunscreen, "
        "wear a hat and avoid direct sun around midday."
    ),
    (
        "Heavy Rain", "precipitation",
        lambda v: v >= THRESHOLDS["rain_heavy"],
        "Heavy rain is forecast ({value:.1f} mm per hour). Avoid low-lying "
        "roads, expect waterlogging and travel only if necessary."
    ),
    (
        "Rain Expected", "precipitation",
        lambda v: THRESHOLDS["rain_light"] <= v < THRESHOLDS["rain_heavy"],
        "Rain is forecast ({value:.1f} mm per hour). Keep an umbrella "
        "with you and drive carefully on wet roads."
    ),
    (
        "High Humidity", "humidity",
        lambda v: v > THRESHOLDS["humidity_high"],
        "Humidity is forecast to be very high ({value:.0f}%). It will feel "
        "hotter than the temperature shows, so rest in shade and drink water."
    ),
    (
        "Very Dry Air", "humidity",
        lambda v: v < THRESHOLDS["humidity_low"],
        "The air is forecast to be very dry ({value:.0f}% humidity). "
        "Drink water often and protect your skin and eyes."
    ),
    (
        "Gale Force Winds", "wind_speed",
        lambda v: v >= THRESHOLDS["wind_gale"],
        "Very strong winds are forecast ({value:.1f} m/s). Stay indoors "
        "and stay away from trees, signboards and power lines."
    ),
    (
        "Strong Winds", "wind_speed",
        lambda v: THRESHOLDS["wind_strong"] <= v < THRESHOLDS["wind_gale"],
        "Strong winds are forecast ({value:.1f} m/s). Secure loose objects "
        "outside and take care when riding or driving."
    ),
]

UNITS = {
    "temperature": "°C",
    "humidity": "%",
    "precipitation": "mm per hour",
    "aqi": "AQI",
    "uv_index": "UV index",
    "wind_speed": "m/s",
}


# ---------------------------------------
# Step 1: Fixed threshold rules
# ---------------------------------------

def is_number(value):

    # NaN is not equal to itself
    return isinstance(value, (int, float)) and value == value


def check_alerts(forecast):

    alerts = []

    for name, key, condition, _ in RULES:

        value = forecast.get(key)

        if is_number(value) and condition(value):
            alerts.append(name)

    return alerts


# ---------------------------------------
# Step 2: Warnings from the LLM (Groq)
# ---------------------------------------

def extract_json(text):

    # The model may add text around the JSON
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("no JSON found in LLM response")

    return json.loads(text[start:end + 1])


def generate_llm_warnings(forecast, alerts):

    # Imported here so that a missing package also uses the fallback
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)

    values = "\n".join(
        f"- {key}: {value} {UNITS.get(key, '')}"
        for key, value in forecast.items()
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.3,
        max_completion_tokens=1500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write short public safety warnings for the people "
                    "of Lahore, Pakistan. Use simple, calm and clear "
                    "language. Use only the numbers you are given."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Forecast for about 24 hours from now:\n{values}\n\n"
                    f"Triggered alerts: {', '.join(alerts)}\n\n"
                    "Write one warning of 1-2 sentences for each alert, "
                    "with one practical advice. Reply with JSON only, "
                    "where each key is the exact alert name and each "
                    "value is the warning text."
                )
            },
        ]
    )

    return extract_json(response.choices[0].message.content)


# ---------------------------------------
# Step 3: Fallback templates
# ---------------------------------------

def template_warning(name, forecast):

    for rule_name, key, _, template in RULES:

        if rule_name == name:
            return template.format(value=forecast[key])


def generate_warnings(forecast, alerts):

    if not alerts:
        return {}, "none"

    try:
        llm = generate_llm_warnings(forecast, alerts)

    except Exception as error:

        # Groq can fail for many reasons, the system must continue
        logger.error(f"LLM call failed, using templates - {error}")

        llm = {}

    warnings = {}
    template_used = 0

    for name in alerts:

        text = llm.get(name)

        if isinstance(text, str) and text.strip():
            warnings[name] = text.strip()

        else:
            warnings[name] = template_warning(name, forecast)
            template_used += 1

    if template_used == len(alerts):
        source = "template"

    elif template_used == 0:
        source = "llm"

    else:
        source = "llm + template"
        logger.warning(f"{template_used} warning(s) used the template")

    return warnings, source


# ---------------------------------------
# Main Alert Agent
# ---------------------------------------

def run_alert_agent(state=None):

    print("\nStarting Alert Agent...")

    state = load_state() if state is None else state

    forecast = state.get("forecast")

    if not forecast:

        logger.error("No forecast in the shared state, alerts skipped")

        return update_state(
            state,
            alerts=[],
            warnings={},
            warning_source="none",
            alert_status="skipped"
        )

    alerts = check_alerts(forecast)

    warnings, source = generate_warnings(forecast, alerts)

    print("\nAlerts:", alerts if alerts else "none")

    for name, text in warnings.items():
        print(f"- {name}: {text}")

    return update_state(
        state,
        alerts=alerts,
        warnings=warnings,
        warning_source=source,
        alerts_checked_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        alert_status="ok"
    )


if __name__ == "__main__":

    run_alert_agent()