import os
import sys

from models.baseline_model import METRICS
from src.agents.forecast_agent import run_forecast_agent


# ---------------------------------------
# Forecast run for GitHub Actions
# ---------------------------------------

def write_summary(state):

    # Shown on the page of the workflow run
    path = os.getenv("GITHUB_STEP_SUMMARY")

    if not path:
        return

    lines = [
        "## Forecast Agent",
        "",
        f"Forecast time: `{state.get('forecast_time')}`  ",
        f"Target time: `{state.get('target_time')}`",
        "",
        "| Metric | Forecast | Model |",
        "| --- | --- | --- |",
    ]

    for metric in METRICS:

        model = state.get("forecast_models", {}).get(metric, {})

        lines.append(
            f"| {metric} | {state.get('forecast', {}).get(metric.lower())} "
            f"| {model.get('model_name')} v{model.get('model_version')} |"
        )

    with open(path, "a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_forecast():

    state = run_forecast_agent()

    if state.get("forecast_status") == "failed":
        print(f"::error::Forecast agent failed - {state.get('forecast_error')}")
        sys.exit(1)

    # The run continues, but the problem is visible in the log
    if state.get("fallback_metrics"):
        print(
            "::warning::Backup (naive) forecast used for: "
            + ", ".join(state["fallback_metrics"])
        )

    if state.get("ledger_saved") is False:
        print("::warning::Forecasts could not be saved to the ledger")

    write_summary(state)

    return state


if __name__ == "__main__":

    run_forecast()