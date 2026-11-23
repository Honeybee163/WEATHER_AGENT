import os
import json
import time
import uuid
import argparse
import pandas as pd
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from models.baseline_model import TIMEZONE
from agents.forecast_agent import run_forecast_agent
from agents.alert_agent import run_alert_agent
from agents.weather_report_agent import run_weather_report_agent
from agents.drift_agent import run_drift_agent
from agents.shared_state import get_logger, load_state, update_state

logger = get_logger("workflow")


# ---------------------------------------
# Configuration
# ---------------------------------------

RUN_LOG_PATH = "data/processed/agent_run_log.csv"
SCHEDULER_STATE_PATH = "data/processed/scheduler_state.json"

# Two separate schedules (Pakistan time):
#   daily       - full workflow every morning, report is always written
#   alert_check - frequent check, a report is written only when a
#                 new alert appears
SCHEDULE = {
    "daily": {"hour": 7, "minute": 0},
    "alert_check": {"every_minutes": 60},
}

# Refresh the data before every scheduled run
REFRESH_DATA_ON_SCHEDULE = True

POLL_SECONDS = 30


# ---------------------------------------
# Shared state of the graph
# ---------------------------------------

class AgentState(TypedDict, total=False):

    # Workflow
    run_id: str
    run_trigger: str
    last_reported_alerts: list
    updated_at: str

    # Forecast Agent
    forecast: dict
    forecast_time: str
    target_time: str
    forecast_models: dict
    fallback_metrics: list
    features_age_hours: float
    ledger_saved: bool
    forecast_status: str
    forecast_error: Any

    # Alert Agent
    alerts: list
    warnings: dict
    warning_source: str
    alerts_checked_at: str
    alert_status: str

    # Weather Report Agent
    weather_report: str
    report_source: str
    report_generated_at: str
    report_saved: dict
    report_status: str

    # Drift Agent
    data_drift: dict
    model_drift: dict
    drift_detected: bool
    drift_actions: Any
    drift_status: str
    drift_error: Any


STATE_KEYS = set(AgentState.__annotations__)


# ---------------------------------------
# Nodes with execution log
# ---------------------------------------

def write_run_log(state, agent, status, started, seconds, message):

    os.makedirs("data/processed", exist_ok=True)

    pd.DataFrame([{
        "run_id": state.get("run_id"),
        "trigger": state.get("run_trigger"),
        "agent": agent,
        "status": status,
        "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(seconds, 2),
        "message": message,
    }]).to_csv(
        RUN_LOG_PATH,
        mode="a",
        header=not os.path.exists(RUN_LOG_PATH),
        index=False
    )

    logger.info(f"{agent}: {status} ({seconds:.1f}s) {message}".strip())


def make_node(name, agent):

    def node(state):

        started = datetime.now()

        begin = time.perf_counter()

        try:
            result = agent(dict(state))

            status = result.get(f"{name}_status", "ok")

            message = result.get(f"{name}_error") or ""

        except Exception as error:

            # One agent failing must not stop the workflow
            logger.error(f"{name} agent crashed - {error}")

            result, status, message = {}, "crashed", str(error)

        write_run_log(
            state,
            name,
            status,
            started,
            time.perf_counter() - begin,
            message
        )

        # LangGraph only accepts keys of the state schema
        return {
            key: value for key, value in result.items()
            if key in STATE_KEYS
        }

    return node


def forecast_node(state):
    return make_node("forecast", run_forecast_agent)(state)


def alert_node(state):

    updates = make_node("alert", run_alert_agent)(state)

    # No alert now: the same alert later is a new alert again
    if not updates.get("alerts"):
        updates["last_reported_alerts"] = []

    return updates


def report_node(state):

    updates = make_node("report", run_weather_report_agent)(state)

    if updates.get("report_status") == "ok":
        updates["last_reported_alerts"] = state.get("alerts", [])

    return updates


def drift_node(state):
    return make_node("drift", run_drift_agent)(state)


# ---------------------------------------
# Routing
# ---------------------------------------

def route_after_alert(state):

    # Daily run: the report is always written
    if state.get("run_trigger") != "alert_check":
        return "report"

    # Alert check: a report only for alerts not reported before
    new_alerts = set(state.get("alerts", [])) - set(
        state.get("last_reported_alerts", [])
    )

    return "report" if new_alerts else END


def route_after_report(state):

    # Drift is checked once a day
    return "drift" if state.get("run_trigger") == "daily" else END


# ---------------------------------------
# Complete graph
# ---------------------------------------

def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("forecast", forecast_node)
    graph.add_node("alert", alert_node)
    graph.add_node("report", report_node)
    graph.add_node("drift", drift_node)

    graph.add_edge(START, "forecast")
    graph.add_edge("forecast", "alert")

    graph.add_conditional_edges(
        "alert",
        route_after_alert,
        {"report": "report", END: END}
    )

    graph.add_conditional_edges(
        "report",
        route_after_report,
        {"drift": "drift", END: END}
    )

    graph.add_edge("drift", END)

    return graph.compile()


# ---------------------------------------
# Run the workflow
# ---------------------------------------

def refresh_features():

    # Imported here: these modules are only needed for a refresh
    from data_pipeline.pipeline import run_pipeline
    from feature_engineering.f_engineering import create_features
    from feature_store.hopsworks_feature_store import run_feature_store

    for name, step in [
        ("data pipeline", run_pipeline),
        ("feature engineering", create_features),
        ("feature store", run_feature_store),
    ]:
        try:
            step()

        # run_pipeline ends with sys.exit when it fails
        except (Exception, SystemExit) as error:
            logger.error(f"Data refresh failed at {name} - {error}")
            return False

    return True


def run_workflow(trigger="daily", refresh=False):

    print(f"\nStarting workflow ({trigger})...")

    if refresh:
        refresh_features()

    state = {
        key: value for key, value in load_state().items()
        if key in STATE_KEYS
    }

    # Unique id, also for two runs in the same second
    state["run_id"] = (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    )
    state["run_trigger"] = trigger

    final_state = build_graph().invoke(state)

    # Keep what the graph added (for example the reported alerts),
    # the next run starts from this state
    update_state(final_state)

    print(f"\nWorkflow finished: {state['run_id']}")

    return final_state


# ---------------------------------------
# Scheduler (daily report and alert checks)
# ---------------------------------------

def load_last_runs():

    if not os.path.exists(SCHEDULER_STATE_PATH):
        return {}

    with open(SCHEDULER_STATE_PATH) as file:
        return {
            job: pd.Timestamp(time_text)
            for job, time_text in json.load(file).items()
        }


def save_last_runs(last_runs):

    os.makedirs("data/processed", exist_ok=True)

    with open(SCHEDULER_STATE_PATH, "w") as file:
        json.dump(
            {job: value.isoformat() for job, value in last_runs.items()},
            file
        )


def due_job(now, last_runs):

    daily = SCHEDULE["daily"]

    today_time = now.replace(
        hour=daily["hour"],
        minute=daily["minute"],
        second=0,
        microsecond=0
    )

    # The daily run also counts as an alert check
    if now >= today_time and (
        "daily" not in last_runs or last_runs["daily"] < today_time
    ):
        return "daily"

    every = pd.Timedelta(minutes=SCHEDULE["alert_check"]["every_minutes"])

    if (
        "alert_check" not in last_runs
        or now - last_runs["alert_check"] >= every
    ):
        return "alert_check"

    return None


def run_scheduler():

    print(
        "\nScheduler started: daily report at "
        f"{SCHEDULE['daily']['hour']:02d}:{SCHEDULE['daily']['minute']:02d}, "
        f"alert check every {SCHEDULE['alert_check']['every_minutes']} "
        "minutes (Ctrl+C to stop)"
    )

    last_runs = load_last_runs()

    while True:

        now = pd.Timestamp.now(tz=TIMEZONE)

        job = due_job(now, last_runs)

        if job:

            run_workflow(job, refresh=REFRESH_DATA_ON_SCHEDULE)

            last_runs[job] = now

            if job == "daily":
                last_runs["alert_check"] = now

            save_last_runs(last_runs)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AgentWeather workflow")

    parser.add_argument(
        "--trigger",
        choices=["daily", "alert_check"],
        default="daily"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="run data pipeline and feature store first"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="keep running with the daily and alert check schedule"
    )

    arguments = parser.parse_args()

    if arguments.schedule:
        run_scheduler()

    else:
        run_workflow(arguments.trigger, refresh=arguments.refresh)