import os
import json
import logging
from datetime import datetime


# ---------------------------------------
# Configuration
# ---------------------------------------

STATE_PATH = "data/processed/agent_state.json"

LOG_PATH = "logs/agents.log"


# ---------------------------------------
# Logging (file and screen)
# ---------------------------------------

def get_logger(name):

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)

    if not logger.handlers:

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        for handler in (
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler()
        ):
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger


# ---------------------------------------
# Shared state of the agents
# ---------------------------------------

def load_state():

    if not os.path.exists(STATE_PATH):
        return {}

    with open(STATE_PATH, encoding="utf-8") as file:
        return json.load(file)


def update_state(state, **changes):

    state.update(changes)

    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)

    with open(STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, default=str)

    return state