from datetime import datetime


_RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

EVENT_LOG = f"main_agent_{_RUN_TIMESTAMP}.log"
DATA_LOG = f"main_data_{_RUN_TIMESTAMP}.log"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_event(message: str):

    with open(EVENT_LOG, "a", encoding="utf-8") as file:
        file.write(f"[{_timestamp()}] {message}\n")


def log_data(label: str, data):

    with open(DATA_LOG, "a", encoding="utf-8") as file:
        file.write(f"\n[{_timestamp()}] {label}\n")
        file.write(f"{data}\n")
        file.write("-" * 100 + "\n")