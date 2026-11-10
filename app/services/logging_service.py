import json
import time
import uuid

from app.config import LOG_DIR


LOG_FILE = LOG_DIR / "events.jsonl"


def log_event(
    event_name: str,
    **details
) -> dict:
    """
    Append one structured event to the local audit log.
    """

    record = {
        "event_id": str(uuid.uuid4()),
        "event": event_name,
        "timestamp": time.time(),
        "details": details
    }

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str
            ) + "\n"
        )

    return record


def read_events(
    limit: int = 100
) -> list[dict]:
    """
    Read the latest audit events.
    """

    if not LOG_FILE.exists():
        return []

    with LOG_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()

    events = []

    for line in lines[-limit:]:

        try:
            events.append(
                json.loads(line)
            )

        except json.JSONDecodeError:
            continue

    return list(reversed(events))