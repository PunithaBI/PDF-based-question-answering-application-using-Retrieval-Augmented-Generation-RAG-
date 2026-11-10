
import json
import time

from app.config import ARTIFACT_DIR


HISTORY_FILE = ARTIFACT_DIR / "qa_history.json"


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(
            HISTORY_FILE.read_text(encoding="utf-8")
        )

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        return []


def save_history(history: list[dict]):
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = HISTORY_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(
            history,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    temporary_file.replace(HISTORY_FILE)


def add_history(result: dict) -> dict:
    history = load_history()

    # Store a timestamp for dashboard display.
    saved_result = {
        **result,
        "timestamp": result.get(
            "timestamp",
            time.strftime("%Y-%m-%dT%H:%M:%S")
        )
    }

    history.insert(0, saved_result)

    # Keep the most recent 500 questions.
    history = history[:500]

    save_history(history)

    return saved_result


def get_history(
    limit: int = 100,
    document_id: str | None = None
) -> list[dict]:

    history = load_history()

    if document_id:
        history = [
            item
            for item in history
            if item.get("document_id") == document_id
        ]

    return history[:limit]


def get_history_item(
    request_id: str
) -> dict | None:

    return next(
        (
            item
            for item in load_history()
            if item.get("request_id") == request_id
        ),
        None
    )
