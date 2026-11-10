import os
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))

    if not path.is_absolute():
        path = BASE_DIR / path

    return path


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

APP_NAME = os.getenv(
    "APP_NAME",
    "PDF Evidence Explorer"
)


# --------------------------------------------------
# OPENROUTER
# --------------------------------------------------

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
).rstrip("/")


# --------------------------------------------------
# EMBEDDING AND RERANKING
# --------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

ENABLE_RERANKER = (
    os.getenv("ENABLE_RERANKER", "true").lower()
    == "true"
)


# --------------------------------------------------
# QDRANT
# --------------------------------------------------

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "pdf_chunks"
)


# --------------------------------------------------
# CHUNKING
# --------------------------------------------------

CHUNK_TARGET_TOKENS = env_int(
    "CHUNK_TARGET_TOKENS", 400
)

CHUNK_OVERLAP_TOKENS = env_int(
    "CHUNK_OVERLAP_TOKENS", 60
)


# --------------------------------------------------
# UPLOAD LIMITS
# --------------------------------------------------

MAX_UPLOAD_MB = env_int(
    "MAX_UPLOAD_MB", 50
)

MAX_PDF_PAGES = env_int(
    "MAX_PDF_PAGES", 1000
)


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

TOP_K = env_int("TOP_K", 5)

CANDIDATE_K = env_int(
    "CANDIDATE_K", 20
)


# --------------------------------------------------
# API
# --------------------------------------------------

REQUEST_TIMEOUT_SECONDS = env_int(
    "REQUEST_TIMEOUT_SECONDS", 60
)


# --------------------------------------------------
# LOCAL STORAGE
# --------------------------------------------------

DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = env_path(
    "UPLOAD_DIR", "data/uploads"
)

ARTIFACT_DIR = env_path(
    "ARTIFACT_DIR", "data/artifacts"
)

LOG_DIR = env_path(
    "LOG_DIR", "data/logs"
)

QDRANT_PATH = env_path(
    "QDRANT_PATH", "data/qdrant"
)


# --------------------------------------------------
# CREATE REQUIRED DIRECTORIES
# --------------------------------------------------

for directory in (
    UPLOAD_DIR,
    ARTIFACT_DIR,
    LOG_DIR,
    QDRANT_PATH
):
    directory.mkdir(
        parents=True,
        exist_ok=True
    )