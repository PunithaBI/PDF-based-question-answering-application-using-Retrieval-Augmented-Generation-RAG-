import time
from datetime import date, datetime
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import (
    APP_NAME,
    MAX_UPLOAD_MB
)

from app.services.document_service import (
    ingest_pdf,
    load_documents,
    get_document,
    get_chunks
)

from app.services.answer_service import (
    answer_question
)

from app.services.history_service import (
    add_history,
    get_history,
    get_history_item
)

from app.services.evaluation_service import (
    run_evaluation,
    run_auto_evaluation,
    run_single_auto_evaluation,
    get_evaluation_status,
    get_evaluation_run,
    get_saved_relevance_labels,
    get_aggregate_evaluation_metrics,
    list_evaluation_runs,
    save_relevance_labels
)

from app.services.logging_service import (
    log_event,
    read_events
)

from app.services.vector_store import (
    count_document
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="Local PDF evidence retrieval and question answering."
)


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# ============================================================
# REQUEST MODELS
# ============================================================

class QuestionRequest(BaseModel):

    document_id: str = Field(
        min_length=1
    )

    question: str = Field(
        min_length=3,
        max_length=3000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )


class EvaluationRequest(BaseModel):

    request_ids: list[str] = Field(
        min_length=1
    )

    k: int = Field(
        default=5,
        ge=1,
        le=20
    )


class AutoEvaluationRequest(BaseModel):

    start_date: date

    end_date: date

    limit: int = Field(
        default=50,
        ge=1,
        le=500
    )

    k: int = Field(
        default=5,
        ge=1,
        le=20
    )


class EvaluationLabelRequest(BaseModel):

    request_id: str

    relevance_grades: dict[str, int]

    judged_all_relevant_chunks: bool = False


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/")
def dashboard():

    return FileResponse(
        STATIC_DIR / "index.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "application": APP_NAME,
        "timestamp": time.time()
    }


# ============================================================
# DOCUMENTS
# ============================================================

@app.get("/api/documents")
def documents():

    return {
        "documents": load_documents()
    }


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...)
):

    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    contents = await file.read(
        MAX_UPLOAD_MB * 1024 * 1024 + 1
    )

    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Maximum upload size is {MAX_UPLOAD_MB} MB."
        )

    try:

        result = ingest_pdf(
            original_filename=filename,
            file_bytes=contents
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        log_event(
            "document_upload_failed",
            filename=filename,
            error_type=type(exc).__name__,
            error=str(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {str(exc)}"
        )


@app.get("/api/documents/{document_id}")
def document_detail(
    document_id: str
):

    document = get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return document


@app.get("/api/documents/{document_id}/chunks")
def document_chunks(
    document_id: str
):

    document = get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return {
        "document_id": document_id,
        "chunks": get_chunks(document_id)
    }


@app.get("/api/documents/{document_id}/index")
def document_index_status(
    document_id: str
):

    document = get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:

        indexed_count = count_document(
            document_id
        )

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=f"Could not check index: {str(exc)}"
        )

    return {
        "document_id": document_id,
        "expected_chunks": document["chunk_count"],
        "indexed_chunks": indexed_count,
        "complete": (
            indexed_count == document["chunk_count"]
        )
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/api/questions")
def ask_question(
    request: QuestionRequest
):

    documents_data = load_documents()

    document = next(
        (
            item for item in documents_data
            if item.get("document_id") == request.document_id
        ),
        None
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Selected document was not found."
        )

    if (
        document.get("status") != "indexed"
        or not document.get("is_active", True)
    ):
        raise HTTPException(
            status_code=409,
            detail="Selected document is not active or indexed."
        )

    try:

        result = answer_question(
            document_id=document["document_id"],
            question=request.question,
            top_k=request.top_k
        )

        add_history(result)

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Question answering failed: {str(exc)}"
        )


# ============================================================
# QUESTION HISTORY
# ============================================================

@app.get("/api/questions")
def question_history(
    limit: int = Query(default=100, ge=1, le=500),
    document_id: str | None = None,
    search: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None
):

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date cannot be after end_date."
        )

    history = get_history(limit=500)

    filtered = []

    for item in history:

        if document_id and item.get("document_id") != document_id:
            continue

        if search:
            if search.casefold() not in item.get("question", "").casefold():
                continue

        timestamp = item.get("timestamp", "")

        try:

            question_date = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            ).date()

        except (ValueError, TypeError):

            if start_date or end_date:
                continue

            question_date = None

        if start_date and question_date and question_date < start_date:
            continue

        if end_date and question_date and question_date > end_date:
            continue

        filtered.append(item)

    return {
        "total": len(filtered),
        "history": filtered[:limit]
    }


@app.get("/api/questions/{request_id}")
def question_detail(
    request_id: str
):

    result = get_history_item(
        request_id
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Question history entry not found."
        )

    return result


@app.get("/api/questions/{request_id}/trace")
def question_trace(
    request_id: str
):

    result = get_history_item(
        request_id
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Question history entry not found."
        )

    return {
        "request_id": request_id,
        "document_id": result["document_id"],
        "processing_time_ms": result["processing_time_ms"],
        "model": result["model"],
        "sources": result["sources"]
    }


# ============================================================
# APPLICATION METRICS
# ============================================================

@app.get("/api/metrics")
def metrics():

    documents_data = load_documents()
    history_data = get_history(limit=500)

    active_documents = [
        item for item in documents_data
        if (
            item.get("status") == "indexed"
            and item.get("is_active", True)
        )
    ]

    return {
        "document_count": len(active_documents),

        "total_pages": sum(
            item.get("page_count", 0)
            for item in active_documents
        ),

        "total_chunks": sum(
            item.get("chunk_count", 0)
            for item in active_documents
        ),

        "question_count": len(history_data),

        "average_answer_time_ms": (
            round(
                sum(
                    item.get("processing_time_ms", 0)
                    for item in history_data
                ) / len(history_data),
                2
            )
            if history_data
            else 0
        )
    }


# ============================================================
# AUDIT LOG
# ============================================================

@app.get("/api/audit")
def audit_events(
    limit: int = Query(
        default=100,
        ge=1,
        le=500
    )
):

    return {
        "events": read_events(limit)
    }


# ============================================================
# DATE-RANGE AUTOMATIC RETRIEVAL EVALUATION
# ============================================================

@app.post("/api/evaluations/auto-run")
def create_auto_evaluation(
    request: AutoEvaluationRequest
):

    if request.start_date > request.end_date:

        raise HTTPException(
            status_code=400,
            detail="start_date cannot be after end_date."
        )

    try:

        return run_auto_evaluation(
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.limit,
            k=request.k
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        log_event(
            "evaluation_auto_run_failed",
            start_date=str(request.start_date),
            end_date=str(request.end_date),
            limit=request.limit,
            k=request.k,
            error_type=type(exc).__name__,
            error=str(exc)
        )

        raise HTTPException(
            status_code=502,
            detail=f"Automatic evaluation failed: {str(exc)}"
        )


# ============================================================
# SAVE RELEVANCE LABELS
# ============================================================

@app.post("/api/evaluations/labels")
def save_evaluation_labels(
    request: EvaluationLabelRequest
):

    try:

        return save_relevance_labels(
            request_id=request.request_id,
            relevance_grades=request.relevance_grades,
            judged_all_relevant_chunks=request.judged_all_relevant_chunks
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


# ============================================================
# MANUAL EVALUATION RUN
# ============================================================

@app.post("/api/evaluations/runs")
def create_evaluation(
    request: EvaluationRequest
):

    try:

        return run_evaluation(
            request_ids=request.request_ids,
            k=request.k
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        log_event(
            "evaluation_run_failed",
            error_type=type(exc).__name__,
            error=str(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(exc)}"
        )


# ============================================================
# LIST SAVED EVALUATION RUNS
# ============================================================

@app.get("/api/evaluations/runs")
def evaluation_runs():

    return {
        "runs": list_evaluation_runs()
    }


# ============================================================
# EVALUATION STATUS / QUESTION FILTERS
# ============================================================

@app.get("/api/evaluations/status")
def evaluation_status(
    search: str | None = Query(
        default=None,
        description="Search question text. Example: trade settlement"
    ),

    request_id: str | None = Query(
        default=None,
        description="Specific question ID. Example: qa-1793786465-825700"
    ),

    start_date: date | None = Query(
        default=None,
        description="Start date inclusive. Format: YYYY-MM-DD."
    ),

    end_date: date | None = Query(
        default=None,
        description="End date inclusive. Format: YYYY-MM-DD."
    ),

    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of questions to return. Range: 1–500"
    )
):

    if start_date and end_date and start_date > end_date:

        raise HTTPException(
            status_code=400,
            detail="start_date cannot be after end_date."
        )

    try:

        return get_evaluation_status(
            search=search,
            request_id=request_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


# ============================================================
# SINGLE QUESTION EVALUATION
# ============================================================

@app.post("/api/evaluations/questions/{request_id}/run")
def evaluate_single_question(
    request_id: str,
    k: int = Query(
        default=5,
        ge=1,
        le=20
    )
):

    try:

        return run_single_auto_evaluation(
            request_id=request_id,
            k=k
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        log_event(
            "single_question_evaluation_failed",
            request_id=request_id,
            error_type=type(exc).__name__,
            error=str(exc)
        )

        raise HTTPException(
            status_code=502,
            detail=f"Single-question evaluation failed: {str(exc)}"
        )


# ============================================================
# EVALUATION REPORT DETAIL
# ============================================================

@app.get("/api/evaluations/runs/{run_id}")
def evaluation_run_detail(
    run_id: str
):

    report = get_evaluation_run(
        run_id
    )

    if not report:

        raise HTTPException(
            status_code=404,
            detail="Evaluation report not found."
        )

    return report


# ============================================================
# SAVED RELEVANCE LABELS
# ============================================================

@app.get("/api/evaluations/labels/{request_id}")
def evaluation_labels_detail(
    request_id: str
):

    labels = get_saved_relevance_labels(
        request_id
    )

    if not labels:

        raise HTTPException(
            status_code=404,
            detail="Relevance labels not found."
        )

    return labels


# ============================================================
# AGGREGATE EVALUATION METRICS
# ============================================================

@app.get("/api/evaluations/metrics")
def evaluation_metrics(
    k: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Retrieval cutoff. Example: 5 for Precision@5 and nDCG@5."
    )
):

    try:

        return get_aggregate_evaluation_metrics(
            k=k
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        log_event(
            "evaluation_metrics_failed",
            error_type=type(exc).__name__,
            error=str(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not calculate evaluation metrics: {str(exc)}"
        )
