import time
from datetime import date
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


class QuestionRequest(BaseModel):

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


@app.get("/")
def dashboard():

    return FileResponse(
        STATIC_DIR / "index.html"
    )


@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "application": APP_NAME,
        "timestamp": time.time()
    }


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


@app.post("/api/questions")
def ask_question(
    request: QuestionRequest
):

    documents = load_documents()

    # Use the most recently uploaded indexed PDF.
    document = next(
        (
            item
            for item in documents
            if item.get("status") == "indexed"
        ),
        None
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="No indexed PDF found. Please upload a PDF first."
        )

    document_id = document["document_id"]

    try:

        result = answer_question(
            document_id=document_id,
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

@app.get("/api/questions")
def question_history(
    limit: int = Query(
        default=100,
        ge=1,
        le=500
    ),
    document_id: str | None = None
):

    return {
        "history": get_history(
            limit=limit,
            document_id=document_id
        )
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


@app.get("/api/metrics")
def metrics():

    documents_data = load_documents()
    history_data = get_history(limit=500)

    return {
        "document_count": len(documents_data),
        "total_pages": sum(
            item.get("page_count", 0)
            for item in documents_data
        ),
        "total_chunks": sum(
            item.get("chunk_count", 0)
            for item in documents_data
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

@app.post("/api/evaluations/auto-run")
def create_auto_evaluation(
    request: AutoEvaluationRequest
):

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
            error_type=type(exc).__name__,
            error=str(exc)
        )

        raise HTTPException(
            status_code=502,
            detail=f"Automatic evaluation failed: {str(exc)}"
        )
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
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(exc)}"
        )


@app.get("/api/evaluations/runs")
def evaluation_runs():

    return {
        "runs": list_evaluation_runs()
    }