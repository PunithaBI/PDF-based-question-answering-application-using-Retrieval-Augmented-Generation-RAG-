
import hashlib
import json
import time
import uuid

from pathlib import Path

from app.config import (
    UPLOAD_DIR,
    ARTIFACT_DIR,
    MAX_UPLOAD_MB
)

from app.services.pdf_service import extract_pdf
from app.services.chunk_service import chunk_pages
from app.services.embedding_service import (
    embed_texts,
    model_limit
)
from app.services.vector_store import (
    upsert_chunks,
    count_document,
    delete_document
)
from app.services.logging_service import log_event


DOCUMENTS_FILE = ARTIFACT_DIR / "documents.json"


def load_documents() -> list[dict]:
    if not DOCUMENTS_FILE.exists():
        return []

    try:
        data = json.loads(
            DOCUMENTS_FILE.read_text(encoding="utf-8")
        )
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_documents(documents: list[dict]):
    temporary_file = DOCUMENTS_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(documents, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    temporary_file.replace(DOCUMENTS_FILE)


def get_document(document_id: str) -> dict | None:
    return next(
        (
            document
            for document in load_documents()
            if document.get("document_id") == document_id
        ),
        None
    )


def get_chunks(document_id: str) -> list[dict]:
    extraction_file = ARTIFACT_DIR / f"{document_id}.json"

    if not extraction_file.exists():
        return []

    data = json.loads(
        extraction_file.read_text(encoding="utf-8")
    )

    return data.get("chunks", [])


def _safe_filename(original_filename: str) -> str:
    return Path(original_filename.replace("\\", "/")).name


def _calculate_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _get_existing_document_hash(document: dict) -> str | None:
    """
    Read the saved hash or calculate it for older document records.
    """
    existing_hash = document.get("content_hash")

    if existing_hash:
        return existing_hash

    document_id = document.get("document_id")

    if not document_id:
        return None

    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"

    if not pdf_path.exists():
        return None

    try:
        existing_hash = _calculate_sha256(pdf_path.read_bytes())
        document["content_hash"] = existing_hash
        return existing_hash
    except OSError:
        return None


def _is_active(document: dict) -> bool:
    return (
        document.get("status") == "indexed"
        and document.get("is_active", True) is not False
    )


def _same_filename(document: dict, filename: str) -> bool:
    return (
        str(document.get("filename", "")).casefold()
        == filename.casefold()
    )


def _mark_superseded(document: dict):
    document["status"] = "superseded"
    document["is_active"] = False
    document["superseded_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def _find_duplicate(
    documents: list[dict],
    content_hash: str
) -> dict | None:
    """
    Find a document with identical file content.
    """
    for document in documents:
        existing_hash = _get_existing_document_hash(document)

        if existing_hash == content_hash:
            return document

    return None


def _cleanup_new_document(document_id: str):
    """
    Remove only the incomplete data belonging to the failed upload.
    """
    try:
        delete_document(document_id)
    except Exception:
        pass

    (UPLOAD_DIR / f"{document_id}.pdf").unlink(missing_ok=True)
    (ARTIFACT_DIR / f"{document_id}.json").unlink(missing_ok=True)


def _return_duplicate(
    document: dict,
    uploaded_filename: str,
    reactivated: bool = False
) -> dict:
    result = dict(document)
    result["duplicate_detected"] = True
    result["indexing_skipped"] = True
    result["uploaded_filename"] = uploaded_filename
    result["reactivated"] = reactivated
    return result


def ingest_pdf(
    original_filename: str,
    file_bytes: bytes
) -> dict:

    started = time.perf_counter()

    if not original_filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are accepted.")

    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(
            f"Maximum upload size is {MAX_UPLOAD_MB} MB."
        )

    safe_filename = _safe_filename(original_filename)

    if not safe_filename:
        raise ValueError("The uploaded filename is invalid.")

    content_hash = _calculate_sha256(file_bytes)
    documents = load_documents()

    # ---------------------------------------------------------
    # 1. Check whether identical content is already stored.
    # ---------------------------------------------------------

    duplicate = _find_duplicate(documents, content_hash)

    if duplicate:
        duplicate_id = duplicate.get("document_id")

        # If this is a superseded version, reactivate it.
        if not _is_active(duplicate):

            for document in documents:
                if (
                    document.get("document_id") != duplicate_id
                    and _same_filename(document, safe_filename)
                    and _is_active(document)
                ):
                    _mark_superseded(document)

            duplicate["status"] = "indexed"
            duplicate["is_active"] = True
            duplicate.pop("superseded_at", None)

            save_documents(
                [duplicate] + [
                    item for item in documents
                    if item.get("document_id") != duplicate_id
                ]
            )

            try:
                log_event(
                    "document_version_reactivated",
                    document_id=duplicate_id,
                    filename=duplicate.get("filename"),
                    uploaded_filename=safe_filename
                )
            except Exception:
                pass

            return _return_duplicate(
                duplicate,
                safe_filename,
                reactivated=True
            )

        # Identical content is already active. Do not index again.
        try:
            log_event(
                "duplicate_document_upload",
                document_id=duplicate_id,
                filename=duplicate.get("filename"),
                uploaded_filename=safe_filename,
                content_hash=content_hash
            )
        except Exception:
            pass

        return _return_duplicate(
            duplicate,
            safe_filename
        )

    # ---------------------------------------------------------
    # 2. Identify the currently active version of this filename.
    # ---------------------------------------------------------

    previous_versions = [
        item for item in documents
        if _same_filename(item, safe_filename)
    ]

    active_previous = [
        item for item in previous_versions
        if _is_active(item)
    ]

    previous_document = (
        active_previous[0] if active_previous else None
    )

    version = max(
        [
            int(item.get("version", 1))
            for item in previous_versions
        ],
        default=0
    ) + 1

    document_id = str(uuid.uuid4())

    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    extraction_path = ARTIFACT_DIR / f"{document_id}.json"

    committed = False

    try:
        # 3. Save the new PDF.
        pdf_path.write_bytes(file_bytes)

        # 4. Extract pages.
        extraction = extract_pdf(pdf_path)
        pages = extraction["pages"]

        # 5. Create chunks.
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError("No searchable text was extracted.")

        # 6. Generate embeddings.
        vectors, embedding_stats = embed_texts(
            [chunk["text"] for chunk in chunks]
        )

        # 7. Index chunks in Qdrant.
        indexed_count = upsert_chunks(
            document_id,
            chunks,
            vectors
        )

        if indexed_count != len(chunks):
            raise RuntimeError(
                f"Indexing incomplete: expected {len(chunks)} chunks, "
                f"but Qdrant accepted {indexed_count}."
            )

        # 8. Save extraction data.
        extraction_path.write_text(
            json.dumps(
                {
                    "metadata": extraction["metadata"],
                    "pages": pages,
                    "chunks": chunks
                },
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        token_counts = [
            chunk["token_count_estimate"]
            for chunk in chunks
        ]

        elapsed_ms = round(
            (time.perf_counter() - started) * 1000,
            2
        )

        # 9. Create new version metadata.
        record = {
            "document_id": document_id,
            "filename": safe_filename,
            "content_hash": content_hash,
            "version": version,
            "is_active": True,
            "supersedes_document_id": (
                previous_document.get("document_id")
                if previous_document else None
            ),
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "indexed_chunk_count": indexed_count,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "indexing_time_ms": elapsed_ms,
            "extraction_method": "PyMuPDF4LLM",
            "embedding": embedding_stats,
            "chunk_statistics": {
                "minimum_tokens": min(token_counts),
                "maximum_tokens": max(token_counts),
                "average_tokens": round(
                    sum(token_counts) / len(token_counts),
                    2
                ),
                "model_max_sequence_length": model_limit()
            },
            "status": "indexed"
        }

        # -----------------------------------------------------
        # 10. Commit the new version.
        # Old versions are preserved, not deleted.
        # -----------------------------------------------------

        for document in documents:
            if (
                _same_filename(document, safe_filename)
                and _is_active(document)
            ):
                _mark_superseded(document)

        updated_documents = [record] + documents

        save_documents(updated_documents)
        committed = True

        try:
            log_event(
                "document_version_created" if previous_document
                else "document_indexed",
                document_id=document_id,
                filename=safe_filename,
                version=version,
                content_hash=content_hash,
                supersedes_document_id=record[
                    "supersedes_document_id"
                ],
                page_count=len(pages),
                chunk_count=len(chunks),
                indexing_time_ms=elapsed_ms
            )
        except Exception:
            pass

        return record

    except Exception:
        if not committed:
            _cleanup_new_document(document_id)

        raise
