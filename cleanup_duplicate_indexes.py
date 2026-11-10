
import hashlib
import shutil
from pathlib import Path

from app.config import UPLOAD_DIR, ARTIFACT_DIR
from app.services.document_service import load_documents, save_documents
from app.services.vector_store import count_document, delete_document


CANONICAL_ID = "c789776b-d708-4b45-b187-107daded4a39"

DUPLICATE_IDS = [
    "c0dffb7d-59a9-40eb-a59b-d002af8fbcc2",
    "d42ac756-fc49-4ddf-b9ae-69b8990d1d8c",
]

EXPECTED_POINTS = 718
EXPECTED_SHA256 = (
    "34d99351ec7e5da97dc709972c94aab0a117e9437f4caa0caae3781c6e865472"
)

DOCUMENTS_FILE = ARTIFACT_DIR / "documents.json"
PROJECT_DIR = ARTIFACT_DIR.parent.parent
BACKUP_ARTIFACTS = (
    PROJECT_DIR / "data_backup_before_dedup_20261105" / "artifacts"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main():
    print("=== DUPLICATE DOCUMENT CLEANUP ===")

    if not DOCUMENTS_FILE.exists():
        raise RuntimeError("documents.json was not found.")

    if not BACKUP_ARTIFACTS.exists():
        raise RuntimeError("The verified backup artifacts folder was not found.")

    documents = load_documents()

    required_ids = [CANONICAL_ID] + DUPLICATE_IDS
    records = {}

    for document_id in required_ids:
        matches = [
            doc for doc in documents
            if doc.get("document_id") == document_id
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one metadata record for {document_id}; "
                f"found {len(matches)}."
            )

        records[document_id] = matches[0]

    print("\n1. Checking PDF files and hashes...")

    hashes = {}

    for document_id in required_ids:
        pdf_path = UPLOAD_DIR / f"{document_id}.pdf"

        if not pdf_path.exists():
            raise RuntimeError(f"PDF is missing: {pdf_path}")

        hashes[document_id] = sha256_file(pdf_path)

        print(document_id, hashes[document_id])

        if hashes[document_id] != EXPECTED_SHA256:
            raise RuntimeError(
                f"Hash mismatch for {document_id}. Nothing was deleted."
            )

    print("\n2. Checking Qdrant point counts...")

    counts = {}

    for document_id in required_ids:
        counts[document_id] = count_document(document_id)
        print(document_id, ":", counts[document_id])

    if counts[CANONICAL_ID] != EXPECTED_POINTS:
        raise RuntimeError(
            "Canonical document does not have exactly 718 points. "
            "Nothing was deleted."
        )

    for document_id in DUPLICATE_IDS:
        if counts[document_id] not in (0, EXPECTED_POINTS):
            raise RuntimeError(
                f"Unexpected point count for {document_id}: "
                f"{counts[document_id]}. Nothing was deleted."
            )

    print("\n3. Checking document filenames...")

    canonical_filename = records[CANONICAL_ID].get("filename")

    for document_id in DUPLICATE_IDS:
        if records[document_id].get("filename") != canonical_filename:
            raise RuntimeError(
                "Document filenames do not match. Nothing was deleted."
            )

    print("\nAll safety checks passed.")
    print("Canonical document:", CANONICAL_ID)
    print("Duplicate documents:", DUPLICATE_IDS)

    confirmation = input(
        '\nType DELETE DUPLICATE VECTORS to continue: '
    )

    if confirmation != "DELETE DUPLICATE VECTORS":
        print("Cancelled. No changes made.")
        return

    print("\n4. Backing up documents.json...")

    backup_file = (
        BACKUP_ARTIFACTS / "documents.json.before_dedup_cleanup"
    )

    shutil.copy2(DOCUMENTS_FILE, backup_file)
    print("Metadata backup:", backup_file)

    print("\n5. Removing duplicate Qdrant vectors...")

    for document_id in DUPLICATE_IDS:
        current_count = count_document(document_id)

        if current_count == EXPECTED_POINTS:
            print("Deleting vectors for:", document_id)
            delete_document(document_id)
        elif current_count == 0:
            print("Already empty:", document_id)

        remaining = count_document(document_id)

        if remaining != 0:
            raise RuntimeError(
                f"Verification failed for {document_id}: "
                f"{remaining} points remain."
            )

        print("Verified zero points:", document_id)

    print("\n6. Updating document metadata...")

    for document_id in DUPLICATE_IDS:
        records[document_id]["is_active"] = False
        records[document_id]["superseded_by_document_id"] = CANONICAL_ID
        records[document_id]["duplicate_content"] = True

    records[CANONICAL_ID]["is_active"] = True
    records[CANONICAL_ID]["content_hash"] = EXPECTED_SHA256

    save_documents(documents)

    print("\n7. Final verification...")

    final_canonical_count = count_document(CANONICAL_ID)

    if final_canonical_count != EXPECTED_POINTS:
        raise RuntimeError(
            f"Canonical count changed unexpectedly: "
            f"{final_canonical_count}"
        )

    for document_id in DUPLICATE_IDS:
        if count_document(document_id) != 0:
            raise RuntimeError(
                f"Duplicate vectors reappeared for {document_id}"
            )

    print("Canonical vectors:", final_canonical_count)
    print("First duplicate vectors:", count_document(DUPLICATE_IDS[0]))
    print("Second duplicate vectors:", count_document(DUPLICATE_IDS[1]))
    print("Original PDFs preserved.")
    print("Extraction JSON files preserved.")
    print("QA history and evaluation files preserved.")
    print("\nCLEANUP COMPLETED.")


if __name__ == "__main__":
    main()
