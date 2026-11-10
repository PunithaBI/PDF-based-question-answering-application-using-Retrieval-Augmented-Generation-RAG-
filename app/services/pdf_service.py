from pathlib import Path

import pymupdf4llm

from app.config import MAX_PDF_PAGES


def extract_pdf(pdf_path: Path) -> dict:
    """
    Extract PDF content using PyMuPDF4LLM.

    Returns:
        {
            "metadata": {...},
            "pages": [
                {
                    "page": 1,
                    "text": "..."
                }
            ]
        }
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    try:
        extracted_pages = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True
        )

    except Exception as exc:
        raise RuntimeError(
            f"PyMuPDF4LLM extraction failed: {type(exc).__name__}"
        ) from exc

    if not extracted_pages:
        raise ValueError(
            "No pages were extracted from the PDF."
        )

    if len(extracted_pages) > MAX_PDF_PAGES:
        raise ValueError(
            f"PDF exceeds the maximum limit of "
            f"{MAX_PDF_PAGES} pages."
        )

    pages = []

    for index, item in enumerate(
        extracted_pages,
        start=1
    ):

        if isinstance(item, dict):
            page_text = item.get("text", "") or ""

            page_metadata = (
                item.get("metadata", {}) or {}
            )

            page_number = page_metadata.get(
                "page",
                index
            )

        else:
            page_text = str(item)
            page_number = index

        pages.append({
            "page": int(page_number),
            "text": page_text.strip()
        })

    total_characters = sum(
        len(page["text"])
        for page in pages
    )

    non_empty_pages = sum(
        1
        for page in pages
        if page["text"].strip()
    )

    if total_characters == 0:
        raise ValueError(
            "No readable text was found. "
            "This version supports text-based PDFs, "
            "not scanned image-only PDFs."
        )

    metadata = {
        "filename": pdf_path.name,
        "page_count": len(pages),
        "non_empty_pages": non_empty_pages,
        "total_characters": total_characters,
        "extraction_method": "PyMuPDF4LLM"
    }

    return {
        "metadata": metadata,
        "pages": pages
    }