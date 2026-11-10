
# PDF Evidence Explorer

A local PDF question-answering application built with FastAPI, PyMuPDF4LLM, BGE embeddings, Qdrant, BM25, hybrid retrieval, cross-encoder reranking, and OpenRouter.

## 1. Architecture

PDF upload
    |
    v
PyMuPDF4LLM extraction
    |
    v
Page-aware chunking
    |
    v
BGE-small embeddings
    |
    v
Local persistent Qdrant
    |
    v
Question
    |
    +---- Dense retrieval
    |
    +---- BM25 keyword retrieval
    |
    v
Reciprocal Rank Fusion
    |
    v
Cross-encoder reranking
    |
    v
Evidence context
    |
    v
OpenRouter LLM
    |
    v
Answer with page and chunk references

## 2. Technology

| Component | Technology |
|---|---|
| Backend | FastAPI |
| PDF extraction | PyMuPDF4LLM |
| Chunking | Custom structure-aware chunker |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector database | Qdrant local persistent mode |
| Keyword retrieval | BM25 |
| Hybrid retrieval | Reciprocal Rank Fusion |
| Reranking | Cross-encoder |
| LLM | OpenRouter |
| Dashboard | HTML, CSS, JavaScript |
| Audit | Local JSONL event log |

## 3. Requirements

- Windows 10 or 11
- Python 3.12 recommended
- Internet connection for initial model downloads and OpenRouter
- OpenRouter API key

## 4. Installation

Open Windows CMD.

Move to the project directory:

    cd /d "YOUR_PROJECT_PATH\pdf_qa_qdrant_final"

Create a virtual environment:

    py -3.12 -m venv .venv

Activate it:

    .venv\Scripts\activate.bat

Upgrade pip:

    python -m pip install --upgrade pip

Install dependencies:

    pip install -r requirements.txt

## 5. Configuration

Copy the example environment file:

    copy .env.example .env

Open `.env` and enter your OpenRouter API key:

    OPENROUTER_API_KEY=your_api_key_here

Do not share or commit your API key.

The initial model configuration is:

    OPENROUTER_MODEL=openrouter/free

The embedding model is:

    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

The reranker is:

    RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

## 6. Start the application

Run:

    run_windows.bat

Open:

    http://127.0.0.1:8000

API documentation:

    http://127.0.0.1:8000/docs

## 7. First use

1. Open the dashboard.
2. Upload a text-based PDF.
3. Wait for extraction and indexing to finish.
4. Select the uploaded document.
5. Enter a question.
6. Review the generated answer.
7. Inspect the supporting page and chunk references.

## 8. Evaluation

Create:

    experiments/benchmark.jsonl

Each line represents one benchmark case.

Example:

    {"case_id":"BENCH-001","document_id":"YOUR_DOCUMENT_ID","question":"What are the main stages of the trade lifecycle?","relevance_grades":{}}

Use actual chunk IDs and human-reviewed relevance grades.

Grades:

- 0: Not relevant
- 1: Partially relevant
- 2: Relevant
- 3: Highly relevant

Run an evaluation from the dashboard or:

    POST /api/evaluations/runs

Metrics include:

- Precision@k
- Recall@k
- MRR
- nDCG@k

Unjudged retrieved chunks are not automatically treated as irrelevant.

## 9. Local storage

The application stores data under:

    data/uploads/
    data/artifacts/
    data/logs/
    data/qdrant/

Do not delete these folders while the application is running.

The Qdrant database is local and persistent. Run the application as a single process. Do not start multiple application instances against the same Qdrant storage directory.

## 10. Current limitations

- Scanned image-only PDFs are not supported in this version.
- OCR is not included.
- Qdrant runs in local persistent mode, not as a separate server.
- OpenRouter requires internet access.
- The application is intended for local development and evaluation, not public deployment.
- Langfuse integration is not included in the current implementation; local JSONL audit logging is provided.

## 11. Security notes

- Keep `.env` private.
- Do not expose the local API publicly.
- Treat uploaded PDF content as untrusted.
- Review answers against their cited source pages.
- Do not use generated answers as a substitute for professional or legally authoritative advice.

## 12. Project status

This is an initial implementation. Run the validation and test steps before relying on the application.
