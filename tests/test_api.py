
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "application" in data


def test_dashboard_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_documents_endpoint():
    response = client.get("/api/documents")

    assert response.status_code == 200
    assert "documents" in response.json()


def test_metrics_endpoint():
    response = client.get("/api/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "document_count" in data
    assert "total_pages" in data
    assert "total_chunks" in data
    assert "question_count" in data


def test_audit_endpoint():
    response = client.get("/api/audit")

    assert response.status_code == 200
    assert "events" in response.json()


def test_upload_rejects_non_pdf():
    response = client.post(
        "/api/documents",
        files={
            "file": (
                "example.txt",
                b"This is not a PDF.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_question_rejects_unknown_document():
    response = client.post(
        "/api/questions",
        json={
            "document_id": "non-existent-document",
            "question": "What is this document about?",
            "top_k": 5
        }
    )

    assert response.status_code == 404


def test_question_rejects_short_question():
    response = client.post(
        "/api/questions",
        json={
            "document_id": "non-existent-document",
            "question": "Hi",
            "top_k": 5
        }
    )

    assert response.status_code == 422


def test_evaluation_rejects_invalid_k():
    response = client.post(
        "/api/evaluations/runs",
        json={
            "k": 0
        }
    )

    assert response.status_code == 422


def test_question_history_endpoint():
    response = client.get("/api/questions")

    assert response.status_code == 200
    assert "history" in response.json()


def test_evaluation_runs_endpoint():
    response = client.get("/api/evaluations/runs")

    assert response.status_code == 200
    assert "runs" in response.json()
