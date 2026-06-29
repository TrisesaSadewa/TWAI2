import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from core.config import settings
from core.main import app


client = TestClient(app)


def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/api/upload",
        files={"file": ("malware.exe", b"MZ fake executable", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported dataset file type" in response.json()["detail"]


def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 8)

    response = client.post(
        "/api/upload",
        files={"file": ("too_large.csv", b"target,value\n0,1\n", "text/csv")},
    )

    assert response.status_code == 413
    assert "too large" in response.json()["detail"]


def test_upload_accepts_valid_csv():
    response = client.post(
        "/api/upload",
        files={"file": ("small.csv", b"target,value\n0,1\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "small.csv"
    assert body["size_bytes"] == len(b"target,value\n0,1\n1,2\n")


def test_metrics_are_private_by_default():
    response = client.get("/metrics")

    assert response.status_code == 403


def test_metrics_allow_service_api_key():
    response = client.get("/metrics", headers={"X-API-Key": settings.SERVICE_API_KEY})

    assert response.status_code == 200
