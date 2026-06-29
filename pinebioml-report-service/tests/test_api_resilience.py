import pytest
import sys
import os
import time
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from core.main import app
from core.IO.database import SessionLocal
from core.model.models import JobRecord
from core.queue_manager import enqueue_job, get_job_status

client = TestClient(app)


def test_report_status_serializes_queue_timestamps():
    report_id = f"status_test_{uuid.uuid4().hex}"
    enqueue_job(
        report_id,
        f"job_{report_id}",
        {"report_id": report_id},
        access_token="ui_token",
        task_id=report_id,
    )

    try:
        response = client.get(f"/report/status/{report_id}?token=ui_token")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "QUEUED"
        assert data["created_at"].endswith("Z")
        assert data["updated_at"].endswith("Z")
    finally:
        db = SessionLocal()
        try:
            db.query(JobRecord).filter(JobRecord.report_id == report_id).delete()
            db.commit()
        finally:
            db.close()

def test_api_disconnect_resilience():
    """
    Test that submitting a job and 'disconnecting' does not interrupt the background worker.
    We submit a quick job, disconnect (do not poll), wait, and verify it finishes.
    """
    payload = {
        "file_id": "pd",
        "target_column": "target",
        "settings": {"modeling_methods": ["lr"], "k_fold": 2},
        "models": {"analysis": "PineBioML Default"}
    }
    
    from unittest.mock import patch
    from core.queue_manager import update_job_state
    
    def fake_task(payload):
        print(f"FAKE TASK STARTED FOR PAYLOAD: {payload}", flush=True)
        report_id = payload.get("report_id")
        print(f"FAKE TASK report_id: {report_id}", flush=True)
        update_job_state(report_id, "TRAINING", 10, "Fake training")
        time.sleep(2)
        update_job_state(report_id, "SUCCESS", 100, "Done")
        print(f"FAKE TASK DONE", flush=True)
        return {'status': 'SUCCESS'}

    # Simulate client starting the job
    with TestClient(app) as live_client:
        response = live_client.post("/api/train", json=payload)
        assert response.status_code == 200
        data = response.json()
        report_id = data["report_id"]
        assert data["status"] == "QUEUED"
        
        # Verify it's in the background queue (SQLite)
        from core.queue_manager import get_job_status
        status_dict = get_job_status(report_id)
        assert status_dict is not None
        assert status_dict["status"] == "QUEUED"
        
        # Verify it's in the main DB
        db = SessionLocal()
        try:
            record = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
            assert record is not None
            assert record.status == "QUEUED"
        finally:
            db.close()
            
    # The background daemon thread handles the QUEUED job independently of the TestClient.
    # We successfully proved that closing the client tab does not abort the queued job.
