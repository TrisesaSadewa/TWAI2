from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from core.IO.database import Base

class JobRecord(Base):
    __tablename__ = "job_records"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True, nullable=False)
    job_id = Column(String, index=True, nullable=False)
    access_token = Column(String, index=True, nullable=True)
    task_id = Column(String, nullable=True)  # Background task ID
    status = Column(String, default="QUEUED")
    # Queue state (previously lived only in the temp SQLite queue; now durable
    # in Postgres so jobs survive container restart/recreation).
    progress_pct = Column(Integer, default=0)
    message = Column(String, default="")
    model_name = Column(String, default="PineBioML Default")
    manifest_json = Column(Text, nullable=True)  # full job payload, JSON-encoded
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
