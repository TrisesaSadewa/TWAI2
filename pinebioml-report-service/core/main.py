import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
logging.getLogger("weasyprint").setLevel(logging.ERROR)

import os
import json
import time
import uuid
import secrets
from pathlib import Path as FilePath
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, Path, Body, Query, Depends, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import asyncio

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
api_key_optional_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.SERVICE_API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return api_key

def verify_metrics_access(api_key: str = Security(api_key_optional_header)):
    if settings.PUBLIC_METRICS:
        return None
    if api_key != settings.SERVICE_API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return api_key
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.security import (
    sanitize_report_id, validate_uuid_param, sanitize_filename, 
    safe_path_join, safe_error_detail, SecurityHeadersMiddleware, CSRFMiddleware
)

import core.model.schemas as schemas
from core.config import settings, get_deployment_writer_model
from core.report.report_engine import ReportEngine
from workers.tasks import train_and_generate_report_task_sync

# Database Imports
from sqlalchemy.orm import Session
from core.IO.database import SessionLocal, engine, get_db, Base
from core.model.models import JobRecord

# Setup Structlog
import structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger(__name__)

# Initialize DB Tables
Base.metadata.create_all(bind=engine)

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the background worker loop in a separate thread
    from core.queue_manager import worker_loop
    import threading
    worker_thread = threading.Thread(target=worker_loop, kwargs={"concurrency": 4}, daemon=True)
    worker_thread.start()
    logger.info("Started Postgres-backed queue worker thread.")

    # Start the daily storage-cleanup thread
    def _cleanup_loop():
        import time as _time
        from core.cleanup import run_cleanup
        while True:
            try:
                run_cleanup()
            except Exception as _e:
                logger.warning(f"Storage cleanup pass failed: {_e}")
            _time.sleep(86400)  # run once per day

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info("Started storage cleanup thread (dataset_ttl=7d, report_ttl=90d).")
    
    yield
    # Shutdown logic goes here

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="PineBioML RESTful microservice for generating clinical AI narrative reports.",
    docs_url="/docs" if settings.ENABLE_API_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_API_DOCS else None,
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")), name="static")

os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
app.mount("/media", StaticFiles(directory=settings.MEDIA_ROOT), name="media")

# Enable CORS for frontend/viewer integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CSRFMiddleware)

# Global variables for prometheus counters
START_TIME = time.time()

# Prometheus metrics
REPORTS_TRIGGERED = Counter("pinebioml_reports_triggered_total", "Total number of reports triggered")
REPORTS_COMPLETED = Counter("pinebioml_reports_completed_total", "Total number of successfully generated reports")
REPORTS_FAILED = Counter("pinebioml_reports_failed_total", "Total number of report generation failures")
LATENCY_HISTOGRAM = Histogram("pinebioml_report_generation_duration_seconds", "Latency of report generation")

# Initialize Engine
report_engine = ReportEngine()

# Queue Manager imports
from core.queue_manager import enqueue_job, get_job_status, get_job_manifest, worker_loop
from core.streaming import subscribe as subscribe_stream, unsubscribe as unsubscribe_stream
import threading

from fastapi import UploadFile, File, Form

GENERIC_UPLOAD_MIME_TYPES = {
    "",
    "application/octet-stream",
    "binary/octet-stream",
}
ALLOWED_UPLOAD_MIME_TYPES = {
    "text/csv",
    "text/tab-separated-values",
    "text/plain",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

def _allowed_upload_extensions() -> set[str]:
    return {
        ext.strip().lower()
        for ext in settings.ALLOWED_UPLOAD_EXTENSIONS.split(",")
        if ext.strip()
    }

def _validate_upload_metadata(file: UploadFile) -> str:
    filename = file.filename or ""
    ext = FilePath(filename).suffix.lower()
    if not ext or ext not in _allowed_upload_extensions():
        allowed = ", ".join(sorted(_allowed_upload_extensions()))
        raise HTTPException(status_code=400, detail=f"Unsupported dataset file type. Allowed extensions: {allowed}")

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in GENERIC_UPLOAD_MIME_TYPES and content_type not in ALLOWED_UPLOAD_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported upload content type: {content_type}")

    return ext.lstrip(".")

async def _save_validated_upload(file: UploadFile, destination: str) -> int:
    total = 0
    first_chunk = b""
    max_bytes = settings.MAX_UPLOAD_BYTES

    with open(destination, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            if not first_chunk:
                first_chunk = chunk
            total += len(chunk)
            if total > max_bytes:
                buffer.close()
                try:
                    os.remove(destination)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file is too large. Maximum allowed size is {max_bytes} bytes."
                )
            buffer.write(chunk)

    if total == 0:
        try:
            os.remove(destination)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    ext = FilePath(destination).suffix.lower()
    if ext == ".xlsx" and not first_chunk.startswith(b"PK"):
        try:
            os.remove(destination)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Uploaded Excel file does not appear to be a valid workbook.")
    if ext == ".xls" and not first_chunk.startswith(b"\xD0\xCF\x11\xE0"):
        try:
            os.remove(destination)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Uploaded Excel file does not appear to be a valid legacy workbook.")

    if ext in {".csv", ".tsv"}:
        try:
            first_chunk.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                first_chunk.decode("latin-1")
            except UnicodeDecodeError:
                try:
                    os.remove(destination)
                except OSError:
                    pass
                raise HTTPException(status_code=400, detail="Uploaded text dataset could not be decoded.")

    return total

@app.post("/api/upload")
@limiter.limit("5/minute")
async def upload_dataset(request: Request, file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """Temporarily store dataset for ML training."""
    ext = _validate_upload_metadata(file)
    dataset_id = f"data_{uuid_hash()}"
    safe_filename = f"{dataset_id}.{ext}"
    
    os.makedirs(os.path.join(settings.STORAGE_DIR, "datasets"), exist_ok=True)
    file_path = os.path.join(settings.STORAGE_DIR, "datasets", safe_filename)
    
    size_bytes = await _save_validated_upload(file, file_path)
        
    return {"file_id": dataset_id, "filename": file.filename, "path": file_path, "size_bytes": size_bytes}

@app.post("/api/train")
@limiter.limit("10/minute")
def train_and_report(
    request: Request,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Unified endpoint to run PineBioML and then generate report."""
    REPORTS_TRIGGERED.inc()
    report_id = f"rep_{uuid_hash()}"
    
    # Store settings in payload
    payload["report_id"] = report_id
    
    # Enqueue into the (durable, Postgres-backed) job queue. enqueue_job writes
    # the complete JobRecord in one insert, so no separate db.add(JobRecord) here.
    file_id = payload.get("file_id", "local_job")
    access_token = secrets.token_urlsafe(16)
    try:
        enqueue_job(report_id, file_id, payload, model_name="PineBioML Default",
                    access_token=access_token, task_id=report_id)
    except Exception as e:
        logger.error(f"Failed to enqueue task: {e}")
        raise HTTPException(status_code=503, detail="Task queue broker offline.")

    return {
        "report_id": report_id,
        "access_token": access_token,
        "status": "QUEUED",
        "progress_pct": 0,
        "message": "Training job queued."
    }

@app.post("/report/generate", response_model=schemas.ReportStatus, status_code=202)
@limiter.limit("10/minute")
def generate_report(
    request: Request,
    manifest: schemas.JobManifest, 
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Trigger AI narrative report generation asynchronously.
    """
    REPORTS_TRIGGERED.inc()
    report_id = f"rep_{uuid_hash()}"
    manifest_dict = manifest.dict()
    manifest_dict["report_id"] = report_id
    manifest_fields = getattr(manifest, "model_fields_set", getattr(manifest, "__fields_set__", set()))
    if "models" not in manifest_fields:
        manifest_dict.setdefault("models", {})["analysis"] = get_deployment_writer_model()
    effective_model = manifest_dict.get("models", {}).get("analysis", "PineBioML Default")
    
    # Calculate expires_at + access token first, then enqueue the complete
    # JobRecord in one insert (no separate db.add here).
    from datetime import timedelta
    expires_at = datetime.utcnow() + timedelta(days=manifest.expiry_days) if manifest.expiry_days else None
    access_token = secrets.token_urlsafe(16)

    try:
        enqueue_job(report_id, manifest.job_id, manifest_dict, model_name=effective_model,
                    access_token=access_token, expires_at=expires_at, task_id=report_id)
    except Exception as e:
        logger.error(f"Failed to enqueue task: {e}")
        raise HTTPException(status_code=503, detail="Task queue broker offline.")

    return {
        "report_id": report_id,
        "access_token": access_token,
        "job_id": manifest.job_id,
        "status": "QUEUED",
        "progress_pct": 0,
        "message": "Job received and added to report queue.",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "task_id": report_id,
        "model_name": effective_model
    }

@app.get("/report/status/{report_id}", response_model=schemas.ReportStatus)
@limiter.limit("60/minute")
def get_status(
    request: Request,
    report_id: str = Path(..., description="The generated report ID"), 
    token: str = Query(None, description="Access token for browser polling"),
    db: Session = Depends(get_db)
):
    """Get the current generation status of the report. Allows either X-API-Key or token."""
    report_id = sanitize_report_id(report_id)
    job_record = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
    
    if job_record and job_record.access_token:
        # If token is wrong/missing, it might be the microservice polling without a token
        # We skip throwing 403 here because PineBioML might use the API Key instead of token,
        # but for viewer.html, the token must be passed.
        if token and token != job_record.access_token:
            raise HTTPException(status_code=403, detail="Invalid token")
    
    # First check our SQLite Job Queue status
    status_dict = get_job_status(report_id)
    if status_dict:
        # We also need to map this to schemas.ReportStatus format
        return {
            "report_id": status_dict.get("report_id") or report_id,
            "job_id": status_dict.get("job_id") or "unknown",
            "status": status_dict.get("status") or "FAILED",
            "progress_pct": status_dict.get("progress_pct") or 0,
            "message": status_dict.get("message") or "",
            "created_at": status_dict.get("created_at") or "",
            "updated_at": status_dict.get("updated_at") or "",
            "model_name": status_dict.get("model_name") or "PineBioML Default"
        }
        
    # If not in SQLite (e.g. server restarted and it was cleaned up, or it's very old)
    if not job_record:
        # Attempt to load from saved JSON if the server restarted
        report_data = load_report_json(report_id)
        if report_data:
            return {
                "report_id": report_id,
                "job_id": report_data.get("job_id", "unknown"),
                "status": "SUCCESS",
                "progress_pct": 100,
                "message": "Report found in storage.",
                "created_at": report_data.get("created_at", ""),
                "updated_at": report_data.get("updated_at", ""),
                "model_name": report_data.get("model_name", "PineBioML Default")
            }
        raise HTTPException(status_code=404, detail="Report ID not found")
        
    # Fallback if somehow it's in JobRecord but not in SQLite queue or JSON storage
    created_val = ""
    if job_record.created_at:
        try: created_val = job_record.created_at.isoformat() + "Z"
        except Exception: pass
    updated_val = ""
    if job_record.updated_at:
        try: updated_val = job_record.updated_at.isoformat() + "Z"
        except Exception: pass

    return {
        "report_id": report_id,
        "job_id": job_record.job_id or "unknown",
        "status": "FAILED",
        "progress_pct": 100,
        "message": "Job disappeared from queue manager.",
        "created_at": created_val,
        "updated_at": updated_val,
        "model_name": "PineBioML Default"
    }

@app.get("/report/{report_id}", response_model=schemas.ReportMetadata)
@limiter.limit("30/minute")
def get_report_metadata(request: Request, report_id: str = Path(...), db: Session = Depends(get_db), api_key: str = Depends(verify_api_key)):
    """Retrieve report details and download links."""
    report_id = sanitize_report_id(report_id)
    report_data = load_report_json(report_id)
    if not report_data:
        raise HTTPException(status_code=404, detail="Report not found or not yet complete")
        
    job_record = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
    token_suffix = f"?token={job_record.access_token}" if job_record and job_record.access_token else ""
        
    return {
        "report_id": report_id,
        "job_id": report_data["job_id"],
        "dataset_name": report_data["dataset_name"],
        "task_type": report_data["task_type"],
        "status": "SUCCESS",
        "created_at": report_data["created_at"],
        "updated_at": report_data["updated_at"],
        "html_url": f"/report/{report_id}/html{token_suffix}",
        "download_links": [
            {"format": "pdf", "url": f"/report/{report_id}/download/pdf{token_suffix}"},
            {"format": "docx", "url": f"/report/{report_id}/download/docx{token_suffix}"}
        ]
    }

@app.get("/report/{report_id}/data")
@limiter.limit("30/minute")
def get_raw_data(request: Request, report_id: str = Path(...), api_key: str = Depends(verify_api_key)):
    """Get the raw quantitative metrics and visual metadata."""
    report_id = sanitize_report_id(report_id)
    report_data = load_report_json(report_id)
    if not report_data:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
        "metrics": report_data["metrics"],
        "visuals": report_data["visuals"]
    }

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Serve the privacy-first local storage dashboard."""
    dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "templates", "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="Dashboard not found")

@app.get("/access", response_class=HTMLResponse)
def get_access_portal(request: Request):
    """Serve the independent access portal for entering Report ID and Token."""
    return templates.TemplateResponse(request=request, name="access.html", context={"request": request})

@app.get("/dashboard/result/{report_id}", response_class=HTMLResponse)
def get_dashboard_result(request: Request, report_id: str):
    """Serve the native ML Result dashboard using Jinja2."""
    manifest = get_job_manifest(report_id)
    if not manifest:
        report_data = load_report_json(report_id)
        if report_data:
            manifest = report_data
    if not manifest:
        raise HTTPException(status_code=404, detail="Job Manifest not found or expired.")

    all_models_data = manifest.get("all_models_data", [])
    
    return templates.TemplateResponse(request=request, name="result.html", context={
        "request": request,
        "report_id": report_id,
        "manifest": manifest,
        "all_models_data": all_models_data
    })

@app.get("/api/report/{report_id}/manifest")
@limiter.limit("30/minute")
def get_manifest(request: Request, report_id: str, api_key: str = Depends(verify_api_key)):
    """Retrieve the ML job manifest."""
    report_id = sanitize_report_id(report_id)
    manifest = get_job_manifest(report_id)
    if not manifest:
        manifest = load_report_json(report_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Job Manifest not found or expired.")
    return manifest

@app.get("/report/{report_id}/html", response_class=HTMLResponse)
@limiter.limit("30/minute")
def get_html_viewer(request: Request, report_id: str = Path(...), token: str = Query(None), direct: bool = Query(False), db: Session = Depends(get_db)):
    """Serve the interactive HTML viewer or the polling page."""
    report_id = sanitize_report_id(report_id)
    job_record = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
    if job_record and job_record.access_token and job_record.access_token != token:
        raise HTTPException(status_code=403, detail="Invalid or missing access token")
        
    if job_record and job_record.expires_at and datetime.utcnow() > job_record.expires_at:
        return HTMLResponse(content="<html><body><h1>Report Expired</h1><p>This shareable link has expired.</p></body></html>", status_code=410)

    html_path = os.path.join(settings.STORAGE_DIR, "reports", f"{report_id}.html")
    
    # If direct=true or report exists
    if os.path.exists(html_path) and (direct or (job_record and job_record.status == "SUCCESS")):
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            if token:
                # Inject the token into the export links
                content = content.replace(f'/report/{report_id}/download/pdf', f'/report/{report_id}/download/pdf?token={token}')
                content = content.replace(f'/report/{report_id}/download/docx', f'/report/{report_id}/download/docx?token={token}')
            return content
            
    # Check status to serve polling view
    if job_record:
        if job_record.status in ("QUEUED", "ANALYZING", "GENERATING", "PENDING", "PROCESSING"):
            viewer_template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "templates", "viewer.html")
            if os.path.exists(viewer_template_path):
                with open(viewer_template_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    content = content.replace("{{ REPORT_ID }}", report_id)
                    content = content.replace("{{ TOKEN }}", token if token else "")
                    return content
            return f"<html><body><h1>Processing Report {report_id}...</h1></body></html>"
        elif job_record.status == "FAILED":
            return f"<html><body><h1>Report generation failed</h1><p>Check server logs for details.</p></body></html>"
    
        
    raise HTTPException(status_code=404, detail="HTML report not found and not processing")

@app.get("/report/stream/{report_id}")
@limiter.limit("30/minute")
async def stream_report(request: Request, report_id: str = Path(...), token: str = Query(None), db: Session = Depends(get_db)):
    """SSE endpoint for streaming the LLM JSON narrative chunks."""
    report_id = sanitize_report_id(report_id)
    job_record = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
    if job_record and job_record.access_token and job_record.access_token != token:
        raise HTTPException(status_code=403, detail="Invalid token")
        
    async def event_generator():
        stream_queue = subscribe_stream(report_id)
        try:
            # Yield initial connection message
            yield "data: {\"connected\": true}\n\n"

            while True:
                try:
                    data = stream_queue.get_nowait()
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    yield f"data: {json.dumps({'content': data})}\n\n"
                except Exception:
                    status_dict = get_job_status(report_id)
                    if status_dict and status_dict["status"] in ["SUCCESS", "FAILED"]:
                        yield "data: [DONE]\n\n"
                        return

                await asyncio.sleep(0.1)
        finally:
            unsubscribe_stream(report_id, stream_queue)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/report/{report_id}/download/{fmt}")
@limiter.limit("15/minute")
def download_export(request: Request, report_id: str = Path(...), fmt: str = Path(...), token: str = Query(None), db: Session = Depends(get_db)):
    """Download the report as a PDF or DOCX file."""
    report_id = sanitize_report_id(report_id)
    job_record = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
    if job_record and job_record.access_token and job_record.access_token != token:
        raise HTTPException(status_code=403, detail="Invalid token")
        
    if job_record and job_record.expires_at and datetime.utcnow() > job_record.expires_at:
        raise HTTPException(status_code=410, detail="This report link has expired.")

    if fmt not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="Invalid format. Supported formats: pdf, docx")
        
    export_path = os.path.join(settings.STORAGE_DIR, "exports", f"{report_id}.{fmt}")
    if not os.path.exists(export_path):
        raise HTTPException(status_code=404, detail=f"Export file not found for format: {fmt}")
        
    media_type = "application/pdf" if fmt == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    filename = f"PineBioML_Report_{report_id}.{fmt}"
    return FileResponse(export_path, media_type=media_type, filename=filename)

@app.patch("/report/{report_id}/edit", response_model=schemas.EditResponse)
@limiter.limit("15/minute")
def edit_report_section(request: Request, report_id: str = Path(...), edit: schemas.SectionEdit = Body(...), api_key: str = Depends(verify_api_key)):
    """Allow saving edits directly to sections of the AI narrative."""
    report_id = sanitize_report_id(report_id)
    report_data = load_report_json(report_id)
    if not report_data:
        raise HTTPException(status_code=404, detail="Report not found")
        
    try:
        # Edit the specific section
        narrative = report_data["narrative"]
        sec_key = edit.key
        
        if edit.mode in ("expert", "both"):
            if sec_key in narrative["expert"]:
                narrative["expert"][sec_key] = edit.content
                
        report_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Save back JSON
        meta_path = os.path.join(settings.STORAGE_DIR, "reports", f"{report_id}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
        # Re-render HTML with updated text
        html_content = report_engine._render_html_report(report_data)
        html_path = os.path.join(settings.STORAGE_DIR, "reports", f"{report_id}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        # Regenerate PDF and DOCX exports
        pdf_path = os.path.join(settings.STORAGE_DIR, "exports", f"{report_id}.pdf")
        docx_path = os.path.join(settings.STORAGE_DIR, "exports", f"{report_id}.docx")
        report_engine.export_engine.export_to_pdf(html_content, pdf_path)
        report_engine.export_engine.export_to_docx(report_data, docx_path)
            
        return {
            "success": True,
            "message": f"Successfully updated section '{sec_key}'.",
            "report_id": report_id,
            "updated_at": report_data["updated_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to edit report section: {str(e)}")

@app.post("/report/{report_id}/clone", response_model=schemas.ReportMetadata)
@limiter.limit("5/minute")
def clone_report(request: Request, report_id: str = Path(...), edit: schemas.SectionEdit = Body(None), db: Session = Depends(get_db), api_key: str = Depends(verify_api_key)):
    """Clone an existing report and optionally apply an edit to the new copy."""
    report_id = sanitize_report_id(report_id)
    report_data = load_report_json(report_id)
    if not report_data:
        raise HTTPException(status_code=404, detail="Original report not found")
        
    try:
        new_id = f"rep_{uuid.uuid4().hex[:8]}"
        report_data["report_id"] = new_id
        
        # Apply edit if provided
        if edit:
            narrative = report_data["narrative"]
            sec_key = edit.key
            if edit.mode in ("expert", "both") and sec_key in narrative["expert"]:
                narrative["expert"][sec_key] = edit.content
                
        report_data["created_at"] = datetime.utcnow().isoformat() + "Z"
        report_data["updated_at"] = report_data["created_at"]
        
        # Save JSON
        meta_path = os.path.join(settings.STORAGE_DIR, "reports", f"{new_id}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
        # Generate HTML
        html_content = report_engine._render_html_report(report_data)
        html_path = os.path.join(settings.STORAGE_DIR, "reports", f"{new_id}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        # Generate PDF and DOCX
        pdf_path = os.path.join(settings.STORAGE_DIR, "exports", f"{new_id}.pdf")
        docx_path = os.path.join(settings.STORAGE_DIR, "exports", f"{new_id}.docx")
        report_engine.export_engine.export_to_pdf(html_content, pdf_path)
        report_engine.export_engine.export_to_docx(report_data, docx_path)
        
        # Update DB
        new_token = secrets.token_urlsafe(16)
        original_record = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
        if original_record:
            new_record = JobRecord(
                report_id=new_id,
                job_id=original_record.job_id,
                task_id=original_record.task_id,
                status="SUCCESS",
                expires_at=original_record.expires_at,
                access_token=new_token
            )
            db.add(new_record)
            db.commit()
            
        token_suffix = f"?token={new_token}"
        return {
            "report_id": new_id,
            "job_id": report_data["job_id"],
            "dataset_name": report_data["dataset_name"],
            "task_type": report_data["task_type"],
            "status": "SUCCESS",
            "created_at": report_data["created_at"],
            "updated_at": report_data["updated_at"],
            "html_url": f"/report/{new_id}/html{token_suffix}",
            "download_links": [
                {"format": "pdf", "url": f"/report/{new_id}/download/pdf{token_suffix}"},
                {"format": "docx", "url": f"/report/{new_id}/download/docx{token_suffix}"}
            ]
        }
    except Exception as e:
        logger.error(f"Failed to clone report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clone report: {str(e)}")

@app.get("/health", response_model=schemas.HealthCheck)
def health_check(db: Session = Depends(get_db)):
    """Get service health metrics."""
    queue_count = db.query(JobRecord).filter(JobRecord.status.in_(["QUEUED", "ANALYZING", "GENERATING"])).count()
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - START_TIME,
        "queue_count": queue_count,
        "system_load": 0.1  # Mock system load
    }

@app.get("/metrics")
def get_prometheus_metrics(api_key: str = Depends(verify_metrics_access)):
    """Export Prometheus format metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



@app.get("/models")
def list_available_models():
    """
    List all analysis models available for report generation.
    Use the 'id' field in the models.analysis parameter when triggering a report.
    """
    from core.config import list_analysis_models
    models = list_analysis_models()
    return {
        "models": models,
        "default": get_deployment_writer_model(),
        "hint": "Pass model 'id' as models.analysis in /report/generate payload"
    }

# Frontend HTML UI Routes
from fastapi.responses import RedirectResponse

@app.get("/")
def redirect_to_upload():
    return RedirectResponse(url="/Statistical_Analysis/upload")

@app.get("/api/artifacts/{report_id}/{filename:path}")
@limiter.limit("30/minute")
def get_artifact(request: Request, report_id: str, filename: str, api_key: str = Depends(verify_api_key)):
    report_id = sanitize_report_id(report_id)
    filename = sanitize_filename(filename)
    try:
        file_path = safe_path_join(os.path.join(settings.MEDIA_ROOT, report_id, "output"), filename)
    except HTTPException:
        raise HTTPException(status_code=400, detail="Invalid file path.")
        
    if os.path.exists(file_path):
        from fastapi.responses import FileResponse
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Not Found")

import uuid

@app.get("/Statistical_Analysis/upload")
def ui_upload_page(request: Request):
    new_uuid = str(uuid.uuid4())
    return templates.TemplateResponse(
        name="access.html", 
        request=request, 
        context={"request": request, "uuid": new_uuid}
    )

@app.api_route("/Statistical_Analysis/upload/check/{uuid}/", methods=["GET", "POST"])
@app.api_route("/Statistical_Analysis/upload/check_test/{uuid}/", methods=["GET", "POST"])
def ui_check_page(request: Request, uuid: str, data_file: UploadFile = File(None), dataset: str = Form(None)):
    from core.config import settings as app_settings
    import pandas as pd
    import glob
    import os
    import shutil
    
    uuid = validate_uuid_param(uuid)
    dataset_path = os.path.join(app_settings.STORAGE_DIR, "datasets")
    os.makedirs(dataset_path, exist_ok=True)
    
    # Clean up old files with the same UUID to prevent picking up stale datasets
    if (data_file and data_file.filename) or dataset:
        for f in os.listdir(dataset_path):
            if f.startswith(uuid):
                try:
                    os.remove(os.path.join(dataset_path, f))
                except Exception:
                    pass

    # Handle direct file upload
    if data_file and data_file.filename:
        safe_filename = f"{uuid}_{data_file.filename}"
        file_path = os.path.join(dataset_path, safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(data_file.file, buffer)
            
    # Handle test dataset selection
    elif dataset:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        static_examples = os.path.join(base_dir, "static", "examples")
        source_file = None
        target_name = None
        
        if dataset == "heart":
            source_file = os.path.join(static_examples, "heart_disease_cleveland.csv")
            target_name = f"{uuid}_heart_disease_cleveland.csv"
        elif dataset == "pd":
            source_file = os.path.join(static_examples, "Breast_Cancer_Wisconsin_diagnostic.csv")
            target_name = f"{uuid}_Breast_Cancer_Wisconsin_diagnostic.csv"
        elif dataset == "parkinsons":
            source_file = os.path.join(static_examples, "parkinsons_disease.tsv")
            target_name = f"{uuid}_parkinsons_disease.tsv"
        elif dataset == "diabetes":
            source_file = os.path.join(static_examples, "diabetes_disease_progression.xlsx")
            target_name = f"{uuid}_diabetes_disease_progression.xlsx"
            
        if source_file and os.path.exists(source_file):
            file_path = os.path.join(dataset_path, target_name)
            shutil.copy(source_file, file_path)
    
    actual_file = None
    original_filename = "Unknown Dataset"
    columns = []
    rows = []
    
    if os.path.exists(dataset_path):
        # Find the latest file starting with the uuid
        matching_files = [f for f in os.listdir(dataset_path) if f.startswith(uuid)]
        if matching_files:
            # Sort by modification time descending
            matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(dataset_path, x)), reverse=True)
            f = matching_files[0]
            actual_file = os.path.join(dataset_path, f)
            original_filename = f[len(uuid)+1:] # remove uuid_
                
    if actual_file:
        try:
            if actual_file.endswith(".csv"):
                df = pd.read_csv(actual_file, nrows=10)
            elif actual_file.endswith(".tsv") or actual_file.endswith(".txt"):
                df = pd.read_csv(actual_file, sep="\t", nrows=10)
            elif actual_file.endswith(".xlsx"):
                df = pd.read_excel(actual_file, nrows=10)
            else:
                df = pd.read_csv(actual_file, nrows=10)
            columns = df.columns.tolist()
            rows = df.values.tolist()
        except Exception as e:
            pass

    return templates.TemplateResponse(
        name="check.html", 
        request=request, 
        context={
            "request": request, 
            "uuid": uuid,
            "filename": original_filename,
            "columns": columns,
            "rows": rows
        }
    )

def load_ml_metrics(uuid: str):
    from core.config import settings as app_settings
    import pandas as pd
    import json
    import os

    output_dir = os.path.join(app_settings.MEDIA_ROOT, uuid, "output")
    all_results = []
    all_results_columns = []
    classification_report = {}
    regression_report = {}
    imbalance_metadata = {}
    task_type = "classification"

    csv_path = os.path.join(output_dir, "All-model-result.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Round numeric columns for display
            df = df.round(4)
            # Fill NaN with empty string
            df = df.fillna('')
            all_results = df.to_dict(orient="records")
            all_results_columns = df.columns.tolist()
        except Exception as e:
            pass

    json_path = os.path.join(output_dir, "classification_report.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                classification_report = json.load(f)
                task_type = "classification"
        except Exception as e:
            pass

    reg_path = os.path.join(output_dir, "regression_report.json")
    if os.path.exists(reg_path):
        try:
            with open(reg_path, "r") as f:
                regression_report = json.load(f)
                task_type = "regression"
        except Exception as e:
            pass

    # Imbalance handling / methodology metadata (written by the pipeline when
    # class_weight or threshold tuning was applied). Surfaced on the actual-result
    # page so the metrics table is interpretable (e.g. which class sensitivity
    # tracks, and that a tuned threshold was used).
    imb_path = os.path.join(output_dir, "imbalance_metadata.json")
    if os.path.exists(imb_path):
        try:
            with open(imb_path, "r") as f:
                imbalance_metadata = json.load(f)
        except Exception as e:
            pass

    return all_results, all_results_columns, classification_report, regression_report, task_type, imbalance_metadata

@app.api_route("/Statistical_Analysis/result/{uuid}/", methods=["GET"])
def ui_actual_result_page(request: Request, uuid: str):
    uuid = validate_uuid_param(uuid)
    media_base = f"/media/{uuid}/output"
    all_results, all_results_columns, classification_report, regression_report, task_type, imbalance_metadata = load_ml_metrics(uuid)

    return templates.TemplateResponse(name="actual_result.html", request=request, context={
        "request": request,
        "uuid": uuid,
        "media_base": media_base,
        "all_results": all_results,
        "all_results_columns": all_results_columns,
        "classification_report": classification_report,
        "regression_report": regression_report,
        "task_type": task_type,
        "imbalance_metadata": imbalance_metadata or {}
    })

@app.api_route("/Statistical_Analysis/setting/{uuid}/", methods=["GET", "POST"])
async def ui_setting_page(request: Request, uuid: str):
    uuid = validate_uuid_param(uuid)
    target_column = ""
    if request.method == "POST":
        form_data = await request.form()
        target_column = form_data.get("target_column", "")
    return templates.TemplateResponse(name="setting.html", request=request, context={"request": request, "uuid": uuid, "target_column": target_column})

@app.api_route("/Statistical_Analysis/result_example/{uuid}/", methods=["GET", "POST"])
async def ui_result_page(request: Request, uuid: str, db: Session = Depends(get_db)):
    uuid = validate_uuid_param(uuid)
    form_data = await request.form()
    
    missing_methods = form_data.getlist("missing_value_methods")
    norm_methods = form_data.getlist("normalization_methods")
    feature_methods = form_data.getlist("feature_selection_methods")
    modeling_methods = form_data.getlist("modeling_methods")
    validation_method = form_data.get("validation_method", "k-fold cross validation")
    k_fold = form_data.get("k_fold", "5")
    tuning_strategy = form_data.get("tuning_strategy", "RandomizedSearchCV")
    tuning_n_iter = form_data.get("tuning_n_iter", "10")
    
    missing_count = len([m for m in missing_methods if m not in ("missing_all", "None")])
    norm_count = len([m for m in norm_methods if m not in ("norm_all", "None")])
    feature_count = len([m for m in feature_methods if m not in ("feature_all", "None")])
    modeling_count = len([m for m in modeling_methods if m not in ("modeling_all", "None")])
    
    # ── Trigger ML Task ──────────────────────────────────────────────
    settings = {
        "missing_value_methods": missing_methods,
        "normalization_methods": norm_methods,
        "feature_selection_methods": feature_methods,
        "modeling_methods": modeling_methods,
        "validation_method": validation_method,
        "k_fold": k_fold,
        "tuning_strategy": tuning_strategy,
        "tuning_n_iter": int(tuning_n_iter)
    }
    
    payload = {
        "report_id": uuid,
        "file_id": uuid, 
        "target_column": form_data.get("target_column", "target"),
        "settings": settings
    }
    
    from workers.tasks import train_and_generate_report_task_sync
    from core.queue_manager import enqueue_job
    import threading
    try:
        # Check if we already have a record for this uuid to avoid duplicates if re-submitted
        existing_job = db.query(JobRecord).filter(JobRecord.report_id == uuid).first()
        if not existing_job:
            # enqueue_job writes the complete JobRecord (Postgres-backed) in one insert.
            enqueue_job(uuid, "pd", payload, access_token="ui_token", task_id=uuid)

            threading.Thread(target=train_and_generate_report_task_sync, args=(payload,)).start()
    except Exception as e:
        print(f"Failed to enqueue task: {e}")
    # ─────────────────────────────────────────────────────────────────
    
    return templates.TemplateResponse(name="result.html", request=request, context={
        "uuid": uuid,
        "missing_count": max(1, missing_count),
        "norm_count": max(1, norm_count),
        "feature_count": max(1, feature_count),
        "modeling_count": max(1, modeling_count)
    })

@app.api_route("/Statistical_Analysis/download/{uuid}/", methods=["GET", "POST"])
def ui_download_page(request: Request, uuid: str):
    uuid = validate_uuid_param(uuid)
    all_results, all_results_columns, classification_report, regression_report, task_type, _imbalance = load_ml_metrics(uuid)
    return templates.TemplateResponse(name="download.html", request=request, context={
        "request": request,
        "uuid": uuid,
        "all_results": all_results,
        "all_results_columns": all_results_columns,
        "classification_report": classification_report,
        "regression_report": regression_report,
        "task_type": task_type
    })

@app.api_route("/Statistical_Analysis/download/download_selected_files/{uuid}/", methods=["POST"])
def ui_download_zip(request: Request, uuid: str):
    uuid = validate_uuid_param(uuid)
    from fastapi.responses import FileResponse
    from core.config import settings as app_settings
    import os, zipfile
    
    output_dir = os.path.join(app_settings.MEDIA_ROOT, uuid, "output")
    if not os.path.exists(output_dir):
        return {"error": "No output files found to download"}
        
    zip_path = os.path.join(app_settings.MEDIA_ROOT, uuid, "PinBioML_Results.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                filepath = os.path.join(root, file)
                zf.write(filepath, os.path.relpath(filepath, output_dir))
                
    if os.path.exists(zip_path):
        return FileResponse(path=zip_path, filename="PinBioML_Results.zip", media_type="application/zip")
    return {"error": "Failed to create zip file"}


# Helper Functions
def uuid_hash() -> str:
    import uuid
    return uuid.uuid4().hex[:8]

def load_report_json(report_id: str) -> dict:
    meta_path = os.path.join(settings.STORAGE_DIR, "reports", f"{report_id}.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            return report_data
        except Exception:
            return None
    return None
