"""
Durable job queue backed by the same Postgres (via SQLAlchemy) as the rest of
the app. Replaces the previous temp-directory SQLite queue, which was lost on
container recreation / restart.

The public API is unchanged (init_db, enqueue_job, update_job_state,
get_job_status, get_job_manifest, fetch_next_job, process_job, worker_loop) so
no caller in main.py needs to change its queue calls — only the redundant
JobRecord inserts were consolidated into enqueue_job (see main.py).
"""
import json
import threading
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from core.IO.database import SessionLocal, engine, Base
from core.model.models import JobRecord


def _isoformat_z(value):
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    return value


def init_db():
    """Create the job_records table (idempotent) and recover interrupted jobs.

    On startup, any job left in TRAINING/GENERATING/STARTING from a previous
    crash or restart is marked FAILED — it won't be picked up again and won't
    hang the queue. Safe to call repeatedly.
    """
    # create_all is idempotent and only adds tables that don't exist; on a fresh
    # NCU deploy this creates job_records with all columns. (It does NOT add
    # columns to an existing table — fine for a fresh start.)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        try:
            db.query(JobRecord).filter(
                JobRecord.status.in_(("TRAINING", "GENERATING", "STARTING"))
            ).update(
                {JobRecord.status: "FAILED",
                 JobRecord.message: "Job interrupted by server crash or restart.",
                 JobRecord.progress_pct: 100},
                synchronize_session=False,
            )
            db.commit()
        except Exception as e:
            logger.error(f"init_db recovery update failed: {e}")
            db.rollback()


def enqueue_job(report_id: str, job_id: str, payload: dict,
                model_name: str = "PineBioML Default",
                access_token: str = None, expires_at: datetime = None,
                task_id: str = None):
    """Insert a complete job record (status=QUEUED) in Postgres.

    This is the single write path for new jobs — main.py no longer does a
    separate db.add(JobRecord(...)) after this, so there's no duplicate insert.
    """
    with SessionLocal() as db:
        try:
            rec = JobRecord(
                report_id=report_id,
                job_id=job_id,
                status="QUEUED",
                progress_pct=0,
                message="Job received and added to report queue.",
                model_name=model_name,
                manifest_json=json.dumps(payload),
                access_token=access_token,
                expires_at=expires_at,
                task_id=task_id,
            )
            db.add(rec)
            db.commit()
        except Exception as e:
            logger.error(f"enqueue_job failed for {report_id}: {e}")
            db.rollback()
            raise


def update_job_state(report_id: str, status: str, progress_pct: int,
                     message: str, model_name: str = None):
    with SessionLocal() as db:
        try:
            rec = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
            if rec:
                rec.status = status
                rec.progress_pct = progress_pct
                rec.message = message
                if model_name:
                    rec.model_name = model_name
                rec.updated_at = datetime.utcnow()
                db.commit()
        except Exception as e:
            logger.error(f"update_job_state failed for {report_id}: {e}")
            db.rollback()


def get_job_status(report_id: str):
    with SessionLocal() as db:
        rec = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
        if rec:
            return {
                "report_id": rec.report_id,
                "job_id": rec.job_id,
                "status": rec.status,
                "progress_pct": rec.progress_pct,
                "message": rec.message,
                "model_name": rec.model_name,
                "created_at": _isoformat_z(rec.created_at),
                "updated_at": _isoformat_z(rec.updated_at),
            }
        return None


def get_job_manifest(report_id: str):
    with SessionLocal() as db:
        rec = db.query(JobRecord).filter(JobRecord.report_id == report_id).first()
        if rec and rec.manifest_json:
            try:
                return json.loads(rec.manifest_json)
            except Exception as e:
                logger.error(f"get_job_manifest: corrupt manifest_json for {report_id}: {e}")
                return None
        return None


def fetch_next_job():
    """Atomically claim the next QUEUED job.

    On Postgres, uses SELECT ... FOR UPDATE SKIP LOCKED so multiple workers can
    claim concurrently without colliding. On SQLite (dev), falls back to a
    SELECT-then-UPDATE within one transaction (SQLite serializes writes, so the
    claim is still safe for single-process dev use). Marks the claimed job
    STARTING so it isn't re-fetched. Returns (report_id, manifest_dict) or
    (None, None).
    """
    from sqlalchemy import text
    with SessionLocal() as db:
        try:
            is_pg = engine.dialect.name == "postgresql"
            if is_pg:
                row = db.execute(
                    text("""
                        SELECT report_id, manifest_json
                        FROM job_records
                        WHERE status = 'QUEUED'
                        ORDER BY created_at ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    """)
                ).first()
            else:
                # SQLite (dev) — plain select; the subsequent UPDATE + commit
                # within this transaction claims it.
                row = db.execute(
                    text("""
                        SELECT report_id, manifest_json
                        FROM job_records
                        WHERE status = 'QUEUED'
                        ORDER BY created_at ASC
                        LIMIT 1
                    """)
                ).first()
            if row:
                report_id, manifest_json = row[0], row[1]
                db.query(JobRecord).filter(JobRecord.report_id == report_id).update(
                    {JobRecord.status: "STARTING",
                     JobRecord.updated_at: datetime.utcnow()},
                    synchronize_session=False,
                )
                db.commit()
                try:
                    manifest = json.loads(manifest_json) if manifest_json else {}
                except Exception:
                    manifest = {}
                return report_id, manifest
            db.commit()  # release any lock when nothing was claimed
            return None, None
        except Exception as e:
            logger.error(f"fetch_next_job failed: {e}")
            db.rollback()
            return None, None


def process_job(report_id, manifest_dict):
    """Run the actual ML+LLM pipeline synchronously."""
    try:
        update_job_state(report_id, "TRAINING", 5, "Initializing task...")
        # Import dynamically to avoid circular imports during startup
        from workers.tasks import train_and_generate_report_task_sync
        train_and_generate_report_task_sync(manifest_dict)
    except Exception as e:
        logger.error(f"Worker failed on job {report_id}: {e}")
        update_job_state(report_id, "FAILED", 100, f"System error: {str(e)}")


def worker_loop(concurrency: int = 4):
    """Background thread loop that pulls jobs and runs them using a bounded thread pool."""
    from concurrent.futures import ThreadPoolExecutor

    logger.info(f"Starting Postgres-backed worker loop with concurrency={concurrency}")
    init_db()

    # We use a semaphore to ensure we only fetch a job from the DB if we have a
    # free worker thread. This prevents marking jobs as STARTING but them sitting
    # in the executor's memory queue.
    semaphore = threading.Semaphore(concurrency)

    def task_wrapper(r_id, manifest):
        try:
            process_job(r_id, manifest)
        finally:
            semaphore.release()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while True:
            try:
                # Wait until we have a free worker slot
                semaphore.acquire()
                report_id, manifest = fetch_next_job()
                if report_id:
                    executor.submit(task_wrapper, report_id, manifest)
                else:
                    # No jobs in queue, release the slot and sleep
                    semaphore.release()
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                semaphore.release()
                time.sleep(5)
