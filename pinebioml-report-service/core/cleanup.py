"""
Retention-based cleanup for storage artifacts.

Two passes run daily (scheduled in main.py) and on-demand:

  Pass 1 — uploaded dataset purge (DATASET_RETENTION_DAYS, default 7 days)
    Datasets are only needed during the pipeline run. Delete them N days after
    the training job completes (or fails), and also N days after upload if no
    matching job record exists (orphaned uploads).

  Pass 2 — report/media/exports purge (REPORT_RETENTION_DAYS, default 90 days)
    Delete all artifacts for reports older than N days:
      storage/reports/<id>.json + <id>.html
      storage/media/<id>/          (entire directory tree)
      storage/exports/<id>.pdf + <id>_layman.pdf + <id>.docx
    Also deletes orphaned media dirs (no matching report JSON, older than N days)
    and stale failed_llm_*.json debug dumps (older than 7 days).

All deletions are logged. Errors on individual files are caught and logged so
one bad deletion never aborts the whole pass.
"""
import os
import json
import shutil
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _mtime_dt(path: str) -> datetime:
    """Return the file/dir mtime as a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _safe_remove(path: str, label: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.info(f"[cleanup] deleted {label}: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            logger.info(f"[cleanup] deleted {label} dir: {path}")
    except Exception as e:
        logger.warning(f"[cleanup] could not delete {label} {path}: {e}")


def _purge_datasets(storage_dir: str, retention_days: int) -> int:
    """Delete uploaded datasets older than retention_days after job completion.

    Filename format: <report_id>_<original_name>.<ext>
    Uses the JobRecord.updated_at (when the job last changed state) as the
    'training completed' timestamp.
    """
    if retention_days <= 0:
        return 0

    from core.IO.database import SessionLocal
    from core.model.models import JobRecord

    datasets_dir = os.path.join(storage_dir, "datasets")
    if not os.path.isdir(datasets_dir):
        return 0

    cutoff = _now() - timedelta(days=retention_days)
    deleted = 0

    with SessionLocal() as db:
        for fname in os.listdir(datasets_dir):
            fpath = os.path.join(datasets_dir, fname)
            if not os.path.isfile(fpath):
                continue
            # Extract report_id: everything before the first underscore that
            # looks like a UUID (8 hex chars + hyphens) or rep_<8hex> prefix.
            parts = fname.split("_", 1)
            report_id = parts[0] if parts else None

            delete = False
            try:
                rec = db.query(JobRecord).filter(
                    JobRecord.report_id == report_id
                ).first() if report_id else None

                terminal = {"DONE", "FAILED", "COMPLETED", "ERROR"}
                if rec and rec.status in terminal:
                    # Use job's last-updated timestamp as "training completed".
                    ref_time = rec.updated_at
                    if ref_time and ref_time.tzinfo is None:
                        ref_time = ref_time.replace(tzinfo=timezone.utc)
                    delete = ref_time is not None and ref_time < cutoff
                else:
                    # No matching record or still running — fall back to mtime.
                    delete = _mtime_dt(fpath) < cutoff
            except Exception as e:
                logger.warning(f"[cleanup] dataset check failed for {fname}: {e}")
                continue

            if delete:
                _safe_remove(fpath, "dataset")
                deleted += 1

    return deleted


def _purge_reports(storage_dir: str, media_root: str, retention_days: int) -> int:
    """Delete report JSON/HTML, media dir, and export files older than retention_days."""
    if retention_days <= 0:
        return 0

    reports_dir = os.path.join(storage_dir, "reports")
    exports_dir = os.path.join(storage_dir, "exports")
    if not os.path.isdir(reports_dir):
        return 0

    cutoff = _now() - timedelta(days=retention_days)
    deleted = 0

    # Build set of report_ids seen in the reports dir (for orphan-media check).
    live_report_ids = set()

    for fname in list(os.listdir(reports_dir)):
        if not fname.endswith(".json"):
            continue
        report_id = fname[:-5]  # strip .json
        json_path = os.path.join(reports_dir, fname)

        # Determine creation time from the report JSON (most accurate) or mtime.
        created_at = None
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("created_at") or data.get("generated_at")
            if raw:
                created_at = datetime.fromisoformat(
                    str(raw).rstrip("Z").replace("Z", "+00:00")
                )
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            pass
        if created_at is None:
            created_at = _mtime_dt(json_path)

        if created_at >= cutoff:
            live_report_ids.add(report_id)
            continue  # not old enough yet

        # Purge all artifacts for this report_id.
        _safe_remove(json_path, "report JSON")
        _safe_remove(os.path.join(reports_dir, f"{report_id}.html"), "report HTML")
        _safe_remove(os.path.join(media_root, report_id), "media dir")
        _safe_remove(os.path.join(exports_dir, f"{report_id}.pdf"), "export PDF")
        _safe_remove(os.path.join(exports_dir, f"{report_id}_layman.pdf"), "export layman PDF")
        _safe_remove(os.path.join(exports_dir, f"{report_id}.docx"), "export DOCX")
        deleted += 1

    # Orphaned media dirs (no matching report JSON, old enough).
    if os.path.isdir(media_root):
        for dname in os.listdir(media_root):
            if dname in live_report_ids:
                continue
            dpath = os.path.join(media_root, dname)
            if os.path.isdir(dpath) and _mtime_dt(dpath) < cutoff:
                _safe_remove(dpath, "orphaned media dir")
                deleted += 1

    # Stale failed_llm_*.json debug dumps (older than 7 days — low retention).
    debug_cutoff = _now() - timedelta(days=7)
    for fname in os.listdir(storage_dir):
        if fname.startswith("failed_llm_") and fname.endswith(".json"):
            fpath = os.path.join(storage_dir, fname)
            if os.path.isfile(fpath) and _mtime_dt(fpath) < debug_cutoff:
                _safe_remove(fpath, "failed_llm debug dump")
                deleted += 1

    return deleted


def run_cleanup() -> dict:
    """Run all retention passes and return a summary dict."""
    from core.config import settings

    storage_dir = settings.STORAGE_DIR
    media_root = settings.MEDIA_ROOT
    dataset_days = settings.DATASET_RETENTION_DAYS
    report_days = settings.REPORT_RETENTION_DAYS

    logger.info(f"[cleanup] starting pass (dataset_ttl={dataset_days}d, report_ttl={report_days}d)")

    n_datasets = _purge_datasets(storage_dir, dataset_days)
    n_reports = _purge_reports(storage_dir, media_root, report_days)

    summary = {"datasets_deleted": n_datasets, "reports_deleted": n_reports}
    logger.info(f"[cleanup] done: {summary}")
    return summary
