import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
logging.getLogger("weasyprint").setLevel(logging.ERROR)

import os
import time
import sys

# Add parent directory to path to import report_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.report.report_engine import ReportEngine
import structlog
from core.queue_manager import update_job_state

logger = structlog.get_logger(__name__)

report_engine = ReportEngine()


def _delete_dataset(path: str) -> None:
    """Delete an uploaded dataset file after training. Only touches files
    inside storage/datasets/ — never the 'pd' synthetic path or other dirs."""
    if not path or path == "pd":
        return
    if "datasets" not in path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.info(f"Deleted uploaded dataset after training: {path}")
    except Exception as e:
        logger.warning(f"Could not delete dataset {path}: {e}")

def train_and_generate_report_task_sync(payload: dict):
    report_id = payload.get("report_id")
    file_id = payload.get("file_id")
    target_col = payload.get("target_column")
    settings = payload.get("settings", {})
    
    from workers.ml_pipeline_runner import run_dynamic_pipeline
    from core.config import settings as app_settings, get_deployment_writer_model
    
    # Locate dataset
    dataset_path = os.path.join(app_settings.STORAGE_DIR, "datasets")
    # Find the actual file extension for this file_id
    actual_file = None
    
    if file_id == "pd":
        actual_file = "pd"
    elif os.path.exists(dataset_path):
        for f in os.listdir(dataset_path):
            if f.startswith(file_id):
                actual_file = os.path.join(dataset_path, f)
                break
                
    if not actual_file:
        update_job_state(report_id, "FAILED", 100, "Dataset file not found.")
        return {'status': 'FAILED'}
        
    update_job_state(report_id, "TRAINING", 10, "Initializing PineBioML pipeline...")

    feature_names = []
    try:
        import pandas as pd
        dataset_columns = list(pd.read_csv(actual_file, nrows=0).columns)
        feature_names = [col for col in dataset_columns if col != target_col]
    except Exception as e:
        logger.warning(f"Could not read dataset feature names for report grounding: {e}")
    
    # Directory to store the plots
    output_dir = os.path.join(app_settings.MEDIA_ROOT, report_id, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        update_job_state(report_id, "TRAINING", 30, "Running auto-ML training and generating plots...")
        pipeline_result = run_dynamic_pipeline(
            report_id=report_id,
            dataset_path=actual_file,
            target_col=target_col,
            settings=settings,
            output_dir=output_dir
        )
        
        # ML is done. Build manifest and generate AI report directly in this task.
        update_job_state(report_id, "GENERATING", 70, "ML completed. Starting AI Report generation...")
        
        import json
        task_type = "classification"
        if isinstance(pipeline_result, dict):
            task_type = pipeline_result.get("task_type", task_type)
        elif os.path.exists(os.path.join(output_dir, "regression_report.json")):
            task_type = "regression"

        metrics = {}
        cr_path = os.path.join(output_dir, "classification_report.json")
        rr_path = os.path.join(output_dir, "regression_report.json")
        if task_type == "regression" and os.path.exists(rr_path):
            with open(rr_path, "r") as f:
                rr = json.load(f)
                metrics["MSE"] = round(rr.get("MSE", 0), 4)
                metrics["MAE"] = round(rr.get("MAE", 0), 4)
                metrics["R2"] = round(rr.get("R2", 0), 4)
                metrics["RMSE"] = round(rr.get("MSE", 0) ** 0.5, 4)
        elif os.path.exists(cr_path):
            with open(cr_path, "r") as f:
                cr = json.load(f)
                metrics["accuracy"] = round(cr.get("accuracy", 0), 4)
                if "macro avg" in cr:
                    metrics["precision"] = round(cr["macro avg"].get("precision", 0), 4)
                    metrics["recall"] = round(cr["macro avg"].get("recall", 0), 4)
                    metrics["f1-score"] = round(cr["macro avg"].get("f1-score", 0), 4)
                    
        artifact_candidates = {
            "model_scores_csv": "All-model-result.csv",
            "regression_report_json": "regression_report.json",
            "classification_report_json": "classification_report.json",
            "confusion_matrix_png": "_Confusion Matrix.png",
            "roc_curve_png": "_ROC Curve.png",
            "true_vs_predicted_png": "_True vs Predicted.png",
            "residuals_png": "_Residuals.png",
            "corr_heatmap_png": "_Correlation Heatmap.png",
            "pca_plot_png": "_PCA.png",
            "pls_plot_png": "_PLS.png",
            "umap_plot_png": "_UMAP.png",
            "feature_importance_html": "feature_importance.html",
            "shap_plot_html": "shap_plot.html",
        }
        artifacts = {}
        for key, filename in artifact_candidates.items():
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                artifacts[key] = path

        manifest_dict = {
            "report_id": report_id,
            "job_id": report_id,
            "dataset_name": os.path.basename(actual_file),
            "task_type": task_type,
            "metrics": metrics,
            "artifacts": artifacts,
            "feature_names": feature_names,
            "models": {"analysis": get_deployment_writer_model()}
        }
        
        def progress_callback(pct, msg, current_model=None):
            update_job_state(report_id, "GENERATING", 70 + int(pct * 0.3), msg, model_name=current_model)
            
        report_engine.generate(manifest_dict, progress_callback=progress_callback, use_cpu_fallback=False)
        
        update_job_state(report_id, "SUCCESS", 100, "Training and Report Generation completed successfully")

        # Eager dataset deletion — the file is no longer needed after training.
        # The 7-day scheduled cleanup (core/cleanup.py) is a belt-and-suspenders
        # fallback; this removes sensitive data as soon as the job completes.
        _delete_dataset(actual_file)

        return {'status': 'SUCCESS', 'report_id': report_id}
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        update_job_state(report_id, "FAILED", 100, f"Pipeline failed: {str(e)}")
        _delete_dataset(actual_file)
        return {'status': 'FAILED'}
