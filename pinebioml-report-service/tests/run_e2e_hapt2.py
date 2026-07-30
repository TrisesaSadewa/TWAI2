import os
import sys
import json
from workers.ml_pipeline_runner import run_dynamic_pipeline
from core.report.report_engine import ReportEngine
from core.config import get_deployment_writer_model

def main():
    dataset_path = r"C:\Users\Trisesa S\Documents\TRS\ITS\IIPP\TWAI2\PineBioML\examples\input\HAPTdata.csv"
    output_dir = os.path.abspath("output/hapt_test")
    report_id = "hapt_e2e_test"
    os.makedirs(output_dir, exist_ok=True)
    
    settings = {
        "modeling_methods": ["rf"], # Just use random forest for speed
        "k_fold": 2
    }
    
    print("Running ML Pipeline...")
    res = run_dynamic_pipeline(
        report_id=report_id,
        dataset_path=dataset_path,
        target_col="target",
        settings=settings,
        output_dir=output_dir
    )
    
    if not res:
        print("ML Pipeline failed!")
        sys.exit(1)
        
    print("Running Report Engine...")
    report_engine = ReportEngine()
    
    task_type = "classification"
    metrics = {}
    cr_path = os.path.join(output_dir, "classification_report.json")
    if os.path.exists(cr_path):
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
        "pr_curve_png": "_PR Curve.png",
        "true_vs_predicted_png": "_True vs Predicted.png",
        "residuals_png": "_Residuals.png",
        "corr_heatmap_png": "_Correlation Heatmap.png",
        "pca_plot_png": "_PCA.png",
        "pls_plot_png": "_PLS.png",
        "umap_plot_png": "_UMAP.png",
        "feature_importance_csv": "feature_importance.csv",
        "feature_importance_html": "feature_importance.html",
        "shap_features_csv": "shap_features.csv",
        "shap_plot_html": "shap_plot.html",
        "imbalance_metadata_json": "imbalance_metadata.json",
    }
    
    artifacts = {}
    for key, filename in artifact_candidates.items():
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            artifacts[key] = path
            
    manifest_dict = {
        "report_id": report_id,
        "job_id": report_id,
        "dataset_name": "HAPTdata.csv",
        "task_type": task_type,
        "metrics": metrics,
        "artifacts": artifacts,
        "feature_names": [],
        "models": {"analysis": get_deployment_writer_model()}
    }
    
    print("Generating report...")
    report_engine.generate(manifest_dict, use_cpu_fallback=False)
    print("E2E Success!")

if __name__ == "__main__":
    main()
