import os
import json
import uuid
import logging
import html as html_lib
from datetime import datetime
import pandas as pd
from core.config import settings
from core.preprocessing.visual_analyzer import VisualAnalyzer
from core.report.narrative_generator import NarrativeGenerator
from core.report.export_engine import ExportEngine

logger = logging.getLogger(__name__)

class ReportEngine:
    """
    Orchestrates the entire report generation lifecycle.
    """
    
    def __init__(self):
        self.visual_analyzer = VisualAnalyzer()
        self.narrative_generator = NarrativeGenerator()
        self.export_engine = ExportEngine()
        
    def generate(self, manifest: dict, progress_callback=None, use_cpu_fallback=False) -> dict:
        """
        Accepts a job manifest, processes it, and generates report assets.
        Supports progress_callback(pct, msg) to report status.
        """
        # Wrap progress_callback to support current_model dynamically
        orig_callback = progress_callback
        if orig_callback:
            import inspect
            try:
                sig = inspect.signature(orig_callback)
                has_current_model = 'current_model' in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            except Exception:
                has_current_model = False
            
            def wrapped_callback(pct, msg, current_model=None):
                if has_current_model:
                    orig_callback(pct, msg, current_model=current_model)
                else:
                    orig_callback(pct, msg)
            progress_callback = wrapped_callback

        report_id = manifest.get("report_id", f"rep_{uuid.uuid4().hex[:8]}")
        job_id = manifest["job_id"]
        dataset_name = manifest["dataset_name"]
        task_type = manifest["task_type"]
        artifacts = manifest["artifacts"]
        
        logger.info(f"Starting report generation for report_id={report_id}, job_id={job_id}...")
        
        def resolve_artifact_path(path):
            if not path:
                return path
            if not os.path.isabs(path) and not os.path.exists(path):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                pinebioml_dir = os.path.join(base_dir, "PineBioML")
                resolved = os.path.normpath(os.path.join(pinebioml_dir, path))
                if os.path.exists(resolved):
                    return resolved
            return path
            
        artifacts = {k: resolve_artifact_path(v) for k, v in artifacts.items()}
        artifacts = self._discover_output_artifacts(report_id, artifacts)

        if progress_callback:
            progress_callback(10, "Reading your analysis results...")
        
        # 1. Start with manifest metrics (normalized keys)
        raw_manifest_metrics = manifest.get("metrics", {})
        metrics = {}
        key_map = {
            "accuracy": "accuracy", "roc-auc": "ROC-AUC", "roc_auc": "ROC-AUC",
            "precision": "precision", "recall": "recall", "f1-score": "F1-Score", "f1": "F1-Score",
            "specificity": "specificity", "mcc": "MCC",
            "r2": "R2", "r²": "R2", "r_squared": "R2", "r-squared": "R2",
            "mse": "MSE", "mae": "MAE", "rmse": "RMSE"
        }
        for k, v in raw_manifest_metrics.items():
            normalized_key = key_map.get(k.lower(), k)
            metrics[normalized_key] = v

        if task_type == "regression":
            parsed_metrics = self._parse_regression_report(artifacts.get("regression_report_json"))
            for k, v in parsed_metrics.items():
                if v != "N/A":
                    metrics[key_map.get(k.lower(), k)] = v
            for k in ["R2", "RMSE", "MAE", "MSE"]:
                if k not in metrics:
                    metrics[k] = "N/A"
        else:
            # 1a. Try to parse from CSV if provided, overwrite if not N/A
            parsed_metrics = self._parse_scores_csv(artifacts.get("optimal_scores_csv") or artifacts.get("scores_csv"))
            for k, v in parsed_metrics.items():
                if v != "N/A":
                    normalized_key = key_map.get(k.lower(), k)
                    metrics[normalized_key] = v

            # Fill defaults for anything still missing
            for k in ["accuracy", "ROC-AUC", "precision", "recall", "F1-Score", "specificity", "MCC"]:
                if k not in metrics:
                    metrics[k] = "N/A"

        all_models = self._parse_all_models_csv(artifacts.get("model_scores_csv"))
        
        # 1b. Compute real ROC-AUC from best model in all_models (replaces hardcoded 0.8650)
        if task_type != "regression":
            real_auc = self._extract_best_auc(all_models)
            if real_auc:
                metrics["ROC-AUC"] = real_auc
            
        # 1b-bis. Fallback for missing metrics
        if task_type != "regression":
            self._fill_missing_metrics_from_all_models(metrics, all_models)
        
        # 1c. Parse per-class breakdown (top 5 most important classes)
        per_class = [] if task_type == "regression" else self._parse_per_class_metrics(artifacts.get("optimal_scores_csv") or artifacts.get("scores_csv"))
        if task_type != "regression" and not per_class:
            per_class = self._parse_per_class_metrics_json(artifacts.get("classification_report_json"))
        
        # 1c-bis. Validate class labels against dataset context to prevent cross-dataset contamination
        per_class = self._validate_class_labels(per_class, dataset_name, all_models)

        imbalance_warning = {} if task_type == "regression" else self._build_imbalance_warning(metrics, per_class, all_models)
        
        # 1d. Compute train/test gap for overfitting analysis
        overfit_analysis = self._compute_overfit_gap(all_models)
        
        # 1e. Parse selected feature names
        selected_features = self._parse_selected_features(artifacts.get("selected_features_csv"))
        manifest_features = manifest.get("feature_names") or manifest.get("selected_features") or []
        if not selected_features and isinstance(manifest_features, list):
            selected_features = [str(feature) for feature in manifest_features if str(feature).strip()]
        
        # 2. Parse Explainability (SHAP/LIME) and Deterministic Anomalies
        shap_features = self._parse_shap_csv(artifacts.get("feature_importance_csv") or artifacts.get("shap_csv"))
        anomaly_flags = self._flag_anomalies(metrics)
        if imbalance_warning:
            anomaly_flags.append(imbalance_warning["message"])
        
        if progress_callback:
            progress_callback(30, "Understanding your charts and plots...")
        # 3. Analyze visual plots
        visuals_summary = self.visual_analyzer.analyze_plots(artifacts, progress_callback=progress_callback)
        html_visuals = {
            key: {"path": path, "description_fallback": ""}
            for key, path in artifacts.items()
            if path and str(path).lower().endswith(".html")
        }
        combined_visuals_summary = {**visuals_summary, **html_visuals}
        
        if progress_callback:
            progress_callback(50, "Consulting biomedical ML guidelines...")
        # (Placeholder for RAG retrieval)
        
        # Extract preprocessing and class-imbalance metadata to feed to the LLM.
        # Prefer manifest metadata, but also load the pipeline artifact written by
        # workers/ml_pipeline_runner.py so the narrative can fact-check the exact
        # imbalance handling that was actually applied.
        prep_meta = manifest.get("imbalance_metadata", {}) or {}
        if not prep_meta and artifacts.get("imbalance_metadata_json"):
            try:
                with open(artifacts["imbalance_metadata_json"], "r", encoding="utf-8") as f:
                    prep_meta = json.load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load imbalance metadata artifact: {e}")
        if str(task_type).lower().find("regression") >= 0:
            report_card_bars_html = self._render_fallback_metrics_table(metrics)
            stars_html = '<span class="star">☆</span><span class="star">☆</span><span class="star">☆</span><span class="star">☆</span><span class="star">☆</span>'
            quality_label = "Regression Performance: Review Metrics"
        elif all_models:
            keys = list(all_models[0].keys())
            acc_key = next((k for k in keys if "accuracy" in k.lower()), None)
            best_model = max(all_models, key=lambda m: float(m.get(acc_key, 0) or 0)) if acc_key else all_models[0]
            
            missing_val = next((best_model[k] for k in keys if k.lower() == "missingvalueprocessing"), None)
            if missing_val: prep_meta["Missing Value Imputation"] = missing_val
            
            std_val = next((best_model[k] for k in keys if k.lower() in ("standarization", "standardization")), None)
            if std_val: prep_meta["Standardization"] = std_val
            
            sel_val = next((best_model[k] for k in keys if k.lower() == "selection"), None)
            if sel_val: prep_meta["Feature Selection"] = sel_val

        if progress_callback:
            progress_callback(70, "Writing your personalized report...")
        # 4. Generate Narratives
        models = manifest.get("models", {})
        narrative = self.narrative_generator.generate_narrative(
            dataset_name=dataset_name,
            task_type=task_type,
            metrics=metrics,
            visuals_summary=combined_visuals_summary,
            shap_features=shap_features,
            anomaly_flags=anomaly_flags,
            models=models,
            use_cpu_fallback=use_cpu_fallback,
            report_id=report_id,
            per_class=per_class,
            overfit_analysis=overfit_analysis,
            selected_features=selected_features,
            imbalance_metadata=prep_meta if prep_meta else None,
            imbalance_warning=imbalance_warning if imbalance_warning else None,
            all_models=all_models
        )
        
        # Save results in report structure
        models = manifest.get("models", {})
        narrative_meta = narrative.get("_meta", {}) if isinstance(narrative, dict) else {}
        narrative_source = narrative_meta.get("source", "llm")
        if narrative_source == "narrative_unavailable":
            narrative_status = "unavailable"
        elif narrative_source == "llm_partial_with_unavailable_sections":
            narrative_status = "partial"
        else:
            narrative_status = "generated"
        report_data = {
            "report_id": report_id,
            "job_id": job_id,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "metrics": metrics,
            "all_models": all_models,
            "per_class": per_class,
            "overfit_analysis": overfit_analysis,
            "imbalance_metadata": prep_meta,
            "imbalance_warning": imbalance_warning,
            "selected_features": selected_features,
            "visuals": {k: v["path"] for k, v in combined_visuals_summary.items()},
            "narrative": narrative,
            "narrative_source": narrative_source,
            "narrative_status": narrative_status,
            "narrative_notice": narrative_meta.get("reason", ""),
            "model_name": models.get("analysis", "PineBioML Default"),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        
        if progress_callback:
            progress_callback(90, "Formatting and finalizing your report...")
        # 5. Render HTML Report Viewer
        html_content = self._render_html_report(report_data)
        
        # Save HTML
        html_path = os.path.join(settings.STORAGE_DIR, "reports", f"{report_id}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        # 5. Export to PDF and DOCX
        pdf_path = os.path.join(settings.STORAGE_DIR, "exports", f"{report_id}.pdf")
        docx_path = os.path.join(settings.STORAGE_DIR, "exports", f"{report_id}.docx")
        
        pdf_ok = self.export_engine.export_to_pdf(html_content, pdf_path)
        docx_ok = self.export_engine.export_to_docx(report_data, docx_path)
        if not pdf_ok or not docx_ok:
            failed = []
            if not pdf_ok:
                failed.append("PDF")
            if not docx_ok:
                failed.append("DOCX")
            raise RuntimeError(f"Failed to generate export file(s): {', '.join(failed)}")
        
        # Save metadata JSON
        meta_path = os.path.join(settings.STORAGE_DIR, "reports", f"{report_id}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
        logger.info(f"Successfully generated report assets for {report_id}.")
        return {
            "report_id": report_id,
            "status": "SUCCESS",
            "message": "Report generated successfully."
        }

    def _discover_output_artifacts(self, report_id: str, artifacts: dict) -> dict:
        """Normalize artifacts and fill missing paths from the standard output folder."""
        output_dir = os.path.join(settings.MEDIA_ROOT, report_id, "output")
        candidates = {
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
        aliases = {
            "all-model-result.csv": "model_scores_csv",
            "all_model_result.csv": "model_scores_csv",
            "all_models.csv": "model_scores_csv",
            "scores.csv": "scores_csv",
            "optimal_scores.csv": "optimal_scores_csv",
            "selected_features.csv": "selected_features_csv",
            "classification_report.json": "classification_report_json",
            "regression_report.json": "regression_report_json",
            "_confusion matrix.png": "confusion_matrix_png",
            "confusion matrix.png": "confusion_matrix_png",
            "confusion_matrix.png": "confusion_matrix_png",
            "_roc curve.png": "roc_curve_png",
            "roc curve.png": "roc_curve_png",
            "roc_curve.png": "roc_curve_png",
            "_true vs predicted.png": "true_vs_predicted_png",
            "true vs predicted.png": "true_vs_predicted_png",
            "true_vs_predicted.png": "true_vs_predicted_png",
            "_residuals.png": "residuals_png",
            "residuals.png": "residuals_png",
            "_correlation heatmap.png": "corr_heatmap_png",
            "correlation heatmap.png": "corr_heatmap_png",
            "correlation_heatmap.png": "corr_heatmap_png",
            "_pca.png": "pca_plot_png",
            "pca.png": "pca_plot_png",
            "pca_plot.html": "pca_plot_html",
            "_pls.png": "pls_plot_png",
            "pls.png": "pls_plot_png",
            "pls_plot.html": "pls_plot_html",
            "_umap.png": "umap_plot_png",
            "umap.png": "umap_plot_png",
            "umap_plot.html": "umap_plot_html",
            "feature_importance.html": "feature_importance_html",
            "feature importance.html": "feature_importance_html",
            "shap_plot.html": "shap_plot_html",
            "shap plot.html": "shap_plot_html",
        }
        semantic_keys = set(candidates) | {
            "scores_csv",
            "optimal_scores_csv",
            "selected_features_csv",
            "feature_importance_csv",
            "shap_csv",
            "pca_plot_html",
            "pls_plot_html",
            "umap_plot_html",
        }

        def normalize_name(value: str) -> str:
            return os.path.basename(str(value or "")).strip().lower()

        discovered = {}
        for raw_key, raw_path in (artifacts or {}).items():
            if not raw_path:
                continue
            raw_key_str = str(raw_key)
            normalized_key = raw_key_str if raw_key_str in semantic_keys else None
            if not normalized_key:
                normalized_key = aliases.get(normalize_name(raw_key_str))
            if not normalized_key:
                normalized_key = aliases.get(normalize_name(raw_path))
            if normalized_key:
                discovered.setdefault(normalized_key, raw_path)
            else:
                discovered[raw_key_str] = raw_path

        if not os.path.isdir(output_dir):
            return discovered

        for key, filename in candidates.items():
            if discovered.get(key):
                continue
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                discovered[key] = path
        return discovered

    def _parse_regression_report(self, json_path: str) -> dict:
        """Parse regression metrics emitted by the PineBioML runner."""
        default_metrics = {"R2": "N/A", "RMSE": "N/A", "MAE": "N/A", "MSE": "N/A"}
        if not json_path or not os.path.exists(json_path):
            return default_metrics

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            metrics = dict(default_metrics)
            if data.get("R2") is not None:
                metrics["R2"] = f"{float(data['R2']):.4f}"
            if data.get("MSE") is not None:
                mse = float(data["MSE"])
                metrics["MSE"] = f"{mse:.4f}"
                metrics["RMSE"] = f"{mse ** 0.5:.4f}"
            if data.get("MAE") is not None:
                metrics["MAE"] = f"{float(data['MAE']):.4f}"
            return metrics
        except Exception as e:
            logger.error(f"Failed to parse regression report {json_path}: {e}")
            return default_metrics

    def _parse_scores_csv(self, csv_path: str) -> dict:
        """Parses precision, recall, f1-score, accuracy from a classification scores CSV."""
        default_metrics = {
            "accuracy": "N/A",
            "ROC-AUC": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "F1-Score": "N/A",
            "specificity": "N/A",
            "MCC": "N/A"
        }
        
        if not csv_path:
            logger.info("No Scores CSV provided in artifacts. Using fallback default metrics.")
            return default_metrics
        if not os.path.exists(csv_path):
            logger.warning(f"Scores CSV not found at {csv_path}. Using fallback default metrics.")
            return default_metrics
            
        try:
            df = pd.read_csv(csv_path)
            # Find row for accuracy or weighted avg
            metrics = {}
            
            # Locate columns
            col_name = df.columns[0]
            
            # Find accuracy row
            acc_row = df[df[col_name].astype(str).str.lower().str.strip() == "accuracy"]
            if not acc_row.empty:
                metrics["accuracy"] = f"{float(acc_row.iloc[0]['f1-score']):.4f}"
            
            # Find weighted avg row
            w_avg_row = df[df[col_name].astype(str).str.lower().str.strip() == "weighted avg"]
            if not w_avg_row.empty:
                metrics["precision"] = f"{float(w_avg_row.iloc[0]['precision']):.4f}"
                metrics["recall"] = f"{float(w_avg_row.iloc[0]['recall']):.4f}"
                metrics["F1-Score"] = f"{float(w_avg_row.iloc[0]['f1-score']):.4f}"
            
            # Find sensitivity and specificity rows (PineBioML generates these for binary tasks)
            sens_row = df[df[col_name].astype(str).str.lower().str.strip() == "sensitivity"]
            if not sens_row.empty:
                metrics["sensitivity"] = f"{float(sens_row.iloc[0]['f1-score']):.4f}"
            spec_row = df[df[col_name].astype(str).str.lower().str.strip() == "specificity"]
            if not spec_row.empty:
                metrics["specificity"] = f"{float(spec_row.iloc[0]['f1-score']):.4f}"
            
            mcc_row = df[df[col_name].astype(str).str.lower().str.strip() == "mcc"]
            if not mcc_row.empty:
                metrics["MCC"] = f"{float(mcc_row.iloc[0]['f1-score']):.4f}"
            
            # ROC-AUC will be overridden by _extract_best_auc from all_models later
            # Set to N/A here so we know it needs to be replaced
            metrics["ROC-AUC"] = "N/A"
                
            # Fill missing keys from defaults
            for k, v in default_metrics.items():
                if k not in metrics:
                    metrics[k] = v
            return metrics
        except Exception as e:
            logger.error(f"Failed to parse scores CSV {csv_path}: {e}")
            return default_metrics
    
    def _extract_best_auc(self, all_models: list) -> str:
        """Extract real ROC-AUC from the best model in the all-models CSV."""
        if not all_models:
            return None
        try:
            keys = list(all_models[0].keys())
            # Find AUC column
            auc_key = next((k for k in keys if k.lower() == "test_auc"), None)
            if not auc_key:
                auc_key = next((k for k in keys if "auc" in k.lower()), None)
            if not auc_key:
                return None
            
            # Find accuracy column to identify best model
            acc_key = next((k for k in keys if k.lower() == "test_accuracy"), None)
            if not acc_key:
                acc_key = next((k for k in keys if "accuracy" in k.lower()), None)
            
            if acc_key:
                best = max(all_models, key=lambda m: float(m.get(acc_key, 0) or 0))
            else:
                best = all_models[0]
            
            auc_val = float(best.get(auc_key, 0) or 0)
            return f"{auc_val:.4f}" if auc_val > 0 else None
        except Exception as e:
            logger.error(f"Failed to extract AUC from all_models: {e}")
            return None

    def _fill_missing_metrics_from_all_models(self, metrics: dict, all_models: list):
        """Fills missing 'N/A' metrics in the metrics dictionary using the best model from all_models."""
        # Check if there are any N/A metrics that need filling
        has_missing = any(metrics.get(k) == "N/A" for k in ["accuracy", "ROC-AUC", "precision", "recall", "F1-Score", "specificity", "MCC"])
        if not all_models or not has_missing:
            return
            
        keys = list(all_models[0].keys())
        acc_key = next((k for k in keys if "accuracy" in k.lower()), None)
        if acc_key:
            best_model = max(all_models, key=lambda m: float(m.get(acc_key, 0) or 0))
        else:
            best_model = all_models[0]
            
        def _get_val(search_term):
            k = next((x for x in keys if search_term in x.lower()), None)
            return best_model.get(k) if k else None
        
        # Only fill metrics that are still N/A    
        if metrics.get("accuracy") == "N/A":
            acc_val = _get_val("accuracy")
            if acc_val is not None: metrics["accuracy"] = f"{float(acc_val):.4f}"
        
        if metrics.get("recall") == "N/A":
            sens_val = _get_val("sensitivity") or _get_val("recall")
            if sens_val is not None:
                metrics["recall"] = f"{float(sens_val):.4f}"
                metrics["sensitivity"] = f"{float(sens_val):.4f}"
            
        if metrics.get("specificity") == "N/A":
            spec_val = _get_val("specificity")
            if spec_val is not None: metrics["specificity"] = f"{float(spec_val):.4f}"
        
        if metrics.get("F1-Score") == "N/A":
            f1_val = _get_val("f1")
            if f1_val is not None: metrics["F1-Score"] = f"{float(f1_val):.4f}"
        
        if metrics.get("MCC") == "N/A":
            mcc_val = _get_val("mcc")
            if mcc_val is not None: metrics["MCC"] = f"{float(mcc_val):.4f}"
        
        if metrics.get("precision") == "N/A":
            prec_val = _get_val("precision")
            if prec_val is not None: metrics["precision"] = f"{float(prec_val):.4f}"
    
    def _parse_per_class_metrics(self, csv_path: str) -> list:
        """Parse top 5 per-class metrics from scores CSV (best + worst performing classes)."""
        if not csv_path or not os.path.exists(csv_path):
            return []
        try:
            df = pd.read_csv(csv_path)
            col_name = df.columns[0]
            
            # Filter out summary rows (accuracy, weighted avg, macro avg, mcc, specificity, blank spacers)
            summary_labels = {"accuracy", "weighted avg", "macro avg", "mcc", "specificity", ""}
            class_rows = df[~df[col_name].astype(str).str.lower().str.strip().isin(summary_labels)].copy()
            class_rows = class_rows[class_rows[col_name].astype(str).str.strip() != ""]
            
            if class_rows.empty:
                return []
            
            # Ensure numeric columns
            for c in ['precision', 'recall', 'f1-score', 'support']:
                if c in class_rows.columns:
                    class_rows[c] = pd.to_numeric(class_rows[c], errors='coerce')
            
            if 'f1-score' not in class_rows.columns:
                return []
            
            # Drop rows where f1-score is NaN
            class_rows = class_rows.dropna(subset=['f1-score'])
            
            # Sort by f1-score to find best and worst
            sorted_best = class_rows.sort_values('f1-score', ascending=False)
            sorted_worst = class_rows.sort_values('f1-score', ascending=True)
            
            # Pick top 3 best + top 2 worst (deduplicated) = up to 5 classes
            top_best = sorted_best.head(3)
            top_worst = sorted_worst.head(2)
            combined = pd.concat([top_best, top_worst]).drop_duplicates(subset=[col_name])
            
            results = []
            for _, row in combined.iterrows():
                class_label = str(row[col_name]).strip()
                entry = {
                    "class": class_label,
                    "precision": float(row.get('precision', 0) or 0),
                    "recall": float(row.get('recall', 0) or 0),
                    "f1": float(row.get('f1-score', 0) or 0),
                    "support": int(row.get('support', 0) or 0)
                }
                results.append(entry)
            
            return results
        except Exception as e:
            logger.error(f"Failed to parse per-class metrics: {e}")
            return []

    def _parse_per_class_metrics_json(self, json_path: str) -> list:
        """Parse per-class metrics from sklearn-style classification_report.json."""
        if not json_path or not os.path.exists(json_path):
            return []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = []
            summary_labels = {"accuracy", "weighted avg", "macro avg", "mcc", "specificity", ""}
            for label, values in data.items():
                label_str = str(label).strip()
                if label_str.lower() in summary_labels or not isinstance(values, dict):
                    continue
                if not any(k in values for k in ("precision", "recall", "f1-score", "support")):
                    continue
                results.append({
                    "class": label_str,
                    "precision": float(values.get("precision", 0) or 0),
                    "recall": float(values.get("recall", 0) or 0),
                    "f1": float(values.get("f1-score", 0) or 0),
                    "support": int(float(values.get("support", 0) or 0)),
                })

            return results
        except Exception as e:
            logger.error(f"Failed to parse per-class metrics JSON: {e}")
            return []

    @staticmethod
    def _metric_to_unit_interval(value):
        if value in (None, "", "N/A"):
            return None
        try:
            text = str(value).replace("%", "").strip()
            num = float(text)
            if "%" in str(value) or num > 1.0:
                return num / 100.0
            return num
        except (TypeError, ValueError):
            return None

    def _build_imbalance_warning(self, metrics: dict, per_class: list, all_models: list) -> dict:
        """
        Detect situations where headline accuracy can hide poor class coverage.
        Returns a structured warning for the prompt, JSON report, and HTML viewer.
        """
        class_rows = []
        for c in per_class or []:
            try:
                support = int(float(c.get("support", 0) or 0))
            except (TypeError, ValueError):
                support = 0
            if support > 0:
                class_rows.append((c, support))
        supports = [support for _, support in class_rows]
        imbalance_ratio = None
        minority_class = None
        minority_recall = None
        majority_share = None

        if supports:
            total_support = sum(supports)
            min_support = min(supports)
            max_support = max(supports)
            imbalance_ratio = round(max_support / min_support, 2) if min_support else None
            majority_share = max_support / total_support if total_support else None
            for c, support in class_rows:
                if support == min_support:
                    minority_class = c.get("class")
                    minority_recall = self._metric_to_unit_interval(c.get("recall"))
                    break

        accuracy = self._metric_to_unit_interval(metrics.get("accuracy"))
        specificity = self._metric_to_unit_interval(metrics.get("specificity"))
        recall = self._metric_to_unit_interval(metrics.get("recall"))
        mcc = self._metric_to_unit_interval(metrics.get("MCC"))

        reasons = []
        if imbalance_ratio and imbalance_ratio >= 3.0:
            share_text = f"{majority_share * 100:.1f}%" if majority_share is not None else "most"
            reasons.append(f"class support is imbalanced by about {imbalance_ratio}:1, with the largest class making up {share_text} of labeled samples")
        if accuracy is not None and accuracy >= 0.85:
            weak_parts = []
            if specificity is not None and specificity < 0.30:
                weak_parts.append(f"specificity is only {specificity * 100:.1f}%")
            if minority_recall is not None and minority_recall < 0.30:
                weak_parts.append(f"minority-class recall for class '{minority_class}' is only {minority_recall * 100:.1f}%")
            if recall is not None and recall < 0.60:
                weak_parts.append(f"macro recall is only {recall * 100:.1f}%")
            if mcc is not None and mcc < 0.30:
                weak_parts.append(f"MCC is low at {mcc:.3f}")
            if weak_parts:
                reasons.append(f"headline accuracy is {accuracy * 100:.1f}%, but " + ", ".join(weak_parts))

        if not reasons:
            return {}

        return {
            "severity": "high" if accuracy is not None and accuracy >= 0.85 and (specificity is not None and specificity < 0.30) else "medium",
            "title": "Accuracy may be misleading because of class imbalance",
            "message": (
                "Do not interpret accuracy alone as clinical readiness. "
                + "; ".join(reasons)
                + ". Use balanced metrics such as specificity, minority-class recall, macro-F1, MCC, ROC-AUC, and threshold analysis before deployment."
            ),
            "imbalance_ratio": imbalance_ratio,
            "majority_share": round(majority_share, 4) if majority_share is not None else None,
            "minority_class": minority_class,
            "minority_recall": round(minority_recall, 4) if minority_recall is not None else None,
        }
    
    # ── Class Label Validation ────────────────────────────────────────────────

    # Known activity recognition labels that should NEVER appear in medical datasets
    _ACTIVITY_LABELS = frozenset({
        'LAYING', 'WALKING', 'STANDING', 'SITTING',
        'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
        'LIE_TO_STAND', 'LIE_TO_SIT', 'SIT_TO_LIE',
        'SIT_TO_STAND', 'STAND_TO_SIT', 'STAND_TO_LIE',
    })

    def _validate_class_labels(self, per_class: list, dataset_name: str, all_models: list) -> list:
        """
        Cross-validate class labels against dataset context.
        Detects and removes contamination from unrelated dataset runs
        (e.g., HAR activity labels leaking into a breast cancer report).
        """
        if not per_class:
            return per_class

        detected_labels = {c['class'].upper().strip() for c in per_class}
        activity_overlap = detected_labels & self._ACTIVITY_LABELS

        if activity_overlap:
            logger.error(
                f"CLASS LABEL CONTAMINATION DETECTED! "
                f"Found activity recognition labels {activity_overlap} in dataset '{dataset_name}'. "
                f"These labels likely came from a different pipeline run. "
                f"Removing contaminated per-class data to prevent hallucinated output."
            )
            return []  # Better no data than wrong data

        # Cross-check: per-class support sum vs actual test set size
        if all_models:
            test_support_sum = sum(c.get('support', 0) for c in per_class)
            expected_test_n = 0
            for m in all_models:
                ts = m.get('test_support', 0)
                if ts:
                    expected_test_n = int(ts)
                    break
            if expected_test_n and test_support_sum > 0:
                ratio = test_support_sum / expected_test_n
                if ratio > 3.0 or ratio < 0.3:
                    logger.warning(
                        f"Per-class support sum ({test_support_sum}) is far from "
                        f"expected test set size ({expected_test_n}). "
                        f"Possible cross-dataset contamination; clearing per-class data."
                    )
                    return []

        return per_class

    def _compute_overfit_gap(self, all_models: list) -> dict:
        """Compute train/test gap across models for overfitting analysis."""
        if not all_models:
            return {}
        try:
            keys = list(all_models[0].keys())
            train_acc_key = next((k for k in keys if k.lower() == "train_accuracy"), None)
            test_acc_key = next((k for k in keys if k.lower() == "test_accuracy"), None)
            if not train_acc_key:
                train_acc_key = next((k for k in keys if k.lower() in ("train_r2", "train_score") or ("train" in k.lower() and "r2" in k.lower())), None)
            if not test_acc_key:
                test_acc_key = next((k for k in keys if k.lower() in ("test_r2", "test_score") or ("test" in k.lower() and "r2" in k.lower())), None)
            model_key = next((k for k in keys if k.lower() in ("modeling", "model", "model_name")), None)
            if not model_key:
                model_key = next((k for k in keys if "model" in k.lower()), None)
            
            if not train_acc_key or not test_acc_key:
                return {}
            
            gaps = []
            for m in all_models:
                train_val = float(m.get(train_acc_key, 0) or 0)
                test_val = float(m.get(test_acc_key, 0) or 0)
                gap = train_val - test_val
                name = m.get(model_key, "Unknown") if model_key else "Unknown"
                gaps.append({
                    "model": name,
                    "train_accuracy": round(train_val, 4),
                    "test_accuracy": round(test_val, 4),
                    "gap": round(gap, 4),
                    "gap_pct": f"{gap * 100:.1f}%"
                })
            
            max_gap = max(gaps, key=lambda g: abs(g["gap"]))
            is_overfitting = max_gap["gap"] > 0.10  # >10% gap = overfitting concern
            
            return {
                "models": gaps,
                "worst_gap": max_gap,
                "is_overfitting": is_overfitting
            }
        except Exception as e:
            logger.error(f"Failed to compute overfit gap: {e}")
            return {}
    
    def _parse_selected_features(self, csv_path: str) -> list:
        """Parse selected feature names from the feature selection CSV."""
        if not csv_path or not os.path.exists(csv_path):
            return []
        try:
            df = pd.read_csv(csv_path)
            if 'Feature' in df.columns:
                return df['Feature'].tolist()
            return df.iloc[:, 0].tolist()
        except Exception as e:
            logger.error(f"Failed to parse selected features: {e}")
            return []

    def _parse_all_models_csv(self, csv_path: str) -> list:
        """Parses the all-model-result.csv for model comparisons."""
        if not csv_path or not os.path.exists(csv_path):
            return []
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to parse all models CSV {csv_path}: {e}")
            return []

    def _parse_shap_csv(self, csv_path: str) -> list:
        """
        Robustly parses a generic SHAP/feature importance CSV.
        Assumes first column is feature name, and any subsequent numeric column is importance.
        Returns top 5 features with their scores.
        """
        if not csv_path or not os.path.exists(csv_path):
            return []
            
        try:
            df = pd.read_csv(csv_path)
            if df.empty or len(df.columns) < 2:
                return []
                
            # Assume first column is feature name, find first numeric column for importance
            feature_col = df.columns[0]
            importance_col = None
            
            for col in df.columns[1:]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    importance_col = col
                    break
                    
            if not importance_col:
                return []
                
            # Sort by absolute importance descending
            df['abs_importance'] = df[importance_col].abs()
            top_df = df.sort_values(by='abs_importance', ascending=False).head(5)
            
            features = []
            for _, row in top_df.iterrows():
                features.append({
                    "feature": str(row[feature_col]),
                    "importance": float(row[importance_col])
                })
            return features
        except Exception as e:
            logger.error(f"Failed to parse SHAP CSV {csv_path}: {e}")
            return []

    def _flag_anomalies(self, metrics: dict) -> list:
        """
        Deterministically evaluates metrics and flags critical underperformance or potential data leakage.
        """
        flags = []
        
        def parse_pct(val):
            if not val: return None
            val_str = str(val).replace("%", "").strip()
            try:
                num = float(val_str)
                # if string had % or is large, divide by 100
                if "%" in str(val) or num > 1.0:
                    return num / 100.0
                return num
            except ValueError:
                return None
                
        if any(k in metrics for k in ("R2", "RMSE", "MAE", "MSE")) and metrics.get("accuracy") in (None, "N/A"):
            try:
                r2 = float(str(metrics.get("R2", "N/A")).strip())
                if r2 < 0:
                    flags.append("Critical Underperformance: R2 is below 0, so the model performs worse than predicting the mean target value.")
                elif r2 < 0.30:
                    flags.append("Weak Fit: R2 is below 0.30, so the regression model explains only a limited share of target variation.")
                elif r2 > 0.99:
                    flags.append("Potential Data Leakage: R2 exceeds 0.99. This is unusual and should be checked against target leakage or duplicated variables.")
            except ValueError:
                pass
            return flags

        acc = parse_pct(metrics.get("accuracy"))
        auc = parse_pct(metrics.get("ROC-AUC"))
        recall = parse_pct(metrics.get("recall"))
        
        # Heuristics
        if acc is not None:
            if acc < 0.60:
                flags.append("Critical Underperformance: Accuracy is below 60%. Model is barely better than random chance.")
            elif acc > 0.99:
                flags.append("Potential Data Leakage: Accuracy exceeds 99%. Model might be overfitting or learning from the target variable.")
                
        if auc is not None:
            if auc < 0.60:
                flags.append("Critical Underperformance: AUC is below 0.60. Model struggles to separate classes.")
            elif auc > 0.99:
                flags.append("Potential Data Leakage: AUC exceeds 0.99. Highly unusual for clinical datasets.")
                
        if recall is not None and recall < 0.50:
            flags.append("High False Negative Rate: Recall is below 50%. Model is missing the majority of positive cases.")
            
        return flags

    def _render_fallback_metrics_table(self, metrics: dict) -> str:
        """Render selected optimal-model metrics when no comparison table is available."""
        import html as html_lib

        if not metrics:
            return "<p>No model performance data available yet.</p>"

        if any(k in metrics for k in ("R2", "RMSE", "MAE", "MSE")) and metrics.get("accuracy") in (None, "N/A"):
            metric_specs = [
                ("R2", metrics.get("R2")),
                ("RMSE", metrics.get("RMSE")),
                ("MAE", metrics.get("MAE")),
                ("MSE", metrics.get("MSE")),
            ]
        else:
            metric_specs = [
                ("Accuracy", metrics.get("accuracy")),
                ("Precision", metrics.get("precision")),
                ("Recall / Sensitivity", metrics.get("recall")),
                ("F1-Score", metrics.get("F1-Score")),
                ("ROC-AUC", metrics.get("ROC-AUC")),
                ("MCC", metrics.get("MCC")),
            ]
        rows = [
            (label, str(value))
            for label, value in metric_specs
            if value not in (None, "", "N/A")
        ]
        if not rows:
            return "<p>No model performance data available yet.</p>"

        html = (
            '<div style="margin-bottom:1.25rem;padding:0.85rem 1.5rem;background:rgba(16,185,129,0.08);'
            'border-radius:12px;border:1px solid rgba(16,185,129,0.2);text-align:center;font-size:1rem;'
            'box-shadow:0 4px 12px rgba(16,185,129,0.05);">'
            '<span style="background:rgba(16,185,129,0.15);color:#10b981;padding:3px 10px;border-radius:20px;'
            'font-size:0.85em;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;margin-right:10px;">'
            'Optimal Model Summary</span>'
            '<span style="color:#94a3b8;">Showing the selected PineBioML model metrics from the final training result.</span>'
            '</div>'
            '<div style="overflow-x:auto;"><table class="model-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
        )
        for label, value in rows:
            html += f"<tr><td><strong>{html_lib.escape(label)}</strong></td><td>{html_lib.escape(value)}</td></tr>"
        html += "</tbody></table></div>"
        return html

    def _render_regression_model_performance_table(self, all_models: list, metrics: dict = None) -> str:
        """Server-side render a regression model comparison table."""
        if not all_models:
            return self._render_fallback_metrics_table(metrics or {})

        keys = list(all_models[0].keys())
        model_key = next((k for k in keys if k.lower() in ("modeling", "model", "model_name")), None)
        if not model_key:
            model_key = next((k for k in keys if "model" in k.lower()), None)
        if not model_key:
            model_key = next((k for k in keys if isinstance(all_models[0].get(k), str)), None)
        if not model_key:
            model_key = keys[0]

        def find_metric_key(candidates):
            lowered = {k.lower(): k for k in keys}
            for candidate in candidates:
                if candidate in lowered:
                    return lowered[candidate]
            return None

        score_key = next((k for k in keys if k.lower() == "test_r2"), None)
        if not score_key:
            score_key = next((k for k in keys if "r2" in k.lower()), None)
        if not score_key:
            score_key = next((k for k in keys if k.lower() == "test_score"), None)
        if not score_key:
            score_key = next((k for k in keys if "score" in k.lower()), None)

        cols = [("Model", model_key)]
        if score_key:
            cols.append(("Test R2" if "r2" in score_key.lower() else "Score", score_key))
        for label, needle in (("Test RMSE", "rmse"), ("Test MAE", "mae"), ("Test MSE", "mse")):
            key = next((k for k in keys if needle in k.lower()), None)
            if key:
                cols.append((label, key))
        for label, match in (
            ("Missing Value",     lambda k: k.lower() in ("missing", "missingvalueprocessing")),
            ("Normalization",     lambda k: k.lower() in ("normalization", "standarization", "standardization")),
            ("Feature Selection", lambda k: k.lower() in ("selection", "feature_selection")),
            ("Hyperparameters",   lambda k: k.lower() in ("best_params", "params", "hyperparameters")),
        ):
            key = next((k for k in keys if match(k)), None)
            if key:
                cols.append((label, key))

        def score(row):
            try:
                return float(row.get(score_key, 0) or 0) if score_key else 0
            except Exception:
                return 0

        sorted_models = sorted(all_models, key=score, reverse=True)
        best = sorted_models[0]
        best_name = best.get(model_key, "Unknown")
        best_score = f"{score(best):.4f}" if score_key else "N/A"

        html = (
            '<div style="margin-bottom:1.25rem;padding:0.85rem 1.5rem;background:rgba(16,185,129,0.08);'
            'border-radius:12px;border:1px solid rgba(16,185,129,0.2);text-align:center;font-size:1rem;">'
            '<span style="background:rgba(16,185,129,0.15);color:#10b981;padding:3px 10px;border-radius:20px;'
            'font-size:0.85em;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;margin-right:10px;">'
            'Best Regression Model</span>'
            f'<strong style="color:#f8fafc;font-size:1.05rem;font-weight:600;">{html_lib.escape(str(best_name))}</strong>'
        )
        if score_key:
            label = "Test R2" if "r2" in score_key.lower() else "Score"
            html += f'<span style="color:#94a3b8;margin:0 10px;">&bull;</span><span style="color:#94a3b8;">{label}:</span> <strong style="color:#10b981;font-size:1.05rem;">{best_score}</strong>'
        html += '</div><div style="overflow-x:auto;"><table class="model-table"><thead><tr>'
        for label, _ in cols:
            html += f"<th>{html_lib.escape(str(label))}</th>"
        html += "</tr></thead><tbody>"
        for row in sorted_models:
            is_best = row is best
            row_style = 'background:rgba(16,185,129,0.12);border-left:3px solid #10b981;' if is_best else ''
            html += f'<tr style="{row_style}">'
            for _, key in cols:
                val = row.get(key, "N/A")
                if isinstance(val, (int, float)):
                    val = f"{val:.4f}"
                html += f"<td>{html_lib.escape(str(val))}</td>"
            html += "</tr>"
        html += "</tbody></table></div>"
        return html

    def _render_model_performance_table(self, all_models: list, metrics: dict = None, task_type: str = "") -> str:
        """Server-side render a clean model comparison table with key columns only."""
        if "regression" in str(task_type or "").lower():
            return self._render_regression_model_performance_table(all_models, metrics)
        if not all_models:
            return self._render_fallback_metrics_table(metrics or {})
        
        keys = list(all_models[0].keys())
        
        # Find model name key
        model_key = next((k for k in keys if k.lower() in ("modeling", "model", "model_name")), None)
        if not model_key:
            model_key = next((k for k in keys if "model" in k.lower()), None)
        if not model_key:
            model_key = next((k for k in keys if isinstance(all_models[0].get(k), str)), None)
        if not model_key:
            model_key = keys[0]
        
        def find_metric_key(candidates):
            lowered = {k.lower(): k for k in keys}
            for candidate in candidates:
                if candidate in lowered:
                    return lowered[candidate]
            # Fallback to substring matching (e.g. test_precision_macro)
            for candidate in candidates:
                for k_low, k_orig in lowered.items():
                    if candidate in k_low:
                        return k_orig
            return None

        col_specs = [
            ("MODEL",             [k for k in keys if k == model_key]),
            ("TEST ACCURACY",     [find_metric_key(("test_accuracy", "test_acc", "cv_accuracy", "accuracy", "acc"))]),
            ("TEST AUC",          [find_metric_key(("test_auc", "auc", "roc_auc", "roc-auc"))]),
            ("TEST F1",           [find_metric_key(("test_f1", "f1", "f1_score", "f1-score"))]),
        ]
        
        # Pipeline components
        params_key = next((k for k in keys if k.lower() in ("best_params", "params", "hyperparameters")), None)

        cols = [(label, key_list[0]) for label, key_list in col_specs if key_list and key_list[0]]
        
        acc_col_key = next((k for label, k in cols if label == "TEST ACCURACY"), None)
        best_idx = 0
        if acc_col_key:
            best_val = -float("inf")
            for i, m in enumerate(all_models):
                try:
                    v = float(m.get(acc_col_key, 0) or 0)
                    if v > best_val:
                        best_val = v
                        best_idx = i
                except Exception:
                    pass
        
        def fmt_val(label, val):
            if val is None:
                return "N/A"
            if isinstance(val, (int, float)):
                if 0.0 <= val <= 1.0:
                    return f"{val * 100:.1f}%"
                return f"{val:.2f}%"
            return str(val)

        def get_pipeline_html(row, is_callout=False):
            p_val = row.get(params_key)
            p_dict = None
            if isinstance(p_val, str):
                import ast
                try:
                    p_dict = ast.literal_eval(p_val)
                except Exception:
                    pass
            elif isinstance(p_val, dict):
                p_dict = p_val
                
            if p_dict:
                p_chunks = []
                for k, v in p_dict.items():
                    clean_k = str(k).replace("clf__", "").replace("selector__", "")
                    p_chunks.append(f'<span style="background:rgba(255,255,255,0.04); padding:4px 10px; border-radius:6px; border:1px solid rgba(255,255,255,0.08); font-size:0.85em;"><span style="color:#10b981;font-family:monospace;font-weight:600;">{html_lib.escape(str(clean_k))}</span> <span style="color:#94a3b8; font-size:0.9em; margin:0 4px;">&rarr;</span> <span style="font-family:monospace; color:#f8fafc; font-weight:600;">{html_lib.escape(str(v))}</span></span>')
                margin_top = "0px" if not is_callout else "6px"
                return f'<div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:{margin_top};">{"".join(p_chunks)}</div>'
            else:
                margin_top = "0px" if not is_callout else "6px"
                return f'<div style="margin-top:{margin_top};">{html_lib.escape(str(p_val))}</div>' if p_val else ''

        best = all_models[best_idx]
        best_name = best.get(model_key, "Unknown")
        best_acc = fmt_val("TEST ACCURACY", best.get(acc_col_key)) if acc_col_key else "N/A"
        
        callout_html = (
            f'<div style="margin-bottom:1.25rem;padding:1rem 1.5rem;background:rgba(16,185,129,0.08);'
            f'border-radius:12px;border:1px solid rgba(16,185,129,0.2);text-align:center;font-size:1rem;'
            f'box-shadow: 0 4px 12px rgba(16,185,129,0.05);">'
            f'<span style="background:rgba(16,185,129,0.15);color:#10b981;padding:4px 12px;border-radius:20px;font-size:0.85em;font-weight:800;letter-spacing:0.05em;text-transform:uppercase;margin-right:12px;">BEST MODEL</span>'
            f'<strong style="color:#f8fafc;font-size:1.1rem;font-weight:700;">{html_lib.escape(str(best_name))}</strong>'
            f'<span style="color:#94a3b8;margin:0 12px;font-size:1.2rem;vertical-align:middle;">&bull;</span>'
            f'<span style="color:#94a3b8;">Test Accuracy:</span> <strong style="color:#10b981;font-size:1.1rem;margin-left:4px;">{best_acc}</strong>'
            f'</div>'
        )

        html = callout_html + '<div style="overflow-x:auto;"><table class="model-table"><thead><tr>'
        for label, _ in cols:
            html += f"<th>{html_lib.escape(str(label))}</th>"
        html += "</tr></thead><tbody>"
        
        for i, row in enumerate(all_models):
            is_best = (i == best_idx)
            row_style = 'background:rgba(16,185,129,0.12);border-left:3px solid #10b981;' if is_best else ''
            
            pipeline_html = get_pipeline_html(row, is_callout=False)
            has_pipeline = bool(pipeline_html)
            
            html += f'<tr style="{row_style}">'
            for label, key in cols:
                val = row.get(key)
                cell_val = fmt_val(label, val)
                cell_style = "font-weight:700;color:#10b981;" if is_best and label != "MODEL" else ""
                if has_pipeline:
                    cell_style += " border-bottom:none;"
                html += f'<td style="{cell_style}">{html_lib.escape(str(cell_val))}</td>'
            html += "</tr>"
            
            if has_pipeline:
                bg_col = 'background:rgba(16,185,129,0.05);border-left:3px solid #10b981;' if is_best else 'background:rgba(255,255,255,0.015);'
                html += f'<tr style="{bg_col}"><td colspan="{len(cols)}" style="padding-top:8px; padding-bottom:16px; font-size:0.9em; color:#94a3b8; border-top:none; text-align:center;">{pipeline_html}</td></tr>'
        
        html += "</tbody></table></div>"
        return html

    def _render_html_report(self, data: dict) -> str:
        """Render the HTML report using the report_viewer.html template."""
        template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources", "templates", "report_viewer.html")
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found at {template_path}")
            
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
            
        # Replace template placeholders
        # Meta info
        html = template.replace("{{ report_id }}", html_lib.escape(data["report_id"]))
        html = html.replace("{{ job_id }}", html_lib.escape(data["job_id"]))
        html = html.replace("{{ dataset_name }}", html_lib.escape(data["dataset_name"]))
        html = html.replace("{{ task_type }}", html_lib.escape(data["task_type"].replace("_", " ").title()))
        html = html.replace("{{ generated_at }}", html_lib.escape(data["created_at"]))
        html = html.replace("{{ model_name }}", html_lib.escape(data.get("model_name", "PineBioML Default")))
        
        # Diagnostic metrics
        metrics = data["metrics"]
        html = html.replace("{{ accuracy }}", html_lib.escape(str(metrics.get("accuracy", "N/A"))))
        html = html.replace("{{ roc_auc }}", html_lib.escape(str(metrics.get("ROC-AUC", "N/A"))))
        html = html.replace("{{ precision }}", html_lib.escape(str(metrics.get("precision", "N/A"))))
        html = html.replace("{{ recall }}", html_lib.escape(str(metrics.get("recall", "N/A"))))
        html = html.replace("{{ f1_score }}", html_lib.escape(str(metrics.get("F1-Score", "N/A"))))
        html = html.replace("{{ specificity }}", html_lib.escape(str(metrics.get("specificity", "N/A"))))
        html = html.replace("{{ mcc }}", html_lib.escape(str(metrics.get("MCC", "N/A"))))
        html = html.replace("{{ r2 }}", html_lib.escape(str(metrics.get("R2", "N/A"))))
        html = html.replace("{{ rmse }}", html_lib.escape(str(metrics.get("RMSE", "N/A"))))
        html = html.replace("{{ mae }}", html_lib.escape(str(metrics.get("MAE", "N/A"))))
        html = html.replace("{{ mse }}", html_lib.escape(str(metrics.get("MSE", "N/A"))))

        imbalance_warning = data.get("imbalance_warning") or {}
        if imbalance_warning:
            warning_title = html_lib.escape(str(imbalance_warning.get("title", "Accuracy may be misleading")))
            warning_message = html_lib.escape(str(imbalance_warning.get("message", "")))
            warning_html = (
                '<div class="report-warning" role="note">'
                f'<div class="report-warning-title">{warning_title}</div>'
                f'<div class="report-warning-body">{warning_message}</div>'
                '</div>'
            )
        else:
            warning_html = ""
        html = html.replace("{{ imbalance_warning_html }}", warning_html)

        narrative_source = data.get("narrative_source", "llm")
        if narrative_source != "llm":
            notice_reason = html_lib.escape(str(data.get("narrative_notice") or "llm_unavailable_or_failed_validation"))
            if narrative_source == "llm_partial_with_unavailable_sections":
                notice_title = "LLM narrative partially unavailable"
                notice_body = (
                    "Some written sections were generated by the LLM, but one or more sections were replaced with an explicit unavailable notice. "
                    "The deterministic ML metrics, plots, and model tables remain available."
                )
            else:
                notice_title = "LLM narrative unavailable"
                notice_body = (
                    "This report still includes deterministic ML metrics, plots, and model tables, "
                    "but the written narrative was not generated by an LLM."
                )
            narrative_notice_html = (
                '<div class="report-notice" role="status">'
                f'<div class="report-notice-title">{notice_title}</div>'
                '<div class="report-notice-body">'
                f'{html_lib.escape(notice_body)} '
                f'Reason: {notice_reason}.'
                '</div>'
                '</div>'
            )
        else:
            narrative_notice_html = ""
        html = html.replace("{{ narrative_notice_html }}", narrative_notice_html)
        
        # Narratives for expert (rendered as direct strings for JS block)
        expert = data["narrative"]["expert"]
        
        # Load external glossary
        glossary_path = os.path.join(os.path.dirname(__file__), "glossaries", "default.json")
        try:
            with open(glossary_path, "r", encoding="utf-8") as f:
                glossary = json.load(f)
        except Exception:
            glossary = data["narrative"].get("glossary", {})
        
        # Escape quotes/newlines for Javascript
        def js_escape(text) -> str:
            if isinstance(text, list):
                text = "\n".join([str(item) for item in text])
            elif not isinstance(text, str):
                text = str(text) if text is not None else ""
            return text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
            
        html = html.replace("{{ expert_executive_summary }}", js_escape(expert["executive_summary"]))
        html = html.replace("{{ expert_preprocessing_and_data_quality }}", js_escape(expert.get("preprocessing_and_data_quality", "")))
        html = html.replace("{{ expert_findings }}", js_escape(expert["findings"]))
        html = html.replace("{{ expert_conclusion }}", js_escape(expert.get("conclusion", "")))
        html = html.replace("{{ expert_recommendations }}", js_escape(expert.get("recommendations", "")))

        # Inject JSON configs
        import json
        html = html.replace("{{ glossary_json }}", json.dumps(glossary))
        html = html.replace("{{ all_models_json }}", json.dumps(data.get("all_models", [])))
        html = html.replace("{{ overfit_json }}", json.dumps(data.get("overfit_analysis", {})))
        
        # Markdown replacement deferred until formatted_visuals is computed
        
        # Server-side pre-render Model Performance table
        all_models = data.get("all_models", [])
        model_perf_html = self._render_model_performance_table(all_models, metrics, data.get("task_type", ""))
        html = html.replace("{{ model_performance_table }}", model_perf_html)
        
        # Server-side pre-render of Clinical Insights Report Card for static viewers (like PDF)
        all_models = data.get("all_models", [])
        report_card_bars_html = ""
        stars_html = ""
        quality_label = "Overall Clarity & Resolution: Pending"
        
        if all_models:
            keys = list(all_models[0].keys())
            # Prefer test_accuracy, then exact 'accuracy'/'acc', then any key containing 'accuracy'
            acc_key = next((k for k in keys if k.lower() == "test_accuracy"), None)
            if not acc_key:
                acc_key = next((k for k in keys if k.lower() in ("accuracy", "acc")), None)
            if not acc_key:
                acc_key = next((k for k in keys if "accuracy" in k.lower()), None)
            if not acc_key:
                # Fall back to first numeric key
                acc_key = next((k for k in keys if isinstance(all_models[0].get(k), (int, float))), None)
            if not acc_key:
                acc_key = keys[1] if len(keys) > 1 else keys[0]
                
            # Prefer exact match 'Modeling'/'model_name', then any key containing 'model'
            model_key = next((k for k in keys if k.lower() in ("modeling", "model", "model_name")), None)
            if not model_key:
                model_key = next((k for k in keys if "model" in k.lower()), None)
            if not model_key:
                # Fall back to first string key
                model_key = next((k for k in keys if isinstance(all_models[0].get(k), str)), None)
            if not model_key:
                model_key = keys[0]
                
            try:
                sorted_models = sorted(all_models, key=lambda x: float(x.get(acc_key, 0) or 0), reverse=True)
            except Exception:
                sorted_models = all_models
                
            top3 = sorted_models[:3]
            
            report_card_bars_html = '<div class="bar-chart-container" style="margin-top: 0;">'
            for m in top3:
                model_name = m.get(model_key, "N/A")
                try:
                    val = float(m.get(acc_key, 0) or 0)
                    val_pct = val * 100.0 if val <= 1.0 else val
                    val_pct = min(max(val_pct, 0.0), 100.0)
                    pct_str = f"{val_pct:.1f}"
                except Exception:
                    pct_str = "0.0"
                
                esc_name = html_lib.escape(str(model_name))
                report_card_bars_html += f"""
                <div class="bar-row">
                    <div class="bar-label" title="{esc_name}">{esc_name}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {pct_str}%;">{pct_str}%</div>
                    </div>
                </div>"""
            report_card_bars_html += '</div>'
            
            # Calculate star rating
            total_score = 0.0
            count = 0
            for key in ["accuracy", "precision", "recall", "ROC-AUC"]:
                val_str = metrics.get(key, "0")
                if not val_str or val_str == "N/A":
                    continue
                try:
                    val_clean = val_str.replace("%", "").strip()
                    val = float(val_clean)
                    if "%" in val_str or val > 1.0:
                        val = val / 100.0
                    total_score += val
                    count += 1
                except Exception:
                    pass
                    
            if count > 0:
                avg_score = total_score / count
                if avg_score >= 0.90:
                    stars = 5
                    quality_label = "Overall Quality: Excellent"
                elif avg_score >= 0.80:
                    stars = 4
                    quality_label = "Overall Quality: Strong"
                elif avg_score >= 0.70:
                    stars = 3
                    quality_label = "Overall Quality: Moderate"
                elif avg_score >= 0.60:
                    stars = 2
                    quality_label = "Overall Quality: Weak"
                else:
                    stars = 1
                    quality_label = "Overall Quality: Poor"
            else:
                stars = 3
                quality_label = "Overall Quality: Moderate"

            if imbalance_warning:
                stars = min(stars, 3)
                quality_label = "Overall Quality: Needs Balanced-Metric Review"
                
            for i in range(5):
                if i < stars:
                    stars_html += '<span class="star filled">★</span>'
                else:
                    stars_html += '<span class="star">☆</span>'
        else:
            report_card_bars_html = self._render_fallback_metrics_table(metrics)
            stars_html = '<span class="star">☆</span><span class="star">☆</span><span class="star">☆</span><span class="star">☆</span><span class="star">☆</span>'
            quality_label = "Overall Quality: Pending"
            
        # Replace placeholders in template HTML
        html = html.replace('<div id="report-card-bars"></div>', f'<div id="report-card-bars">{report_card_bars_html}</div>')
        html = html.replace(
            '<span class="star">★</span><span class="star">★</span><span class="star">★</span><span class="star">★</span><span class="star">★</span>',
            stars_html
        )
        html = html.replace(
            '<div class="rating-label" id="rating-label">Overall Clarity & Resolution: Pending</div>',
            f'<div class="rating-label" id="rating-label">{quality_label}</div>'
        )
        
        # Embed visuals
        visuals_dict = data["visuals"]
        formatted_visuals = {}
        for key, val in visuals_dict.items():
            if not val:
                formatted_visuals[key] = ""
                continue
            if str(val).lower().endswith(".html"):
                formatted_visuals[key] = self._artifact_url(data["report_id"], val)
                continue
            b64_str = VisualAnalyzer.encode_image_to_base64(val)
            formatted_visuals[key] = f"data:image/png;base64,{b64_str}" if b64_str else ""
                
        # Embed visuals section
        plots_grid_html = ""
        for key, visual_src in formatted_visuals.items():
            if visual_src:
                title = key.replace('_png', '').replace('_html', '').replace('.png', '').replace('.html', '').replace('_', ' ').title()
                if str(visual_src).lower().endswith(".html"):
                    plots_grid_html += f"""
                    <div class="plot-container">
                        <iframe class="plot-img" src="{visual_src}" title="{key}" style="border:0;"></iframe>
                        <div class="plot-title">{title}</div>
                    </div>
                    """
                else:
                    plots_grid_html += f"""
                    <div class="plot-container" onclick="zoomPlot('{visual_src}')">
                        <img class="plot-img" src="{visual_src}" alt="{key}">
                        <div class="plot-title">{title}</div>
                    </div>
                    """
        # Inject custom plots grid html using a robust regex that replaces the entire template plots-grid
        import re
        pattern = r'<div class="plots-grid"[^>]*>.*?{% endfor %}\s*</div>'
        replacement = f'<div class="plots-grid" style="display:none;">{plots_grid_html}</div>'
        html = re.sub(pattern, lambda _: replacement, html, flags=re.DOTALL)

        # Embed visuals placeholders in body and pre-render them
        exec_summary_html = self._replace_plots_with_html(self._markdown_to_html(expert.get("executive_summary", "")), formatted_visuals)
        preprocessing_html = self._replace_plots_with_html(self._markdown_to_html(expert.get("preprocessing_and_data_quality", "")), formatted_visuals)
        findings_html = self._replace_plots_with_html(self._markdown_to_html(expert.get("findings", "")), formatted_visuals)
        conclusion_html = self._replace_plots_with_html(self._markdown_to_html(expert.get("conclusion", "")), formatted_visuals)
        recs_html = self._replace_plots_with_html(self._markdown_to_html(expert.get("recommendations", "")), formatted_visuals)

        html = html.replace("{{ executive_summary }}", exec_summary_html)
        html = html.replace("{{ preprocessing_and_data_quality }}", preprocessing_html)
        html = html.replace("{{ findings }}", findings_html)
        html = html.replace("{{ conclusion }}", conclusion_html)
        html = html.replace("{{ recommendations }}", recs_html)
        
        return html


    def _markdown_to_html(self, text) -> str:
        """Converts basic markdown to HTML for server-side rendering (e.g. for PDF export)."""
        from core.security import sanitize_html_content
        import markdown
        if not text:
            return ""
            
        if isinstance(text, list):
            text = "\n\n".join([str(t) for t in text])
        elif not isinstance(text, str):
            text = str(text)
        
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Sanitize dangerous HTML tags/event handlers from LLM output
        text = sanitize_html_content(text)
        
        html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'nl2br'])
        return html

    def _replace_plots_with_html(self, html_str: str, formatted_visuals: dict) -> str:
        if not html_str:
            return ""
        import re
        
        # Merge consecutive plot tags so they group horizontally
        html_str = re.sub(r'\[PLOT:\s*(.*?)\s*\]\s*(?:<br\s*/?>|</p>\s*<p>|\n)+\s*\[PLOT:\s*(.*?)\s*\]', r'[PLOT: \1, \2]', html_str, flags=re.IGNORECASE)
        html_str = re.sub(r'\[PLOT:\s*(.*?)\s*\]\s*(?:<br\s*/?>|</p>\s*<p>|\n)+\s*\[PLOT:\s*(.*?)\s*\]', r'[PLOT: \1, \2]', html_str, flags=re.IGNORECASE)
        
        def sub_match(match):
            plot_keys_str = match.group(1)
            keys = [k.strip().lower() for k in plot_keys_str.split(',')]
            plots_html = '<div class="inline-plots-container">'
            def normalize_plot_key(value: str) -> str:
                value = value.lower().strip()
                for suffix in ("_png", "_html", ".png", ".html"):
                    value = value.replace(suffix, "")
                value = value.replace("_", "").replace("-", "").replace(" ", "")
                aliases = {
                    "shapsummary": "shapplot",
                    "shap": "shapplot",
                    "featureimportance": "featureimportance",
                    "roccurve": "roccurve",
                    "prcurve": "precisionrecallcurve",
                    "confusionmatrix": "confusionmatrix",
                    "correlationheatmap": "corrheatmap",
                    "pca2d": "pcaplot",
                    "pls2d": "plsplot",
                    "umap2d": "umapplot",
                }
                return aliases.get(value, value)
            for k in keys:
                clean_k = normalize_plot_key(k)
                match_key = next((vk for vk in formatted_visuals if normalize_plot_key(vk) == clean_k or clean_k in normalize_plot_key(vk) or normalize_plot_key(vk) in clean_k), None)
                if match_key and formatted_visuals[match_key]:
                    val = formatted_visuals[match_key]
                    title = match_key.replace('_png', '').replace('_html', '').replace('.png', '').replace('.html', '').replace('_', ' ').title()
                    if str(val).lower().endswith(".html"):
                        plots_html += f"""
                        <div class="plot-container inline-plot">
                            <iframe class="plot-img" src="{val}" title="{match_key}" style="border:0;"></iframe>
                            <div class="plot-title">{title}</div>
                        </div>
                        """
                    else:
                        plots_html += f"""
                        <div class="plot-container inline-plot" onclick="zoomPlot('{val}')">
                            <img class="plot-img" src="{val}" alt="{match_key}">
                            <div class="plot-title">{title}</div>
                        </div>
                        """
            plots_html += '</div>'
            return plots_html
            
        return re.sub(r'\[PLOT:\s*(.*?)\s*\]', sub_match, html_str, flags=re.IGNORECASE)

    def _artifact_url(self, report_id: str, artifact_path: str) -> str:
        """Return a browser URL for an artifact path in the report output folder."""
        try:
            output_dir = os.path.join(settings.MEDIA_ROOT, report_id, "output")
            filename = os.path.basename(artifact_path)
            if os.path.abspath(artifact_path).startswith(os.path.abspath(output_dir)):
                from urllib.parse import quote
                return f"/media/{report_id}/output/{quote(filename)}"
        except Exception:
            pass
        return artifact_path
