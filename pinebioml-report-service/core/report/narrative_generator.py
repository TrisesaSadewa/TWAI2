import os
import json
import logging
import re
import requests
from typing import Dict, Any, List, Optional
from core.config import settings, get_model_tier, get_model_config, get_deployment_writer_model
from core.streaming import publish as publish_stream

class HallucinationError(Exception):
    def __init__(self, message, bad_sections=None):
        super().__init__(message)
        self.bad_sections = bad_sections or []

class PromptLeakageError(Exception): pass

logger = logging.getLogger(__name__)

# ── Known instruction phrases that should NEVER appear in LLM output ─────────
_LEAKAGE_PHRASES = [
    "dynamically generate",
    "do not use generic explanations",
    "leverage your expert knowledge",
    "explain how the model hyperparameters",
    "list all selected features and",
    "you must return a raw json",
    "format the output strictly as",
    "you are an expert clinical data scientist",
    "must be strictly one of",
    "generate a dictionary of 5-10",
    "must be generated in both english",
    "you must include this exact structured explanation",
]

_PERFECT_CLAIM_RE = re.compile(
    r"\b(perfect|flawless|error[- ]?free|no false positives|no false negatives|caught everything|missed nothing|100\s*%|100\s+percent)\b",
    re.IGNORECASE,
)

_COMMON_CLINICAL_FEATURE_RE = {
    "age": re.compile(r"\bage\b", re.IGNORECASE),
    "blood pressure": re.compile(r"\bblood\s+pressure\b", re.IGNORECASE),
    "cholesterol": re.compile(r"\bcholesterol\b", re.IGNORECASE),
    "glucose": re.compile(r"\bglucose\b", re.IGNORECASE),
    "bmi": re.compile(r"\bbmi\b|\bbody\s+mass\s+index\b", re.IGNORECASE),
    "heart rate": re.compile(r"\bheart\s+rate\b", re.IGNORECASE),
    "smoking": re.compile(r"\bsmoking\b|\bsmoker\b", re.IGNORECASE),
}


def _is_negated_match(content: str, match: re.Match) -> bool:
    """Return True for phrases like "not perfect" that are cautions, not claims."""
    prefix = content[max(0, match.start() - 12):match.start()].lower()
    return bool(re.search(r"\b(not|isn't|is not|wasn't|was not|cannot be|not yet)\s*$", prefix))


def _feature_allowed(feature_label: str, allowed_features: set[str]) -> bool:
    compact_label = re.sub(r"[^a-z0-9]+", "", feature_label.lower())
    return any(compact_label in feat or feat in compact_label for feat in allowed_features)


class NarrativeGenerator:
    """
    Generates expert narratives using an LLM (OpenAI-compatible).
    If the LLM is unavailable or fails validation, it returns a transparent
    narrative-unavailable notice instead of pretending to provide AI analysis.

    Uses a tiered prompt strategy:
      - Tier 1 (basic models like qwen2.5-coder): Simple data-focused prompt,
        no example prose, explicit JSON schema
      - Tier 2 (strong reasoners like deepseek-r1): Rich prompt with full
        clinical context and structural examples
    """

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_API_BASE_URL
        self.model = get_deployment_writer_model()

    def _is_regression_task(self, task_type: str) -> bool:
        return "regression" in str(task_type or "").lower()

    def generate_narrative(
        self,
        dataset_name: str,
        task_type: str,
        metrics: dict,
        visuals_summary: dict,
        shap_features: list = None,
        anomaly_flags: list = None,
        models: dict = None,
        use_cpu_fallback: bool = False,
        report_id: str = None,
        per_class: list = None,
        overfit_analysis: dict = None,
        selected_features: list = None,
        **kwargs
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate both Expert and Layman reports.
        """
        formatted_metrics = json.dumps(metrics, indent=2)
        visual_names = list(visuals_summary.keys())
        imbalance_metadata = kwargs.get("imbalance_metadata")
        imbalance_warning = kwargs.get("imbalance_warning")

        is_custom_url = self.base_url and "api.openai.com" not in self.base_url
        if self.api_key or is_custom_url:
            try:
                models = models or {}
                model_key = models.get("analysis")
                if model_key and model_key in settings.SUPPORTED_MODELS:
                    resolved_model = settings.SUPPORTED_MODELS[model_key]
                else:
                    resolved_model = self.model
                    logger.warning(f"Requested model alias '{model_key}' not found. Falling back to default: {resolved_model}")

                tier = get_model_tier(model_key or resolved_model)
                model_cfg = get_model_config(model_key or resolved_model)
                logger.info(f"Triggering LLM narrative generation using {resolved_model} (tier {tier})...")

                fallback_llm = None
                qwen_tag = settings.SUPPORTED_MODELS.get("qwen2.5-coder-14b", "qwen2.5-coder:14b")
                if resolved_model != qwen_tag:
                    fallback_llm = qwen_tag

                result = self._generate_via_llm(
                    dataset_name, task_type, formatted_metrics, visual_names,
                    shap_features, anomaly_flags, metrics, resolved_model,
                    use_cpu_fallback=use_cpu_fallback, report_id=report_id,
                    per_class=per_class, overfit_analysis=overfit_analysis,
                    selected_features=selected_features,
                    tier=tier, model_cfg=model_cfg,
                    visuals_summary=visuals_summary,
                    imbalance_metadata=imbalance_metadata,
                    imbalance_warning=imbalance_warning,
                    fallback_model=fallback_llm
                )
                return result
            except Exception as e:
                logger.error(f"LLM generation failed; returning narrative-unavailable notice: {e}")
                return self._generate_rule_based(
                    dataset_name, task_type, metrics, visuals_summary, shap_features, anomaly_flags,
                    per_class=per_class, overfit_analysis=overfit_analysis, selected_features=selected_features,
                    imbalance_metadata=imbalance_metadata,
                    imbalance_warning=imbalance_warning
                )
        else:
            logger.info("No LLM API Key set. Returning narrative-unavailable notice.")
            return self._generate_rule_based(
                dataset_name, task_type, metrics, visuals_summary, shap_features, anomaly_flags,
                per_class=per_class, overfit_analysis=overfit_analysis, selected_features=selected_features,
                imbalance_metadata=imbalance_metadata,
                imbalance_warning=imbalance_warning
            )

    # ── Prompt Building ──────────────────────────────────────────────────────

    def _build_data_block(
        self, dataset_name, task_type, formatted_metrics,
        shap_features, anomaly_flags, per_class, overfit_analysis,
        selected_features, visual_descriptions, imbalance_metadata=None,
        imbalance_warning=None
    ) -> str:
        """Build the factual DATA section — same for all tiers."""
        sections = [
            f"DATASET: {dataset_name}",
            f"ML TASK: {task_type}",
            f"METRICS:\n{formatted_metrics}",
        ]

        if shap_features:
            lines = "\n".join(f"  - {f['feature']}: importance={f['importance']}" for f in shap_features)
            sections.append(f"TOP FEATURES (SHAP):\n{lines}")

        if per_class:
            lines = "\n".join(
                f"  - Class '{c['class']}': Precision={c['precision']:.3f}, Recall={c['recall']:.3f}, F1={c['f1']:.3f}, Support={c['support']}"
                for c in per_class
            )
            sections.append(f"PER-CLASS PERFORMANCE:\n{lines}")

        if overfit_analysis and overfit_analysis.get("models"):
            lines = "\n".join(
                f"  - {g['model']}: Train={g['train_accuracy']:.4f}, Test={g['test_accuracy']:.4f}, Gap={g['gap_pct']}"
                for g in overfit_analysis["models"]
            )
            warning = " [WARNING: >10% gap detected]" if overfit_analysis.get("is_overfitting") else ""
            sections.append(f"OVERFITTING ANALYSIS:{warning}\n{lines}")

        if selected_features:
            feat_list = ", ".join(selected_features[:15])
            sections.append(
                f"FEATURES AVAILABLE TO MODEL ({len(selected_features)} total): {feat_list}\n"
                "Only discuss feature names from this list. If a feature is absent from this list, do not mention it."
            )

        if anomaly_flags:
            lines = "\n".join(f"  - {a}" for a in anomaly_flags)
            sections.append(f"ANOMALY FLAGS:\n{lines}")

        if imbalance_warning:
            sections.append(
                "ACCURACY / CLASS IMBALANCE WARNING:\n"
                f"  - Severity: {imbalance_warning.get('severity', 'medium')}\n"
                f"  - {imbalance_warning.get('message', '')}\n"
                "  - The report must explicitly warn that accuracy alone may be misleading."
            )

        if visual_descriptions:
            lines = "\n".join(f"  - {name}: {desc}" for name, desc in visual_descriptions.items())
            sections.append(f"VISUAL PLOT ANALYSES:\n{lines}")

        if imbalance_metadata:
            lines = []
            for k, v in imbalance_metadata.items():
                if isinstance(v, dict):
                    v_str = ", ".join(f"{vk}={vv}" for vk, vv in v.items())
                    lines.append(f"  - {k}: {v_str}")
                else:
                    lines.append(f"  - {k}: {v}")
            # Explicit methodology facts so the LLM states them plainly instead
            # of inferring from the raw metadata dump.
            minority = imbalance_metadata.get("class_distribution")
            if minority:
                # The minority class label is the one with the smallest percentage.
                try:
                    minor_label = min(minority, key=lambda k: float(str(minority[k]).split('%')[0]))
                    lines.append(f"  - METRIC DEFINITION: sensitivity, specificity, and ROC-AUC are computed with the MINORITY class ('{minor_label}') as the positive class. This is deliberate so the report reflects detection of the rare/clinically-important cases, not the majority class.")
                except Exception:
                    pass
            if imbalance_metadata.get("class_weight_applied"):
                lines.append("  - IMBALANCE HANDLING: class_weight='balanced' (cost-sensitive learning) was applied during training to counter class imbalance. State this as a fact; do NOT claim SMOTE/oversampling unless 'SMOTE' appears in tool_used.")
            tt = imbalance_metadata.get("tuned_threshold")
            tm = imbalance_metadata.get("threshold_metric")
            if tt is not None:
                lines.append(f"  - DECISION THRESHOLD: the binary decision threshold was tuned to {tt} (maximizing {tm or 'f1'} for the minority class on out-of-fold predictions) instead of the default 0.5/argmax. Reported sensitivity/specificity reflect this tuned threshold.")
            if lines:
                sections.append(f"DATA QUALITY & PREPROCESSING:\n" + "\n".join(lines))
        elif "regression" not in str(task_type or "").lower():
            sections.append(
                "DATA QUALITY & PREPROCESSING:\n"
                "  - CLASS IMBALANCE METADATA: not recorded in the pipeline output for this run.\n"
                "  - IMBALANCE HANDLING: not recorded. Do not claim the data are balanced, and do not claim that SMOTE, oversampling, class weighting, or threshold tuning was or was not used unless explicitly stated elsewhere in DATA.\n"
                "  - If discussing class balance, use only PER-CLASS PERFORMANCE support counts when available; otherwise say that class-distribution metadata were not available."
            )

        return "\n\n".join(sections)

    def _build_visual_descriptions(self, metrics, shap_features, visuals_summary, task_type: str = "") -> dict:
        """Pre-compute metric-aware descriptions of visual plots for the prompt."""
        accuracy = metrics.get("accuracy", "N/A")
        roc_auc = metrics.get("ROC-AUC", "N/A")
        recall = metrics.get("recall", "N/A")
        r2 = metrics.get("R2", "N/A")
        rmse = metrics.get("RMSE", "N/A")
        is_regression = self._is_regression_task(task_type)

        descriptions = {}
        for key in visuals_summary:
            name = key.replace("_png", "").replace("_", " ").title()
            if is_regression and "true_vs_predicted" in key:
                descriptions[name] = (
                    f"True-vs-predicted regression plot. Points near the diagonal indicate accurate predictions. "
                    f"R2={r2}, RMSE={rmse}."
                )
            elif is_regression and "residuals" in key:
                descriptions[name] = (
                    "Residual plot for regression errors. Random scatter around zero supports stable fit; "
                    "curvature, funnels, or outliers suggest bias, heteroscedasticity, or unusual cases."
                )
            elif "confusion_matrix" in key:
                descriptions[name] = (
                    f"Shows true/false positive/negative distribution. "
                    f"Accuracy={accuracy}, Recall={recall}. Strong diagonal expected."
                )
            elif "roc_curve" in key:
                descriptions[name] = (
                    f"ROC curve with AUC={roc_auc}. Curve bending toward top-left indicates "
                    f"strong discriminative power above random baseline (0.5)."
                )
            elif "feature_importance" in key:
                if shap_features:
                    top = shap_features[0]
                    descriptions[name] = (
                        f"Ranked bar chart of predictive features. Top predictor: "
                        f"{top['feature']} (importance={top['importance']}). "
                        f"Steep drop-off indicates sparse, interpretable feature set."
                    )
                else:
                    descriptions[name] = "Ranked bar chart of feature predictive importance."
            elif "shap" in key:
                if shap_features:
                    top3 = ", ".join(f['feature'] for f in shap_features[:3])
                    descriptions[name] = (
                        f"SHAP beeswarm plot. Top drivers: {top3}. "
                        f"Red=high feature value pushing prediction right, blue=low pushing left."
                    )
                else:
                    descriptions[name] = "SHAP summary plot showing feature importance and directionality."
            else:
                descriptions[name] = visuals_summary[key].get("description_fallback", "Model validation plot.")
        return descriptions

    def _build_regression_prompt(self, data_block: str, dataset_name: str) -> list:
        system = (
            "You are a clinical biostatistics AI. You analyze regression ML training results and write professional reports. "
            "You output ONLY valid JSON. Never include your instructions in the output."
        )
        user = f"""Analyze this REGRESSION ML training run and generate a clinical narrative report.

=== TRAINING RUN DATA ===
{data_block}
=== END DATA ===

Generate a JSON object with two keys: "expert" and "glossary". Every section value must be a single Markdown string.

EXPERT REPORT:
- "executive_summary": Summarize the continuous prediction task for dataset {dataset_name}. Discuss R2, RMSE, MAE, and MSE only if present in the DATA. Do not mention accuracy, ROC-AUC, confusion matrices, false positives, false negatives, sensitivity, specificity, or class imbalance.
- "preprocessing_and_data_quality": Explain preprocessing and data quality using only the DATA section. If class-distribution metadata is absent, do not invent it.
- "findings": Start with [PLOT: true_vs_predicted, residuals]. Then render a Markdown table with exactly two columns, | Metric | Value |, using only regression metrics from DATA such as R2, RMSE, MAE, and MSE. After the table, explain how close predictions are to observed values, what residual spread means, and whether unusual cases or systematic bias may be present. Then include [PLOT: feature_importance, shap_summary] and explain key drivers if available.
- "visuals_analysis": Explain each available regression plot with [PLOT: plot_name] directly above the explanation. Focus on true-vs-predicted, residuals, feature importance, SHAP, PCA, PLS, UMAP, and correlation plots when present.
- "conclusion": Give a careful conclusion about regression model reliability, limitations, and whether additional validation is needed.
- "recommendations": Give practical recommendations for improving continuous-target prediction and validating the model.

GLOSSARY:
Include definitions for R2, RMSE, MAE, MSE, residual, and SHAP.

RULES:
1. Use ONLY the metric values from the DATA section. Do not invent numbers.
2. Do not discuss classification concepts for this regression task.
3. Output ONLY the JSON object. Do not wrap it in markdown fences."""

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    def _build_prompt_tier1(self, data_block: str, imbalance_metadata: dict = None, has_overfit_data: bool = False) -> list:
        """
        Tier 1: Simple, explicit prompt for basic models (qwen2.5-coder, gpt-4o-mini).
        No example prose. Just data + strict schema.
        """
        data_quality_text = "Class-imbalance handling is not recorded in the DATA. Do not infer balanced classes or absence of correction."
        if imbalance_metadata:
            data_quality_text = "Data quality and preprocessing analysis identified the following key metadata: " + ", ".join([f"{k} ({v})" for k,v in imbalance_metadata.items() if not isinstance(v, dict)])
            
        system = (
            "You are a clinical biostatistics AI. You analyze ML training results and write professional reports. "
            "You output ONLY valid JSON. Never include your instructions in the output."
        )
        user = f"""Analyze this ML training run data and write a clinical report as JSON.

=== DATA ===
{data_block}
=== END DATA ===

Write a JSON object with this EXACT structure. All values MUST be a single Markdown-formatted string. Do NOT use nested objects or arrays (for tables, use standard Markdown table syntax, not JSON arrays). Fill every field with substantive clinical analysis (NOT placeholders). Use the actual metric values from the DATA above.

{{
  "expert": {{
    "executive_summary": "1) Short Summary of results. 2) Example of the performance (e.g., 'In 100 patients, the model can...'). 3) Key Points Identified. Readable in 30 seconds. NO dramatic language.",
    "preprocessing_and_data_quality": "Write 4-5 sentences explaining the data quality. You MUST explicitly discuss this metadata: {data_quality_text}",
    "findings": "Quantitative Analysis & Findings. MUST BE A SINGLE MARKDOWN STRING. 1) Format Accuracy, Precision, Sensitivity/Recall, Specificity, F1-Score, MCC, ROC-AUC as a standard Markdown table with EXACTLY two columns: Metric and Value. DO NOT include an interpretation column. This table MUST BE AT THE VERY TOP of the section. 2) Put [PLOT: roc_curve]. 3) Write a 3-5 sentence explanation of the ROC curve. 4) Put [PLOT: pr_curve]. 5) Write a 3-5 sentence explanation of the PR curve. 6) Put [PLOT: confusion_matrix]. 7) Write a 3-5 sentence explanation of the Confusion Matrix. 8) Put [PLOT: correlation_heatmap]. 9) Write a 3-5 sentence explanation of the Correlation Heatmap (top 5-10 features). 10) Put [PLOT: feature_importance]. 11) Write a 3-5 sentence explanation of Feature Importance (top 5-10 features). 12) Put [PLOT: shap_summary]. 13) Write a 3-5 sentence explanation of SHAP (top 5-10 features). 14) Discuss dimensionality reduction plots ([PLOT: pca_2d], [PLOT: pls_2d], [PLOT: umap_2d]) with a 3-5 sentence explanation for each if present.{' 15) Add Overfitting analysis.' if has_overfit_data else ''}",
    "visuals_analysis": "Explain each available plot as a SINGLE MARKDOWN STRING. Put the relevant [PLOT: plot_name] placeholder directly above each explanation. Explain how to read the plot axes, colors, bars, or clusters. Do not invent metrics or repeat unsupported performance claims.",
    "conclusion": "Conclusion of the model. MUST BE A SINGLE STRING of at least 5-10 sentences.",
    "recommendations": "1) Dataset fix / how to improve data quality. 2) Recommendations on model usage. 3) Other tips. MUST BE A SINGLE STRING of at least 5-10 sentences."
  }},
  "glossary": {{
    "Accuracy": "English definition",
    "AUC-ROC": "English definition",
    "false alarms": "English definition",
    "missed cases": "English definition"
  }}
}}

RULES:
1. Use ONLY the metric values from the DATA section. Do not invent numbers.
2. The expert findings MUST reference specific values (e.g., "Precision of 77.31%").
3. If per-class data is provided, include a per-class breakdown table.
    "findings": "Quantitative Analysis & Findings. MUST BE A SINGLE MARKDOWN STRING. 1) Format Accuracy, Precision, Sensitivity/Recall, Specificity, F1-Score, MCC, ROC-AUC as a standard Markdown table with EXACTLY two columns: Metric and Value. DO NOT include an interpretation column. This table MUST BE AT THE VERY TOP of the section. 2) Put [PLOT: roc_curve]. 3) Write a 3-5 sentence explanation of the ROC curve. 4) Put [PLOT: pr_curve]. 5) Write a 3-5 sentence explanation of the PR curve. 6) Put [PLOT: confusion_matrix]. 7) Write a 3-5 sentence explanation of the Confusion Matrix. 8) Put [PLOT: correlation_heatmap]. 9) Write a 3-5 sentence explanation of the Correlation Heatmap (top 5-10 features). 10) Put [PLOT: feature_importance]. 11) Write a 3-5 sentence explanation of Feature Importance (top 5-10 features). 12) Put [PLOT: shap_summary]. 13) Write a 3-5 sentence explanation of SHAP (top 5-10 features). 14) Discuss dimensionality reduction plots ([PLOT: pca_2d], [PLOT: pls_2d], [PLOT: umap_2d]) with a 3-5 sentence explanation for each if present.{' 15) Add Overfitting analysis.' if has_overfit_data else ''}",
    "visuals_analysis": "Explain each available plot as a SINGLE MARKDOWN STRING. Put the relevant [PLOT: plot_name] placeholder directly above each explanation. Explain how to read the axes, colors, bars, clusters. Do not invent metrics or repeat unsupported performance claims.",
    "conclusion": "Conclusion of the model. MUST BE A SINGLE STRING of at least 5-10 sentences.",
    "recommendations": "1) Dataset fix / how to improve data quality. 2) Recommendations on model usage. 3) Other tips. MUST BE A SINGLE STRING of at least 5-10 sentences."
  }},
  "glossary": {{
    "Accuracy": "English definition",
    "AUC-ROC": "English definition",
    "false alarms": "English definition",
    "missed cases": "English definition"
  }}
}}

RULES:
1. Use ONLY the metric values from the DATA section. Do not invent numbers.
2. The expert findings MUST reference specific values (e.g., "Precision of 77.31%").
3. If per-class data is provided, include a per-class breakdown table.
4. If anomaly flags are present, explain them in findings.
5. If an ACCURACY / CLASS IMBALANCE WARNING is present, explicitly state that accuracy alone may be misleading in the expert executive_summary and expert findings.
6. Write at least 150 words per section.
7. Do NOT claim 100%, perfect, flawless, or error-free performance unless that exact value is present in the DATA.
8. For class balance and imbalance correction, use only DATA QUALITY & PREPROCESSING facts. If imbalance metadata says "not recorded", write that it was not recorded; do NOT infer the data are balanced or that no correction was used.
9. Do NOT repeat these instructions in your output."""

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    def _build_prompt_tier2(self, data_block: str, dataset_name: str, has_overfit_data: bool = False) -> list:
        """
        Tier 2: Rich prompt for strong reasoners (deepseek-r1, gpt-4o).
        More nuanced instructions, clinical depth.
        """
        system = (
            "You are an expert Clinical Data Scientist and Biostatistician at PineBioML. "
            "You write comprehensive, peer-review-quality clinical ML validation reports. "
            "You output ONLY valid JSON structures. Never echo instructions."
        )
        user = f"""Analyze this ML training run and generate a comprehensive clinical narrative report.

=== TRAINING RUN DATA ===
{data_block}
=== END DATA ===

Generate a JSON object with two keys: "expert" and "glossary".

EXPERT REPORT (Follow these EXACT sections and instructions. All section values MUST be a single Markdown string, NO nested JSON objects or arrays):
- "executive_summary":
   1) Short Summary of results.
   2) Example of the performance (e.g., "In 100 patients, the model can...").
   3) Key Points Identified.
   *Rule: Must be readable in 30 seconds. NO dramatic language or grand proclamations (e.g., avoid "struck a balance", "harmonic balance").*

- "preprocessing_and_data_quality":
   1) Provide a detailed, definitive, and actionable explanation of the input data quality and preprocessing. MUST BE EXACTLY 4-5 SENTENCES.
   *Rule: You MUST explicitly mention the class distribution and exact imbalance strategy/tools only when they are recorded in DATA QUALITY & PREPROCESSING or PER-CLASS PERFORMANCE. If imbalance metadata is "not recorded", say it was not recorded and do NOT infer balanced classes, no imbalance correction, SMOTE, oversampling, class weighting, or threshold tuning. Do NOT use speculative language like "likely", "appears", or "I feel like". State facts based ONLY on the provided DATA section.*

- "findings":
   **MUST BE A SINGLE MARKDOWN STRING OF AT LEAST 10 SENTENCES. DO NOT OUTPUT AS NESTED JSON OBJECTS.**
   **Quantitative Analysis & Findings**
   1) Render the Overall Performance Metrics (Accuracy, ROC-AUC, Precision, Recall/Sensitivity, Specificity, F1-Score, MCC) as a standard Markdown table with EXACTLY two columns: "Metric" and "Value". DO NOT include an interpretation column. This table MUST BE AT THE VERY TOP of the section. **CRITICAL RULE: YOU MUST USE THE EXACT NUMBERS PROVIDED IN THE `PRE-COMPUTED METRICS` SECTION. DO NOT INVENT OR HALLUCINATE NUMBERS. If Accuracy is 99.12%, you must write 99.12%.**
   2) Put [PLOT: roc_curve] directly below the metrics table.
   3) Write a 3-5 sentence explanation of the ROC curve and its implications.
   4) Put [PLOT: pr_curve] directly below the ROC curve explanation.
   5) Write a 3-5 sentence explanation of the PR curve.
   6) Put [PLOT: confusion_matrix] directly below the PR curve explanation.
   7) Write a 3-5 sentence explanation of the Confusion Matrix (TNs, TPs, FPs, FNs) and clinical impact.
   8) Put [PLOT: correlation_heatmap] directly below the confusion matrix explanation.
   9) Write a 3-5 sentence explanation of the Correlation Heatmap, focusing on the top 5-10 features.
   10) Put [PLOT: feature_importance] directly below the correlation heatmap explanation.
   11) Write a 3-5 sentence explanation of Feature Importance, focusing on the top 5-10 features.
   12) Put [PLOT: shap_summary] directly below the feature importance explanation.
   13) Write a 3-5 sentence explanation of SHAP, focusing on the top 5-10 features.
   14) Discuss dimensionality reduction plots ([PLOT: pca_2d], [PLOT: pls_2d], [PLOT: umap_2d]) with a 3-5 sentence explanation for each plot if present.{f'''
   15) Write an Overfitting Analysis paragraph based on train/test gaps.''' if has_overfit_data else ''}

- "visuals_analysis":
   Explain any other available plots with the relevant [PLOT: plot_name] placeholder directly above the explanation. Focus on how to read the axes, colors, bars, clusters, or curves. Do not invent values. Do not repeat unsupported performance scores.

- "conclusion":
   - Detailed conclusion of what has been discussed. Read in 60 seconds. NO dramatic language. Avoid technical/convoluted words. MUST BE A SINGLE STRING OF AT LEAST 5-10 SENTENCES.

- "recommendations":
   1) Dataset fix / how to improve the quality of the data.
   2) Recommendations on model usage.
   3) Other tips and Recommendations. MUST BE A SINGLE STRING OF AT LEAST 5-10 SENTENCES.

Output ONLY the JSON object. Do NOT include markdown code fences around it."""

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    # ── LLM Generation ───────────────────────────────────────────────────────

    def _generate_via_llm(
        self, dataset_name, task_type, formatted_metrics, visual_names,
        shap_features, anomaly_flags, raw_metrics, resolved_model,
        use_cpu_fallback=False, report_id=None, per_class=None,
        overfit_analysis=None, selected_features=None,
        tier=1, model_cfg=None, visuals_summary=None, imbalance_metadata=None,
        fallback_model=None, imbalance_warning=None
    ) -> Dict[str, Dict[str, str]]:
        """Query LLM for narrative generation with tier-aware prompting."""
        api_key = self.api_key or "local-llm-key"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Pre-compute visual descriptions as data (not asking LLM to interpret images)
        visual_descriptions = {}
        if visuals_summary:
            visual_descriptions = self._build_visual_descriptions(
                raw_metrics, shap_features, visuals_summary, task_type
            )

        # Build factual data block
        data_block = self._build_data_block(
            dataset_name, task_type, formatted_metrics,
            shap_features, anomaly_flags, per_class, overfit_analysis,
            selected_features, visual_descriptions, imbalance_metadata,
            imbalance_warning
        )

        has_overfit_data = bool(overfit_analysis and overfit_analysis.get("models"))
        
        # Select prompt tier
        if self._is_regression_task(task_type):
            messages = self._build_regression_prompt(data_block, dataset_name)
        elif tier >= 2:
            messages = self._build_prompt_tier2(data_block, dataset_name, has_overfit_data)
        else:
            messages = self._build_prompt_tier1(data_block, imbalance_metadata, has_overfit_data)

        model_cfg = model_cfg or {}
        payload = {
            "model": resolved_model,
            "messages": messages,
            "temperature": model_cfg.get("temperature", 0.2)
        }

        # Set Ollama-specific options if local endpoint is used
        options = {}
        if use_cpu_fallback:
            options["num_gpu"] = 0
        
        # Override context window size based on model config
        context_tokens = model_cfg.get("context_tokens") or model_cfg.get("max_tokens")
        if context_tokens:
            options["num_ctx"] = context_tokens
            
        if options:
            payload["options"] = options

        url = f"{self.base_url}/chat/completions"
        request_timeout = settings.LLM_REQUEST_TIMEOUT_SECONDS

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            if report_id:
                payload["stream"] = True
                resp = requests.post(url, headers=headers, json=payload, timeout=request_timeout, stream=True)
                resp.raise_for_status()

                result_json = ""
                for line in resp.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                chunk_data = json.loads(line[6:])
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    result_json += content
                            except json.JSONDecodeError:
                                pass
                # Stream DONE notification will happen after quality gates pass
            else:
                resp = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
                resp.raise_for_status()
                result_json = resp.json()["choices"][0]["message"]["content"]

            import re
            cleaned_response = result_json
            # Strip reasoning blocks. Reasoning models (deepseek-r1) wrap
            # chain-of-thought in think/open-think ... close-think tags. Two
            # failure modes: (a) balanced tags -> remove the whole block;
            # (b) UNCLOSED opening tag (truncated / stream-split across
            # chunks) -> raw deliberation (which often invents numbers) leaks
            # into the parsed JSON and trips the hallucination gate. Drop
            # everything from the opening tag to end-of-string in that case.
            _OPEN = chr(60) + "think" + chr(62)
            _CLOSE = chr(60) + "/" + "think" + chr(62)
            cleaned_response = re.sub(
                re.escape(_OPEN) + r".*?" + re.escape(_CLOSE),
                "", cleaned_response, flags=re.DOTALL
            ).strip()
            if _OPEN in cleaned_response:
                cleaned_response = cleaned_response.split(_OPEN, 1)[0].rstrip()
            # (legacy think_pattern regex removed — superseded by the _OPEN/_CLOSE
            #  handling above, which also covers unclosed tags.)

            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            try:
                parsed_json = json.loads(cleaned_response.strip())

                def _flatten_to_markdown(val):
                    if isinstance(val, str): return val
                    if isinstance(val, list):
                        # If list of lists (e.g., table rows), format as table-ish or bullet points
                        if all(isinstance(i, list) for i in val):
                            return "\n".join("- " + ": ".join(str(x) for x in row) for row in val)
                        return "\n\n".join(str(i) for i in val)
                    if isinstance(val, dict):
                        parts = []
                        for k, v in val.items():
                            parts.append(f"**{str(k).replace('_', ' ').title()}**")
                            parts.append(_flatten_to_markdown(v))
                        return "\n\n".join(parts)
                    return str(val)

                # Normalize sections that must be single markdown strings
                for mode in ("expert",):
                    if mode in parsed_json:
                        for section in ("findings", "visuals_analysis", "preprocessing_and_data_quality"):
                            if section in parsed_json[mode]:
                                if isinstance(parsed_json[mode][section], (dict, list)):
                                    parsed_json[mode][section] = _flatten_to_markdown(parsed_json[mode][section])

                # ── Quality Gates ────────────────────────────────────────────────
                try:
                    self._validate_no_hallucinations(
                        parsed_json,
                        raw_metrics,
                        per_class,
                        overfit_analysis,
                        imbalance_metadata,
                        selected_features,
                    )
                except HallucinationError as e:
                    try:
                        debug_path = os.path.join(settings.STORAGE_DIR, f"failed_llm_{report_id or 'unknown'}.json")
                        with open(debug_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "error": str(e),
                                "llm_output": parsed_json,
                                "raw_metrics": raw_metrics,
                                "per_class": per_class,
                                "overfit_analysis": overfit_analysis
                            }, f, indent=4)
                        logger.warning(f"Saved failed LLM output due to hallucination check to {debug_path}")
                    except Exception as debug_err:
                        logger.error(f"Failed to save debug LLM output: {debug_err}")
                    raise e
                    
                quality_result = self._validate_output_quality(parsed_json, raw_metrics, task_type)

                if quality_result["leaked"]:
                    raise PromptLeakageError(f"Prompt instructions leaked into output: {quality_result['leaked_phrases']}")

                break
                
            except (HallucinationError, PromptLeakageError, json.JSONDecodeError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    # Append the CLEANED response (think-tags / code fences stripped)
                    # as the assistant turn — not the raw result_json, which may
                    # contain leaked chain-of-thought that would confuse the model
                    # into repeating the same broken format on retry.
                    messages.append({"role": "assistant", "content": cleaned_response})
                    messages.append({
                        "role": "user",
                        "content": f"Your previous response failed validation: {str(e)}. Please correct this and generate a new JSON strictly adhering to the provided metrics."
                    })
                    if report_id:
                        msg = f"\n\n[System] Quality check failed ({str(e)}). Retrying...\n\n"
                        publish_stream(report_id, msg)
                    continue
                else:
                    if fallback_model:
                        logger.warning(f"Primary model {resolved_model} failed after {MAX_RETRIES} attempts. Cascading to backup LLM: {fallback_model}")
                        if report_id:
                            publish_stream(report_id, f"\n\n[System] Primary model failed. Cascading to backup LLM ({fallback_model})...\n\n")
                        
                        fb_tier = get_model_tier(fallback_model)
                        fb_cfg = get_model_config(fallback_model)
                        
                        return self._generate_via_llm(
                            dataset_name, task_type, formatted_metrics, visual_names,
                            shap_features, anomaly_flags, raw_metrics, fallback_model,
                            use_cpu_fallback=use_cpu_fallback, report_id=report_id, per_class=per_class,
                            overfit_analysis=overfit_analysis, selected_features=selected_features,
                            tier=fb_tier, model_cfg=fb_cfg, visuals_summary=visuals_summary, 
                            imbalance_metadata=imbalance_metadata,
                            imbalance_warning=imbalance_warning,
                            fallback_model=None  # Prevent infinite cascade
                        )

                    # Fallback after max retries
                    logger.error(f"Failed after {MAX_RETRIES} attempts. Returning narrative-unavailable notice.")
                    fallback = self._generate_rule_based(
                        dataset_name, task_type, raw_metrics, visuals_summary or {},
                        shap_features, anomaly_flags, per_class=per_class, 
                        overfit_analysis=overfit_analysis, selected_features=selected_features,
                        imbalance_warning=imbalance_warning
                    )
                    
                    if isinstance(e, HallucinationError) and getattr(e, 'bad_sections', None) and 'parsed_json' in locals():
                        logger.warning(f"Max retries reached with hallucination flags in: {e.bad_sections}. Discarding untrusted LLM narrative.")

                    if report_id:
                        publish_stream(report_id, json.dumps(fallback))
                        publish_stream(report_id, "[DONE]")
                    return fallback

        # If partially valid, merge good LLM sections with rule-based fallback
        if quality_result["incomplete_sections"]:
            logger.warning(f"LLM output has incomplete sections: {quality_result['incomplete_sections']}. Merging with rule-based fallback.")
            fallback = self._generate_rule_based(
                dataset_name, task_type, raw_metrics, visuals_summary or {},
                shap_features, anomaly_flags,
                per_class=per_class, overfit_analysis=overfit_analysis,
                selected_features=selected_features,
                imbalance_warning=imbalance_warning
            )
            parsed_json = self._merge_with_fallback(parsed_json, fallback, quality_result["incomplete_sections"])

        if report_id:
            publish_stream(report_id, json.dumps(parsed_json))
            publish_stream(report_id, "[DONE]")

        return parsed_json

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_no_hallucinations(
        self,
        narrative: dict,
        metrics: dict,
        per_class: list = None,
        overfit_analysis: dict = None,
        imbalance_metadata: dict = None,
        selected_features: list = None,
    ):
        """Check that all percentage values in the output exist in the input data."""
        allowed = set()
        allowed_features = {
            re.sub(r"[^a-z0-9]+", "", str(feature).lower())
            for feature in (selected_features or [])
            if str(feature).strip()
        }
        for v in metrics.values():
            v_str = str(v).replace('%', '').strip()
            try:
                val = float(v_str)
                allowed.add(val)
                if 0 <= val <= 1.0:
                    allowed.add(round(val * 100, 2))
                    allowed.add(round(val * 100, 1))
                    allowed.add(round(val * 100, 0))
            except (ValueError, TypeError):
                pass

        # Also allow per-class metric values
        if per_class:
            for c in per_class:
                for key in ('precision', 'recall', 'f1'):
                    try:
                        val = float(c[key])
                        allowed.add(round(val * 100, 2))
                        allowed.add(round(val * 100, 1))
                        allowed.add(round(val, 4))
                    except (ValueError, TypeError, KeyError):
                        pass

        # Allow overfitting analysis gap percentages and train/test accuracies
        if overfit_analysis and "models" in overfit_analysis:
            for m in overfit_analysis["models"]:
                # Allow gap_pct values
                gap_str = str(m.get("gap_pct", "")).replace('%', '').strip()
                try:
                    val = float(gap_str)
                    allowed.add(val)
                    allowed.add(-val)
                    allowed.add(abs(val))
                except (ValueError, TypeError):
                    pass
                # Allow train_accuracy and test_accuracy percentages
                for key in ("train_accuracy", "test_accuracy"):
                    try:
                        val = float(m.get(key, 0))
                        allowed.add(round(val * 100, 2))
                        allowed.add(round(val * 100, 1))
                        allowed.add(round(val * 100, 0))
                    except (ValueError, TypeError):
                        pass

        if imbalance_metadata:
            for k, v in imbalance_metadata.items():
                v_str = str(v).replace('%', '').strip()
                try:
                    val = float(v_str)
                    allowed.add(round(val * 100, 2))
                    allowed.add(round(val * 100, 1))
                    allowed.add(round(val, 4))
                except (ValueError, TypeError):
                    pass

        bad_sections = []
        error_messages = []
        has_true_perfect_metric = any(
            abs(a - 100.0) < 0.1 or abs(a - 1.0) < 0.001
            for a in allowed
        )

        for mode in ("expert",):
            mode_data = narrative.get(mode, {})
            if not isinstance(mode_data, dict):
                continue
            for section, content in mode_data.items():
                if not isinstance(content, str):
                    continue
                
                percentages = re.findall(r'(?<![\d\.])(\d+(?:\.\d+)?)%', content)
                section_hallucinated = False
                hard_hallucination = False
                perfect_claim = _PERFECT_CLAIM_RE.search(content)
                if perfect_claim and not has_true_perfect_metric and not _is_negated_match(content, perfect_claim):
                    section_hallucinated = True
                    hard_hallucination = True
                    error_messages.append(f"{mode}.{section} (unsupported 100%/perfect claim: {perfect_claim.group(0)})")

                if not section_hallucinated and allowed_features:
                    unsupported_features = [
                        label
                        for label, pattern in _COMMON_CLINICAL_FEATURE_RE.items()
                        if pattern.search(content) and not _feature_allowed(label, allowed_features)
                    ]
                    if unsupported_features:
                        section_hallucinated = True
                        hard_hallucination = True
                        error_messages.append(
                            f"{mode}.{section} (unsupported feature mention: {', '.join(unsupported_features)})"
                        )
                
                for pct_str in percentages:
                    if section_hallucinated:
                        break
                    try:
                        pct_val = float(pct_str)
                        found = False
                        if abs(pct_val - 100.0) < 0.1 and not has_true_perfect_metric:
                            section_hallucinated = True
                            hard_hallucination = True
                            error_messages.append(f"{mode}.{section} (unsupported 100% claim)")
                            break
                        for a in allowed:
                            if abs(a - pct_val) <= 1.0:
                                found = True; break
                            if abs(a*100 - pct_val) <= 1.0 or abs(a - pct_val*100) <= 1.0:
                                found = True; break
                            # Allow mathematical complements (e.g. 100 - X% or 1 - X)
                            if 0 <= a <= 1.0:
                                comp = 1.0 - a
                                if abs(comp - pct_val) <= 1.0 or abs(comp*100 - pct_val) <= 1.0 or abs(comp - pct_val*100) <= 1.0:
                                    found = True; break
                            elif 1.0 < a <= 100.0:
                                comp = 100.0 - a
                                if abs(comp - pct_val) <= 1.0 or abs(comp*100 - pct_val) <= 1.0 or abs(comp - pct_val*100) <= 1.0:
                                    found = True; break
                        if not found:
                            # Allow common structural percentages that may refer to baselines
                            # or absence of events, but do not blanket-allow 100% because
                            # that can mask invented "perfect model" claims.
                            if pct_val in (0, 50):
                                continue
                            # Allow percentages derived from AUC (e.g., AUC 0.8650 → 86.5%)
                            auc_val = metrics.get("ROC-AUC", "")
                            try:
                                auc_f = float(str(auc_val).strip())
                                if abs(auc_f * 100 - pct_val) < 0.1:
                                    continue
                            except (ValueError, TypeError):
                                pass
                            
                            section_hallucinated = True
                            hard_hallucination = True
                            error_messages.append(f"{mode}.{section} ({pct_str}%)")
                            break
                    except ValueError:
                        pass
                        
                if section_hallucinated:
                    # Explicit unsupported percentages / perfect claims are hard
                    # failures. Do not spend minutes asking a second LLM to clear
                    # metric values that are not present in the factual input.
                    if hard_hallucination:
                        bad_sections.append(f"{mode}.{section}")
                        continue

                    logger.info(f"Tier 1 flagged potential hallucination in {mode}.{section}. Running Tier 2 Semantic Check...")
                    from core.preprocessing.rag import semantic_verify_hallucinations
                    ground_truth = {
                        "metrics": metrics,
                        "per_class": per_class,
                        "overfit_analysis": overfit_analysis,
                        "imbalance_metadata": imbalance_metadata
                    }
                    is_grounded = semantic_verify_hallucinations([content], ground_truth)
                    
                    if is_grounded:
                        logger.info(f"Tier 2 CLEARED hallucination flag for {mode}.{section}")
                        # Remove from error messages since it was cleared
                        error_messages = [msg for msg in error_messages if not msg.startswith(f"{mode}.{section}")]
                    else:
                        bad_sections.append(f"{mode}.{section}")

        if bad_sections:
            raise HallucinationError(f"Hallucinated metric(s) detected in: {', '.join(error_messages)}", bad_sections)

    def _validate_output_quality(self, narrative: dict, metrics: dict, task_type: str = "") -> dict:
        """
        Post-generation quality gate:
        1. Detect prompt instruction leakage
        2. Check section completeness
        3. Verify data anchoring (real metric values appear)
        """
        result = {
            "leaked": False,
            "leaked_phrases": [],
            "incomplete_sections": [],
            "data_anchored": True
        }

        full_text = json.dumps(narrative).lower()

        # 1. Prompt leakage detection
        for phrase in _LEAKAGE_PHRASES:
            if phrase.lower() in full_text:
                result["leaked"] = True
                result["leaked_phrases"].append(phrase)

        # 2. Section completeness (each section should have substantive content)
        min_length = 80  # characters
        for mode in ("expert",):
            mode_data = narrative.get(mode, {})
            required_sections = ["executive_summary", "findings", "visuals_analysis", "recommendations"]
            if mode == "expert":
                required_sections.insert(1, "preprocessing_and_data_quality")
            for section in required_sections:
                content = mode_data.get(section, "")
                if isinstance(content, (list, dict)):
                    content_str = json.dumps(content)
                else:
                    content_str = str(content)
                if len(content_str) < min_length:
                    result["incomplete_sections"].append(f"{mode}.{section}")

        # 3. Data anchoring — at least 2 real metric values must appear in expert sections
        expert = narrative.get("expert", {})
        expert_text = json.dumps(expert)
        anchored_count = 0
        anchor_keys = ("R2", "RMSE", "MAE", "MSE") if self._is_regression_task(task_type) else ("accuracy", "ROC-AUC", "precision", "recall")
        for key in anchor_keys:
            val = str(metrics.get(key, "")).replace("%", "").strip()
            if val and val != "N/A" and val in expert_text:
                anchored_count += 1
        available_anchor_count = sum(1 for key in anchor_keys if str(metrics.get(key, "")).replace("%", "").strip() not in ("", "N/A"))
        required_anchor_count = min(2, available_anchor_count)
        if anchored_count < required_anchor_count:
            result["data_anchored"] = False
            result["incomplete_sections"].append("expert.findings (not data-anchored)")

        return result

    def _merge_with_fallback(self, llm_output: dict, fallback: dict, bad_sections: list) -> dict:
        """
        Keep good LLM sections, substitute rule-based content for failed ones.
        """
        merged = json.loads(json.dumps(llm_output))  # deep copy

        for section_path in bad_sections:
            parts = section_path.split(".")
            if len(parts) == 2:
                mode, section = parts[0], parts[1]
                # Remove any parenthetical annotation like "(not data-anchored)"
                section = section.split(" ")[0]
                if mode in fallback and section in fallback[mode]:
                    if mode not in merged:
                        merged[mode] = {}
                    merged[mode][section] = fallback[mode][section]
                    logger.info(f"Merged rule-based fallback for {mode}.{section}")

        # Ensure glossary exists
        if "glossary" not in merged or not merged["glossary"]:
            merged["glossary"] = fallback.get("glossary", {})

        merged["_meta"] = {
            "source": "llm_partial_with_unavailable_sections",
            "reason": "some_llm_sections_missing_or_incomplete",
            "fallback_sections": bad_sections,
        }

        return merged

    # ══════════════════════════════════════════════════════════════════════════
    # Transparent Fallback Notice
    # ══════════════════════════════════════════════════════════════════════════

    def _generate_rule_based(
        self,
        dataset_name: str,
        task_type: str,
        metrics: dict,
        visuals_summary: dict,
        shap_features: list = None,
        anomaly_flags: list = None,
        per_class: list = None,
        overfit_analysis: dict = None,
        selected_features: list = None,
        imbalance_metadata: dict = None,
        imbalance_warning: dict = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Return a disclosure-only narrative placeholder.

        The normal report body still renders metrics, plots, model tables, and
        warnings from deterministic artifacts. This fallback intentionally does
        not add interpretation so users are not misled into thinking an LLM
        authored the narrative.
        """
        warning_text = ""
        if imbalance_warning:
            warning_text = (
                "\n\n**Accuracy interpretation warning:** "
                f"{imbalance_warning.get('message', 'Accuracy alone may be misleading because class performance is uneven.')}"
            )
        notice = (
            "LLM narrative could not be generated or did not pass validation. "
            "The ML metrics, plots, model comparison table, and deterministic warnings are still available in this report, "
            "but this section was not generated by an LLM and is not an AI-written interpretation."
            f"{warning_text}"
        )

        return {
            "_meta": {
                "source": "narrative_unavailable",
                "reason": "llm_unavailable_or_failed_validation"
            },
            "expert": {
                "executive_summary": notice,
                "preprocessing_and_data_quality": "No LLM-written preprocessing narrative is available. Review the structured preprocessing choices, metrics, and plots shown elsewhere in the report.",
                "findings": "No LLM-written findings are available. Use the metric cards, model performance table, confusion matrix, ROC curve, and generated plots for the factual results.",
                "visuals_analysis": "No LLM-written visual analysis is available. The generated plot files are still embedded in the report for direct inspection.",
                "conclusion": "No LLM-written conclusion is available because narrative generation failed or was not configured.",
                "recommendations": "Review the deterministic ML outputs directly, check the LLM service/model configuration, and rerun narrative generation when the LLM is available."
            },
            "glossary": {
                "LLM narrative": "A written interpretation generated by a language model.",
                "Deterministic outputs": "Metrics, tables, and plots produced directly by the ML pipeline rather than by a language model."
            }
        }
