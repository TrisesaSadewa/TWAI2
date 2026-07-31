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
    r"\b(error[- ]?free|no false positives|no false negatives|caught everything|missed nothing|100\s*%|100\s+percent)\b",
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
    prefix = content[max(0, match.start() - 24):match.start()].lower()
    return bool(re.search(r"\b(not|isn't|is not|wasn't|was not|cannot be|not yet|far from|less than|never|no model is|almost|although not|while not|near-?|nearly)\s*$", prefix))


def _feature_allowed(feature_label: str, allowed_features: set[str]) -> bool:
    compact_label = re.sub(r"[^a-z0-9]+", "", feature_label.lower())
    return any(compact_label in feat or feat in compact_label for feat in allowed_features)


def _normalize_narrative_dict(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        return parsed

    if "expert" in parsed and isinstance(parsed["expert"], dict):
        expert_dict = parsed["expert"]
    else:
        expert_dict = parsed.get("expert_narrative") or parsed.get("narrative") or parsed.get("report") or parsed

    normalized_expert = {}
    alias_map = {
        "executive_summary": "executive_summary",
        "exec_summary": "executive_summary",
        "summary": "executive_summary",
        "overview": "executive_summary",
        "preprocessing_and_data_quality": "preprocessing_and_data_quality",
        "data_quality": "preprocessing_and_data_quality",
        "preprocessing": "preprocessing_and_data_quality",
        "data_preprocessing": "preprocessing_and_data_quality",
        "findings": "findings",
        "key_findings": "findings",
        "results": "findings",
        "analysis": "findings",
        "model_performance": "findings",
        "recommendations": "recommendations",
        "recommendation": "recommendations",
        "clinical_recommendations": "recommendations",
        "conclusion_and_recommendations": "recommendations",
        "conclusion": "conclusion",
        "visuals_analysis": "visuals_analysis",
        "plot_analysis": "visuals_analysis",
        "visuals": "visuals_analysis"
    }

    if isinstance(expert_dict, dict):
        for k, v in expert_dict.items():
            k_clean = str(k).lower().strip().replace(" ", "_").replace("-", "_")
            target_key = alias_map.get(k_clean, k_clean)
            normalized_expert[target_key] = v

    if "executive_summary" in normalized_expert or "findings" in normalized_expert or "recommendations" in normalized_expert or "preprocessing_and_data_quality" in normalized_expert:
        return {"expert": normalized_expert}

    return parsed


def _extract_llm_chunk_text(chunk_data: dict) -> tuple:
    """Extract (content, reasoning) from a streaming chunk.
    
    Returns a 2-tuple: (content_text, reasoning_text).
    Content is the actual JSON/text output; reasoning is chain-of-thought.
    Callers must accumulate them into SEPARATE buffers to avoid contamination.
    """
    if not isinstance(chunk_data, dict):
        return ("", "")
    
    content = ""
    reasoning = ""
    
    choices = chunk_data.get("choices", [])
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            delta = choice.get("delta", {})
            if isinstance(delta, dict):
                # Extract content and reasoning into SEPARATE variables
                content = str(delta.get("content") or delta.get("text") or "")
                reasoning = str(
                    delta.get("reasoning_content") or
                    delta.get("reasoning") or
                    delta.get("thinking") or
                    ""
                )
                if content or reasoning:
                    return (content, reasoning)
            # Try text (OpenAI completions) — no reasoning field here
            text = choice.get("text")
            if text:
                return (str(text), "")
    
    # Try message (Ollama native chat)
    msg = chunk_data.get("message")
    if isinstance(msg, dict):
        content = str(msg.get("content") or msg.get("text") or "")
        reasoning = str(
            msg.get("reasoning_content") or
            msg.get("reasoning") or
            msg.get("thinking") or
            ""
        )
        if content or reasoning:
            return (content, reasoning)
    
    # Try response/content (Ollama native generate or generic)
    val = chunk_data.get("content") or chunk_data.get("response") or chunk_data.get("text")
    return (str(val), "") if val is not None else ("", "")


def _extract_llm_response_text(res_payload: dict) -> tuple:
    """Extract (content, reasoning) from a non-streaming LLM response.
    
    Returns a 2-tuple: (content_text, reasoning_text).
    Content is the actual JSON/text output; reasoning is chain-of-thought.
    Callers should prefer content; use reasoning ONLY as a last-resort fallback.
    """
    if not isinstance(res_payload, dict):
        return ("", "")
    
    choices = res_payload.get("choices", [])
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            # Try message (OpenAI chat)
            msg = choice.get("message", {})
            if isinstance(msg, dict):
                content = str(msg.get("content") or msg.get("text") or "").strip()
                reasoning = str(
                    msg.get("reasoning") or
                    msg.get("reasoning_content") or
                    msg.get("thinking") or
                    ""
                ).strip()
                return (content, reasoning)

            # Try text (OpenAI completions)
            text = choice.get("text")
            if text and str(text).strip():
                return (str(text).strip(), "")

    # Try message (Ollama native chat)
    msg = res_payload.get("message")
    if isinstance(msg, dict):
        content = str(msg.get("content") or msg.get("text") or "").strip()
        reasoning = str(
            msg.get("reasoning") or
            msg.get("reasoning_content") or
            msg.get("thinking") or
            ""
        ).strip()
        return (content, reasoning)

    # Try response/content (Ollama native generate or generic)
    val = res_payload.get("content") or res_payload.get("response") or res_payload.get("text")
    return (str(val).strip(), "") if val is not None else ("", "")


def _clean_and_parse_llm_json(raw_text: str) -> dict:
    """Extracts, cleans, and parses JSON output from LLMs (handling reasoning models,
    plain-text Thinking Process headers, markdown code fences, unescaped control chars/newlines, trailing commas, and truncated JSON)."""
    import re, json
    cleaned = raw_text or ""

    # 1. Strip reasoning blocks (<think>...</think>, including unclosed opening <think> tag)
    _OPEN = chr(60) + "think" + chr(62)
    _CLOSE = chr(60) + "/" + "think" + chr(62)
    cleaned = re.sub(re.escape(_OPEN) + r".*?" + re.escape(_CLOSE), "", cleaned, flags=re.DOTALL).strip()
    if _OPEN in cleaned:
        parts = cleaned.split(_OPEN, 1)
        before_think = parts[0].strip()
        after_think = parts[1].strip() if len(parts) > 1 else ""
        if before_think and ("{" in before_think):
            cleaned = before_think
        elif after_think and ("{" in after_think):
            cleaned = after_think
        else:
            cleaned = before_think or after_think

    # 2. Strip plain-text reasoning prefixes (e.g., "Thinking Process:\n...", "Thinking:\n...", "Reasoning:\n...")
    json_start = re.search(r'(?:```(?:json)?\s*)?\{\s*"[a-zA-Z0-9_]+"\s*:', cleaned)
    if json_start and json_start.start() > 0:
        prefix = cleaned[:json_start.start()].strip()
        if re.search(r'\b(?:thinking|reasoning)\b', prefix, re.IGNORECASE) or len(prefix) > 20:
            cleaned = cleaned[json_start.start():].strip()

    # 3. Extract JSON content inside markdown code fence ```json ... ``` or ``` ... ```
    code_fence_matches = list(re.finditer(r'```(?:json)?\s*(.*?)\s*```', cleaned, re.DOTALL))
    if code_fence_matches:
        for m in reversed(code_fence_matches):
            txt = m.group(1).strip()
            if txt.startswith("{") and "}" in txt:
                target_text = txt
                break
        else:
            target_text = code_fence_matches[-1].group(1).strip()
    else:
        target_text = cleaned.strip()

    # 4. Try json_repair on target_text first
    try:
        import json_repair
        res = json_repair.loads(target_text)
        if isinstance(res, dict) and res:
            return _normalize_narrative_dict(res)
    except Exception:
        pass

    # 5. Extract outermost curly braces for actual JSON object key
    first_brace_match = re.search(r'\{\s*"[a-zA-Z0-9_]+"\s*:', target_text)
    if first_brace_match:
        first_brace = first_brace_match.start()
    else:
        first_brace = target_text.find('{')

    last_brace = target_text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        target_json = target_text[first_brace:last_brace + 1].strip()
    elif first_brace != -1:
        target_json = target_text[first_brace:].strip()
    else:
        target_json = target_text.strip()

    # 6. Strip trailing commas before } or ]
    target_json = re.sub(r',\s*([\}\]])', r'\1', target_json)

    # 7. Try json.loads with strict=False
    try:
        res = json.loads(target_json, strict=False)
        if isinstance(res, dict):
            return _normalize_narrative_dict(res)
    except Exception:
        pass

    # 8. Try json_repair on target_json
    try:
        import json_repair
        res = json_repair.loads(target_json)
        if isinstance(res, dict) and res:
            return _normalize_narrative_dict(res)
    except Exception:
        pass

    # 9. Auto-repair unclosed quotes and missing closing braces/brackets for truncated outputs
    repaired = target_json
    quotes = len(re.findall(r'(?<!\\)"', repaired))
    if quotes % 2 != 0:
        repaired += '"'

    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    repaired = repaired + (']' * max(0, open_brackets)) + ('}' * max(0, open_braces))
    repaired = re.sub(r',\s*([\}\]])', r'\1', repaired)

    try:
        res = json.loads(repaired, strict=False)
        if isinstance(res, dict):
            return _normalize_narrative_dict(res)
    except Exception:
        pass

    # 7. Auto-repair unclosed quotes and missing closing braces/brackets for truncated outputs
    repaired = target_json
    quotes = len(re.findall(r'(?<!\\)"', repaired))
    if quotes % 2 != 0:
        repaired += '"'

    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    repaired = repaired + (']' * max(0, open_brackets)) + ('}' * max(0, open_braces))
    repaired = re.sub(r',\s*([\}\]])', r'\1', repaired)

    try:
        res = json.loads(repaired, strict=False)
        if isinstance(res, dict):
            return _normalize_narrative_dict(res)
    except Exception:
        pass

    # 10. Fallback for reasoning models (e.g., deepseek-r1, qwen3.5) that output plain-text section drafts without JSON braces
    if any(k in raw_text.lower() for k in ("thinking", "executive", "findings", "preprocessing", "recommendations")):
        extracted_expert = {}
        section_patterns = {
            "executive_summary": r"(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:executive\s*summary|verdict)\s*(?:[\#\*\-]+\s*)*(?::|\n)\s*)(.*?)(?=(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:preprocessing|data\s*quality|findings|results|conclusion|recommendations|visuals|glossary))|\Z)",
            "preprocessing_and_data_quality": r"(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:preprocessing\s*(?:and\s*data\s*quality)?|data\s*quality)\s*(?:[\#\*\-]+\s*)*(?::|\n)\s*)(.*?)(?=(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:findings|results|conclusion|recommendations|visuals|glossary))|\Z)",
            "findings": r"(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:findings|results|model\s*performance)\s*(?:[\#\*\-]+\s*)*(?::|\n)\s*)(.*?)(?=(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:conclusion|recommendations|visuals|glossary))|\Z)",
            "conclusion": r"(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:conclusion|overall\s*assessment)\s*(?:[\#\*\-]+\s*)*(?::|\n)\s*)(.*?)(?=(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:recommendations|visuals|glossary))|\Z)",
            "recommendations": r"(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:recommendations)\s*(?:[\#\*\-]+\s*)*(?::|\n)\s*)(.*?)(?=(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:visuals|glossary))|\Z)",
            "visuals_analysis": r"(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:visuals\s*analysis|plot\s*analysis|visuals)\s*(?:[\#\*\-]+\s*)*(?::|\n)\s*)(.*?)(?=(?:(?:^|\n)\s*(?:[\#\*\-]+\s*)*(?:glossary))|\Z)"
        }
        for sec_name, pattern in section_patterns.items():
            match = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)
            if match:
                sec_text = match.group(1).strip()
                sec_text = re.sub(r"^\**[A-Za-z\s]+\**\s*:\s*", "", sec_text)
                min_len = 10 if sec_name in ("preprocessing_and_data_quality", "recommendations") else 20
                if len(sec_text) >= min_len:
                    extracted_expert[sec_name] = sec_text

        if extracted_expert and len(extracted_expert) >= 1:
            logger.info(f"Fallback parser recovered {len(extracted_expert)} section(s) from plain-text LLM output: {list(extracted_expert.keys())}")
            return _normalize_narrative_dict({"expert": extracted_expert})

    # Final attempt: standard json.loads to raise original JSONDecodeError if completely unparseable
    try:
        res = json.loads(target_json)
        return _normalize_narrative_dict(res) if isinstance(res, dict) else res
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parsing failed! Raw LLM text was (first 500 chars): {cleaned[:500]!r}")
        raise e


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

    _DATASET_CONTEXTS = {
        "breast_cancer": "Binary classification of breast tumor malignancy from fine-needle aspirate (FNA) cell nucleus morphology measurements. The clinical goal is to assist pathologists in distinguishing benign from malignant masses.",
        "heart_disease": "Prediction of heart disease presence from patient demographics, vital signs, and clinical test results. Supports early cardiovascular risk stratification.",
        "diabetes": "Classification or regression of diabetes indicators from metabolic and demographic features. Relevant to early screening and preventive care.",
        "lung_cancer": "Detection of lung cancer from clinical and/or imaging-derived features. High-stakes binary classification where false negatives carry significant clinical risk.",
    }

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_API_BASE_URL
        self.model = get_deployment_writer_model()

    def _build_clinical_context(self, dataset_name: str, task_type: str, selected_features: list, additional_context: str = "") -> str:
        """Infer clinical domain context from dataset name and features, complemented by user-supplied metadata."""
        context_parts = []
        name_lower = dataset_name.lower() if dataset_name else ""
        auto_detected = None
        for pattern, context in self._DATASET_CONTEXTS.items():
            if pattern in name_lower:
                auto_detected = f"AUTO-DETECTED CLINICAL DOMAIN:\n  {context}"
                break
        
        if not auto_detected and selected_features:
            feat_sample = ", ".join(selected_features[:10])
            auto_detected = (
                f"AUTO-DETECTED CLINICAL DOMAIN:\n"
                f"  Dataset domain inferable from features: {feat_sample}."
            )

        if auto_detected:
            context_parts.append(auto_detected)

        if additional_context and str(additional_context).strip():
            context_parts.append(f"USER-PROVIDED STUDY METADATA & OBJECTIVES:\n  {str(additional_context).strip()}")

        if context_parts:
            return "CLINICAL & STUDY CONTEXT:\n" + "\n\n".join(context_parts)
        return ""

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
        all_models = kwargs.get("all_models")
        additional_context = kwargs.get("additional_context") or kwargs.get("user_context")

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
                fallback_tag = settings.SUPPORTED_MODELS.get("deepseek-r1:8b", "deepseek-r1:8b")
                if resolved_model != fallback_tag:
                    fallback_llm = fallback_tag

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
                    fallback_model=fallback_llm,
                    all_models=all_models,
                    additional_context=additional_context
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
        imbalance_warning=None, all_models=None, additional_context=None
    ) -> str:
        """Build the factual DATA section — same for all tiers."""
        clinical_ctx = self._build_clinical_context(dataset_name, task_type, selected_features, additional_context=additional_context or "")
        
        sections = []
        if clinical_ctx:
            sections.append(clinical_ctx)
            
        sections.extend([
            f"DATASET: {dataset_name}",
            f"ML TASK: {task_type}",
            f"METRICS:\n{formatted_metrics}",
        ])

        if shap_features:
            lines = []
            for f in shap_features:
                dir_str = f", direction={f['direction']}" if "direction" in f else ""
                lines.append(f"  - {f['feature']}: importance={f['importance']}{dir_str}")
            lines = "\n".join(lines)
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
                "CLASS IMBALANCE & PREPROCESSING NOTE:\n"
                f"  - {imbalance_warning.get('message', '')}\n"
                "  - State in the preprocessing and findings sections that class imbalance was mitigated via sample weighting/cost-sensitive learning, and highlight balanced metrics (F1-Score, MCC, ROC-AUC)."
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

        # Hyperparameter tuning results have been removed per user request (unsupported tables)
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
            "You are a clinical biostatistics AI. You analyze regression ML training results and write professional reports.\n"
            "CRITICAL DIRECTIVE: Do NOT output any thinking process, reasoning preamble, or plan. Do NOT output 'Thinking Process:' or '<think>'. "
            "Begin your response IMMEDIATELY with '{' and output ONLY valid JSON."
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
- "conclusion": Give a careful conclusion about regression model reliability, limitations, and whether additional validation is needed.
- "recommendations": Give practical recommendations for improving continuous-target prediction and validating the model.
- "visuals_analysis": Explain each available regression plot with [PLOT: plot_name] directly above the explanation. Focus on true-vs-predicted, residuals, feature importance, SHAP, PCA, PLS, UMAP, and correlation plots when present.

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
            "You are a clinical biostatistics AI. You analyze ML training results and write professional reports.\n"
            "CRITICAL DIRECTIVE: Do NOT output any thinking process, reasoning preamble, or plan. Do NOT output 'Thinking Process:' or '<think>'. "
            "Begin your response IMMEDIATELY with '{' and output ONLY valid JSON."
        )
        user = f"""Analyze this ML training run data and write a clinical report as JSON.

=== DATA ===
{data_block}
=== END DATA ===

Write a JSON object with this EXACT structure. All values MUST be a single Markdown-formatted string. Do NOT use nested objects or arrays (for tables, use standard Markdown table syntax, not JSON arrays). Fill every field with substantive clinical analysis (NOT placeholders). Use the actual metric values from the DATA above.

{{
  "expert": {{
    "executive_summary": "**CLINICAL BACKGROUND:** 1-2 sentences synthesizing the user-provided study metadata/disease goals if present.\\n\\n**VERDICT:** One sentence declaring clinical readiness ('ready for preliminary screening', 'conditionally suitable', or 'not recommended'). \\n\\n**PERFORMANCE SNAPSHOT:** 2-3 sentences translating key metrics into plain clinical language with a concrete patient-count example.\\n\\n**CRITICAL FLAGS:** 1-2 sentences noting the most important warnings (e.g., imbalance, leakage) or stating 'No critical anomalies detected.'\\n\\nRule: Readable in 30 seconds. Total length: 5-10 sentences.",
    "preprocessing_and_data_quality": "Write 4-5 sentences explaining the data quality. You MUST explicitly discuss this metadata: {data_quality_text}",
    "findings": "MUST BE A SINGLE MARKDOWN STRING. Structure as exactly 5 stages using ### headers:\\n\\n### Stage 1 — Overall Performance\\nRender a Markdown table with exactly two columns (| Metric | Value |) containing all metrics. Below it, 2-3 sentences of clinical context.\\n\\n### Stage 2 — Discrimination Analysis\\n[PLOT: roc_curve]\\n[PLOT: pr_curve]\\n3-5 sentences analyzing ROC and PR curves. Critique high scores (>0.95) for leakage. Discuss precision-recall trade-offs especially for imbalanced classes.\\n> 💡 **Key Insight:** [One sentence takeaway]\\n\\n### Stage 3 — Error Analysis\\n[PLOT: confusion_matrix]\\n3-5 sentences on confusion matrix errors (FP vs FN impact). Include per-class breakdown if available. **CROSS-REFERENCE:** Connect to discrimination analysis.\\n> 💡 **Key Insight:** [One sentence takeaway]\\n\\n### Stage 4 — Feature Intelligence\\n[PLOT: corr_heatmap]\\n3-5 sentences on correlation. Identify any unexpected negative correlations or clustered variables.\\n> 💡 **Key Insight:** [One sentence takeaway]\\n[PLOT: feature_importance]\\n3-5 sentences on key drivers. **CRITICAL**: Do NOT give generic observations. Specify exact variable names, their ranking, and hypothesize biological/clinical reasons for their importance. Highlight any surprises.\\n> 💡 **Key Insight:** [One sentence takeaway]\\n[PLOT: shap_summary]\\n3-5 sentences on SHAP directions. **CRITICAL**: Detail the exact impact (positive/negative) of top features based on their directionality. Analyze complex non-linear effects if visible. Do SHAP results align with feature importance?\\n> 💡 **Key Insight:** [One sentence summarizing the most profound takeaway]\\n\\n### Stage 5 — Generalization & Stability\\nDiscuss dimensionality reduction ([PLOT: pca_plot], [PLOT: pls_plot], [PLOT: umap_plot]) and overfitting if present. **CROSS-REFERENCE:** Synthesize overall evidence.\\n> 💡 **Key Insight:** [One sentence takeaway]\\n\\n**NARRATIVE FLOW ENFORCEMENT:** Each stage MUST begin with a transition sentence connecting to the previous stage.",
    "visuals_analysis": "Explain each available plot as a SINGLE MARKDOWN STRING. Put the relevant [PLOT: plot_name] placeholder directly above each explanation. Explain how to read the plot axes, colors, bars, or clusters. Do not invent metrics or repeat unsupported performance claims.",
    "conclusion": "MUST BE A SINGLE MARKDOWN STRING.\\n\\n**OVERALL ASSESSMENT:** 2-3 sentences providing definitive clinical judgment.\\n\\n**KEY STRENGTHS:** 2-3 bullet points with specific metric references.\\n\\n**KEY LIMITATIONS:** 2-3 bullet points with specific metric references.\\n\\n**BEFORE DEPLOYMENT:** 1-3 concrete actionable steps required.",
    "recommendations": "MUST BE A SINGLE MARKDOWN STRING.\\n\\n**DATA QUALITY IMPROVEMENTS:** 2-3 suggestions.\\n\\n**MODEL ARCHITECTURE CONSIDERATIONS:** 2-3 suggestions.\\n\\n**VALIDATION PROTOCOL:** 2-3 concrete steps.\\n\\n**CLINICAL INTEGRATION PATHWAY:** 2-3 practical considerations. Every suggestion must be specific to THIS dataset and model."
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
8. DO NOT hallucinate arbitrary percentages (e.g., PCA variance) or exact true/false positive counts unless explicitly provided in the DATA.
9. For class balance and imbalance correction, use only DATA QUALITY & PREPROCESSING facts. If imbalance metadata says "not recorded", write that it was not recorded; do NOT infer the data are balanced or that no correction was used.
10. Do NOT repeat these instructions in your output.
11. If USER-PROVIDED STUDY METADATA & OBJECTIVES are present in the DATA section, actively reference and synthesize the user's research goals, disease background, and feature meanings throughout the executive_summary, findings, and recommendations."""

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
   **Must contain exactly four clearly labeled paragraphs:**
   **CLINICAL BACKGROUND:** 1-2 sentences synthesizing the specific disease, cohort, and research goals IF USER-PROVIDED STUDY METADATA is present in the DATA (otherwise state "No specific clinical context provided.").
   **VERDICT:** One sentence declaring clinical readiness: 'This model is [ready for preliminary screening / conditionally suitable — requires further validation / not recommended for clinical use] because [rationale].'
   **PERFORMANCE SNAPSHOT:** 2-3 sentences translating key metrics into plain clinical language with a concrete patient-count example (e.g., 'For every 100 patients...').
   **CRITICAL FLAGS:** 1-2 sentences noting the most important warnings (e.g., imbalance, leakage) or stating 'No critical anomalies detected.'
   *Rule: Readable in 30 seconds. NO dramatic language. Total length: 6-10 sentences.*

- "preprocessing_and_data_quality":
   1) Provide a detailed, definitive, and actionable explanation of the input data quality and preprocessing. MUST BE EXACTLY 4-5 SENTENCES.
   *Rule: You MUST explicitly mention the class distribution and exact imbalance strategy/tools only when they are recorded in DATA QUALITY & PREPROCESSING or PER-CLASS PERFORMANCE. If imbalance metadata is "not recorded", say it was not recorded and do NOT infer balanced classes, no imbalance correction, SMOTE, oversampling, class weighting, or threshold tuning. Do NOT use speculative language like "likely", "appears", or "I feel like". State facts based ONLY on the provided DATA section.*

- "findings":
   **MUST BE A SINGLE MARKDOWN STRING. Structure as exactly 5 stages using ### headers:**
   
   ### Stage 1 — Overall Performance
   Render a Markdown table with exactly two columns (| Metric | Value |) containing all metrics. Below it, write 2-3 sentences of clinical context interpretation.
   
   ### Stage 2 — Discrimination Analysis
   [PLOT: roc_curve]
   [PLOT: pr_curve]
   Critique discrimination power. If ROC-AUC > 0.95, heavily warn about potential data leakage or overfitting. Highlight the specific trade-offs between sensitivity and precision, especially observing the PR curve if the dataset is imbalanced.
   > 💡 **Key Insight:** [One sentence summarizing the most important takeaway from this stage.]
   
   ### Stage 3 — Error Analysis
   [PLOT: confusion_matrix]
   Analyze exactly where the model fails. Which class has the highest false positives? False negatives? What is the clinical implication of missing the minority class vs. over-diagnosing the majority class?
   > 💡 **Key Insight:** [One sentence summarizing the most important takeaway from this stage.]
   
   ### Stage 4 — Feature Intelligence
   [PLOT: corr_heatmap]
   Discuss multivariate correlations and potential multicollinearity (2-3 sentences). Identify any unexpected negative correlations or clustered variables.
   [PLOT: feature_importance]
   **CRITICAL:** Do NOT write generic summaries like "feature X is the most important". You MUST name the exact top 3-5 features and provide deep analytical insight. Discuss their clinical plausibility, why they mathematically dominate the model's decisions, and if any unexpected features appear at the top.
   [PLOT: shap_summary]
   **CRITICAL:** Provide deep analytical value. Specify exactly which features push predictions positive vs. negative based on their directionality. Analyze complex non-linear effects if visible (e.g., high values decrease risk, but low values don't increase it). State whether this aligns with clinical reality or if the model relies on confounding variables.
   > 💡 **Key Insight:** [One sentence summarizing the most profound, non-obvious takeaway from this stage.]
   
   ### Stage 5 — Generalization & Stability
   Discuss dimensionality reduction ([PLOT: pca_plot], [PLOT: pls_plot], [PLOT: umap_plot]) with 2-3 sentences each if present. Comment on cluster separability.{f'''
   Write an Overfitting Analysis paragraph based on train/test gaps.''' if has_overfit_data else ''}
   **CROSS-REFERENCE:** Synthesize — does overall evidence paint a consistent picture?
   > 💡 **Key Insight:** [One sentence summarizing the most important takeaway from this stage.]
   
   **NARRATIVE FLOW ENFORCEMENT:** Each stage MUST begin with a transition sentence connecting to the previous stage. Use phrases like "Building on the discrimination analysis above...", "The error patterns below confirm/challenge...", "Consistent with the feature analysis...".

- "conclusion":
   **MUST BE A SINGLE MARKDOWN STRING. Structure into exactly four labeled parts using bold text:**
   **OVERALL ASSESSMENT:** 2-3 sentences providing definitive clinical judgment.
   **KEY STRENGTHS:** 2-3 bullet points with specific metric references.
   **KEY LIMITATIONS:** 2-3 bullet points with specific metric references.
   **BEFORE DEPLOYMENT:** 1-3 concrete actionable steps required.
   *Rule: Total length 8-12 sentences. No dramatic language.*

- "recommendations":
   **MUST BE A SINGLE MARKDOWN STRING. Structure into exactly four labeled categories using bold text:**
   **DATA QUALITY IMPROVEMENTS:** 2-3 specific suggestions based on actual features and data quality.
   **MODEL ARCHITECTURE CONSIDERATIONS:** 2-3 suggestions based on the models tested.
   **VALIDATION PROTOCOL:** 2-3 concrete validation steps appropriate for the dataset.
   **CLINICAL INTEGRATION PATHWAY:** 2-3 practical considerations for deployment.
   *Rule: Every suggestion must be actionable and specific to THIS dataset and model.*

- "visuals_analysis":
   Explain any other available plots with the relevant [PLOT: plot_name] placeholder directly above the explanation. Focus on how to read the axes, colors, bars, clusters, or curves. Do not invent values. Do not repeat unsupported performance scores.

Your output MUST be a valid JSON object matching this exact schema (replace "..." with your actual generated markdown string, do NOT output literal "..." placeholders):
{{
  "expert": {{
    "executive_summary": "...",
    "preprocessing_and_data_quality": "...",
    "findings": "...",
    "conclusion": "...",
    "recommendations": "...",
    "visuals_analysis": "..."
  }},
  "glossary": {{
    "Term 1": "Definition",
    "Term 2": "Definition"
  }}
}}

RULES:
1. Use ONLY the metric values from the DATA section. Do not invent numbers.
2. The expert findings MUST reference specific values (e.g., "Precision of 0.93").
3. Do NOT claim 100%, perfect, flawless, or error-free performance unless that exact value is present in the DATA.
4. DO NOT hallucinate arbitrary percentages (e.g. PCA variance) or exact true/false positive counts unless explicitly provided in the DATA.
5. If an ACCURACY / CLASS IMBALANCE WARNING is present, explicitly state that accuracy alone may be misleading.

6. If USER-PROVIDED STUDY METADATA & OBJECTIVES are present in the DATA section, actively synthesize the user's research goals, disease background, and feature definitions into the executive_summary, findings, and recommendations.

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
        fallback_model=None, imbalance_warning=None, all_models=None,
        additional_context=None
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
            imbalance_warning, all_models=all_models,
            additional_context=additional_context
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
        
        max_tokens = model_cfg.get("max_tokens", 8192)
        
        payload = {
            "model": resolved_model,
            "messages": messages,
            "temperature": model_cfg.get("temperature", 0.2),
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "think": False,
            "reasoning_effort": "none"
        }

        # Set Ollama-specific options if local endpoint is used
        options = {}
        if use_cpu_fallback:
            options["num_gpu"] = 0
            
        options["num_predict"] = max_tokens
        
        # Override context window size based on model config
        context_tokens = model_cfg.get("context_tokens") or 8192
        if context_tokens:
            options["num_ctx"] = context_tokens
            
        # Disable chain-of-thought thinking mode for structured JSON tasks (Ollama / Qwen3.5 / DeepSeek-R1)
        options["think"] = False
            
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

                content_buf = ""
                reasoning_buf = ""
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if not line_str or line_str == "data: [DONE]":
                            continue
                        
                        json_str = line_str[6:].strip() if line_str.startswith("data: ") else line_str
                        try:
                            chunk_data = json.loads(json_str)
                            chunk_content, chunk_reasoning = _extract_llm_chunk_text(chunk_data)
                            if chunk_content:
                                content_buf += chunk_content
                            if chunk_reasoning:
                                reasoning_buf += chunk_reasoning
                        except Exception as parse_err:
                            logger.error(f"Error parsing streaming chunk: {parse_err}. Chunk string was: {json_str[:200]!r}")
                
                # Prefer content; fall back to reasoning only when content is empty
                if content_buf.strip():
                    result_json = content_buf
                elif reasoning_buf.strip():
                    logger.warning(f"Streaming content buffer empty, falling back to reasoning buffer ({len(reasoning_buf)} chars). Model: {resolved_model}")
                    # Attempt to extract embedded JSON from reasoning buffer
                    json_match = re.search(r'(\{.*?\})', reasoning_buf, re.DOTALL)
                    if json_match and len(json_match.group(1)) > 50:
                        result_json = json_match.group(1)
                    else:
                        result_json = reasoning_buf
                else:
                    result_json = ""
                # Stream DONE notification will happen after quality gates pass
            else:
                resp = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
                resp.raise_for_status()
                resp_content, resp_reasoning = _extract_llm_response_text(resp.json())
                # Prefer content; fall back to reasoning only when content is empty
                if resp_content.strip():
                    result_json = resp_content
                elif resp_reasoning.strip():
                    logger.warning(f"Non-streaming content empty, falling back to reasoning ({len(resp_reasoning)} chars). Model: {resolved_model}")
                    json_match = re.search(r'(\{.*?\})', resp_reasoning, re.DOTALL)
                    if json_match and len(json_match.group(1)) > 50:
                        result_json = json_match.group(1)
                    else:
                        result_json = resp_reasoning
                else:
                    result_json = ""

            # Log empty response before throwing JSONDecodeError
            if not result_json.strip():
                logger.error(f"result_json is completely empty after LLM API call. Model: {resolved_model}")


            try:
                parsed_json = _clean_and_parse_llm_json(result_json)

                def _flatten_to_markdown(val):
                    if isinstance(val, str): return val
                    if isinstance(val, list):
                        # If list of lists (e.g., table rows), render as a proper markdown table
                        if all(isinstance(i, list) for i in val) and len(val) >= 2:
                            header = "| " + " | ".join(str(x) for x in val[0]) + " |"
                            sep = "| " + " | ".join("---" for _ in val[0]) + " |"
                            rows = "\n".join("| " + " | ".join(str(x) for x in row) + " |" for row in val[1:])
                            return f"{header}\n{sep}\n{rows}"
                        return "\n".join(f"- {i}" for i in val)
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
                        for section in ("findings", "visuals_analysis", "preprocessing_and_data_quality", "conclusion", "recommendations"):
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
                    
                quality_result = self._validate_output_quality(parsed_json, raw_metrics, task_type, visuals_summary)

                if quality_result["leaked"]:
                    raise PromptLeakageError(f"Prompt instructions leaked into output: {quality_result['leaked_phrases']}")

                break
                
            except (HallucinationError, PromptLeakageError, json.JSONDecodeError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    # Strip reasoning / thinking preambles before appending assistant turn
                    # so the model is not forced into repeating the plain-text preamble loop.
                    cleaned_assistant_turn = result_json
                    json_match = re.search(r'(?:```(?:json)?\s*)?\{\s*"[a-zA-Z0-9_]+"\s*:', cleaned_assistant_turn)
                    if json_match and json_match.start() > 0:
                        cleaned_assistant_turn = cleaned_assistant_turn[json_match.start():].strip()

                    messages.append({"role": "assistant", "content": cleaned_assistant_turn})
                    messages.append({
                        "role": "user",
                        "content": f"Your previous response failed validation ({str(e)}). Do NOT output plain-text reasoning or preamble. Output ONLY a valid JSON object matching the requested schema."
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
                            fallback_model=None,  # Prevent infinite cascade
                            all_models=all_models
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

        # ── Targeted retry for missing sections ─────────────────────────────
        if quality_result["incomplete_sections"]:
            retryable = [s for s in quality_result["incomplete_sections"]
                         if s.startswith("expert.") and "(" not in s]
            if retryable:
                logger.info(f"LLM output has incomplete sections: {quality_result['incomplete_sections']}. Attempting targeted retry for: {retryable}")
                
                # Priority order: conclusion & recommendations first (no fallback),
                # visuals_analysis last (has deterministic fallback descriptions).
                priority_order = ["expert.conclusion", "expert.recommendations",
                                  "expert.findings", "expert.executive_summary",
                                  "expert.preprocessing_and_data_quality",
                                  "expert.visuals_analysis"]
                retryable.sort(key=lambda s: next(
                    (i for i, p in enumerate(priority_order) if s.startswith(p)), 99
                ))
                
                # Increase max retries to ensure we rely on the LLM to generate all sections
                # (including visuals_analysis) and only use fallback if there's a persistent error/offline.
                TARGETED_MAX_RETRIES = 5
                # Batch sections into groups of 2 to prevent token exhaustion
                BATCH_SIZE = 2
                
                for target_attempt in range(TARGETED_MAX_RETRIES):
                    if not retryable:
                        break
                    
                    # Take the next batch of sections to retry
                    batch = retryable[:BATCH_SIZE]
                    
                    try:
                        # Slightly increase temperature on subsequent retries to avoid repeating the exact same reasoning trap
                        retry_model_cfg = dict(model_cfg or {})
                        if target_attempt > 0:
                            retry_model_cfg["temperature"] = min(0.7, retry_model_cfg.get("temperature", 0.2) + (target_attempt * 0.15))
                            logger.info(f"Targeted retry attempt {target_attempt + 1}: increasing temperature to {retry_model_cfg['temperature']:.2f}")

                        logger.info(f"Targeted retry attempt {target_attempt + 1}: requesting batch {batch}")
                        patched = self._retry_missing_sections(
                            batch, parsed_json, data_block,
                            resolved_model, headers, url, request_timeout,
                            retry_model_cfg, report_id
                        )
                        if patched:
                            parsed_json = patched
                            quality_result = self._validate_output_quality(parsed_json, raw_metrics, task_type, visuals_summary, is_retry=True)
                            if not quality_result["incomplete_sections"]:
                                logger.info(f"Targeted retry successfully filled all missing sections on attempt {target_attempt + 1}.")
                                break
                            else:
                                # Update retryable list with whatever is still missing
                                retryable = [s for s in quality_result["incomplete_sections"]
                                             if s.startswith("expert.") and "(" not in s]
                                retryable.sort(key=lambda s: next(
                                    (i for i, p in enumerate(priority_order) if s.startswith(p)), 99
                                ))
                                if not retryable:
                                    break
                    except Exception as e:
                        logger.warning(f"Targeted retry for missing sections failed on attempt {target_attempt + 1}: {e}")

        # If partially valid, merge good LLM sections with rule-based fallback
        if quality_result["incomplete_sections"]:
            logger.warning(f"LLM output has incomplete sections: {quality_result['incomplete_sections']}. Merging with rule-based fallback.")
            try:
                debug_path = os.path.join(settings.STORAGE_DIR, f"incomplete_llm_{report_id or 'unknown'}.json")
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "raw_response": result_json,
                        "parsed_json": parsed_json,
                        "quality_result": quality_result
                    }, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to dump incomplete LLM output: {e}")
            fallback = self._generate_rule_based(
                dataset_name, task_type, raw_metrics, visuals_summary or {},
                shap_features, anomaly_flags,
                per_class=per_class, overfit_analysis=overfit_analysis,
                selected_features=selected_features,
                imbalance_warning=imbalance_warning
            )
            parsed_json = self._merge_with_fallback(parsed_json, fallback, quality_result["incomplete_sections"])

        # ── Deterministic Plot Injection ─────────────────────────────────
        # Ensure all available plots are referenced even if the LLM
        # forgot to emit the corresponding [PLOT: ...] tags.
        if visuals_summary:
            parsed_json = self._inject_missing_plot_tags(
                parsed_json, visuals_summary, task_type,
                metrics=raw_metrics, shap_features=shap_features
            )

        if report_id:
            publish_stream(report_id, json.dumps(parsed_json))
            publish_stream(report_id, "[DONE]")

        return parsed_json

    # ── Deterministic Plot Injection ────────────────────────────────────────

    # Canonical plot order and the keywords the LLM typically uses when
    # discussing them.  Used by _inject_missing_plot_tags to decide *where*
    # to place a tag the LLM omitted.
    _CLASSIFICATION_PLOT_ORDER = [
        ("roc_curve",           ["roc curve", "roc-auc", "auc", "receiver operating"]),
        ("pr_curve",            ["precision-recall", "pr curve", "precision recall", "auc-pr"]),
        ("confusion_matrix",    ["confusion matrix", "true positive", "false positive", "tn", "tp", "fn", "fp"]),
        ("corr_heatmap",        ["correlation heatmap", "correlation matrix", "heatmap", "correlated"]),
        ("feature_importance",  ["feature importance", "important feature", "predictive feature"]),
        ("shap",                ["shap", "shapley"]),
        ("pca",                 ["pca", "principal component"]),
        ("pls",                 ["pls", "partial least"]),
        ("umap",                ["umap", "uniform manifold"]),
    ]

    _REGRESSION_PLOT_ORDER = [
        ("true_vs_predicted",   ["true vs predicted", "true-vs-predicted", "actual vs predicted", "predicted vs actual"]),
        ("residuals",           ["residual", "residuals"]),
        ("corr_heatmap",        ["correlation heatmap", "correlation matrix", "heatmap"]),
        ("feature_importance",  ["feature importance", "important feature"]),
        ("shap",                ["shap", "shapley"]),
        ("pca",                 ["pca", "principal component"]),
        ("pls",                 ["pls", "partial least"]),
        ("umap",                ["umap", "uniform manifold"]),
    ]

    @staticmethod
    def _normalize_visual_key(key: str) -> str:
        """Collapse a visuals_summary key to its bare plot concept."""
        k = key.lower()
        for suffix in ("_png", "_html", ".png", ".html"):
            k = k.replace(suffix, "")
        return re.sub(r"[^a-z0-9]", "", k)

    def _inject_missing_plot_tags(
        self, parsed_json: dict, visuals_summary: dict, task_type: str,
        metrics: dict = None, shap_features: list = None
    ) -> dict:
        """
        Ensure all available plots are referenced in the narrative.
        It prioritizes visuals_analysis, searching for keywords to inject
        the tag directly before the sentence that mentions the plot.
        When no keyword match is found (orphan chart), a deterministic
        fallback description is appended alongside the tag.
        """
        expert_dict = parsed_json.get("expert", {})
        findings = expert_dict.get("findings", "")
        visuals = expert_dict.get("visuals_analysis", "")

        # Pre-compute fallback descriptions for orphan charts
        fallback_descs = {}
        if metrics and visuals_summary:
            fallback_descs = self._build_visual_descriptions(
                metrics, shap_features or [], visuals_summary, task_type
            )

        # To avoid duplicating plots, check if the tag is anywhere in the expert narrative
        full_text = "\n".join(str(v) for v in expert_dict.values())

        plot_order = (
            self._REGRESSION_PLOT_ORDER
            if self._is_regression_task(task_type)
            else self._CLASSIFICATION_PLOT_ORDER
        )

        available_normalised = {
            self._normalize_visual_key(k) for k in visuals_summary
        }

        def _find_insert_pos(text: str, kw_idx: int) -> int:
            """Finds the start of the sentence or paragraph containing kw_idx."""
            for i in range(kw_idx - 1, -1, -1):
                if text[i] == '\n':
                    return i + 1
                if i > 0 and text[i] == ' ' and text[i-1] in ('.', '!', '?', ';'):
                    return i + 1
            return 0

        injected = 0

        for plot_key, keywords in plot_order:
            normalised_plot = re.sub(r"[^a-z0-9]", "", plot_key.lower())

            has_plot = any(
                normalised_plot in av or av in normalised_plot
                for av in available_normalised
            )
            if not has_plot:
                continue

            existing_tags = re.findall(r"\[PLOT:\s*(.*?)\s*\]", full_text, re.IGNORECASE)
            def normalize_tag(val: str) -> str:
                val = val.lower().strip()
                val = re.sub(r"[^a-z0-9]", "", val)
                for suffix in ("plot", "curve", "png", "html", "2d"):
                    if val.endswith(suffix) and val != suffix:
                        val = val[:-len(suffix)]
                return val
                
            normalized_existing = {normalize_tag(tag) for tag in existing_tags}
            target_normalized = normalize_tag(plot_key)
            if target_normalized in normalized_existing:
                continue

            tag = f"[PLOT: {plot_key}]"
            inserted = False

            def try_inject(text: str) -> tuple[bool, str]:
                if not isinstance(text, str) or not text.strip():
                    return False, text
                text_lower = text.lower()
                for kw in keywords:
                    idx = text_lower.find(kw)
                    if idx != -1:
                        insert_pos = _find_insert_pos(text, idx)
                        preceding = text[max(0, insert_pos - 60):insert_pos]
                        if "[PLOT:" in preceding.upper():
                            continue
                        new_text = (
                            text[:insert_pos]
                            + f"\n\n{tag}\n\n"
                            + text[insert_pos:]
                        ).strip()
                        return True, new_text
                return False, text

            inserted, visuals = try_inject(visuals)
            
            if not inserted:
                inserted, findings = try_inject(findings)
                
            if not inserted:
                # Build a fallback description for the orphan chart
                pretty_name = plot_key.replace("_", " ").title()
                fallback_text = ""
                for desc_key, desc_val in fallback_descs.items():
                    if plot_key.replace("_", "") in desc_key.replace(" ", "").lower():
                        fallback_text = f"\n{desc_val}"
                        break
                
                block = f"{tag}{fallback_text}"
                if isinstance(visuals, str) and visuals.strip():
                    visuals = visuals.rstrip() + f"\n\n{block}\n"
                elif isinstance(findings, str) and findings.strip():
                    findings = findings.rstrip() + f"\n\n{block}\n"
                else:
                    findings = block
                injected += 1
            else:
                injected += 1

        if injected:
            logger.info(f"Deterministically injected {injected} missing [PLOT:] tag(s)")
            if findings: parsed_json.setdefault("expert", {})["findings"] = findings
            if visuals: parsed_json.setdefault("expert", {})["visuals_analysis"] = visuals

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
        data_block: str = None,
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

        # Also allow per-class metric values and support percentages
        if per_class:
            total_support = sum(float(c.get("support", 0)) for c in per_class if c.get("support") is not None)
            for c in per_class:
                for key in ('precision', 'recall', 'f1'):
                    try:
                        val = float(c[key])
                        allowed.add(round(val * 100, 2))
                        allowed.add(round(val * 100, 1))
                        allowed.add(round(val, 4))
                    except (ValueError, TypeError, KeyError):
                        pass
                
                # Allow the percentage representation of this class's support
                try:
                    support = float(c.get("support", 0))
                    if total_support > 0:
                        pct = (support / total_support) * 100
                        allowed.add(round(pct, 2))
                        allowed.add(round(pct, 1))
                        allowed.add(round(pct, 0))
                except (ValueError, TypeError):
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

        def _extract_pct(val_str):
            for num_str in re.findall(r'\b\d+(?:\.\d+)?\b', str(val_str)):
                try:
                    val = float(num_str)
                    allowed.add(round(val * 100, 2))
                    allowed.add(round(val * 100, 1))
                    allowed.add(round(val, 4))
                    allowed.add(round(val, 2))
                    allowed.add(round(val, 1))
                except ValueError:
                    pass

        if imbalance_metadata:
            for k, v in imbalance_metadata.items():
                if isinstance(v, dict):
                    for sub_v in v.values():
                        _extract_pct(sub_v)
                else:
                    _extract_pct(v)
                    
        # Also extract all numbers from metrics dict string values
        if metrics:
            for v in metrics.values():
                if isinstance(v, dict):
                    for sub_v in v.values():
                        _extract_pct(sub_v)
                else:
                    _extract_pct(v)
                    
        # Extract ALL numbers directly from the raw data block to guarantee no false positives
        if data_block:
            for num_str in re.findall(r'\b\d+(?:\.\d+)?\b', str(data_block)):
                try:
                    val = float(num_str)
                    allowed.add(round(val * 100, 2))
                    allowed.add(round(val * 100, 1))
                    allowed.add(round(val, 4))
                    allowed.add(round(val, 2))
                    allowed.add(round(val, 1))
                except ValueError:
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
                            # Allow common structural percentages (e.g. train/test splits, thresholds, baselines)
                            if pct_val in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95):
                                continue
                            # Allow percentages derived from any raw metric value in the metrics dictionary
                            for m_v in metrics.values():
                                try:
                                    m_f = float(str(m_v).replace('%', '').strip())
                                    if abs(m_f * 100 - pct_val) <= 1.0 or abs(m_f - pct_val) <= 1.0:
                                        found = True
                                        break
                                except (ValueError, TypeError):
                                    pass
                            if found:
                                continue
                            
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

    def _retry_missing_sections(
        self, missing_sections: list, current_json: dict, data_block: str,
        model: str, headers: dict, url: str, timeout: int,
        model_cfg: dict, report_id: str = None
    ) -> Optional[dict]:
        """
        Make a short, targeted LLM call to generate only the missing expert sections.
        Returns the patched narrative dict, or None if the retry fails.
        """
        section_names = []
        for s in missing_sections:
            if "." in s:
                clean_sec = s.split(".", 1)[1].split("(")[0].strip()
                if clean_sec and clean_sec not in section_names:
                    section_names.append(clean_sec)
            elif s and s not in section_names:
                section_names.append(s.split("(")[0].strip())

        section_list = ", ".join(f'"{s}"' for s in section_names)

        existing_keys = list(current_json.get("expert", {}).keys())

        # Truncate data_block — the full data was already used for the other sections;
        # a summary suffices for the missing ones.
        truncated_data = data_block[:3000] if len(data_block) > 3000 else data_block

        section_instructions = []
        for name in section_names:
            if name == "recommendations":
                section_instructions.append(
                    '"recommendations": A single Markdown string structured as:\n'
                    '  **DATA QUALITY IMPROVEMENTS:** 2-3 specific suggestions.\n'
                    '  **MODEL ARCHITECTURE CONSIDERATIONS:** 2-3 suggestions.\n'
                    '  **VALIDATION PROTOCOL:** 2-3 concrete steps.\n'
                    '  **CLINICAL INTEGRATION PATHWAY:** 2-3 practical considerations.\n'
                    '  Every suggestion must be specific to THIS dataset and model.'
                )
            elif name == "conclusion":
                section_instructions.append(
                    '"conclusion": A single Markdown string structured as:\n'
                    '  **OVERALL ASSESSMENT:** 2-3 sentences.\n'
                    '  **KEY STRENGTHS:** 2-3 bullet points with metric references.\n'
                    '  **KEY LIMITATIONS:** 2-3 bullet points with metric references.\n'
                    '  **BEFORE DEPLOYMENT:** 1-3 actionable steps.'
                )
            elif name == "visuals_analysis":
                section_instructions.append(
                    '"visuals_analysis": A single Markdown string explaining available plots '
                    'with [PLOT: plot_name] placeholders. Focus on axes, colors, and interpretation.'
                )
            elif name == "findings":
                section_instructions.append(
                    '"findings": A detailed Markdown string analyzing model performance. '
                    'Include a Markdown table with metrics. Discuss discrimination, errors, and feature importance.'
                )
            else:
                section_instructions.append(
                    f'"{name}": A substantive Markdown string (at least 100 words).'
                )

        instructions_text = "\n".join(f"- {inst}" for inst in section_instructions)

        messages = [
            {"role": "system", "content": "You are a clinical biostatistics AI. CRITICAL: Do NOT write any thinking process or reasoning preamble. Output ONLY valid JSON directly starting immediately with '{'."},
            {"role": "user", "content": (
                f"A clinical ML report was generated but is missing these sections: {section_list}.\n"
                f"The report already contains: {', '.join(existing_keys)}.\n\n"
                f"=== DATA ===\n{truncated_data}\n=== END DATA ===\n\n"
                f"Generate a JSON object containing ONLY these keys:\n{instructions_text}\n\n"
                f"Use ONLY metric values from the DATA. Output ONLY the JSON object."
            )}
        ]

        retry_max_tokens = max(model_cfg.get("max_tokens", 8192), 4096)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": model_cfg.get("temperature", 0.2),
            "max_tokens": retry_max_tokens,
            "response_format": {"type": "json_object"},
            "think": False,
            "reasoning_effort": "none"
        }

        options = {"num_predict": retry_max_tokens}
        # Always set context window — without this, models like qwen3.5
        # default to 4096 total tokens which is far too small for retries.
        context_tokens = model_cfg.get("context_tokens") or 8192
        options["num_ctx"] = max(context_tokens, 8192)
        # Disable chain-of-thought for structured JSON retry tasks
        options["think"] = False
        payload["options"] = options

        logger.info(f"Targeted retry: requesting {section_list} from {model}")

        if report_id:
            publish_stream(report_id, "\n\n[System] Generating missing section(s)...\n\n")

        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        
        raw_resp = resp.json()
        resp_content, resp_reasoning = _extract_llm_response_text(raw_resp)
        
        # Prefer content; fall back to reasoning only when content is empty
        if resp_content.strip():
            result = resp_content
        elif resp_reasoning.strip():
            logger.warning(f"Targeted retry content empty, falling back to reasoning ({len(resp_reasoning)} chars)")
            json_match = re.search(r'(\{[\s\S]*\})', resp_reasoning)
            if json_match and len(json_match.group(1)) > 30:
                result = json_match.group(0)
            else:
                result = ""
        else:
            result = ""
        
        if not result.strip():
            logger.error(f"Targeted retry returned empty response. Raw API payload: {raw_resp}")

        patch = _clean_and_parse_llm_json(result)

        # Merge patched sections into current_json
        patched = json.loads(json.dumps(current_json))  # deep copy
        patch_expert = patch.get("expert", patch) if isinstance(patch, dict) else {}
        if isinstance(patch, dict) and "expert" in patch and isinstance(patch["expert"], dict):
            patch_expert = patch["expert"]

        alias_map = {
            "executive_summary": ["executive_summary", "exec_summary", "summary", "overview"],
            "preprocessing_and_data_quality": ["preprocessing_and_data_quality", "preprocessing", "data_quality", "data_preprocessing"],
            "findings": ["findings", "results", "analysis", "model_performance"],
            "recommendations": ["recommendations", "recommendation", "clinical_recommendations", "conclusion_and_recommendations"],
            "conclusion": ["conclusion", "overall_assessment"],
            "visuals_analysis": ["visuals_analysis", "plot_analysis", "visuals"]
        }

        for name in section_names:
            possible_keys = alias_map.get(name, [name])
            content = None
            for key in possible_keys:
                if key in patch_expert:
                    content = patch_expert[key]
                    break

            if content is not None:
                if isinstance(content, (dict, list)):
                    if isinstance(content, list):
                        content = "\n".join(f"- {i}" for i in content)
                    elif isinstance(content, dict):
                        parts = []
                        for k, v in content.items():
                            parts.append(f"**{str(k).replace('_', ' ').title()}**")
                            parts.append(str(v))
                        content = "\n\n".join(parts)
                min_len = 10 if name in ("preprocessing_and_data_quality", "recommendations") else 30
                if isinstance(content, str) and len(content) >= min_len:
                    patched.setdefault("expert", {})[name] = content
                    logger.info(f"Targeted retry: patched expert.{name} ({len(content)} chars)")
                else:
                    logger.warning(f"Targeted retry: expert.{name} too short or wrong type, skipping")

        return patched

    def _validate_output_quality(self, narrative: dict, metrics: dict, task_type: str = "", visuals_summary: dict = None, is_retry: bool = False) -> dict:
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
        for mode in ("expert",):
            mode_data = narrative.get(mode, {})
            required_sections = ["executive_summary", "findings", "conclusion", "recommendations"]
            if mode == "expert":
                required_sections.insert(1, "preprocessing_and_data_quality")
                if visuals_summary:
                    required_sections.append("visuals_analysis")
            for section in required_sections:
                content = mode_data.get(section, "")
                if isinstance(content, (list, dict)):
                    content_str = json.dumps(content)
                else:
                    content_str = str(content)
                
                # Use a smaller min_length for sections that could legitimately be very brief
                min_length = 30
                if section in ("preprocessing_and_data_quality", "recommendations"):
                    min_length = 10
                    
                if len(content_str) < min_length:
                    result["incomplete_sections"].append(f"{mode}.{section}")

        # 3. Data anchoring — at least 2 real metric values must appear in expert sections
        expert = narrative.get("expert", {})
        expert_text = json.dumps(expert)
        # Extract all numeric values from text for robust matching
        extracted_nums = []
        for n_str in re.findall(r'\b\d+(?:\.\d+)?\b', expert_text):
            try:
                extracted_nums.append(float(n_str))
            except ValueError:
                pass

        anchored_count = 0
        anchor_keys = ("R2", "RMSE", "MAE", "MSE") if self._is_regression_task(task_type) else ("accuracy", "ROC-AUC", "precision", "recall")
        for key in anchor_keys:
            raw_val = metrics.get(key, "")
            if not raw_val or raw_val == "N/A":
                continue
            val_str = str(raw_val).replace("%", "").strip()
            
            try:
                v_float = float(val_str)
                matched = False
                for num in extracted_nums:
                    # Match exact, or scaled by 100 (percentage), or rounded to 1/2/3 decimals
                    # Using a tolerance of 0.015 for floats, and 1.5 for percentages to allow for rounding
                    if abs(num - v_float) <= 0.015 or abs(num - (v_float * 100)) <= 1.5:
                        matched = True
                        break
                        
                if matched or val_str in expert_text:
                    anchored_count += 1
            except ValueError:
                if val_str in expert_text:
                    anchored_count += 1
                
        available_anchor_count = sum(1 for key in anchor_keys if str(metrics.get(key, "")).replace("%", "").strip() not in ("", "N/A"))
        required_anchor_count = min(2, available_anchor_count)
        if anchored_count < required_anchor_count:
            result["data_anchored"] = False
            logger.warning(f"Data anchoring check: only {anchored_count}/{required_anchor_count} metrics matched in expert narrative.")
            findings_len = len(str(expert.get("findings", "")))
            if findings_len < 30:
                result["incomplete_sections"].append("expert.findings")
            else:
                result["incomplete_sections"].append("expert.findings (not data-anchored)")

        # 4. Plot-tag presence — the findings should reference available plots
        findings_text = str(expert.get("findings", "")).lower()
        if self._is_regression_task(task_type):
            expected_plots = ["true_vs_predicted", "residuals"]
        else:
            expected_plots = ["roc_curve", "confusion_matrix", "feature_importance", "shap"]
        missing_plots = []
        for p in expected_plots:
            # Allow spaces or hyphens instead of underscores in the tag
            pattern = r"\[plot:\s*" + p.replace("_", r"[_\-\s]*") + r"(?:_png|\.png)?\s*\]"
            if not re.search(pattern, findings_text):
                missing_plots.append(p)
        # Flag only when more than half of expected plots are missing
        if len(missing_plots) > len(expected_plots) // 2 and not is_retry:
            logger.info(f"LLM omitted plot tags from findings: {missing_plots}. Will rely on _inject_missing_plots to append them.")

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
