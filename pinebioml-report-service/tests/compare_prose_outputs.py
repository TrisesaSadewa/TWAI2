import os
import sys
import json
from pathlib import Path

project_dir = str(Path(__file__).resolve().parent.parent)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

os.environ.setdefault("SERVICE_API_KEY", "test-dev-service-key")

from core.report.narrative_generator import NarrativeGenerator
from core.config import settings, MODEL_REGISTRY

MODELS_TO_TEST = ["granite4.1:8b", "ministral-3:8b", "llama3.1:8b"]

for m in MODELS_TO_TEST:
    settings.SUPPORTED_MODELS[m] = m
    MODEL_REGISTRY[m] = {
        "ollama_tag": m,
        "tier": 2,
        "vram_gb": 8.0,
        "roles": ["analysis"],
        "description": f"Test {m}",
        "max_tokens": 4096,
        "temperature": 0.2,
        "repeat_penalty": 1.15,
        "top_p": 0.95,
        "num_predict": 4096
    }

gen = NarrativeGenerator()

mock_metrics = {
    "accuracy": 0.9450,
    "ROC-AUC": 0.9820,
    "recall": 0.9380,
    "precision": 0.9510,
    "f1": 0.9444,
    "MCC": 0.8910
}

mock_per_class = [
    {"class": "Benign", "precision": 0.97, "recall": 0.96, "f1": 0.965, "support": 357},
    {"class": "Malignant", "precision": 0.94, "recall": 0.95, "f1": 0.945, "support": 212}
]

mock_visuals = {
    "roc_curve_png": {"description_fallback": "ROC Curve shows AUC=0.982 with strong TPR retention at low FPR."},
    "confusion_matrix_png": {"description_fallback": "Confusion Matrix: 343 TN, 14 FP, 11 FN, 201 TP."},
    "feature_importance_png": {"description_fallback": "Feature importance highlights mean concave points, worst perimeter, and worst radius."}
}

mock_shap = [
    {"feature": "mean concave points", "importance": 0.42, "direction": "positive"},
    {"feature": "worst perimeter", "importance": 0.31, "direction": "positive"},
    {"feature": "worst radius", "importance": 0.28, "direction": "positive"}
]

selected_feats = ["mean concave points", "worst perimeter", "worst radius", "worst texture", "area error"]

print("=" * 80)
print("EXTRACTING REAL PROSE SAMPLE COMPARISON")
print("=" * 80)

for m in MODELS_TO_TEST:
    print(f"\n>>> Model: [{m}] <<<")
    res = gen.generate_narrative(
        dataset_name="breast_cancer_diagnostic",
        task_type="classification",
        metrics=mock_metrics,
        visuals_summary=mock_visuals,
        shap_features=mock_shap,
        anomaly_flags=[],
        models={"analysis": m},
        per_class=mock_per_class,
        selected_features=selected_feats,
        report_id=f"inspect_{m}"
    )
    
    expert = res.get("expert", {})
    if isinstance(expert, dict):
        total_words = 0
        for section, text in expert.items():
            words = len(str(text).split())
            total_words += words
            print(f"  - Section '{section}': {words} words")
        print(f"  TOTAL EXPERT WORDS: {total_words}")
        print("\n  SAMPLE EXCERPT (Findings section):")
        findings_text = str(expert.get("findings", ""))[:600]
        print(f"  --------------------------------------------------")
        print(f"  {findings_text}...")
        print(f"  --------------------------------------------------")
