import os
import sys
import time
import json
import urllib.request
import requests
from pathlib import Path

project_dir = str(Path(__file__).resolve().parent.parent)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

os.environ.setdefault("SERVICE_API_KEY", "test-dev-service-key")

from core.report.narrative_generator import NarrativeGenerator
from core.config import settings, MODEL_REGISTRY

MODELS_TO_TEST = [
    "granite4.1:8b",
    "ministral-3:8b",
    "llama3.1:8b",
    "gemma4:12b",
    "deepseek-r1:8b",
    "qwen3.5:9b"
]

captured_telemetry = []
original_post = requests.post

class TelemetryStreamWrapper:
    def __init__(self, response):
        self.response = response
        self.raw = response.raw

    def __getattr__(self, name):
        return getattr(self.response, name)

    def iter_lines(self, *args, **kwargs):
        for line in self.response.iter_lines(*args, **kwargs):
            if line:
                try:
                    line_str = line.decode('utf-8').strip()
                    json_str = line_str[6:].strip() if line_str.startswith("data: ") else line_str
                    data = json.loads(json_str)
                    if isinstance(data, dict) and data.get("done") is True:
                        eval_count = data.get("eval_count") or 0
                        prompt_eval_count = data.get("prompt_eval_count") or 0
                        eval_dur = data.get("eval_duration") or 0
                        prompt_dur = data.get("prompt_eval_duration") or 0
                        load_dur = data.get("load_duration") or 0
                        total_dur = data.get("total_duration") or 0
                        
                        captured_telemetry.append({
                            "eval_count": eval_count,
                            "prompt_eval_count": prompt_eval_count,
                            "eval_duration": eval_dur,
                            "prompt_eval_duration": prompt_dur,
                            "load_duration": load_dur,
                            "total_duration": total_dur
                        })
                except Exception:
                    pass
            yield line

def telemetry_aware_post(*args, **kwargs):
    req_start = time.time()
    resp = original_post(*args, **kwargs)
    req_dur_ns = int((time.time() - req_start) * 1e9)
    
    if kwargs.get("stream"):
        return TelemetryStreamWrapper(resp)
    
    try:
        data = resp.json()
        if isinstance(data, dict):
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            eval_count = data.get("eval_count") or usage.get("completion_tokens") or 0
            prompt_eval_count = data.get("prompt_eval_count") or usage.get("prompt_tokens") or 0
            total_dur = data.get("total_duration") or req_dur_ns
            eval_dur = data.get("eval_duration") or req_dur_ns
            prompt_dur = data.get("prompt_eval_duration") or 0
            load_dur = data.get("load_duration") or 0
            
            captured_telemetry.append({
                "eval_count": eval_count,
                "prompt_eval_count": prompt_eval_count,
                "eval_duration": eval_dur,
                "prompt_eval_duration": prompt_dur,
                "load_duration": load_dur,
                "total_duration": total_dur
            })
    except Exception:
        pass
    return resp

requests.post = telemetry_aware_post

def check_ollama():
    try:
        urllib.request.urlopen(settings.OLLAMA_BASE_URL, timeout=2)
        return True
    except Exception:
        return False

def run_bench():
    if not check_ollama():
        print(f"Error: Ollama unavailable at {settings.OLLAMA_BASE_URL}")
        sys.exit(1)

    for m in MODELS_TO_TEST:
        settings.SUPPORTED_MODELS[m] = m
        MODEL_REGISTRY[m] = {
            "ollama_tag": m,
            "tier": 2,
            "vram_gb": 8.0,
            "roles": ["analysis"],
            "description": f"Test Model {m}",
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
        "confusion_matrix_png": {"description_fallback": "Confusion Matrix: 343 TN, 14 FP, 11 FN, 201 TP."}
    }

    mock_shap = [
        {"feature": "mean concave points", "importance": 0.42, "direction": "positive"},
        {"feature": "worst perimeter", "importance": 0.31, "direction": "positive"}
    ]

    selected_feats = ["mean concave points", "worst perimeter", "worst radius", "worst texture", "area error"]

    results = {}

    print("=" * 95)
    print("      COMPREHENSIVE MULTI-MODEL BENCHMARK (SPEED, VERBOSITY, TOKENS/SEC)")
    print("=" * 95)

    for model in MODELS_TO_TEST:
        print(f"\nEvaluating Model: [{model}]...")
        global captured_telemetry
        captured_telemetry = []

        start_time = time.time()
        error_str = None
        res = None

        try:
            res = gen.generate_narrative(
                dataset_name="breast_cancer_diagnostic",
                task_type="classification",
                metrics=mock_metrics,
                visuals_summary=mock_visuals,
                shap_features=mock_shap,
                anomaly_flags=[],
                models={"analysis": model},
                per_class=mock_per_class,
                selected_features=selected_feats,
                report_id=f"allbench_{model}"
            )
            wall_time = time.time() - start_time
        except Exception as e:
            wall_time = time.time() - start_time
            error_str = str(e)
            print(f"  FAILED in {wall_time:.2f}s: {error_str}")
            results[model] = {"status": "failed", "wall_time": wall_time, "error": error_str}
            continue

        tot_eval_count = sum(t.get("eval_count", 0) for t in captured_telemetry)
        tot_eval_dur_s = sum(t.get("eval_duration", 0) for t in captured_telemetry) / 1e9
        tot_prompt_count = sum(t.get("prompt_eval_count", 0) for t in captured_telemetry)
        tot_prompt_dur_s = sum(t.get("prompt_eval_duration", 0) for t in captured_telemetry) / 1e9

        eval_speed = (tot_eval_count / tot_eval_dur_s) if tot_eval_dur_s > 0 else 0
        prompt_speed = (tot_prompt_count / tot_prompt_dur_s) if tot_prompt_dur_s > 0 else 0

        expert = res.get("expert", {}) if isinstance(res, dict) else {}
        total_words = 0
        section_breakdown = {}
        if isinstance(expert, dict):
            for sec_k, sec_v in expert.items():
                w = len(str(sec_v).split())
                total_words += w
                section_breakdown[sec_k] = w

        print(f"  Status:         SUCCESS")
        print(f"  Wall Time:      {wall_time:.2f} s")
        print(f"  Gen Speed:      {eval_speed:.2f} tok/s ({tot_eval_count} tokens generated)")
        print(f"  Prompt Speed:   {prompt_speed:.2f} tok/s ({tot_prompt_count} prompt tokens)")
        print(f"  Total Words:    {total_words} words (Verbosity Score: {total_words / (tot_eval_count or 1):.2f} words/tok)")
        print(f"  Section Words:  {section_breakdown}")

        results[model] = {
            "status": "success",
            "wall_time": wall_time,
            "gen_speed": eval_speed,
            "prompt_speed": prompt_speed,
            "eval_tokens": tot_eval_count,
            "prompt_tokens": tot_prompt_count,
            "word_count": total_words,
            "sections": section_breakdown
        }

    print("\n" + "=" * 105)
    print("                                   FINAL COMPARISON SUMMARY TABLE")
    print("=" * 105)
    print(f"{'Model Name':<16} | {'Gen Speed (tok/s)':<18} | {'Total Words':<12} | {'Eval Tokens':<12} | {'Wall Time':<10} | {'Status'}")
    print("-" * 105)

    for m, r in results.items():
        if r["status"] == "success":
            print(f"{m:<16} | {r['gen_speed']:<18.2f} | {r['word_count']:<12} | {r['eval_tokens']:<12} | {r['wall_time']:<10.2f}s | SUCCESS")
        else:
            print(f"{m:<16} | {'N/A':<18} | {'N/A':<12} | {'N/A':<12} | {r['wall_time']:<10.2f}s | FAILED")

    print("=" * 105)

if __name__ == "__main__":
    run_bench()
