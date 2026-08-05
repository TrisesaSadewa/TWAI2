import os
import sys
import time
import json
import urllib.request
import requests
from pathlib import Path

# Add project root directory to sys.path
project_dir = str(Path(__file__).resolve().parent.parent)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

os.environ.setdefault("SERVICE_API_KEY", "test-dev-service-key")

from core.report.narrative_generator import NarrativeGenerator
from core.config import settings, MODEL_REGISTRY

MODELS_TO_TEST = [
    "llama3.1:8b",
    "granite4.1:8b",
    "ministral-3:8b"
]

captured_telemetry = []
original_post = requests.post

def telemetry_aware_post(*args, **kwargs):
    req_start = time.time()
    resp = original_post(*args, **kwargs)
    req_dur_ns = int((time.time() - req_start) * 1e9)
    
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

def check_ollama_is_running():
    try:
        urllib.request.urlopen(settings.OLLAMA_BASE_URL, timeout=2)
        return True
    except Exception:
        return False

def format_duration(seconds):
    if seconds is None:
        return "N/A"
    return f"{seconds:.2f}s"

def run_benchmark(num_runs=2):
    if not check_ollama_is_running():
        print(f"Error: Ollama server not reachable at {settings.OLLAMA_BASE_URL}")
        sys.exit(1)

    # Register models in settings so NarrativeGenerator supports them directly
    for m in MODELS_TO_TEST:
        settings.SUPPORTED_MODELS[m] = m
        MODEL_REGISTRY[m] = {
            "ollama_tag": m,
            "tier": 2,
            "vram_gb": 8.0,
            "roles": ["analysis"],
            "description": f"Benchmark Model {m}",
            "max_tokens": 4096,
            "temperature": 0.2,
            "repeat_penalty": 1.15,
            "top_p": 0.95,
            "num_predict": 4096
        }

    gen = NarrativeGenerator()

    # Realistic production input context
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
        "feature_importance_png": {"description_fallback": "Feature importance highlights mean concave points, worst perimeter, and worst radius as primary drivers."}
    }

    mock_shap = [
        {"feature": "mean concave points", "importance": 0.42, "direction": "positive"},
        {"feature": "worst perimeter", "importance": 0.31, "direction": "positive"},
        {"feature": "worst radius", "importance": 0.28, "direction": "positive"},
        {"feature": "worst texture", "importance": 0.15, "direction": "positive"},
        {"feature": "area error", "importance": 0.12, "direction": "positive"}
    ]

    mock_anomalies = [
        "Class imbalance detected (Malignant represents 37.2% of total dataset).",
        "Minor false negative count (11 cases) requires clinical risk mitigation note."
    ]

    selected_feats = ["mean concave points", "worst perimeter", "worst radius", "worst texture", "area error", "mean texture", "worst area"]

    results = {}

    print("=" * 85)
    print("      PRODUCTION REPORT GENERATION SPEED BENCHMARK: OLLAMA 8B CLASS MODELS")
    print("=" * 85)
    print(f"Models: {', '.join(MODELS_TO_TEST)}")
    print(f"Iterations per model: {num_runs} (Warmup + Production Runs)\n")

    for model in MODELS_TO_TEST:
        print(f"-----------------------------------------------------------------------------------")
        print(f"Testing Model: [{model}]")
        print(f"-----------------------------------------------------------------------------------")
        
        run_stats = []
        
        for run_idx in range(1, num_runs + 1):
            run_type = "Warmup" if run_idx == 1 else f"Run {run_idx - 1}"
            print(f"  > Executing {run_type}...", end="", flush=True)
            
            global captured_telemetry
            captured_telemetry = []
            
            start_wall = time.time()
            error_msg = None
            generated_json = None
            
            try:
                res = gen.generate_narrative(
                    dataset_name="breast_cancer_diagnostic",
                    task_type="classification",
                    metrics=mock_metrics,
                    visuals_summary=mock_visuals,
                    shap_features=mock_shap,
                    anomaly_flags=mock_anomalies,
                    models={"analysis": model},
                    per_class=mock_per_class,
                    selected_features=selected_feats,
                    report_id=f"bench_{model}_{run_idx}"
                )
                end_wall = time.time()
                wall_time = end_wall - start_wall
                generated_json = res
            except Exception as e:
                end_wall = time.time()
                wall_time = end_wall - start_wall
                error_msg = str(e)
                print(f" FAILED in {wall_time:.2f}s ({error_msg})")
                run_stats.append({"run_type": run_type, "status": "failed", "wall_time": wall_time, "error": error_msg})
                continue
            
            # Aggregate telemetry
            tot_eval_count = sum(t.get("eval_count", 0) for t in captured_telemetry)
            tot_eval_dur_s = sum(t.get("eval_duration", 0) for t in captured_telemetry) / 1e9
            tot_prompt_count = sum(t.get("prompt_eval_count", 0) for t in captured_telemetry)
            tot_prompt_dur_s = sum(t.get("prompt_eval_duration", 0) for t in captured_telemetry) / 1e9
            tot_load_dur_s = sum(t.get("load_duration", 0) for t in captured_telemetry) / 1e9
            
            prompt_speed = (tot_prompt_count / tot_prompt_dur_s) if tot_prompt_dur_s > 0 else 0
            gen_speed = (tot_eval_count / tot_eval_dur_s) if tot_eval_dur_s > 0 else 0
            
            # Character/Word length of generated text
            narrative_text = json.dumps(res) if isinstance(res, dict) else str(res)
            word_count = len(narrative_text.split())
            
            print(f" SUCCESS!")
            print(f"      Wall Clock Time:    {wall_time:.2f} s")
            print(f"      Model Load Time:    {tot_load_dur_s:.2f} s")
            print(f"      Prompt Processing:  {tot_prompt_count} tokens @ {prompt_speed:.2f} tok/s ({tot_prompt_dur_s:.2f}s)")
            print(f"      Response Generation:{tot_eval_count} tokens @ {gen_speed:.2f} tok/s ({tot_eval_dur_s:.2f}s)")
            print(f"      Generated Length:   {word_count} words ({len(narrative_text)} chars)\n")
            
            run_stats.append({
                "run_type": run_type,
                "status": "success",
                "wall_time": wall_time,
                "load_time": tot_load_dur_s,
                "prompt_tokens": tot_prompt_count,
                "prompt_speed": prompt_speed,
                "eval_tokens": tot_eval_count,
                "gen_speed": gen_speed,
                "word_count": word_count,
                "char_count": len(narrative_text)
            })

        results[model] = run_stats

    # Print Summary Table
    print("\n" + "=" * 95)
    print("                             BENCHMARK SUMMARY & COMPARISON")
    print("=" * 95)
    header = f"{'Model':<16} | {'Warmup Time':<12} | {'Prod Run Time':<14} | {'Gen Speed (tok/s)':<18} | {'Output Toks':<12} | {'Status'}"
    print(header)
    print("-" * 95)

    for m, stats in results.items():
        succ_runs = [r for r in stats if r["status"] == "success"]
        if not succ_runs:
            print(f"{m:<16} | {'FAILED':<12} | {'FAILED':<14} | {'N/A':<18} | {'N/A':<12} | FAILED")
            continue
        
        warmup_time = f"{stats[0]['wall_time']:.2f}s" if len(stats) > 0 and stats[0]['status'] == "success" else "N/A"
        prod_runs = [r for r in stats if r['run_type'] != 'Warmup' and r['status'] == 'success']
        if not prod_runs:
            prod_runs = succ_runs
            
        avg_prod_time = sum(r['wall_time'] for r in prod_runs) / len(prod_runs)
        avg_gen_speed = sum(r['gen_speed'] for r in prod_runs) / len(prod_runs)
        avg_eval_toks = sum(r['eval_tokens'] for r in prod_runs) / len(prod_runs)
        
        print(f"{m:<16} | {warmup_time:<12} | {avg_prod_time:<14.2f}s | {avg_gen_speed:<18.2f} | {avg_eval_toks:<12.0f} | SUCCESS")

    print("=" * 95)

if __name__ == "__main__":
    run_benchmark(num_runs=3)
