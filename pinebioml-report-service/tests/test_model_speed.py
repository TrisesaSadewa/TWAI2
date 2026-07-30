import os
import sys
from pathlib import Path

# Add project directory to sys.path so imports like `from core...` resolve anywhere
project_dir = str(Path(__file__).resolve().parent)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Ensure required env vars are set before importing config if running outside repo root
os.environ.setdefault("SERVICE_API_KEY", "test-dev-service-key")

import argparse
import time
import json
import urllib.request
import requests

from core.report.narrative_generator import NarrativeGenerator
from core.config import settings, MODEL_REGISTRY

MODELS_TO_TEST = [
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "qwen3.5:9b"
]

# Global store for captured Ollama response metrics per request
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

# Patch requests.post globally for timing capture
requests.post = telemetry_aware_post

def check_ollama_is_running():
    try:
        urllib.request.urlopen(settings.OLLAMA_BASE_URL, timeout=2)
        return True
    except Exception:
        return False

def format_ns_duration(ns):
    if ns is None:
        return "N/A"
    seconds = ns / 1e9
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    return f"{seconds:.2f}s"

def run_test(verbose=False):
    if not check_ollama_is_running():
        print(f"Warning: Ollama doesn't seem to be reachable at {settings.OLLAMA_BASE_URL}")

    # Register the models dynamically so the NarrativeGenerator doesn't fallback
    for m in MODELS_TO_TEST:
        if m not in settings.SUPPORTED_MODELS:
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
        "accuracy": 0.95,
        "ROC-AUC": 0.97,
        "recall": 0.94,
        "precision": 0.96,
        "f1": 0.95
    }
    
    mock_per_class = [
        {"class": "0", "precision": 0.96, "recall": 0.96, "f1": 0.96, "support": 100},
        {"class": "1", "precision": 0.94, "recall": 0.94, "f1": 0.94, "support": 100}
    ]
    
    results = {}
    
    print("=" * 60)
    print("Starting AI Report Generation Telemetry Benchmark")
    print("=" * 60)
    
    for model in MODELS_TO_TEST:
        print(f"\n[{model}] Starting test...")
        global captured_telemetry
        captured_telemetry = []
        
        start_time = time.time()
        
        try:
            res = gen.generate_narrative(
                dataset_name="breast_cancer",
                task_type="classification",
                metrics=mock_metrics,
                visuals_summary={"roc_curve_png": {"description_fallback": "ROC Curve"}},
                shap_features=[{"feature": "mean radius", "importance": 0.45, "direction": "positive"}],
                anomaly_flags=[],
                models={"analysis": model},
                per_class=mock_per_class,
                selected_features=["mean radius", "mean texture"]
            )
            
            end_time = time.time()
            wall_duration = end_time - start_time
            
            # Extract aggregate metrics across all LLM requests made (e.g. retries/targeted fixes)
            total_eval_count = sum(t.get("eval_count", 0) for t in captured_telemetry)
            total_eval_dur_ns = sum(t.get("eval_duration", 0) for t in captured_telemetry)
            total_prompt_count = sum(t.get("prompt_eval_count", 0) for t in captured_telemetry)
            total_prompt_dur_ns = sum(t.get("prompt_eval_duration", 0) for t in captured_telemetry)
            total_load_dur_ns = sum(t.get("load_duration", 0) for t in captured_telemetry)
            total_ollama_dur_ns = sum(t.get("total_duration", 0) for t in captured_telemetry) or int(wall_duration * 1e9)

            # If native prompt_eval_duration is missing, infer prefill time from total duration minus eval duration
            if total_prompt_dur_ns == 0 and total_ollama_dur_ns > total_eval_dur_ns:
                total_prompt_dur_ns = total_ollama_dur_ns - total_eval_dur_ns

            prompt_eval_rate = (total_prompt_count / (total_prompt_dur_ns / 1e9)) if total_prompt_dur_ns > 0 else 0
            eval_rate = (total_eval_count / (total_eval_dur_ns / 1e9)) if total_eval_dur_ns > 0 else 0

            print(f"[{model}] Success!")
            print(f"  total duration:       {format_ns_duration(total_ollama_dur_ns)}")
            print(f"  load duration:        {format_ns_duration(total_load_dur_ns)}")
            print(f"  prompt eval count:    {total_prompt_count} token(s)")
            print(f"  prompt eval duration: {format_ns_duration(total_prompt_dur_ns)}")
            print(f"  prompt eval rate:     {prompt_eval_rate:.2f} tokens/s")
            print(f"  eval count:           {total_eval_count} token(s)")
            print(f"  eval duration:        {format_ns_duration(total_eval_dur_ns)}")
            print(f"  eval rate:            {eval_rate:.2f} tokens/s")
            
            if verbose:
                print(f"\n[{model}] Generated Output:\n")
                print(json.dumps(res, indent=2))
                print("-" * 40)
                
            results[model] = {
                "wall_duration": wall_duration,
                "total_duration_ns": total_ollama_dur_ns,
                "load_duration_ns": total_load_dur_ns,
                "prompt_eval_count": total_prompt_count,
                "prompt_eval_duration_ns": total_prompt_dur_ns,
                "prompt_eval_rate": prompt_eval_rate,
                "eval_count": total_eval_count,
                "eval_duration_ns": total_eval_dur_ns,
                "eval_rate": eval_rate,
                "status": "success"
            }
        except Exception as e:
            end_time = time.time()
            wall_duration = end_time - start_time
            print(f"[{model}] Failed! Took {wall_duration:.2f}s")
            print(f"[{model}] Error: {str(e)}")
            results[model] = {
                "wall_duration": wall_duration,
                "error": str(e),
                "status": "failed"
            }

    print("\n" + "=" * 80)
    print("DETAILED OLLAMA TELEMETRY COMPARISON")
    print("=" * 80)
    print(f"{'Model':<18} | {'Load Dur':<10} | {'Prompt Tok':<10} | {'Prompt Rate':<13} | {'Eval Tok':<10} | {'Eval Rate':<12}")
    print("-" * 80)
    for m, stats in results.items():
        if stats["status"] == "success":
            print(
                f"{m:<18} | "
                f"{format_ns_duration(stats['load_duration_ns']):<10} | "
                f"{stats['prompt_eval_count']:<10} | "
                f"{stats['prompt_eval_rate']:<13.2f} | "
                f"{stats['eval_count']:<10} | "
                f"{stats['eval_rate']:<12.2f}"
            )
        else:
            print(f"{m:<18} | {'FAILED':<10} | N/A        | N/A           | N/A        | N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test token speed and AI report generation performance across models.")
    parser.add_argument("--verbose", action="store_true", help="Print the full JSON output of the generated reports.")
    args = parser.parse_args()
    
    run_test(verbose=args.verbose)

