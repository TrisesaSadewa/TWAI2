import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Dict, Any

# ── Centralized Model Registry ──────────────────────────────────────────────
# Fields per model:
#   ollama_tag:   The Ollama model tag (or cloud model ID for OpenAI-compat)
#   tier:         Prompt complexity tier (1 = simpler prompts for basic models,
#                 2 = full rich prompts for strong reasoners)
#   vram_gb:      Approximate VRAM consumption (for UI display / scheduling)
#   roles:        What the model can be used for: "analysis", "vision", "embedding"
#   description:  Human-readable label for UI display / model picker
#   max_tokens:   Recommended max output tokens for this model
#   context_tokens: Recommended Ollama context window (num_ctx)
#   temperature:  Recommended temperature for clinical report generation

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Local Ollama models ───────────────────────────────────────────────
    "deepseek-r1-14b": {
        "ollama_tag":   "deepseek-r1:14b",
        "tier":         2,
        "vram_gb":      8.7,
        "roles":        ["analysis"],
        "description":  "DeepSeek-R1 14B",
        "max_tokens":   8192,
        "temperature":  0.2,
    },
    "qwen2.5-coder-14b": {
        "ollama_tag":   "qwen2.5-coder:14b",
        "tier":         1,
        "vram_gb":      8.7,
        "roles":        ["analysis"],
        "description":  "Qwen 2.5 Coder 14B",
        "max_tokens":   8192,
        "temperature":  0.15,
    },
    "qwen3.5:9b": {
        "ollama_tag":   "qwen3.5:9b",
        "tier":         2,
        "vram_gb":      6.0,
        "roles":        ["analysis"],
        "description":  "Qwen 3.5 9B",
        "max_tokens":   8192,
        "temperature":  0.15,
    },
    "gemma4:12b": {
        "ollama_tag":   "gemma4:12b",
        "tier":         2,
        "vram_gb":      7.0,
        "roles":        ["analysis"],
        "description":  "Gemma4 12B",
        "max_tokens":   8192,
        "context_tokens": 32768,
        "temperature":  0.15,
    },


    # ── Vision models  ────────────────
    "glm-ocr:latest": {
        "ollama_tag":   "glm-ocr:latest",
        "tier":         1,
        "vram_gb":      5.0,
        "roles":        ["vision"],
        "description":  "GLM-OCR — Vision/chart analysis",
        "max_tokens":   8192,
        "temperature":  0.3,
    },
    "qwen3-vl:8b": {
        "ollama_tag":   "qwen3-vl:8b",
        "tier":         1,
        "vram_gb":      6.0,
        "roles":        ["vision"],
        "description":  "Qwen3-VL 8B — Vision/chart analysis",
        "max_tokens":   8192,
        "temperature":  0.3,
    },
        "llava-7b": {
        "ollama_tag":   "llava:7b",
        "tier":         1,
        "vram_gb":      4.7,
        "roles":        ["vision"],
        "description":  "LLaVA 7B — Vision/chart analysis",
        "max_tokens":   2048,
        "temperature":  0.3,
    },
    "moondream2": {
        "ollama_tag":   "moondream2",
        "tier":         1,
        "vram_gb":      1.6,
        "roles":        ["vision"],
        "description":  "Moondream2 — Lightweight vision",
        "max_tokens":   2048,
        "temperature":  0.3,
    }
}

# Auto-derive the SUPPORTED_MODELS lookup (alias → ollama_tag) for backward compat
_SUPPORTED_MODELS = {alias: m["ollama_tag"] for alias, m in MODEL_REGISTRY.items()}
# Also map the actual ollama tags to themselves so they are recognized directly
for m in MODEL_REGISTRY.values():
    _SUPPORTED_MODELS[m["ollama_tag"]] = m["ollama_tag"]
# Also add legacy aliases so old trigger scripts don't break
_SUPPORTED_MODELS["deepseek-14b"] = "deepseek-r1:14b"
_SUPPORTED_MODELS["llava"] = "llava:7b"
_SUPPORTED_MODELS["moondream"] = "moondream2"
_SUPPORTED_MODELS["glm-ocr"] = "glm-ocr:latest"
_SUPPORTED_MODELS["gemma4"] = "gemma4:12b"
_SUPPORTED_MODELS["qwen3-vl"] = "qwen3-vl:8b"
_SUPPORTED_MODELS["qwen3.5"] = "qwen3.5:9b"
_SUPPORTED_MODELS["qwen2.5-coder"] = "qwen2.5-coder:14b"
_SUPPORTED_MODELS["deepseek-r1"] = "deepseek-r1:14b"


def get_model_tier(model_alias: str) -> int:
    """Return the prompt-complexity tier for a model alias. Defaults to 1 (simple)."""
    entry = MODEL_REGISTRY.get(model_alias)
    if entry:
        return entry["tier"]
    # Try to find by ollama_tag
    for m in MODEL_REGISTRY.values():
        if m["ollama_tag"] == model_alias:
            return m["tier"]
    return 1  # default to simple prompts for unknown models


def get_model_config(model_alias: str) -> dict:
    """Return the full config dict for a model alias, or sensible defaults."""
    entry = MODEL_REGISTRY.get(model_alias)
    if entry:
        return entry
    # Try by ollama_tag
    for m in MODEL_REGISTRY.values():
        if m["ollama_tag"] == model_alias:
            return m
    return {
        "ollama_tag": model_alias,
        "tier": 1,
        "vram_gb": 0,
        "roles": ["analysis"],
        "description": f"Unknown model: {model_alias}",
        "max_tokens": 4096,
        "temperature": 0.2,
    }


def list_analysis_models() -> list:
    """Return a list of models available for report analysis, for UI/API display."""
    result = []
    for alias, m in MODEL_REGISTRY.items():
        if "analysis" in m["roles"]:
            result.append({
                "id": alias,
                "name": m["description"],
                "tier": m["tier"],
                "vram_gb": m["vram_gb"],
                "ollama_tag": m["ollama_tag"],
            })
    return result


def get_deployment_writer_model() -> str:
    """Resolve the default report writer for the configured deployment hardware."""
    deployment = (settings.WRITER_DEPLOYMENT or "gpu").strip().lower()
    if deployment in {"cpu", "cpu-only", "cpu_only"}:
        return settings.CPU_DEPLOYMENT_WRITER_MODEL
    return settings.LLM_MODEL


class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "PineBioML AI Report Microservice"
    API_VERSION: str = "1.0.0"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # ── Local Ollama endpoint (RTX 5050 Mobile, CUDA) ────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # LLM Settings (supports both Ollama local and OpenAI-compatible cloud API)
    LLM_API_KEY: Optional[str] = None
    LLM_API_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "deepseek-r1:14b"
    WRITER_DEPLOYMENT: str = "gpu"
    CPU_DEPLOYMENT_WRITER_MODEL: str = "deepseek-r1:14b"
    LLM_REQUEST_TIMEOUT_SECONDS: int = 900

    # Tier-2 hallucination verifier model. This must be a FAST, NON-REASONING
    # model: the verifier answers a single YES/NO grounding question per flagged
    # sentence, so a reasoning model (deepseek-r1) is both wasteful and too slow
    # (it times out on CPU-only hosts). GPU deployments keep LLM_MODEL on the
    # reasoning model; CPU deployments switch the writer via WRITER_DEPLOYMENT.
    VERIFIER_MODEL: str = "qwen2.5-coder:14b"

    # Threshold tuning for binary classification on imbalanced data. After fit,
    # the decision threshold for the minority class is searched (0.05-0.95) to
    # maximize this metric on out-of-fold probabilities. Supported: "f1"
    # (balanced, default), "sensitivity" (screening-first / recall), "mcc".
    THRESHOLD_OPTIMIZE_METRIC: str = "f1"

    # Embedding model served via Ollama
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Vision / multimodal model for chart & plot analysis
    VISION_MODEL: str = "glm-ocr:latest"
    ENABLE_VISION_ANALYSIS: bool = False
    VISION_ANALYSIS_TIMEOUT_SECONDS: int = 30

    # Production surface hardening
    ENABLE_API_DOCS: bool = False
    PUBLIC_METRICS: bool = False
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024
    ALLOWED_UPLOAD_EXTENSIONS: str = ".csv,.tsv,.xlsx,.xls"

    # Backward-compatible lookup (auto-derived from MODEL_REGISTRY)
    SUPPORTED_MODELS: dict = _SUPPORTED_MODELS

    # Storage settings
    STORAGE_DIR: str = "./storage"
    MEDIA_ROOT: str = "./storage/media"
    DATABASE_URL: str = "sqlite:///./storage/databases/pinebioml_reports.db"

    # Retention policy for uploaded data and generated artifacts.
    # Set to 0 to disable automatic deletion for that category.
    DATASET_RETENTION_DAYS: int = 7    # delete uploaded datasets N days after training
    REPORT_RETENTION_DAYS: int = 90    # delete reports/media/exports N days after creation

    # Security/Webhooks
    CALLBACK_API_KEY: Optional[str] = None
    SERVICE_API_KEY: str
    ALLOWED_ORIGINS: str = "https://pinebioml.ncu.edu.tw"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Ensure storage directories exist
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_DIR, "exports"), exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_DIR, "databases"), exist_ok=True)
os.makedirs(os.path.join(settings.STORAGE_DIR, "datasets"), exist_ok=True)
