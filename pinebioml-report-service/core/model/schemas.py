from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class ModelConfig(BaseModel):
    analysis: str = Field("deepseek-r1:14b", description="Model for general narrative generation")
    interpretability: str = Field("deepseek-r1:14b", description="Model for SHAP/LIME explanation")
    reasoning: str = Field("deepseek-r1:14b", description="Model for advanced statistical reasoning / CoT")

class JobManifest(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the ML run")
    dataset_name: str = Field(..., description="Name of the dataset analyzed")
    task_type: str = Field(..., description="Type of ML task: binary_classification, multiclass_classification, regression")
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Pre-computed metrics from the ML pipeline (accuracy, precision, recall, etc.)")
    artifacts: Dict[str, str] = Field(..., description="Map of artifact identifiers to absolute or relative file paths")
    imbalance_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata regarding class distribution and imbalance strategies")
    all_models_data: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Performance data for all evaluated models")
    selected_features: Optional[List[str]] = Field(default_factory=list, description="List of features selected by the pipeline")
    callback_url: Optional[str] = Field(None, description="Optional webhook URL to call when report generation is complete")
    api_key: Optional[str] = Field(None, description="Optional API key for authenticating the callback request")
    expiry_days: Optional[int] = Field(7, description="Number of days until the shared link expires (default 7)")
    models: Optional[ModelConfig] = Field(default_factory=ModelConfig, description="Specific models to use for each section")

class ReportStatus(BaseModel):
    report_id: str
    access_token: Optional[str] = None
    job_id: str
    status: str = Field(..., description="QUEUED, ANALYZING, GENERATING, SUCCESS, FAILED")
    progress_pct: int = Field(0, ge=0, le=100)
    message: str
    created_at: str
    updated_at: str
    model_name: Optional[str] = Field(None, description="The AI model configured for this report analysis")

class SectionEdit(BaseModel):
    key: str = Field(..., description="Section key to edit, e.g. executive_summary, findings, or recommendations")
    content: str = Field(..., description="New markdown content for the section")
    mode: str = Field("expert", description="expert only (layman mode removed)")

class EditResponse(BaseModel):
    success: bool
    message: str
    report_id: str
    updated_at: str

class DownloadLink(BaseModel):
    format: str = Field(..., description="pdf or docx")
    url: str = Field(..., description="URL path to download the report in this format")

class ReportMetadata(BaseModel):
    report_id: str
    job_id: str
    dataset_name: str
    task_type: str
    status: str
    created_at: str
    updated_at: str
    html_url: str = Field(..., description="URL to the interactive HTML viewer")
    download_links: List[DownloadLink]

class MetricOverview(BaseModel):
    metrics: Dict[str, Any]
    visuals: List[Dict[str, str]]

class HealthCheck(BaseModel):
    status: str
    uptime_seconds: float
    queue_count: int
    system_load: float

class GPUInfo(BaseModel):
    device_id: int
    name: str
    vram_total_mb: float
    vram_used_mb: float
    vram_free_mb: float
    temperature_c: float

class GPUStatus(BaseModel):
    has_gpu: bool
    devices: List[GPUInfo]
    safe_to_inference: bool
