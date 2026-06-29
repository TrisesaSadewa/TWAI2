# PineBioML AI Report Microservice

A highly performant, beautiful, RESTful FastAPI microservice that generates professional, dual-audience (Clinical Expert & Layman/Public) AI-narrative reports directly from **PineBioML** machine learning runs.

## 🚀 Key Features

- **Asynchronous Processing**: Immediate validation and job scheduling to prevent notebook blocking.
- **Dual-Audience Narrative Engine**: Utilizes state-of-the-art LLMs to generate high-quality clinical biostatistics text alongside simplified plain-language patient-centric summaries.
- **Visual Artifact Analysis**: Automatically base64 encodes and analyzes plots (Confusion Matrices, ROC Curves, PCA / UMAP clusters) to write contextual graph critiques.
- **Premium Interactive Viewer**: Includes a stunning glassmorphic HTML viewer with interactive light/dark mode toggles, real-time SSE streaming generation, and hover zoom effects.
- **Multiple Document Exports**: Generates production-ready PDF (via WeasyPrint) and Word DOCX formats with a single download endpoint.
- **Scalability & Resiliency**: Built-in SQLite-backed job queue, in-process background workers, and automatic CPU inference fallback if GPU VRAM is exhausted.
- **Security First**: 100% local on-premise execution. Report links are protected by cryptographically secure tokens and automatically expire after 30 days.
- **Telemetry & Hardware Auditing**: Exposes live Prometheus monitoring metrics, standard health status, and live CUDA/NVIDIA GPU VRAM monitoring.
---

## 🛠️ Getting Started

### 1. Install Dependencies
Make sure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables
Copy the `.env.example` file to `.env`:
```bash
copy .env.example .env
```
Open `.env` and configure your settings. To use LLM narrative analysis, set:
```env
LLM_API_KEY=your-openai-api-key-here
```
*Note: If no API key is provided, the microservice automatically falls back to an ultra-detailed, rule-based biostatistics generator so the service functions fully out-of-the-box!*

### 3. Start the Server
Run the FastAPI development server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
You can access the interactive Swagger API documentation at: **`http://localhost:8000/docs`**

---

## 📖 API Documentation

FastAPI auto-generates comprehensive OpenAPI/Swagger documentation.
Once the server is running, navigate to:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Core Endpoints
- `POST /report/generate`: Accepts a `JobManifest` containing dataset details and artifacts (CSV/Images). Returns `202 Accepted` with a `report_id` and secure `access_token`.
- `GET /report/stream/{report_id}?token=XYZ`: Real-time Server-Sent Events (SSE) streaming the LLM JSON narrative as it's generated.
- `GET /report/status/{report_id}?token=XYZ`: Poll this endpoint to get live generation progress (`QUEUED`, `ANALYZING`, `SUCCESS`).
- `GET /report/{report_id}/html?token=XYZ`: Returns the interactive, self-contained HTML Single-Page Application viewer.
- `GET /report/{report_id}/download/{fmt}?token=XYZ`: Download the report as `pdf` or `docx`.

---

## 🧪 Testing

We use `pytest` for unit, integration, and End-to-End (E2E) testing. The E2E tests hit the local Ollama instance to ensure the full pipeline runs flawlessly.

To run the test suite:
```bash
pip install pytest pytest-asyncio pytest-timeout httpx
python -m pytest tests/ -v
```

The E2E tests mock a **Breast Cancer Wisconsin** dataset run and verify the generation of the HTML, PDF, and DOCX assets.

---

## 📚 User Guide

Are you a researcher or clinician? Check out our comprehensive [User Guide](USER_GUIDE.md) to understand how to read and interact with the AI-generated reports, including how to interpret SHAP values and add your own clinical recommendations.
