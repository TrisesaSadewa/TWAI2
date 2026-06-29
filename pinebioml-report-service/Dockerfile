FROM python:3.11-slim

# Install system dependencies for WeasyPrint (PDF rendering)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-cffi \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y pandoc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PineBioML and ALL its dependencies (optuna, xgboost, lightgbm, catboost, etc.)
# Note: If running completely decoupled, you will need to map or install PineBioML here
COPY PineBioML /tmp/PineBioML
RUN pip install --no-cache-dir --default-timeout=1000 /tmp/PineBioML && rm -rf /tmp/PineBioML

COPY pinebioml-report-service/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY pinebioml-report-service .

EXPOSE 8000

CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]
