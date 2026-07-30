# NCU Production Deployment Guide

This guide covers the full deployment of the PineBioML report service on the NCU server,
from cloning the repository to a running Docker stack.

Use `docker-compose.prod.yml` for production. Keep `docker-compose.yml` for local development only.

---

## Prerequisites

The NCU server must have:

- Git
- Docker Engine (20.10+)
- Docker Compose plugin (`docker compose`, not the legacy `docker-compose`)
- NVIDIA Container Toolkit (required for Docker GPU passthrough on Linux)
- At least 50 GB free disk space (Ollama models are large)
- Outbound internet access (to pull Docker images and Ollama models on first boot)

---

## 1. Clone The Repository

```bash
git clone https://github.com/TrisesaSadewa/TWAI2.git /opt/pinebioml
cd /opt/pinebioml
```

The relevant layout after cloning:

```text
/opt/pinebioml/
  PineBioML/                  Python package installed into the container
  pinebioml-report-service/   The service — Dockerfile, compose, source code
```

---

## 2. Create The Production Environment File

```bash
cd /opt/pinebioml/pinebioml-report-service
cp .env.production.example .env.production
nano .env.production
```

Fill in the three required secrets:

```env
SERVICE_API_KEY=replace_with_a_long_random_secret
POSTGRES_PASSWORD=replace_with_a_strong_database_password
ALLOWED_ORIGINS=https://pinebioml.ncu.edu.tw
```

Generate a strong `SERVICE_API_KEY`:

```bash
openssl rand -hex 32
```

Important rules:

- Never commit `.env.production` to Git.
- Do not use `ALLOWED_ORIGINS=*` in production.
- If multiple frontend origins are needed, use a comma-separated list:

```env
ALLOWED_ORIGINS=https://pinebioml.ncu.edu.tw,https://www.pinebioml.ncu.edu.tw
```

---

## 3. Create The Docker Shared Network

Run this once on the server. If the network already exists, Docker will say so — that is fine.

```bash
docker network create pinebioml-shared-network
```

---

## 4. Build And Start The Stack

The build context is the repo root (`..` relative to the compose file), so run the compose command
from inside `pinebioml-report-service/`:

```bash
cd /opt/pinebioml/pinebioml-report-service

docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

This will:

1. Build the API image (installs system deps, pandoc, PineBioML, Python requirements)
2. Pull the Postgres and Ollama images
3. Start all four services: `db`, `ollama`, `ollama-init`, `api`
4. `ollama-init` pulls `deepseek-r1:14b`, `qwen2.5-coder:14b`, and `nomic-embed-text` — this takes
   a while on first run depending on server bandwidth

---

## 5. Monitor Startup

Check that all services come up healthy:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml ps
```

Expected state once ready:

```text
NAME            STATUS
db              running (healthy)
ollama          running (healthy)
ollama-init     exited (0)
api             running (healthy)
```

Follow the API logs:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml logs -f api
```

Follow Ollama model download progress:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml logs -f ollama-init
```

Health check from the server:

```bash
curl http://localhost:8001/health
```

Expected response: `{"status":"ok"}`

---

## 6. Reverse Proxy (Nginx)

Point the NCU DNS record to the server:

```text
pinebioml.ncu.edu.tw  ->  server public IP
```

Install Nginx and create a site config:

```bash
sudo nano /etc/nginx/sites-available/pinebioml
```

Paste:

```nginx
server {
    listen 80;
    server_name pinebioml.ncu.edu.tw;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name pinebioml.ncu.edu.tw;

    ssl_certificate     /etc/letsencrypt/live/pinebioml.ncu.edu.tw/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/pinebioml.ncu.edu.tw/privkey.pem;

    client_max_body_size 50m;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;

        proxy_buffering off;
        proxy_read_timeout 1200s;
        proxy_send_timeout 1200s;
    }
}
```

Enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/pinebioml /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 7. Smoke Test

Before giving the URL to users, run through this checklist:

1. `https://pinebioml.ncu.edu.tw/health` — should return `{"status":"ok"}`
2. `https://pinebioml.ncu.edu.tw/` — should redirect to the upload page
3. `https://pinebioml.ncu.edu.tw/Statistical_Analysis/upload` — upload page loads
4. Upload a small `.csv` test dataset, select a target column, run a job
5. ML results page loads: `/Statistical_Analysis/result/{uuid}/`
6. Plots load from `/media/...`
7. Report viewer loads: `/report/{report_id}/html`
8. PDF download works: `/report/{report_id}/download/pdf`
9. DOCX download works: `/report/{report_id}/download/docx`
10. Restart containers and confirm reports still load:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml restart
```

---

## 8. Updating To A New Version

```bash
cd /opt/pinebioml
git pull
cd pinebioml-report-service
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

Only the `api` image rebuilds if Postgres/Ollama images are unchanged.

---

## 9. Operational Commands

| Task | Command |
| ---- | ------- |
| View API logs | `docker compose --env-file .env.production -f docker-compose.prod.yml logs -f api` |
| View all logs | `docker compose --env-file .env.production -f docker-compose.prod.yml logs -f` |
| Restart all | `docker compose --env-file .env.production -f docker-compose.prod.yml restart` |
| Stop all | `docker compose --env-file .env.production -f docker-compose.prod.yml down` |
| Stop + wipe volumes | `docker compose --env-file .env.production -f docker-compose.prod.yml down -v` ⚠️ destroys data |
| Check Ollama models | `docker compose --env-file .env.production -f docker-compose.prod.yml exec ollama ollama list` |
| Open DB shell | `docker compose --env-file .env.production -f docker-compose.prod.yml exec db psql -U pinebioml -d pinebioml_db` |

---

## 10. LLM / Ollama Settings / GPU Acceleration

To ensure the LLM generation uses the GPU and avoids massive CPU spikes, you need to properly wire CUDA into the server.

### 10.1 Wiring CUDA on Linux Servers (Ubuntu/Debian)

If the NCU server does not have GPU pass-through configured for Docker yet, follow these steps:

1. **Install NVIDIA Drivers (if not already installed):**
   ```bash
   sudo apt update
   sudo apt install -y ubuntu-drivers-common
   sudo ubuntu-drivers autoinstall
   sudo reboot
   ```
   *After rebooting, verify your GPU is detected by running `nvidia-smi`.*

2. **Install NVIDIA Container Toolkit:**
   This bridges the host's CUDA drivers with Docker containers.
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

3. **Configure Docker to use NVIDIA runtime:**
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. **Verify GPU Passthrough:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

### 10.2 Wiring CUDA on Windows Laptops

Docker Desktop with the WSL2 backend automatically passes the GPU to Docker, so no extra drivers or toolkits are needed. Just ensure you have the latest NVIDIA Game Ready or Studio drivers installed.

---

Production `.env.production` configuration for GPU usage:

```env
LLM_MODEL=qwen3.5:9b
WRITER_DEPLOYMENT=gpu
CPU_DEPLOYMENT_WRITER_MODEL=qwen3.5:9b
VERIFIER_MODEL=qwen3.5:9b
EMBEDDING_MODEL=ibm-granite/granite-embedding-311m-multilingual-r2
VISION_MODEL=glm-ocr:latest
ENABLE_VISION_ANALYSIS=false
LLM_REQUEST_TIMEOUT_SECONDS=900
```

- Keep `ENABLE_VISION_ANALYSIS=false` unless a fast vision model is confirmed working.
- Disk usage: plan for 30+ GB for `ollama_data` volume.

---

## 11. Storage Volumes

| Volume | Contents |
| ------ | -------- |
| `app_storage` | Reports, exports, media plots, uploaded datasets |
| `postgres_data` | Postgres database |
| `ollama_data` | Ollama models |

Back up these volumes per NCU policy if data must survive server migration.
Do not store generated reports in the Git repository.

---

## 12. Known Follow-Ups

These are not blockers for a first deployment but should be planned:

- Add rate limiting at the reverse proxy.
- Add authentication if the service should not be publicly accessible.
- Move the background job queue from SQLite to Postgres or Redis for durability.
- Monitor disk usage for `app_storage` and `ollama_data`.
- Set up volume backups.
