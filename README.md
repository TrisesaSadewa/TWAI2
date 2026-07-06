# TWAI2 — PineBioML Report Service

An AI-powered biomedical machine learning report generation service, built for NCU deployment.

The repository contains two main components:

- **[PineBioML/](PineBioML/)** — Python ML package (feature selection, classification, regression, survival analysis)
- **[pinebioml-report-service/](pinebioml-report-service/)** — FastAPI service that runs ML jobs, generates narrative reports via LLM, and exports PDF/DOCX

---

## Deployment (NCU Production)

### Prerequisites

The server must have:

- Git
- Docker Engine 20.10+
- Docker Compose plugin (`docker compose`, not legacy `docker-compose`)
- NVIDIA Container Toolkit (required for Docker GPU passthrough on Linux)
- At least 50 GB free disk space (Ollama models are large)
- Outbound internet access (to pull Docker images and Ollama models on first boot)

---

### 1. Clone the Repository

```bash
git clone https://github.com/TrisesaSadewa/TWAI2.git /opt/pinebioml
cd /opt/pinebioml
```

Repository layout after cloning:

```
/opt/pinebioml/
  PineBioML/                  Python package installed into the container
  pinebioml-report-service/   Service — Dockerfile, compose files, source code
```

---

### 2. Create the Production Environment File

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

Rules:
- Never commit `.env.production` to Git.
- Do not use `ALLOWED_ORIGINS=*` in production.
- For multiple frontend origins, use a comma-separated list.

---

### 3. Create the Docker Shared Network

Run once on the server. If the network already exists, Docker will say so — that is fine.

```bash
docker network create pinebioml-shared-network
```

---

### 4. Build and Start the Stack

```bash
cd /opt/pinebioml/pinebioml-report-service
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

This will:
1. Build the API image (installs system deps, pandoc, PineBioML, Python requirements)
2. Pull the Postgres and Ollama images
3. Start all four services: `db`, `ollama`, `ollama-init`, `api`
4. `ollama-init` pulls `deepseek-r1:14b`, `qwen2.5-coder:14b`, and `nomic-embed-text` — takes a while on first run

---

### 5. Monitor Startup

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml ps
```

Expected state once ready:

```
NAME            STATUS
db              running (healthy)
ollama          running (healthy)
ollama-init     exited (0)
api             running (healthy)
```

Follow logs:

```bash
# API logs
docker compose --env-file .env.production -f docker-compose.prod.yml logs -f api

# Ollama model download progress
docker compose --env-file .env.production -f docker-compose.prod.yml logs -f ollama-init
```

Health check:

```bash
curl http://localhost:8001/health
# Expected: {"status":"ok"}
```

---

### 6. Reverse Proxy (Nginx)

Point the NCU DNS record to the server's public IP, then install Nginx:

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

### 7. Smoke Test

1. `https://pinebioml.ncu.edu.tw/health` — returns `{"status":"ok"}`
2. `https://pinebioml.ncu.edu.tw/` — redirects to the upload page
3. `https://pinebioml.ncu.edu.tw/Statistical_Analysis/upload` — upload page loads
4. Upload a small `.csv` test dataset, select a target column, run a job
5. ML results page loads: `/Statistical_Analysis/result/{uuid}/`
6. Plots load from `/media/...`
7. Report viewer loads: `/report/{report_id}/html`
8. PDF download works: `/report/{report_id}/download/pdf`
9. DOCX download works: `/report/{report_id}/download/docx`
10. Restart and confirm reports still load:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml restart
```

---

### 8. Updating to a New Version

```bash
cd /opt/pinebioml
git pull
cd pinebioml-report-service
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

Only the `api` image rebuilds if Postgres/Ollama images are unchanged.

---

### Operational Commands

| Task | Command |
|------|---------|
| View API logs | `docker compose --env-file .env.production -f docker-compose.prod.yml logs -f api` |
| View all logs | `docker compose --env-file .env.production -f docker-compose.prod.yml logs -f` |
| Restart all | `docker compose --env-file .env.production -f docker-compose.prod.yml restart` |
| Stop all | `docker compose --env-file .env.production -f docker-compose.prod.yml down` |
| Stop + wipe volumes | `docker compose --env-file .env.production -f docker-compose.prod.yml down -v` ⚠️ destroys data |
| Check Ollama models | `docker compose --env-file .env.production -f docker-compose.prod.yml exec ollama ollama list` |
| Open DB shell | `docker compose --env-file .env.production -f docker-compose.prod.yml exec db psql -U pinebioml -d pinebioml_db` |

---

### Storage Volumes

| Volume | Contents |
|--------|----------|
| `app_storage` | Reports, exports, media plots, uploaded datasets |
| `postgres_data` | Postgres database |
| `ollama_data` | Ollama models (~30 GB) |

Back up these volumes per NCU policy if data must survive server migration.

---

For the full deployment reference including LLM/Ollama configuration options, see [pinebioml-report-service/DEPLOY_NCU.md](pinebioml-report-service/DEPLOY_NCU.md).
