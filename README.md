# MythForge Video API

Production-ready Docker stack: **MP3 → HD video** (black 1280x720, synced audio). CPU-only, no GPU.

---

## Alfred deploy (copy-paste)

```bash
cd /opt && git clone https://github.com/andy-arrow/mythforge-api.git mythforge-api
cd mythforge-api && docker compose up -d
```

Deploy in ~60 seconds. No terminal expertise required.

---

## Test (after deploy)

```bash
# Create a 10s test tone (inside API container)
docker compose exec api ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -c:a libmp3lame -y /tmp/test.mp3

# Render (from host; or use container path /tmp/test.mp3 from inside)
curl -X POST -F "mp3=@/tmp/test.mp3" http://localhost:8000/api/render

# Response: {"job_id":"xxxxxxxx","url":"/exports/xxxxxxxx/output.mp4"}
# Download result (replace JOB_ID)
curl -o output.mp4 "http://localhost/JOB_ID/output.mp4"
# Or: http://localhost/exports/JOB_ID/output.mp4
```

---

## Endpoints

| URL | Method | Description |
|-----|--------|-------------|
| `http://VPS_IP/health` | GET | Health check (via API) |
| `http://VPS_IP/api/render` | POST | Upload `mp3` file → returns `job_id` and `url` |
| `http://VPS_IP/exports/<job_id>/output.mp4` | GET | Download rendered video |

---

## Stack

- **API**: Flask + ffmpeg-python, Python 3.12-slim, port 8000
- **Nginx**: Alpine, port 80; serves `/exports/` and proxies `/api/` to API
- **Volume**: `exports` persistent for job outputs

CPU-optimized for 8-core, 24GB RAM VPS. Phase 2–ready (e.g. add faster-whisper in Dockerfile).
