# MythForge Video API

Production-ready Docker stack: **MP3 → HD video** (1280×720, synced audio).  
CPU-only VPS (8-core, 24 GB RAM). Phase 3-ready: Whisper + SDXL.

---

## Live instance

| | |
|-|-|
| **Health** | `curl http://$VPS_IP/health` |
| **Render** | `curl -X POST -F "mp3=@song.mp3" http://$VPS_IP/api/render` |
| **Video** | `http://$VPS_IP/exports/<job_id>/output.mp4` |
| **Status** | `curl http://$VPS_IP/api/status/<job_id>` |

---

## Deploy (copy-paste)

```bash
# 1. Clone on VPS
cd /opt
git clone https://github.com/andy-arrow/mythforge-api.git mythforge-api
cd mythforge-api

# 2. Configure (copy and edit)
cp .env.example .env
nano .env            # set VPS_IP and optionally API_KEY

# 3. Deploy + test
chmod +x deploy.sh
./deploy.sh
```

---

## Re-deploy after changes

```bash
cd /opt/mythforge-api
./deploy.sh          # pulls latest, rebuilds, smoke-tests
```

---

## API reference

### `POST /api/render`
Upload an MP3, receive a video URL.

**Request:** `multipart/form-data`, field `mp3`  
**Auth:** `X-API-Key: <key>` header (if `API_KEY` is set in `.env`)

```bash
curl -X POST \
  -H "X-API-Key: YOUR_KEY" \
  -F "mp3=@episode1.mp3" \
  http://$VPS_IP/api/render
```

**Response:**
```json
{
  "success": true,
  "job_id": "8c7436df",
  "url": "http://VPS_IP/exports/8c7436df/output.mp4",
  "duration_s": 312.4,
  "phase": "2-ai-installed",
  "message": "Video created. AI pipeline (Whisper + SDXL) coming next."
}
```

### `GET /api/status/<job_id>`
Check job state.

```bash
curl http://$VPS_IP/api/status/8c7436df
```

### `GET /health`
Returns `{"status":"healthy","ffmpeg":true,...}` or 503 if degraded.

### `GET /exports/<job_id>/output.mp4`
Download or stream the rendered video (served by nginx, cached 1h).

---

## Environment variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VPS_IP` | `51.83.154.112` | Public IP of VPS |
| `API_KEY` | _(empty)_ | Optional API key; empty = open |
| `FLASK_ENV` | `production` | `production` or `development` |
| `MAX_UPLOAD_MB` | `500` | Max MP3 upload size |
| `FFMPEG_TIMEOUT` | `600` | Max render time (seconds) |
| `JOB_TTL_HOURS` | `48` | Auto-delete jobs older than this |

---

## Stack

| Service | Image | Role |
|---------|-------|------|
| `api` | Python 3.12-slim | Flask API + FFmpeg |
| `nginx` | nginx:alpine | Reverse proxy, `/exports/` static serving |

- Port **80** public (nginx). Port **8000** internal only (not exposed to host).
- Shared bind mount: `./exports` → `/app/exports` (API) and `/exports:ro` (nginx).
- Resource limits: 6 CPU / 20 GB RAM for API.
- Log rotation: 10 MB × 3 files per service.

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | MP3 → black background video | ✅ Done |
| 2 | AI dependencies installed | ✅ Done |
| 3a | Whisper transcription | ⏳ Next |
| 3b | SDXL image generation | ⏳ Soon |
| 3c | Captions (burn subtitles) | ⏳ Soon |
| 3d | Orchestral background music | ⏳ Later |
