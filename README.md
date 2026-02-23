# MythForge Video API

**Phase 3 live.**  
MP3 → Whisper transcription → styled caption frames → HD video (1280×720).  
CPU-only VPS (8-core, 24 GB RAM).

---

## Pipeline

```
MP3 upload
  │
  ├─ ffprobe          probe duration
  ├─ faster-whisper   transcription → timestamped segments + SRT
  ├─ PIL              one 1280×720 dark-themed caption frame per segment
  └─ FFmpeg           concat frames + audio → output.mp4 (faststart)
```

Typical render time for a 4-min audio: **50–90 seconds**.

---

## Live instance

| | |
|-|-|
| **Health** | `curl http://$VPS_IP/health` |
| **Render** | `curl -X POST -F "mp3=@song.mp3" http://$VPS_IP/api/render` |
| **Video** | `http://$VPS_IP/exports/<job_id>/output.mp4` |
| **Subtitles** | `http://$VPS_IP/exports/<job_id>/subtitles.srt` |
| **Status** | `curl http://$VPS_IP/api/status/<job_id>` |

---

## Deploy (copy-paste)

```bash
# 1. Clone on VPS
cd /opt
git clone https://github.com/andy-arrow/mythforge-api.git mythforge-api
cd mythforge-api

# 2. Configure
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
./deploy.sh          # pulls latest, rebuilds, waits healthy, smoke-tests
```

---

## API reference

### `POST /api/render`

Upload an MP3, receive a video URL.

**Request:** `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `mp3` | yes | MP3 audio file |
| `title` | no | Episode label shown at top of frames (max 60 chars) |

**Auth:** `X-API-Key: <key>` header (if `API_KEY` is set in `.env`)

```bash
curl -X POST \
  -H "X-API-Key: YOUR_KEY" \
  -F "mp3=@hera_ep1.mp3" \
  -F "title=HERA — Episode 1" \
  http://$VPS_IP/api/render
```

**Response:**
```json
{
  "success": true,
  "job_id": "8c7436df",
  "url": "http://VPS_IP/exports/8c7436df/output.mp4",
  "subtitles_url": "http://VPS_IP/exports/8c7436df/subtitles.srt",
  "duration_s": 253.74,
  "segments": 87,
  "phase": "3-ai-pipeline",
  "message": "Video generated: Whisper transcription + styled caption frames."
}
```

### `GET /api/status/<job_id>`

```bash
curl http://$VPS_IP/api/status/8c7436df
```

### `GET /health`

Returns `{"status":"healthy","whisper_ready":true,...}` or 503 if degraded.

### `GET /exports/<job_id>/output.mp4`
### `GET /exports/<job_id>/subtitles.srt`

Served by nginx (MP4 cached 1h, SRT available for download).

---

## Environment variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VPS_IP` | `51.83.154.112` | Public IP shown in URLs |
| `API_KEY` | _(empty)_ | Optional auth key; empty = open |
| `FLASK_ENV` | `production` | `production` or `development` |
| `MAX_UPLOAD_MB` | `500` | Max MP3 upload size |
| `FFMPEG_TIMEOUT` | `660` | Max render time (seconds) |
| `JOB_TTL_HOURS` | `48` | Auto-delete jobs older than this |
| `WHISPER_MODEL` | `tiny` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `WHISPER_LANGUAGE` | `en` | ISO 639-1 code, or `auto` for detection |

---

## Stack

| Service | Image | Role |
|---------|-------|------|
| `api` | Python 3.12-slim | gunicorn + Flask + FFmpeg + Whisper + PIL |
| `nginx` | nginx:alpine | Reverse proxy, `/exports/` static serving |

- Port **80** public (nginx). Port **8000** internal only.
- Shared bind mounts: `./exports` (read-write API, read-only nginx), `./models` (Whisper cache).
- Resource limits: 6 CPU / 20 GB RAM.
- Log rotation: 10 MB × 3 files per service.

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | MP3 → black background video | ✅ Done |
| 2 | AI dependencies installed | ✅ Done |
| 3a | Whisper transcription → SRT | ✅ Done |
| 3b | PIL styled caption frames | ✅ Done |
| 3c | FFmpeg concat + faststart | ✅ Done |
| 4a | SDXL scene images (GPU) | ⏳ Next |
| 4b | Per-scene image backgrounds | ⏳ Soon |
| 4c | Orchestral background layer | ⏳ Later |
