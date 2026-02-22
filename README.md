# MythForge Video API

Production-ready Docker stack: **MP3 → HD video** (black 1280x720, synced audio). CPU-only, no GPU.

---

## Production (live)

**Base URL:** `http://51.83.154.112`

| Check | Command |
|-------|---------|
| Health | `curl http://51.83.154.112/health` |
| Render | `curl -s -X POST -F "mp3=@song.mp3" http://51.83.154.112/api/render \| jq -r '.url'` |
| Video | `http://51.83.154.112/exports/{job_id}/output.mp4` |

Upload any MP3 → get `job_id` → video ready in ~8s. Share the exports URL.

**Phase 2:** API returns `ai_status`, `phase: "2-ai-installed"` in health. Full whisper+diffusers integration in progress.

---

## Phase 2: AI upgrade (3-min build)

```bash
cd /opt/mythforge-api

# 1. Check current dependencies
docker exec $(docker compose ps -q api) pip list | grep -E "(whisper|diffusers|torch)"

# 2. Rebuild with AI deps (requirements.txt already includes faster-whisper, diffusers, etc.)
docker compose down
docker compose up -d --build

sleep 10
docker compose ps
docker compose logs api --tail=10

# 3. Verify AI packages loaded
chmod +x verify-ai.sh && ./verify-ai.sh
```

**Test with Hera MP3:**

```bash
curl -X POST -F "mp3=@/path/to/hera_full_audio_combined.mp3" http://51.83.154.112/api/render | jq
# Monitor: watch -n 5 "ls -la /opt/mythforge-api/exports/<JOB_ID>/"
```

**Success criteria:** `docker compose ps` healthy, `verify-ai.sh` prints Whisper/Diffusers/PyTorch versions, API accepts MP3 and returns `job_id` + full URL.

---

## Alfred deploy (copy-paste)

**One-time deploy + build + test from your machine (replace `YOUR_VPS_IP`):**

```bash
ssh root@YOUR_VPS_IP "cd /opt && rm -rf mythforge-api && git clone https://github.com/andy-arrow/mythforge-api.git mythforge-api && cd /opt/mythforge-api && mkdir -p exports && docker compose up -d --build && sleep 15 && docker compose ps && chmod +x deploy-and-test.sh && ./deploy-and-test.sh"
```

**Or on the VPS only:**

```bash
cd /opt && git clone https://github.com/andy-arrow/mythforge-api.git mythforge-api
cd mythforge-api
mkdir -p exports && chown ubuntu:ubuntu exports   # or chown to your app user; skip if root-only
docker compose up -d
```

Deploy in ~60 seconds. No terminal expertise required.

---

## Test (after deploy)

**Option A – run the script (creates tone, renders, prints public URL):**

```bash
cd /opt/mythforge-api && chmod +x deploy-and-test.sh && ./deploy-and-test.sh
```

**Option B – manual steps:**

```bash
# Create a 10s test tone inside the API container (writes to shared ./exports)
docker compose exec api ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -c:a libmp3lame -y /app/exports/test.mp3

# Render (must run inside container so /app/exports/test.mp3 is available)
docker compose exec api curl -s -X POST -F "mp3=@/app/exports/test.mp3" http://localhost:8000/api/render

# Response: {"job_id":"xxxxxxxx","url":"/exports/xxxxxxxx/output.mp4"}
# Public URL: http://YOUR_VPS_IP/exports/xxxxxxxx/output.mp4
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
- **Nginx**: Alpine, port 80; serves `/exports/` (read-only) and proxies `/api/` to API; `client_max_body_size 500M`
- **Exports**: Host `./exports` → API `/app/exports`, Nginx `/exports:ro` (shared bind mount)

CPU-optimized for 8-core, 24GB RAM VPS. Phase 2–ready (e.g. add faster-whisper in Dockerfile).
