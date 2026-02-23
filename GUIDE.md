# MythForge — Guide for Andreas

## Status

- ✅ Phase 1: MP3 → video
- ✅ Phase 2: AI deps installed
- ✅ Phase 3: Whisper transcription + styled caption frames + FFmpeg assembly

---

## Step 1 — Deploy (one command)

```bash
ssh ubuntu@vps-4d43058a.vps.ovh.net \
  "cd /opt/mythforge-api && git fetch origin && git reset --hard origin/main && chmod +x deploy.sh && ./deploy.sh"
```

What it does:
1. Pulls latest code (hard reset, no merge conflicts)
2. Rebuilds Docker image (installs DejaVu fonts, all AI deps)
3. Starts gunicorn (2 workers)
4. **Pre-downloads Whisper model** in background (~39 MB for `tiny`)
5. Runs smoke test: 5-second tone → video + SRT subtitles

**First deploy after Phase 3:** The build downloads all AI packages (~3 GB total). Allow 5 minutes.

---

## Step 2 — Render the Hera MP3

```bash
ssh ubuntu@vps-4d43058a.vps.ovh.net \
  "curl -s -X POST \
     -F 'mp3=@/opt/openclawworkspace/mythforge/hera_full_audio_combined.mp3' \
     -F 'title=HERA — The Birth of War' \
     http://localhost/api/render"
```

Response:
```json
{
  "success": true,
  "job_id": "abc12345",
  "url": "http://51.83.154.112/exports/abc12345/output.mp4",
  "subtitles_url": "http://51.83.154.112/exports/abc12345/subtitles.srt",
  "duration_s": 253.74,
  "segments": 87,
  "phase": "3-ai-pipeline"
}
```

Open both URLs in your browser:
- `output.mp4` → the full video with caption frames
- `subtitles.srt` → the Whisper transcript (download and review)

**Expected render time: ~50–90 seconds** for 4-minute audio.

---

## Step 3 — Check job status

```bash
curl http://51.83.154.112/api/status/abc12345
```

---

## Troubleshooting

| Problem | Command |
|---------|---------|
| Containers down | `ssh ubuntu@... "cd /opt/mythforge-api && ./deploy.sh"` |
| Check API logs | `ssh ubuntu@... "cd /opt/mythforge-api && sudo docker compose logs api --tail=30"` |
| Check nginx logs | `ssh ubuntu@... "cd /opt/mythforge-api && sudo docker compose logs nginx --tail=20"` |
| Find Hera MP3 | `ssh ubuntu@... "find /opt /home -name 'hera*.mp3' 2>/dev/null"` |
| Quick render test | `ssh ubuntu@... "curl -s -X POST -F 'mp3=@/tmp/test.mp3' http://localhost/api/render"` |
| Check Whisper ready | `curl http://51.83.154.112/health` → look for `"whisper_ready": true` |
| Force model download | `ssh ubuntu@... "sudo docker compose exec api python -c 'from simple_api import get_whisper_model; get_whisper_model()'"`|

---

## What Phase 3 output looks like

Each frame (1280×720):
```
┌─────────────────────────────────────────┐
│              HERA — Episode 1            │  ← gold label, 20px
│    ────────────────────────────────     │  ← thin gold separator
│                                          │
│                                          │
│     "She rose from the sea of chaos,    │  ← cream white 48px text
│      eldest daughter of Cronus,          │    word-wrapped, drop shadow
│      queen before she had a throne."     │
│                                          │
│                                          │
│    ────────────────────────────────     │  ← thin gold accent bar
│████████████                              │  ← progress bar (gold)
└─────────────────────────────────────────┘
```

---

## Phase 4 roadmap

| Step | What happens |
|------|-------------|
| 4a: SDXL (GPU) | Generate one image per scene — visual illustrations |
| 4b: Scene backgrounds | Replace dark background with AI-generated scene art |
| 4c: Background music | Mix orchestral layer under narration |

Say "go" to start Phase 4.
