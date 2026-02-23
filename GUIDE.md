# MythForge — Guide for Andreas

## Goal
Hera Episode 1: narration audio + AI-generated visuals + captions.  
URL: `http://VPS_IP/exports/<JOB_ID>/output.mp4`

---

## Status
- ✅ Phase 1: API working — MP3 → video
- ✅ Phase 2: AI dependencies installed (Whisper, SDXL)
- ⏳ Phase 3: AI pipeline — next

---

## Step 1 — Deploy (one command)

Open Terminal on your Mac and paste:

```bash
ssh ubuntu@vps-4d43058a.vps.ovh.net \
  "cd /opt/mythforge-api && git pull && chmod +x deploy.sh && ./deploy.sh"
```

Done. It pulls latest code, rebuilds, waits for healthy, smoke-tests.

---

## Step 2 — Render the Hera MP3

```bash
ssh ubuntu@vps-4d43058a.vps.ovh.net \
  "curl -s -X POST \
     -F 'mp3=@/opt/openclawworkspace/mythforge/hera_full_audio_combined.mp3' \
     http://localhost:8000/api/render"
```

Response:
```json
{
  "success": true,
  "job_id": "8c7436df",
  "url": "http://51.83.154.112/exports/8c7436df/output.mp4"
}
```

Open the URL in your browser to watch.

---

## Step 3 — Check job status

```bash
curl http://51.83.154.112/api/status/8c7436df
```

---

## Troubleshooting

| Problem | Command |
|---------|---------|
| Containers down | `ssh ubuntu@... "cd /opt/mythforge-api && ./deploy.sh"` |
| Check logs | `ssh ubuntu@... "cd /opt/mythforge-api && sudo docker compose logs api --tail=30"` |
| Find Hera MP3 | `ssh ubuntu@... "find /opt /home -name 'hera*.mp3' 2>/dev/null"` |
| Test with 5s tone | `ssh ubuntu@... "curl -s -X POST -F 'mp3=@/tmp/test.mp3' http://localhost:8000/api/render"` |

---

## Phase 3 roadmap

| Step | What happens |
|------|-------------|
| 3a: Whisper | Audio → timestamped transcript |
| 3b: SDXL | Transcript → Hera images per scene |
| 3c: Captions | Subtitles burned onto video |
| 3d: Music | Orchestral background layer |

Say "go" to start Phase 3.
