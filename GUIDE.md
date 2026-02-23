# MYTHFORGE — Simple Step-by-Step Guide for Andreas

## Final Goal
A Hera Episode 1 video with:
1. Your narration (MP3)
2. AI-generated images of Hera *(Phase 3)*
3. Captions from the transcript *(Phase 3)*
4. URL: `http://51.83.154.112/exports/<JOB_ID>/output.mp4`

---

## What's done
- ✅ Phase 1: Basic API (MP3 → black background video)
- ✅ Phase 2: AI software installed (Whisper, SDXL)
- ❌ Phase 3: AI pipeline not wired yet

---

## STEP 1 — Log in and check the API

```bash
ssh ubuntu@vps-4d43058a.vps.ovh.net
cd /opt/mythforge-api
sudo docker compose ps
```

Expected output:
```
NAME                    STATUS
mythforge-api-api-1     Up (healthy)   0.0.0.0:8000->8000/tcp
mythforge-api-nginx-1   Up             0.0.0.0:80->80/tcp
```

---

## STEP 2 — Pull latest code and restart

```bash
cd /opt/mythforge-api
git pull
sudo docker compose restart api
sleep 10
curl http://localhost:8000/health
```

Expected:
```json
{"status":"healthy","phase":"2-ai-installed"}
```

If containers are down:
```bash
sudo docker compose down
sudo docker compose up -d --build
```

---

## STEP 3 — Test with Hera MP3

```bash
# Go to where the MP3 is
cd /home/node/.openclaw/workspace/mythforge

# Upload to API
curl -X POST -F "mp3=@hera_full_audio_combined.mp3" \
  http://51.83.154.112/api/render

# You'll get:
# {
#   "success": true,
#   "job_id": "abc12345",
#   "url": "http://51.83.154.112/exports/abc12345/output.mp4"
# }
```

Open the URL in any browser to watch the video.

---

## Troubleshooting

### "permission denied" on docker
```bash
sudo usermod -aG docker ubuntu   # one-time fix; log out and back in
# or just prefix every docker command with: sudo
```

### No MP3 file found
```bash
find /home /opt -name "hera*.mp3" 2>/dev/null
```

### API not responding
```bash
sudo docker compose logs api --tail=30
sudo docker compose restart api
```

### Test with a tiny file first
```bash
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -c:a libmp3lame -y /tmp/test.mp3
curl -X POST -F "mp3=@/tmp/test.mp3" http://localhost:8000/api/render
```

---

## What to report back
1. `sudo docker compose ps` output
2. `curl http://localhost:8000/health` output
3. Job ID from Step 3 and whether the video plays
4. Any errors

---

## Roadmap

| Phase | What | When |
|-------|------|------|
| ✅ 1 | MP3 → video | Done |
| ✅ 2 | AI software installed | Done |
| ⏳ 3a | Whisper transcription (audio → text) | Next |
| ⏳ 3b | SDXL images (Hera visuals) | After 3a |
| ⏳ 3c | Captions on screen | After 3b |
| ⏳ 3d | Orchestral background music | After 3c |
