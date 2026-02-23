# MythForge Video API — Phase 6.2

**Production-ready AI video generation stack on CPU-only VPS.**

MP3 → ElevenLabs (re-voice) → Whisper transcription → AI-generated video clips (Grok Imagine) → Ken Burns frames → FFmpeg → HD MP4

---

## Live Instance

| | |
|-|-|
| **VPS** | `ubuntu@vps-4d43058a` / `51.83.154.112` |
| **Health** | `curl http://51.83.154.112/health` |
| **Repo** | `https://github.com/andy-arrow/mythforge-api` |
| **API path** | `/opt/mythforge-api` on VPS |

---

## Current Pipeline (Phase 6.2)

```
Original MP3
  │
  ├─ [revoice.py step 1] POST /api/transcribe
  │     └─ faster-whisper (tiny, int8, CPU) → full script text + timestamps
  │
  ├─ [revoice.py step 2] POST kie.ai ElevenLabs TTS
  │     └─ voice: Bill (very deep) — default
  │     └─ model: elevenlabs/text-to-speech-multilingual-v2
  │     └─ polls until state=success → audio URL
  │
  ├─ [revoice.py step 3] Download ElevenLabs MP3
  │     └─ /tmp/hera_elevenlabs.mp3
  │
  └─ [revoice.py step 4] POST /api/render
        ├─ Whisper transcribe (again, on new audio)
        ├─ Phase 6: Kie.ai Grok Imagine T2I — 1 AI image per segment
        │     └─ cinematic prompts with Greek figure detection
        │     └─ 16:9 aspect ratio
        ├─ Phase 6.2: Kie.ai Grok Imagine I2V — AI video clip per image
        │     └─ duration: "6" or "10" seconds (Grok whitelist)
        │     └─ cinematic motion prompts (push, parallax, fog)
        │     └─ falls back to AI image if I2V fails
        ├─ DRAMATIC Ken Burns effect (12% zoom, 8 pan directions, smoothstep easing)
        │     └─ 4 keyframes/second for silky smooth animation
        │     └─ LANCZOS resampling
        └─ FFmpeg concat → output.mp4 (1280×720, H.264, AAC 192k, faststart)
```

---

## Stack

| Service | Image | Role |
|---------|-------|------|
| `api` | Python 3.12-slim | gunicorn + Flask + FFmpeg + Whisper + PIL |
| `nginx` | nginx:alpine | Reverse proxy, 1-hour timeout for AI video |

- Port **80** public (nginx). Port **8000** internal only.
- Gunicorn: 2 workers, **3600s timeout** (1 hour for AI video generation)
- nginx `/api/render`: **3600s** proxy timeout
- Shared bind mounts: `./exports`, `./models` (Whisper cache persists across deploys)
- Resources: 6 CPU / 20 GB RAM

---

## Deploy

```bash
# Full deploy (pull + rebuild + restart)
cd /opt/mythforge-api && git pull && \
  KIE_KEY=YOUR_KIE_KEY sudo -E docker compose up -d --build
```

---

## Primary Command

```bash
# Full pipeline: ElevenLabs re-voice + AI video generation
python3 revoice.py \
  --audio /opt/openclawworkspace/mythforge/hera_full_audio_combined.mp3 \
  --title "HERA — The Birth of War" \
  --kie-key YOUR_KIE_KEY \
  --api-url http://localhost

# With limited AI segments (faster test, ~5 min)
python3 revoice.py ... --max-ai-segs 5

# AI images only (no video, ~$0.50, fast)
python3 revoice.py ... --ai-visuals images

# Legacy paintings (free)
python3 revoice.py ... --ai-visuals none
```

---

## API Reference

### `POST /api/render`

| Field | Required | Description |
|-------|----------|-------------|
| `mp3` | yes | MP3 audio file |
| `title` | no | Episode label (max 60 chars) |
| `ai_visuals` | no | `"video"` (default), `"images"`, `"none"` |
| `kie_key` | no | Kie.ai API key (or set KIE_KEY env) |
| `max_ai_segs` | no | Limit AI generation (0 = all) |
| `bgm` | no | Background music file |
| `bgm_volume` | no | BGM level 0.0–1.0 (default 0.15) |

**Response:**
```json
{
  "success": true,
  "job_id": "9dbf9938",
  "url": "http://51.83.154.112/exports/9dbf9938/output.mp4",
  "subtitles_url": "http://51.83.154.112/exports/9dbf9938/subtitles.srt",
  "duration_s": 196.88,
  "segments": 51,
  "ai_images": 3,
  "ai_videos": 3,
  "phase": "6.2-ai-video"
}
```

### `POST /api/transcribe`
Returns full text + timestamped segments + SRT URL. No video generated.

### `POST /api/test-kie`
Tests Kie.ai connection by generating one image. Returns image URL.

### `GET /health`
Returns status, phase, kie_configured, whisper_ready.

---

## ElevenLabs Voices (Kie.ai whitelist only)

| Voice | Type | Notes |
|-------|------|-------|
| `Bill` | Male, very deep | **Default. Best for mythology** |
| `George` | Male, British deep | Authoritative |
| `Brian` | Male, American warm | Good narrator |
| `Daniel` | Male, British mid | Clear |
| `Callum` | Male, Scottish | Distinctive |

> ⚠️ Community voice IDs (e.g. Matthew Schmitz) do NOT work — Kie.ai whitelist only.

---

## Kie.ai API Notes

| API | Model | Valid params |
|-----|-------|-------------|
| Text-to-Image | `grok-imagine/text-to-image` | `aspect_ratio: "16:9"`, `n: 1` |
| Image-to-Video | `grok-imagine/image-to-video` | `duration: "6"` or `"10"` only, `resolution: "480p"/"720p"`, `mode: "fun"/"normal"/"spicy"` |
| ElevenLabs TTS | `elevenlabs/text-to-speech-multilingual-v2` | `voice`, `stability`, `similarity_boost`, `style`, `speed` |

> ⚠️ I2V does NOT accept `"5"` for duration — only `"6"` or `"10"`.  
> ⚠️ External image URLs cannot use `"spicy"` mode.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KIE_KEY` | `` | Kie.ai API key for AI visuals |
| `VPS_IP` | `51.83.154.112` | Public IP shown in URLs |
| `API_KEY` | `` | Optional auth (empty = open) |
| `FFMPEG_TIMEOUT` | `660` | FFmpeg encode timeout (s) |
| `JOB_TTL_HOURS` | `48` | Auto-delete jobs older than this |
| `WHISPER_MODEL` | `tiny` | tiny/base/small/medium/large-v3 |

---

## Phases Complete

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | MP3 → black background video | ✅ |
| 2 | AI dependencies installed | ✅ |
| 3 | Whisper transcription + styled caption frames | ✅ |
| 4 | Themed scenes (8 themes) + Ken Burns | ✅ |
| 5 | Public-domain painting backgrounds | ✅ |
| 6 | AI-generated images per segment (Grok T2I) | ✅ |
| 6.1 | Cinematic prompt engineering + voice tuning | ✅ |
| 6.2 | AI video clips per segment (Grok I2V) + dramatic Ken Burns | ✅ |

---

## Known Issues

- Wikimedia paintings: some 404/429 — falls back to gradient (non-issue since Phase 6 uses AI images)
- Kie.ai daily quota: AI video generation uses significant credits
- AI Videos blocked by quota → falls back to AI images automatically
