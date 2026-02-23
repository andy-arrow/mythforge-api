# MythForge — Quick Reference Guide

## Current State: Phase 6.2

**What it does:** Takes the original Hera MP3, re-voices it with ElevenLabs (Bill — very deep), generates AI video clips per segment via Kie.ai Grok Imagine, and assembles a full HD video with dramatic Ken Burns.

---

## One-Command Deploy

```bash
cd /opt/mythforge-api && git pull && \
  KIE_KEY=ad6a8f65013f2534d7a4c8a809e714ce sudo -E docker compose up -d --build
```

---

## Primary Run Command

```bash
# Full pipeline (AI video + re-voice) — ~20-40 min for full video
python3 revoice.py \
  --audio /opt/openclawworkspace/mythforge/hera_full_audio_combined.mp3 \
  --title "HERA — The Birth of War" \
  --kie-key ad6a8f65013f2534d7a4c8a809e714ce \
  --api-url http://localhost
```

```bash
# Fast test — AI images only, 5 segments (~3 min)
python3 revoice.py \
  --audio /opt/openclawworkspace/mythforge/hera_full_audio_combined.mp3 \
  --title "HERA — The Birth of War" \
  --kie-key ad6a8f65013f2534d7a4c8a809e714ce \
  --api-url http://localhost \
  --ai-visuals images \
  --max-ai-segs 5
```

---

## Defaults (as of Phase 6.2)

| Setting | Value |
|---------|-------|
| Voice | `Bill` (very deep American) |
| AI mode | `video` (Grok I2V, falls back to images) |
| Ken Burns zoom | 12% dramatic push-in |
| Pan directions | 8 (varies per segment) |
| Resolution | 1280×720 |
| Video codec | libx264, AAC 192k |

---

## Voices Available (Kie.ai whitelist)

Best for mythology narration:
- `Bill` — very deep ← **default**
- `George` — British, authoritative
- `Brian` — American, warm
- `Daniel` — British, mid

```bash
--voice George   # change voice
```

---

## AI Visual Modes

```bash
--ai-visuals video    # AI video clips per segment (~$6 total) ← default
--ai-visuals images   # AI images per segment (~$0.50 total)
--ai-visuals none     # free, uses painting backgrounds
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `504 Gateway Time-out` | nginx/gunicorn now set to 3600s — redeploy |
| `voice is not within range` | Use whitelisted voices only (Bill, George, Brian, Daniel) |
| `duration not within range` | Grok I2V only accepts "6" or "10" — fixed in code |
| `daily limit exceeded` | Kie.ai quota hit — wait for reset or use new key |
| AI Videos: 0 | Usually quota — AI images still generated as fallback |
| Paintings 404/429 | Wikimedia rate limit — non-issue, AI images are used instead |

---

## Check Logs

```bash
# All recent activity
sudo docker logs mythforge-api-api-1 --tail 100

# Errors only
sudo docker logs mythforge-api-api-1 --tail 100 | grep -i "error\|fail\|warn"

# Video/AI generation
sudo docker logs mythforge-api-api-1 --tail 100 | grep -i "video\|AI\|kie"
```

---

## Recent Videos

| Job ID | Voice | AI | Notes |
|--------|-------|-----|-------|
| `9dbf9938` | Bill | 3 images | Latest — quota blocked videos |
| `dd9993ab` | George | 32 images | Phase 6, full AI images |
| `336a3ca3` | George | 0 (Phase 5) | Paintings + Ken Burns |

Latest: http://51.83.154.112/exports/9dbf9938/output.mp4
