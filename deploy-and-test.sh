#!/usr/bin/env bash
# Run from /opt/mythforge-api after docker compose up. Creates test MP3, calls render, prints public URL.
# Ensure host dir exists: mkdir -p exports && chown ubuntu:ubuntu exports  # optional chown if not root
set -e
cd "$(dirname "$0")"
mkdir -p exports
echo "Building test tone..."
docker exec "$(docker compose ps -q api)" ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -c:a libmp3lame -y /app/exports/test.mp3 -loglevel error
echo "Rendering..."
RESP=$(docker exec "$(docker compose ps -q api)" curl -s -X POST -F "mp3=@/app/exports/test.mp3" http://localhost:8000/api/render)
JOB_ID=$(echo "$RESP" | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
if [ -z "$JOB_ID" ]; then
  echo "Render failed or no job_id in response: $RESP"
  exit 1
fi
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s icanhazip.com 2>/dev/null || echo "VPS_IP")
echo "Public URL: http://${PUBLIC_IP}/exports/${JOB_ID}/output.mp4"
