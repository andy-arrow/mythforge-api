#!/bin/bash
# MythForge — single deploy + test script.
# Works as root (Docker directly) or as ubuntu user (uses sudo).
# Usage:  ./deploy.sh [--no-test]
set -euo pipefail

# ---- detect sudo need ----
if [ -w /var/run/docker.sock ] 2>/dev/null; then
  DC="docker compose"
  DOCKER="docker"
else
  DC="sudo docker compose"
  DOCKER="sudo docker"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== MythForge deploy ==="
echo "DIR: $SCRIPT_DIR"
echo "DC:  $DC"

# ---- pull latest code ----
git pull

# ---- prepare exports dir ----
mkdir -p exports
# Only chown if running as root; ubuntu owns it via bind mount otherwise
[ "$(id -u)" -eq 0 ] && chown -R 1000:1000 exports 2>/dev/null || true

# ---- load .env if present ----
[ -f .env ] && set -a && . .env && set +a && echo "Loaded .env"

# ---- rebuild + start ----
$DC down --remove-orphans
$DC up -d --build

echo ""
echo "=== Waiting for health (up to 60s) ==="
for i in $(seq 1 12); do
  sleep 5
  STATUS=$(curl -s http://localhost:8000/health 2>/dev/null | grep -o '"healthy"' || true)
  if [ "$STATUS" = '"healthy"' ]; then
    echo "API healthy after $((i * 5))s"
    break
  fi
  echo "  ... waiting ($((i * 5))s)"
done

echo ""
$DC ps
echo ""
$DC logs api --tail=8

# ---- optional test ----
if [ "${1:-}" = "--no-test" ]; then
  echo "Skipping test (--no-test)"
  exit 0
fi

echo ""
echo "=== Smoke test ==="

# Check ffmpeg on host, else run inside container
if command -v ffmpeg &>/dev/null; then
  ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -c:a libmp3lame -y /tmp/mythforge_test.mp3 -loglevel error
  MP3_SRC="/tmp/mythforge_test.mp3"
else
  $DOCKER exec "$($DC ps -q api)" ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -c:a libmp3lame -y /app/exports/test.mp3 -loglevel error
  MP3_SRC="@/app/exports/test.mp3"
  # Run curl inside container using container path
  RESP=$($DOCKER exec "$($DC ps -q api)" curl -s -X POST -F "mp3=@/app/exports/test.mp3" http://localhost:8000/api/render)
  echo "$RESP"
  JOB_ID=$(echo "$RESP" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
  [ -n "$JOB_ID" ] && [ "$JOB_ID" != "null" ] && \
    echo "Public: http://${VPS_IP:-51.83.154.112}/exports/${JOB_ID}/output.mp4"
  exit 0
fi

RESP=$(curl -s -X POST -F "mp3=@${MP3_SRC}" http://localhost:8000/api/render)
echo "$RESP"
JOB_ID=$(echo "$RESP" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
[ -n "$JOB_ID" ] && [ "$JOB_ID" != "null" ] && \
  echo "Public: http://${VPS_IP:-51.83.154.112}/exports/${JOB_ID}/output.mp4"

echo ""
echo "=== Deploy complete ==="
