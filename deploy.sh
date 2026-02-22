#!/bin/bash
set -e
cd /opt/mythforge-api
mkdir -p exports
chown ubuntu:ubuntu exports

# Ensure ffmpeg-python present (append only if missing)
grep -qx 'ffmpeg-python' requirements.txt 2>/dev/null || echo 'ffmpeg-python' >> requirements.txt

sudo docker compose down
sudo docker compose up -d --build --force-recreate

sleep 60  # Heavy deps

sudo docker compose ps
sudo docker compose logs api | tail -20
sudo docker compose logs nginx | tail -10

# Test (host must have ffmpeg, curl, jq)
ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -y test.mp3 -loglevel error
RESP=$(curl -s -X POST -F "mp3=@test.mp3" http://localhost:8000/api/render)
echo "$RESP" | jq .
JOB_ID=$(echo "$RESP" | jq -r '.job_id')
sleep 10
ls -la exports/
if [ -n "$JOB_ID" ] && [ "$JOB_ID" != "null" ]; then
  echo "Public: http://51.83.154.112/exports/${JOB_ID}/output.mp4"
fi
