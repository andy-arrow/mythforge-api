#!/bin/bash
# Phase 2: pull latest, check deps, rebuild with AI, verify, test.
# Run as ubuntu: use sudo for docker (or add ubuntu to docker group).
set -e
cd /opt/mythforge-api
D="docker compose"
DOCKER="docker"
[ -w /var/run/docker.sock ] 2>/dev/null || { D="sudo docker compose"; DOCKER="sudo docker"; }

git pull

echo "=== 1. Current dependencies ==="
$D ps -q api >/dev/null 2>&1 && $DOCKER exec $($D ps -q api) pip list | grep -E "(whisper|diffusers|torch)" || echo "(containers down or none yet)"

echo ""
echo "=== 2. Rebuild with AI deps (~3 min) ==="
$D down
$D up -d --build

sleep 10
$D ps
$D logs api --tail=10

echo ""
echo "=== 3. Verify AI ==="
chmod +x verify-ai.sh 2>/dev/null || true
./verify-ai.sh 2>/dev/null || $DOCKER exec $($D ps -q api) python -c "
import faster_whisper
import diffusers
import torch
print('✅ AI dependencies loaded')
print('Whisper:', faster_whisper.__version__)
print('Diffusers:', diffusers.__version__)
print('PyTorch:', torch.__version__)
"

echo ""
echo "=== 4. Test health & render ==="
curl -s http://51.83.154.112/health | jq .
echo ""
echo "Upload an MP3 to test render:"
echo "  curl -s -X POST -F \"mp3=@/path/to/your.mp3\" http://51.83.154.112/api/render | jq"
