#!/bin/bash
# Verify AI dependencies inside the API container (Phase 2).
# Run after: sudo docker compose up -d --build  (or docker compose if root)
set -e
cd "$(dirname "$0")"
[ -w /var/run/docker.sock ] 2>/dev/null && D="docker compose" || D="sudo docker compose"
[ -w /var/run/docker.sock ] 2>/dev/null && DOCKER="docker" || DOCKER="sudo docker"
API_CID=$($D ps -q api)
if [ -z "$API_CID" ]; then
  echo "API container not running. Run: $D up -d --build"
  exit 1
fi
echo "Checking AI dependencies in API container..."
$DOCKER exec "$API_CID" python -c "
import faster_whisper
import diffusers
import torch
print('✅ AI dependencies loaded')
print('Whisper:', faster_whisper.__version__)
print('Diffusers:', diffusers.__version__)
print('PyTorch:', torch.__version__)
"
echo ""
echo "Current pip packages:"
$DOCKER exec "$API_CID" pip list | grep -E "(whisper|diffusers|torch|accelerate|transformers)" || true
