FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY simple_api.py .

RUN mkdir -p /app/exports /app/models /app/assets/bg /app/assets/ai && \
    chown -R appuser:appuser /app/exports /app/models /app/assets

USER appuser

# Phase 6: Kie.ai API key (optional, enables AI-generated visuals)
ENV KIE_KEY=""

EXPOSE 8000

# start_period gives the Whisper model time to download on first boot
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 2 workers: one renders while the other handles health/status requests
# timeout 3600 (1 hour) for AI video generation which can be very slow
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "3600", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "simple_api:app"]
