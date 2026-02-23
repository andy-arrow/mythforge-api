FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simple_api.py .

# Exports dir owned by appuser
RUN mkdir -p /app/exports && chown appuser:appuser /app/exports

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "simple_api.py"]
