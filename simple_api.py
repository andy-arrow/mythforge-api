"""
MythForge Video API — production-ready.
MP3 → HD video (black 1280x720, synced audio).
Docker WORKDIR /app; exports volume at /app/exports.
"""
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPORTS_ROOT   = Path(os.environ.get("EXPORTS_ROOT", "/app/exports"))
VPS_IP         = os.environ.get("VPS_IP", "51.83.154.112")
API_KEY        = os.environ.get("API_KEY", "")           # empty = no auth (dev)
MAX_UPLOAD_MB  = int(os.environ.get("MAX_UPLOAD_MB", 500))
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", 600))   # seconds
JOB_TTL_HOURS  = int(os.environ.get("JOB_TTL_HOURS", 48))     # auto-cleanup age
FLASK_ENV      = os.environ.get("FLASK_ENV", "production")
DEBUG          = FLASK_ENV == "development"

EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_handlers = [logging.StreamHandler()]
log_file = os.environ.get("LOG_FILE", "")
if log_file:
    log_handlers.append(
        RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=3)
    )
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("mythforge")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            return f(*args, **kwargs)                # auth disabled in dev
        key = request.headers.get("X-API-Key") or request.args.get("api_key", "")
        if key != API_KEY:
            logger.warning("Unauthorized request from %s", request.remote_addr)
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_filename(filename: str) -> str:
    """Remove path traversal sequences."""
    return re.sub(r"[^a-zA-Z0-9_\-./]", "", filename).lstrip("/.")

def get_duration(mp3_path: Path) -> float:
    """Probe audio duration with ffprobe; raise on failure."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(mp3_path),
        ],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(result.stdout.strip())

def cleanup_old_jobs():
    """Remove job dirs older than JOB_TTL_HOURS to prevent disk exhaustion."""
    cutoff = datetime.now(tz=timezone.utc).timestamp() - JOB_TTL_HOURS * 3600
    removed = 0
    for job_dir in EXPORTS_ROOT.iterdir():
        if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
            shutil.rmtree(job_dir, ignore_errors=True)
            removed += 1
    if removed:
        logger.info("Cleaned up %d expired job(s)", removed)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check — verifies ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        ffmpeg_ok = True
    except Exception:
        ffmpeg_ok = False
    status = "healthy" if ffmpeg_ok else "degraded"
    code   = 200 if ffmpeg_ok else 503
    return jsonify({
        "status":   status,
        "phase":    "2-ai-installed",
        "ffmpeg":   ffmpeg_ok,
        "exports":  str(EXPORTS_ROOT),
    }), code


@app.route("/api/render", methods=["POST"])
@require_api_key
def render():
    """Accept MP3 upload; return job_id and full public URL of output.mp4."""
    logger.info("RENDER START from %s", request.remote_addr)

    # --- input validation ---
    if "mp3" not in request.files:
        return jsonify({"error": "No MP3 file (field name: mp3)"}), 400
    mp3_file = request.files["mp3"]
    if not mp3_file.filename:
        return jsonify({"error": "No file selected"}), 400

    # Basic MIME / extension check
    allowed = {"audio/mpeg", "audio/mp3", "audio/x-mpeg", "application/octet-stream"}
    if mp3_file.content_type not in allowed and not mp3_file.filename.lower().endswith(".mp3"):
        return jsonify({"error": "File must be an MP3"}), 415

    # --- setup job ---
    cleanup_old_jobs()

    job_id  = str(uuid.uuid4())[:8]
    job_dir = EXPORTS_ROOT / job_id
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Failed to create job dir: %s", e)
        return jsonify({"error": "Server error: could not create job directory"}), 500

    mp3_path    = job_dir / "input.mp3"
    output_path = job_dir / "output.mp4"

    # --- save upload ---
    try:
        mp3_file.save(str(mp3_path))
        logger.info("Saved upload: %s (%.1f KB)", mp3_path, mp3_path.stat().st_size / 1024)
    except OSError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.error("Failed to save upload: %s", e)
        return jsonify({"error": "Server error: could not save file"}), 500

    # --- probe duration ---
    try:
        duration = get_duration(mp3_path)
        logger.info("Audio duration: %.1fs", duration)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.error("Duration probe failed: %s", e)
        return jsonify({"error": f"Could not read MP3 duration: {e}"}), 422

    # --- render video ---
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=1280x720:r=30:d={duration}",
        "-i", str(mp3_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    logger.info("Running FFmpeg (timeout %ss)", FFMPEG_TIMEOUT)
    try:
        subprocess.run(
            cmd, capture_output=True, text=True,
            check=True, timeout=FFMPEG_TIMEOUT,
        )
        logger.info("Video ready: %s (%.1f MB)",
                    output_path, output_path.stat().st_size / (1024 * 1024))
    except subprocess.TimeoutExpired:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.error("FFmpeg timed out after %ss", FFMPEG_TIMEOUT)
        return jsonify({"error": "Render timed out"}), 504
    except subprocess.CalledProcessError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.error("FFmpeg failed: %s", e.stderr[-500:])
        return jsonify({"error": f"FFmpeg failed: {e.stderr[-500:]}"}), 500

    return jsonify({
        "success":    True,
        "job_id":     job_id,
        "url":        f"http://{VPS_IP}/exports/{job_id}/output.mp4",
        "duration_s": round(duration, 2),
        "phase":      "2-ai-installed",
        "message":    "Video created. AI pipeline (Whisper + SDXL) coming next.",
    })


@app.route("/api/status/<job_id>", methods=["GET"])
@require_api_key
def job_status(job_id: str):
    """Check whether a job's output file exists."""
    # Sanitize job_id (only hex chars expected from uuid4[:8])
    if not re.fullmatch(r"[0-9a-f]{8}", job_id):
        return jsonify({"error": "Invalid job_id"}), 400
    output = EXPORTS_ROOT / job_id / "output.mp4"
    if output.exists():
        return jsonify({
            "job_id": job_id,
            "status": "done",
            "url":    f"http://{VPS_IP}/exports/{job_id}/output.mp4",
            "size_mb": round(output.stat().st_size / (1024 * 1024), 2),
        })
    job_dir = EXPORTS_ROOT / job_id
    if job_dir.exists():
        return jsonify({"job_id": job_id, "status": "processing"})
    return jsonify({"job_id": job_id, "status": "not_found"}), 404


@app.route("/exports/<path:filename>", methods=["GET"])
def serve_exports(filename: str):
    """Serve a rendered file; sanitised to prevent path traversal."""
    safe = safe_filename(filename)
    if safe != filename or ".." in safe:
        return jsonify({"error": "Invalid path"}), 400
    return send_from_directory(str(EXPORTS_ROOT), safe)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large (max {MAX_UPLOAD_MB} MB)"}), 413

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled exception")
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("MythForge API starting (env=%s debug=%s)", FLASK_ENV, DEBUG)
    app.run(host="0.0.0.0", port=8000, debug=DEBUG)
