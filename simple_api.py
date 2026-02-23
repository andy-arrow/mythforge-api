"""
MythForge Video API — Phase 3.

Pipeline:  MP3  →  Whisper transcription  →  PIL caption frames  →  FFmpeg MP4
           Each subtitle segment becomes one styled 1280×720 frame.
           All frames are stitched with the original audio via FFmpeg concat.

Docker: WORKDIR /app, exports at /app/exports, models at /app/models.
Run via gunicorn: gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 660 simple_api:app
"""
import logging
import os
import re
import shutil
import subprocess
import textwrap
import threading
import uuid
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPORTS_ROOT    = Path(os.environ.get("EXPORTS_ROOT",    "/app/exports"))
MODELS_DIR      = Path(os.environ.get("MODELS_DIR",      "/app/models"))
VPS_IP          = os.environ.get("VPS_IP",                "51.83.154.112")
API_KEY         = os.environ.get("API_KEY",               "")
MAX_UPLOAD_MB   = int(os.environ.get("MAX_UPLOAD_MB",     500))
FFMPEG_TIMEOUT  = int(os.environ.get("FFMPEG_TIMEOUT",    660))
JOB_TTL_HOURS   = int(os.environ.get("JOB_TTL_HOURS",    48))
FLASK_ENV       = os.environ.get("FLASK_ENV",             "production")
WHISPER_MODEL   = os.environ.get("WHISPER_MODEL",         "tiny")
_wl             = os.environ.get("WHISPER_LANGUAGE",      "en")
WHISPER_LANG    = None if _wl.lower() in ("", "auto", "none") else _wl
DEBUG           = FLASK_ENV == "development"

# DejaVu fonts — installed via fonts-dejavu-core in Dockerfile
FONT_REGULAR    = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD       = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
            return f(*args, **kwargs)
        key = request.headers.get("X-API-Key") or request.args.get("api_key", "")
        if key != API_KEY:
            logger.warning("Unauthorized from %s", request.remote_addr)
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def safe_filename(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-./]", "", filename).lstrip("/.")

def get_duration(mp3_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(mp3_path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    raw = result.stdout.strip()
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"ffprobe returned non-numeric duration: {raw!r}")

def cleanup_old_jobs() -> None:
    cutoff = datetime.now(tz=timezone.utc).timestamp() - JOB_TTL_HOURS * 3600
    removed = 0
    for job_dir in EXPORTS_ROOT.iterdir():
        if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
            shutil.rmtree(job_dir, ignore_errors=True)
            removed += 1
    if removed:
        logger.info("Cleaned %d expired job(s)", removed)

# ---------------------------------------------------------------------------
# Whisper — thread-safe lazy load, background pre-download at startup
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_lock  = threading.Lock()

def get_whisper_model():
    """Load Whisper model (once per process, thread-safe)."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                logger.info("Loading Whisper '%s' (int8, CPU) ...", WHISPER_MODEL)
                _whisper_model = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(MODELS_DIR),
                )
                logger.info("Whisper ready")
    return _whisper_model

def _prefetch_whisper() -> None:
    """Download model files at startup so first render isn't slow."""
    try:
        get_whisper_model()
    except Exception as exc:
        logger.warning("Whisper prefetch failed: %s", exc)

# Pre-download model in background when each gunicorn worker starts
threading.Thread(target=_prefetch_whisper, daemon=True, name="whisper-prefetch").start()

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def transcribe(mp3_path: Path) -> list[dict]:
    """
    Run faster-whisper on mp3_path.
    Returns list of {start, end, text} for every non-empty segment.
    """
    model = get_whisper_model()
    segments_iter, info = model.transcribe(
        str(mp3_path),
        beam_size=1,                        # fastest; plenty accurate for narration
        language=WHISPER_LANG,
        condition_on_previous_text=False,   # reduces hallucination on repeated text
        vad_filter=True,                    # skip silent regions
        vad_parameters={"min_silence_duration_ms": 400},
    )
    result = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            result.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    logger.info("Transcribed %d segment(s) (lang=%s, prob=%.2f)",
                len(result), info.language, info.language_probability)
    return result

def write_srt(segments: list[dict], path: Path) -> None:
    """Write an SRT subtitle file from segments."""
    def _ts(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")

    with open(path, "w", encoding="utf-8") as fh:
        for idx, seg in enumerate(segments, 1):
            fh.write(f"{idx}\n{_ts(seg['start'])} --> {_ts(seg['end'])}\n{seg['text']}\n\n")

# ---------------------------------------------------------------------------
# PIL frame generation
# ---------------------------------------------------------------------------
def _font(path: str, size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(path, size)
    except (IOError, OSError):
        logger.warning("Font not found (%s), using PIL default", path)
        return ImageFont.load_default()

def make_frame(text: str, title: str, frame_idx: int, total_frames: int,
               width: int = 1280, height: int = 720):
    """
    Render one 1280×720 caption frame:
      - Deep navy-black background with subtle centre-glow gradient
      - Gold 'MYTHFORGE' / episode title label at top with separator
      - Cream white body text, centred, drop-shadowed
      - Thin gold accent bar at bottom
      - Progress bar across the full bottom edge
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (6, 6, 14))
    draw = ImageDraw.Draw(img)

    # Subtle radial-ish vignette: darker at top/bottom, slightly lighter centre
    for y in range(height):
        t = 1.0 - abs(2.0 * y / height - 1.0)   # 0 at edges, 1 at centre
        v = int(6 + 20 * t)
        draw.line([(0, y), (width, y)], fill=(v // 3, v // 3, v))

    font_label  = _font(FONT_REGULAR, 20)
    font_sep    = _font(FONT_REGULAR, 14)
    font_main   = _font(FONT_BOLD,    48)

    # Episode / branding label (top centre)
    label = title.upper() if title else "MYTHFORGE"
    lb = draw.textbbox((0, 0), label, font=font_label)
    lw = lb[2] - lb[0]
    draw.text(((width - lw) // 2, 36), label, fill=(190, 162, 82), font=font_label)

    # Thin gold separator line beneath label
    draw.rectangle([(width // 4, 68), (3 * width // 4, 70)], fill=(160, 136, 60))

    if text:
        # Wrap long lines; 42 chars gives ~3–4 words per line at 48px
        wrapped = textwrap.fill(text, width=42)

        mb = draw.multiline_textbbox(
            (0, 0), wrapped, font=font_main, align="center", spacing=16
        )
        tw, th = mb[2] - mb[0], mb[3] - mb[1]
        x = (width - tw) // 2
        y = (height - th) // 2 + 8   # slight downward offset from geometric centre

        # Drop shadow (2px offset, very dark)
        draw.multiline_text(
            (x + 2, y + 2), wrapped,
            fill=(2, 2, 8), font=font_main, align="center", spacing=16,
        )
        # Main caption text (cream white)
        draw.multiline_text(
            (x, y), wrapped,
            fill=(240, 234, 214), font=font_main, align="center", spacing=16,
        )

    # Bottom accent bar
    draw.rectangle(
        [(width // 4, height - 44), (3 * width // 4, height - 41)],
        fill=(160, 136, 60),
    )

    # Progress bar (full-width, 3px, gold, scales with frame_idx)
    if total_frames > 0:
        bar_w = int(width * frame_idx / total_frames)
        draw.rectangle([(0, height - 4), (bar_w, height - 1)], fill=(190, 162, 82))

    return img

# ---------------------------------------------------------------------------
# Frame sequence builder
# ---------------------------------------------------------------------------
def build_frame_sequence(
    segments: list[dict],
    duration: float,
    title: str,
    frames_dir: Path,
) -> Path:
    """
    Generate one PNG per caption event (segment + silence gaps).
    Returns path to the FFmpeg concat list file.

    Gap strategy:
      - Any silence > 0.4s between segments becomes a blank frame (no text).
      - The last segment is extended to the end of the audio if needed.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Build ordered event list: (start, end, text)
    events: list[tuple[float, float, str]] = []
    prev_end = 0.0
    for seg in segments:
        if seg["start"] > prev_end + 0.1:
            events.append((prev_end, seg["start"], ""))     # silence gap
        events.append((seg["start"], seg["end"], seg["text"]))
        prev_end = seg["end"]
    if prev_end < duration - 0.1:
        events.append((prev_end, duration, ""))              # trailing silence

    if not events:
        events = [(0.0, duration, "")]                       # fallback: blank video

    total = len(events)
    lines: list[str] = []

    for idx, (start, end, text) in enumerate(events):
        dur = max(end - start, 0.04)                         # min 40 ms (FFmpeg floor)
        frame_path = frames_dir / f"{idx:05d}.png"
        img = make_frame(text, title, idx, total)
        img.save(str(frame_path), format="PNG", optimize=False)
        lines.append(f"file '{frame_path}'")
        lines.append(f"duration {dur:.4f}")

    # FFmpeg concat demuxer requires the last file entry to appear twice
    last_file = next(l for l in reversed(lines) if l.startswith("file "))
    lines.append(last_file)

    concat_path = frames_dir.parent / "concat.txt"
    concat_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Built %d frame(s) for %.1fs of audio", total, duration)
    return concat_path

# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------
def assemble_video(
    concat_path: Path,
    mp3_path: Path,
    output_path: Path,
) -> None:
    """
    Stitch frames + audio → output.mp4.

    -movflags +faststart  →  moov atom at start of file, enables browser streaming
    -crf 23               →  good quality/size trade-off for libx264
    -preset fast          →  reasonable encode speed on CPU
    -vf scale             →  guard: ensures output is exactly 1280×720
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-i", str(mp3_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=1280:720:flags=lanczos",
        "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ]
    logger.info("FFmpeg assembly start (timeout=%ss) ...", FFMPEG_TIMEOUT)
    subprocess.run(
        cmd, capture_output=True, text=True, check=True, timeout=FFMPEG_TIMEOUT,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Video ready: %.1f MB", size_mb)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check — confirms ffmpeg and whisper model are available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        ffmpeg_ok = True
    except Exception:
        ffmpeg_ok = False

    whisper_ready = _whisper_model is not None

    status = "healthy" if ffmpeg_ok else "degraded"
    code   = 200 if ffmpeg_ok else 503
    return jsonify({
        "status":        status,
        "phase":         "3-ai-pipeline",
        "ffmpeg":        ffmpeg_ok,
        "whisper_model": WHISPER_MODEL,
        "whisper_ready": whisper_ready,
        "exports":       str(EXPORTS_ROOT),
    }), code


@app.route("/api/render", methods=["POST"])
@require_api_key
def render():
    """
    Accept MP3 upload.
    Transcribes with Whisper, generates caption frames with PIL,
    assembles with FFmpeg.  Returns job_id, video URL, and subtitle URL.

    Optional form field:
      title   — episode/branding label shown at top of every frame (max 60 chars)
    """
    logger.info("RENDER START from %s", request.remote_addr)

    # ---- input validation ----
    if "mp3" not in request.files:
        return jsonify({"error": "No MP3 file (field name: mp3)"}), 400
    mp3_file = request.files["mp3"]
    if not mp3_file.filename:
        return jsonify({"error": "No file selected"}), 400

    allowed_mime = {"audio/mpeg", "audio/mp3", "audio/x-mpeg", "application/octet-stream"}
    if (mp3_file.content_type not in allowed_mime
            and not mp3_file.filename.lower().endswith(".mp3")):
        return jsonify({"error": "File must be an MP3"}), 415

    title = (request.form.get("title", "") or "").strip()[:60]

    # ---- setup job ----
    cleanup_old_jobs()

    job_id  = str(uuid.uuid4())[:8]
    job_dir = EXPORTS_ROOT / job_id
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("mkdir failed: %s", exc)
        return jsonify({"error": "Server error: could not create job directory"}), 500

    mp3_path    = job_dir / "input.mp3"
    srt_path    = job_dir / "subtitles.srt"
    frames_dir  = job_dir / "frames"
    output_path = job_dir / "output.mp4"
    segments    = []

    try:
        # ---- save upload ----
        mp3_file.save(str(mp3_path))
        logger.info("Saved %.1f KB → %s", mp3_path.stat().st_size / 1024, mp3_path)

        # ---- probe duration ----
        duration = get_duration(mp3_path)
        logger.info("Duration: %.1fs", duration)

        # ---- Phase 3a: transcribe ----
        logger.info("Phase 3a: Whisper transcription ...")
        segments = transcribe(mp3_path)
        write_srt(segments, srt_path)
        logger.info("SRT written: %d segment(s)", len(segments))

        # ---- Phase 3b: generate frames ----
        logger.info("Phase 3b: Generating %d caption frames ...", len(segments))
        concat_path = build_frame_sequence(segments, duration, title, frames_dir)

        # ---- Phase 3c: assemble video ----
        logger.info("Phase 3c: FFmpeg assembly ...")
        assemble_video(concat_path, mp3_path, output_path)

    except subprocess.TimeoutExpired:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.error("Render timed out (job %s)", job_id)
        return jsonify({"error": "Render timed out"}), 504

    except subprocess.CalledProcessError as exc:
        logger.error("FFmpeg failed (job %s): %s", job_id, exc.stderr[-1000:])
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": "Render failed — check server logs"}), 500

    except Exception as exc:
        logger.exception("Render error (job %s): %s", job_id, exc)
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": "Render failed — check server logs"}), 500

    finally:
        mp3_path.unlink(missing_ok=True)          # never keep the raw upload
        shutil.rmtree(frames_dir, ignore_errors=True)   # frames used; discard PNGs

    return jsonify({
        "success":       True,
        "job_id":        job_id,
        "url":           f"http://{VPS_IP}/exports/{job_id}/output.mp4",
        "subtitles_url": f"http://{VPS_IP}/exports/{job_id}/subtitles.srt",
        "duration_s":    round(duration, 2),
        "segments":      len(segments),
        "phase":         "3-ai-pipeline",
        "message":       "Video generated: Whisper transcription + styled caption frames.",
    })


@app.route("/api/status/<job_id>", methods=["GET"])
@require_api_key
def job_status(job_id: str):
    """Check whether a job has completed."""
    if not re.fullmatch(r"[0-9a-f]{8}", job_id):
        return jsonify({"error": "Invalid job_id"}), 400
    output = EXPORTS_ROOT / job_id / "output.mp4"
    if output.exists():
        srt = EXPORTS_ROOT / job_id / "subtitles.srt"
        return jsonify({
            "job_id":        job_id,
            "status":        "done",
            "url":           f"http://{VPS_IP}/exports/{job_id}/output.mp4",
            "subtitles_url": f"http://{VPS_IP}/exports/{job_id}/subtitles.srt"
                             if srt.exists() else None,
            "size_mb":       round(output.stat().st_size / (1024 * 1024), 2),
        })
    if (EXPORTS_ROOT / job_id).exists():
        return jsonify({"job_id": job_id, "status": "processing"})
    return jsonify({"job_id": job_id, "status": "not_found"}), 404


@app.route("/exports/<path:filename>", methods=["GET"])
def serve_exports(filename: str):
    """Serve export files (MP4, SRT); sanitised against path traversal."""
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
def internal_error(_):
    logger.exception("Unhandled exception")
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point (local dev only — production uses gunicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("MythForge API starting (env=%s debug=%s)", FLASK_ENV, DEBUG)
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=DEBUG,
        use_reloader=DEBUG,
        use_debugger=False,
    )
