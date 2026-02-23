"""
MythForge Video API — Phase 5.

Pipeline:
  MP3  →  Whisper transcription
       →  keyword-based scene-theme detection per segment (8 themes)
       →  real public-domain painting backgrounds (Wikimedia Commons) + Ken Burns
       →  optional background music mixing (bgm field)
       →  FFmpeg concat → MP4

New in Phase 5:
  - Real painting backgrounds: one public-domain masterwork per theme, downloaded
    on first use, cached to /app/assets/bg/, graceful gradient fallback on failure
  - Ken Burns now zooms the actual painting (not a gradient), looking cinematic
  - Dark vignette overlay ensures caption text is always readable over any image
  - New POST /api/transcribe endpoint: transcription only, no video rendering

Docker: WORKDIR /app, exports at /app/exports, models at /app/models.
Run via gunicorn: gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 720 simple_api:app
"""
import logging
import math
import os
import re
import shutil
import subprocess
import textwrap
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPORTS_ROOT   = Path(os.environ.get("EXPORTS_ROOT",   "/app/exports"))
MODELS_DIR     = Path(os.environ.get("MODELS_DIR",     "/app/models"))
ASSETS_DIR     = Path(os.environ.get("ASSETS_DIR",     "/app/assets"))
VPS_IP         = os.environ.get("VPS_IP",               "51.83.154.112")
API_KEY        = os.environ.get("API_KEY",              "")
MAX_UPLOAD_MB  = int(os.environ.get("MAX_UPLOAD_MB",    500))
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT",   660))
JOB_TTL_HOURS  = int(os.environ.get("JOB_TTL_HOURS",   48))
FLASK_ENV      = os.environ.get("FLASK_ENV",            "production")
WHISPER_MODEL  = os.environ.get("WHISPER_MODEL",        "tiny")
_wl            = os.environ.get("WHISPER_LANGUAGE",     "en")
WHISPER_LANG   = None if _wl.lower() in ("", "auto", "none") else _wl
DEBUG          = FLASK_ENV == "development"

# DejaVu fonts — installed via fonts-dejavu-core in Dockerfile
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
try:
    (ASSETS_DIR / "bg").mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # created by Dockerfile; tolerate read-only mounts gracefully

# ---------------------------------------------------------------------------
# Scene themes
# ---------------------------------------------------------------------------
SCENE_THEMES: dict[str, dict] = {
    "war": {
        "bg_top": (12, 2,  2),  "bg_bot": (30, 7,  5),
        "accent": (200, 50, 40), "title": (220, 160, 60), "text": (245, 230, 218),
    },
    "heaven": {
        "bg_top": (2,  4,  18), "bg_bot": (8,  15, 52),
        "accent": (200, 185, 90), "title": (240, 220, 130), "text": (248, 244, 230),
    },
    "sea": {
        "bg_top": (2,  8,  20), "bg_bot": (4,  22, 48),
        "accent": (40, 150, 190), "title": (100, 210, 230), "text": (228, 244, 248),
    },
    "death": {
        "bg_top": (4,  2,  8),  "bg_bot": (12, 6,  24),
        "accent": (90, 45, 130), "title": (170, 125, 210), "text": (230, 220, 240),
    },
    "love": {
        "bg_top": (12, 2,  8),  "bg_bot": (28, 7,  20),
        "accent": (200, 85, 105), "title": (230, 145, 165), "text": (248, 230, 236),
    },
    "fire": {
        "bg_top": (12, 4,  2),  "bg_bot": (34, 12, 2),
        "accent": (230, 100, 20), "title": (245, 165, 40), "text": (250, 235, 215),
    },
    "earth": {
        "bg_top": (4,  8,  2),  "bg_bot": (10, 24, 6),
        "accent": (85, 145, 60), "title": (145, 185, 105), "text": (235, 245, 228),
    },
    "default": {
        "bg_top": (2,  2,  8),  "bg_bot": (6,  6,  26),
        "accent": (160, 136, 60), "title": (190, 162, 82), "text": (240, 234, 214),
    },
}

THEME_KEYWORDS: dict[str, set[str]] = {
    "war":    {"war","battle","fight","sword","blood","rage","fury","weapon","army","kill",
               "slay","conquer","clash","combat","warrior","storm","thunder","shield","spear"},
    "heaven": {"heaven","olympus","divine","sacred","holy","olympian","throne","immortal",
               "eternal","sky","cloud","sun","light","glory","god","goddess","mount"},
    "sea":    {"sea","ocean","water","wave","tide","deep","shore","sail","ship","poseidon",
               "river","flood","stream","flow","coast","island","nymph","foam"},
    "death":  {"death","dead","die","dying","underworld","hades","shadow","darkness","grave",
               "tomb","soul","spirit","persephone","night","doom","fate","end"},
    "love":   {"love","heart","beauty","beautiful","aphrodite","desire","passion","beloved",
               "embrace","tender","gentle","birth","born","child","mother","joy"},
    "fire":   {"fire","flame","burn","heat","forge","volcanic","hephaestus","prometheus",
               "torch","blaze","inferno","eruption","molten","ash","ember"},
    "earth":  {"earth","ground","mountain","forest","nature","gaia","soil","stone","rock",
               "harvest","grow","root","tree","land","field","grain"},
}


# ---------------------------------------------------------------------------
# Phase 5: public-domain painting backgrounds (Wikimedia Commons)
# One masterwork per theme — downloaded once, cached to ASSETS_DIR/bg/.
# Graceful fallback to gradient if download/decode fails.
# ---------------------------------------------------------------------------
THEME_BG_URLS: dict[str, str] = {
    # Albrecht Altdorfer – The Battle of Alexander at Issus (1529)
    "war":     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Albrecht_Altdorfer_-_The_Battle_of_Alexander_at_Issus_-_WGA0243.jpg/1280px-Albrecht_Altdorfer_-_The_Battle_of_Alexander_at_Issus_-_WGA0243.jpg",
    # Jean-Auguste-Dominique Ingres – Jupiter and Thetis (1811)
    "heaven":  "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Jean_Auguste_Dominique_Ingres%2C_Jupiter_and_Thetis.jpg/856px-Jean_Auguste_Dominique_Ingres%2C_Jupiter_and_Thetis.jpg",
    # William-Adolphe Bouguereau – The Birth of Venus (1879)
    "sea":     "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Bouguereau_The_Birth_of_Venus_detail.jpg/1036px-Bouguereau_The_Birth_of_Venus_detail.jpg",
    # Gustave Moreau – Orpheus (1865) — Orpheus in the Underworld
    "death":   "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Gustave_Moreau_-_Orpheus.jpg/767px-Gustave_Moreau_-_Orpheus.jpg",
    # William-Adolphe Bouguereau – Cupidon (1875)
    "love":    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Bouguereau-Cupidon.jpg/900px-Bouguereau-Cupidon.jpg",
    # Jacob Jordaens – Prometheus Bound (1640)
    "fire":    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Prometheus_chained.jpg/1280px-Prometheus_chained.jpg",
    # Jean-François Millet – The Gleaners (1857)
    "earth":   "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Jean-Fran%C3%A7ois_Millet_-_Gleaners_-_Google_Art_Project_2.jpg/1280px-Jean-Fran%C3%A7ois_Millet_-_Gleaners_-_Google_Art_Project_2.jpg",
    # Raphael – The School of Athens (1511)
    "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/%22The_School_of_Athens%22_by_Raffaello_Sanzio_da_Urbino.jpg/1280px-%22The_School_of_Athens%22_by_Raffaello_Sanzio_da_Urbino.jpg",
}

# In-process image cache — keyed by "theme_WxH" to avoid repeated disk reads
_bg_cache: dict[str, "PIL.Image.Image"] = {}


def get_theme_bg_image(theme: str, width: int = 1280, height: int = 720):
    """
    Return a PIL Image (RGB, width×height) for the given theme.

    First call per theme:
      1. Checks ASSETS_DIR/bg/{theme}.jpg on disk.
      2. Downloads from THEME_BG_URLS if not present.
      3. Cover-crops to width×height and caches in _bg_cache.
    On any failure: silently falls back to the numpy gradient.
    """
    from PIL import Image

    cache_key = f"{theme}_{width}x{height}"
    if cache_key in _bg_cache:
        return _bg_cache[cache_key]

    bg_dir    = ASSETS_DIR / "bg"
    disk_path = bg_dir / f"{theme}.jpg"

    # Download if missing
    if not disk_path.exists():
        url = THEME_BG_URLS.get(theme, THEME_BG_URLS["default"])
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "MythForge/5.0 (public-domain art)"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                disk_path.write_bytes(resp.read())
            logger.info(
                "Downloaded painting for theme '%s' (%d KB)",
                theme, disk_path.stat().st_size // 1024,
            )
        except Exception as exc:
            logger.warning("Painting download failed for '%s': %s — gradient fallback", theme, exc)
            _bg_cache[cache_key] = _make_background(width, height, theme)
            return _bg_cache[cache_key]

    # Load, cover-resize, center-crop
    try:
        img    = Image.open(disk_path).convert("RGB")
        ir, tr = img.width / img.height, width / height
        if ir > tr:
            new_h, new_w = height, int(img.width * height / img.height)
        else:
            new_w, new_h = width, int(img.height * width / img.width)
        img  = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x, y = (new_w - width) // 2, (new_h - height) // 2
        img  = img.crop((x, y, x + width, y + height))
        _bg_cache[cache_key] = img
        return img
    except Exception as exc:
        logger.warning("Painting load failed for '%s': %s — gradient fallback", theme, exc)
        disk_path.unlink(missing_ok=True)   # remove corrupt file so next call retries
        _bg_cache[cache_key] = _make_background(width, height, theme)
        return _bg_cache[cache_key]


def detect_theme(text: str) -> str:
    """Return the best-matching scene theme for a transcript segment."""
    words  = set(re.sub(r"[^\w\s]", "", text.lower()).split())
    scores = {theme: len(words & kws) for theme, kws in THEME_KEYWORDS.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else "default"


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
    cutoff  = datetime.now(tz=timezone.utc).timestamp() - JOB_TTL_HOURS * 3600
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
        beam_size=1,
        language=WHISPER_LANG,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )
    result = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            result.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    logger.info(
        "Transcribed %d segment(s) (lang=%s, prob=%.2f)",
        len(result), info.language, info.language_probability,
    )
    return result


def write_srt(segments: list[dict], path: Path) -> None:
    """Write an SRT subtitle file from segments."""
    def _ts(s: float) -> str:
        h   = int(s // 3600)
        m   = int((s % 3600) // 60)
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")

    with open(path, "w", encoding="utf-8") as fh:
        for idx, seg in enumerate(segments, 1):
            fh.write(f"{idx}\n{_ts(seg['start'])} --> {_ts(seg['end'])}\n{seg['text']}\n\n")

# ---------------------------------------------------------------------------
# PIL frame generation — Phase 4: themed backgrounds + Ken Burns
# ---------------------------------------------------------------------------
def _font(path: str, size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(path, size)
    except (IOError, OSError):
        logger.warning("Font not found (%s), using PIL default", path)
        return ImageFont.load_default()


def _make_background(width: int, height: int, theme: str):
    """
    Build a theme-coloured gradient background using numpy vectorisation.
    ~20× faster than the pixel-by-pixel PIL loop it replaces.
    Returns a PIL Image (RGB, width × height).
    """
    from PIL import Image

    t_cfg = SCENE_THEMES.get(theme, SCENE_THEMES["default"])
    top   = np.array(t_cfg["bg_top"], dtype=np.float32)
    bot   = np.array(t_cfg["bg_bot"], dtype=np.float32)

    # Linear gradient: top colour at row 0, bottom colour at row height-1
    t        = np.linspace(0.0, 1.0, height, dtype=np.float32)          # (H,)
    gradient = top * (1.0 - t[:, np.newaxis]) + bot * t[:, np.newaxis]  # (H, 3)

    # Subtle vignette: add brightness at vertical midpoint (sin peaks at 0.5)
    vignette = np.sin(np.pi * t).reshape(-1, 1) * 18.0                  # (H, 1)
    gradient = gradient + vignette

    # Broadcast to full width and convert to uint8
    arr = np.clip(
        gradient[:, np.newaxis, :].repeat(width, axis=1), 0, 255
    ).astype(np.uint8)                                                    # (H, W, 3)
    return Image.fromarray(arr, "RGB")


def _draw_scene_deco(draw, theme: str, width: int, height: int) -> None:
    """
    Subtle decorative elements that reinforce each scene theme.
    Uses a deterministic RNG seed so every frame in a segment looks identical.
    """
    import random
    rnd = random.Random(theme)  # same seed → same decoration on every frame

    if theme == "heaven":
        for _ in range(90):
            x = rnd.randint(0, width)
            y = rnd.randint(0, int(height * 0.55))
            r = rnd.choice([1, 1, 1, 2])
            b = rnd.randint(140, 220)
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(b, b, b // 2))

    elif theme == "sea":
        for i in range(6):
            base_y = int(height * (0.40 + i * 0.07))
            pts = [
                (x, base_y + int(7 * math.sin(x * 0.04 + i * 1.3)))
                for x in range(0, width + 20, 20)
            ]
            for j in range(len(pts) - 1):
                draw.line([pts[j], pts[j + 1]], fill=(30, 90, 120), width=1)

    elif theme == "fire":
        for _ in range(35):
            x = rnd.randint(0, width)
            y = rnd.randint(int(height * 0.50), height)
            r = rnd.choice([1, 1, 2])
            draw.ellipse(
                [(x - r, y - r), (x + r, y + r)],
                fill=(220, rnd.randint(70, 160), 0),
            )

    elif theme == "death":
        for _ in range(25):
            x = rnd.randint(0, width)
            y = rnd.randint(0, height)
            draw.point((x, y), fill=(55, 28, 75))

    elif theme == "war":
        for _ in range(6):
            x1 = rnd.randint(0, width)
            y1 = rnd.randint(0, height)
            draw.line(
                [(x1, y1), (x1 + rnd.randint(-80, 80), y1 + rnd.randint(15, 70))],
                fill=(70, 18, 18), width=1,
            )

    elif theme == "earth":
        for _ in range(12):
            x = rnd.randint(0, width)
            y = rnd.randint(int(height * 0.70), height)
            r = rnd.randint(2, 5)
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(50, 80, 30))


def make_frame(
    text: str,
    title: str,
    frame_idx: int,
    total_frames: int,
    theme: str = "default",
    zoom: float = 0.0,
    width: int = 1280,
    height: int = 720,
):
    """
    Render one 1280×720 caption frame.

    Phase 5: uses a real public-domain painting as background.
    Ken Burns is applied to the painting BEFORE drawing text, so the
    zoom effect animates the artwork, not the caption.
    A dark vignette overlay is composited over the image to ensure
    caption text is always legible regardless of the painting's content.

    theme: one of SCENE_THEMES keys — drives colour palette + UI chrome
    zoom:  0.0 (no zoom) → 1.0 (3% centred push-in) for Ken Burns
    """
    from PIL import Image, ImageDraw

    t_cfg = SCENE_THEMES.get(theme, SCENE_THEMES["default"])

    # 1. Background painting (cover-cropped to 1280×720, cached in memory)
    img = get_theme_bg_image(theme, width, height).copy()

    # 2. Ken Burns: zoom into the painting before compositing text
    if zoom > 0.001:
        z      = zoom * 0.03          # max 3% push-in
        crop_w = int(width  * (1.0 - z))
        crop_h = int(height * (1.0 - z))
        x0     = (width  - crop_w) // 2
        y0     = (height - crop_h) // 2
        img    = img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(
            (width, height), Image.Resampling.BILINEAR
        )

    # 3. Dark vignette overlay — ensures text legibility on any painting
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ov      = ImageDraw.Draw(overlay)
    ov.rectangle([(0, 0), (width, height)],                     fill=(0, 0, 0, 70))   # global dim
    ov.rectangle([(0, int(height * 0.25)), (width, int(height * 0.82))],
                 fill=(0, 0, 0, 110))                                                   # text band
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(img)

    font_label = _font(FONT_REGULAR, 20)
    font_main  = _font(FONT_BOLD,    48)

    # 4. Episode title bar (top centre)
    label = title.upper() if title else "MYTHFORGE"
    lb    = draw.textbbox((0, 0), label, font=font_label)
    lw    = lb[2] - lb[0]
    draw.text(((width - lw) // 2, 36), label, fill=t_cfg["title"], font=font_label)
    draw.rectangle([(width // 4, 68), (3 * width // 4, 70)], fill=t_cfg["accent"])

    # 5. Caption text (centred, drop-shadowed)
    if text:
        wrapped = textwrap.fill(text, width=42)
        mb      = draw.multiline_textbbox(
            (0, 0), wrapped, font=font_main, align="center", spacing=16
        )
        tw, th = mb[2] - mb[0], mb[3] - mb[1]
        x = (width  - tw) // 2
        y = (height - th) // 2 + 8

        draw.multiline_text(
            (x + 2, y + 2), wrapped,
            fill=(0, 0, 0), font=font_main, align="center", spacing=16,
        )
        draw.multiline_text(
            (x, y), wrapped,
            fill=t_cfg["text"], font=font_main, align="center", spacing=16,
        )

    # 6. Bottom accent bar + progress bar
    draw.rectangle(
        [(width // 4, height - 44), (3 * width // 4, height - 41)],
        fill=t_cfg["accent"],
    )
    if total_frames > 0:
        bar_w = int(width * frame_idx / total_frames)
        draw.rectangle([(0, height - 4), (bar_w, height - 1)], fill=t_cfg["title"])

    return img

# ---------------------------------------------------------------------------
# Frame sequence builder — Phase 4: per-second keyframes + theme detection
# ---------------------------------------------------------------------------
def build_frame_sequence(
    segments: list[dict],
    duration: float,
    title: str,
    frames_dir: Path,
) -> tuple[Path, dict[str, int]]:
    """
    Generate PNG frames for every caption event.

    Speech segments:
      - Detect scene theme from text (keyword scoring)
      - Generate 1 keyframe per second of duration (capped at 12)
      - Each keyframe has a linearly-interpolated zoom from 0.0 → 1.0
        (Ken Burns push-in; max 3% crop via make_frame)

    Silence gaps:
      - Single blank frame (theme=default, zoom=0)

    Returns (concat_path, theme_distribution_dict).
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Build ordered event list: (start, end, text)
    events:   list[tuple[float, float, str]] = []
    prev_end  = 0.0
    for seg in segments:
        if seg["start"] > prev_end + 0.1:
            events.append((prev_end, seg["start"], ""))    # silence gap
        events.append((seg["start"], seg["end"], seg["text"]))
        prev_end = seg["end"]
    if prev_end < duration - 0.1:
        events.append((prev_end, duration, ""))              # trailing silence

    if not events:
        events = [(0.0, duration, "")]                       # fallback: blank video

    total_events  = len(events)
    lines:        list[str]      = []
    frame_counter = 0
    theme_counts: dict[str, int] = {}

    for evt_idx, (start, end, text) in enumerate(events):
        evt_dur = max(end - start, 0.04)   # min 40 ms (FFmpeg floor)
        theme   = detect_theme(text) if text else "default"
        theme_counts[theme] = theme_counts.get(theme, 0) + 1

        if text and evt_dur >= 1.0:
            # Ken Burns: 1 keyframe per second, capped at 12 per segment
            n_kf    = min(int(math.ceil(evt_dur)), 12)
            kf_dur  = evt_dur / n_kf
            for kf in range(n_kf):
                zoom = kf / (n_kf - 1) if n_kf > 1 else 0.0
                img  = make_frame(
                    text, title, evt_idx, total_events,
                    theme=theme, zoom=zoom,
                )
                p = frames_dir / f"{frame_counter:05d}.png"
                img.save(str(p), format="PNG", optimize=False)
                lines.append(f"file '{p}'")
                lines.append(f"duration {kf_dur:.4f}")
                frame_counter += 1
        else:
            # Short segment or silence: single frame, no zoom
            img = make_frame(text, title, evt_idx, total_events, theme=theme, zoom=0.0)
            p   = frames_dir / f"{frame_counter:05d}.png"
            img.save(str(p), format="PNG", optimize=False)
            lines.append(f"file '{p}'")
            lines.append(f"duration {evt_dur:.4f}")
            frame_counter += 1

    # FFmpeg concat demuxer requires the last file entry to appear twice
    last_file = next(l for l in reversed(lines) if l.startswith("file "))
    lines.append(last_file)

    concat_path = frames_dir.parent / "concat.txt"
    concat_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "Built %d frame(s) across %d event(s) for %.1fs audio",
        frame_counter, total_events, duration,
    )
    return concat_path, theme_counts

# ---------------------------------------------------------------------------
# Video assembly — Phase 4: optional BGM mixing
# ---------------------------------------------------------------------------
def assemble_video(
    concat_path: Path,
    mp3_path: Path,
    output_path: Path,
    bgm_path: Path | None = None,
    bgm_volume: float = 0.15,
) -> None:
    """
    Stitch frames + narration → output.mp4.

    If bgm_path is provided, the BGM is looped (stream_loop -1) and mixed under
    the narration at bgm_volume (0.0–1.0).  FFmpeg amix uses duration=shortest
    so the video never runs past the narration track.

    -movflags +faststart  →  moov atom at file start, enables browser streaming
    -crf 23               →  good quality/size trade-off for libx264
    -preset fast          →  reasonable encode speed on CPU
    """
    if bgm_path:
        # inputs: 0=video concat  1=BGM (looped)  2=narration
        audio_filter = (
            f"[1:a]volume={bgm_volume:.3f}[bgm];"
            "[bgm][2:a]amix=inputs=2:duration=shortest:dropout_transition=2[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-stream_loop", "-1", "-i", str(bgm_path),
            "-i", str(mp3_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=1280:720:flags=lanczos",
            "-crf", "23", "-preset", "fast",
            "-filter_complex", audio_filter,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            str(output_path),
        ]
    else:
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

    logger.info(
        "FFmpeg assembly start (bgm=%s, timeout=%ss) ...",
        bgm_path is not None, FFMPEG_TIMEOUT,
    )
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

    cached_themes = [p.stem for p in (ASSETS_DIR / "bg").glob("*.jpg")]
    status = "healthy" if ffmpeg_ok else "degraded"
    return jsonify({
        "status":          status,
        "phase":           "5-cinematic-paintings",
        "ffmpeg":          ffmpeg_ok,
        "whisper_model":   WHISPER_MODEL,
        "whisper_ready":   _whisper_model is not None,
        "exports":         str(EXPORTS_ROOT),
        "cached_paintings": cached_themes,
    }), 200 if ffmpeg_ok else 503


@app.route("/api/render", methods=["POST"])
@require_api_key
def render():
    """
    Accept MP3 + optional BGM upload and produce a themed lyric-caption video.

    Required form field:
      mp3          — narration audio (MP3)

    Optional form fields:
      title        — episode label shown at top of every frame (max 60 chars)
      bgm          — background music file (MP3 or WAV) mixed under narration
      bgm_volume   — BGM level 0.0–1.0  (default 0.15 ≈ −16 dBFS)
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

    # ---- BGM ----
    bgm_file = request.files.get("bgm")
    try:
        bgm_volume = max(0.0, min(1.0, float(request.form.get("bgm_volume", "0.15"))))
    except ValueError:
        bgm_volume = 0.15

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
    bgm_path    = job_dir / "bgm.mp3" if (bgm_file and bgm_file.filename) else None
    srt_path    = job_dir / "subtitles.srt"
    frames_dir  = job_dir / "frames"
    output_path = job_dir / "output.mp4"
    segments:     list[dict]      = []
    theme_counts: dict[str, int]  = {}
    duration = 0.0

    try:
        # ---- save uploads ----
        mp3_file.save(str(mp3_path))
        logger.info("Saved %.1f KB → %s", mp3_path.stat().st_size / 1024, mp3_path)

        if bgm_path and bgm_file:
            bgm_file.save(str(bgm_path))
            logger.info("BGM saved: %.1f KB", bgm_path.stat().st_size / 1024)

        # ---- probe duration ----
        duration = get_duration(mp3_path)
        logger.info("Duration: %.1fs", duration)

        # ---- Phase 3a: transcribe ----
        logger.info("Phase 3a: Whisper transcription ...")
        segments = transcribe(mp3_path)
        write_srt(segments, srt_path)
        logger.info("SRT written: %d segment(s)", len(segments))

        # ---- Phase 4a: generate themed Ken Burns frames ----
        logger.info("Phase 4a: Generating themed Ken Burns frames ...")
        concat_path, theme_counts = build_frame_sequence(
            segments, duration, title, frames_dir
        )

        # ---- Phase 4b: assemble video (with optional BGM) ----
        logger.info("Phase 4b: FFmpeg assembly (bgm=%s) ...", bgm_path is not None)
        assemble_video(
            concat_path, mp3_path, output_path,
            bgm_path=bgm_path, bgm_volume=bgm_volume,
        )

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
        mp3_path.unlink(missing_ok=True)
        if bgm_path:
            bgm_path.unlink(missing_ok=True)
        shutil.rmtree(frames_dir, ignore_errors=True)

    return jsonify({
        "success":       True,
        "job_id":        job_id,
        "url":           f"http://{VPS_IP}/exports/{job_id}/output.mp4",
        "subtitles_url": f"http://{VPS_IP}/exports/{job_id}/subtitles.srt",
        "duration_s":    round(duration, 2),
        "segments":      len(segments),
        "themes":        theme_counts,
        "phase":         "5-cinematic-paintings",
        "message":       "Video generated: real painting backgrounds + Ken Burns + optional BGM.",
    })


@app.route("/api/transcribe", methods=["POST"])
@require_api_key
def transcribe_only():
    """
    Transcribe an uploaded MP3 and return segments as JSON.
    Much faster than /api/render — no frame generation or video encoding.

    Required form field:
      mp3   — narration audio (MP3)

    Returns:
      full_text    — plain script text (all segments joined)
      segments     — list of {start, end, text} dicts
      srt_url      — hosted SRT file URL
      duration_s   — audio duration in seconds
    """
    if "mp3" not in request.files:
        return jsonify({"error": "No MP3 file (field name: mp3)"}), 400
    mp3_file = request.files["mp3"]
    if not mp3_file.filename:
        return jsonify({"error": "No file selected"}), 400

    job_id  = str(uuid.uuid4())[:8]
    job_dir = EXPORTS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = job_dir / "input.mp3"
    srt_path = job_dir / "subtitles.srt"

    try:
        mp3_file.save(str(mp3_path))
        duration = get_duration(mp3_path)
        segments = transcribe(mp3_path)
        write_srt(segments, srt_path)
        full_text = " ".join(seg["text"] for seg in segments)
        return jsonify({
            "success":    True,
            "job_id":     job_id,
            "duration_s": round(duration, 2),
            "full_text":  full_text,
            "segments":   segments,
            "srt_url":    f"http://{VPS_IP}/exports/{job_id}/subtitles.srt",
        })
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.exception("Transcription error: %s", exc)
        return jsonify({"error": "Transcription failed"}), 500
    finally:
        mp3_path.unlink(missing_ok=True)


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
