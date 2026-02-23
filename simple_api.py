"""
MythForge Video API — Phase 2 (simple + AI-ready)
MP3 → HD video. Docker WORKDIR /app; exports at /app/exports.
"""
from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
EXPORTS_ROOT = "/app/exports"
VPS_IP = os.environ.get("VPS_IP", "51.83.154.112")


def get_duration(mp3_path: str) -> float:
    """Probe audio duration with ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            mp3_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "phase": "2-ai-installed"})


@app.route("/api/render", methods=["POST"])
def render():
    print("RENDER START")
    if "mp3" not in request.files:
        return jsonify({"error": "No MP3 file"}), 400
    mp3_file = request.files["mp3"]
    if mp3_file.filename == "":
        return jsonify({"error": "No MP3 file selected"}), 400

    job_id = str(uuid.uuid4())[:8]
    job_dir = Path(EXPORTS_ROOT) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    mp3_path = job_dir / "input.mp3"
    mp3_file.save(str(mp3_path))
    print(f"SAVED {mp3_path}")

    output_path = job_dir / "output.mp4"

    try:
        duration = get_duration(str(mp3_path))
        print(f"Audio duration: {duration}s")
    except Exception as e:
        print(f"Duration probe failed: {e}")
        duration = 30  # safe fallback

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

    print(f"Running FFmpeg: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"VIDEO generated: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return jsonify({"error": f"FFmpeg failed: {e.stderr}"}), 500

    return jsonify({
        "success": True,
        "job_id": job_id,
        "url": f"http://{VPS_IP}/exports/{job_id}/output.mp4",
        "message": "Video created. AI pipeline (Whisper + SDXL) coming next.",
        "ai_status": "whisper+diffusers installed, integration in progress",
    })


@app.route("/exports/<path:filename>")
def serve_exports(filename):
    return send_from_directory(EXPORTS_ROOT, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
