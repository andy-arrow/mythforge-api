"""
MythForge Video API: MP3 → HD video (black background, synced audio).
Docker WORKDIR /app; exports volume mounted at /app/exports.
"""
from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import logging
import ffmpeg

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
EXPORTS_ROOT = "/app/exports"


def mp3_duration(path: str) -> float:
    """Return duration in seconds of audio file (probe)."""
    probe = ffmpeg.probe(path)
    return float(probe["format"]["duration"])


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
    job_dir = os.path.join(EXPORTS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    mp3_path = os.path.join(job_dir, "input.mp3")
    mp3_file.save(mp3_path)
    print(f"SAVED {mp3_path}")

    # GENERATE VIDEO: 1280x720 black background synced to audio
    output_path = os.path.join(job_dir, "output.mp4")
    try:
        duration = mp3_duration(mp3_path)
        print(f"Audio duration: {duration}s")
        video = ffmpeg.input(
            f"color=c=black:s=1280x720:r=30:d={duration}",
            f="lavfi",
        )
        audio = ffmpeg.input(mp3_path)
        (
            ffmpeg.output(
                video,
                audio,
                output_path,
                vcodec="libx264",
                acodec="aac",
                pix_fmt="yuv420p",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"VIDEO generated: {output_path}")
    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        print(f"FFmpeg error: {stderr}")
        return jsonify({"error": f"FFmpeg failed: {stderr}"}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "success": True,
        "job_id": job_id,
        "url": f"http://{request.host}/exports/{job_id}/output.mp4",
        "message": "AI pipeline placeholder - full AI coming next",
        "ai_status": "whisper+diffusers installed, integration in progress",
    })


@app.route("/exports/<path:filename>")
def serve_exports(filename):
    """Serve rendered files (volume at /app/exports; nginx serves /exports/ in production)."""
    return send_from_directory(EXPORTS_ROOT, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
