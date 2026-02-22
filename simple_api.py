"""
MythForge Video API: MP3 → HD video (black background, synced audio).
Production-ready Flask app for CPU-only VPS (8-core, 24GB RAM).
"""
from flask import Flask, request, jsonify, send_from_directory
import uuid
import os
import ffmpeg

app = Flask(__name__)
EXPORTS_ROOT = "/exports"


def mp3_duration(path: str) -> float:
    """Return duration in seconds of audio file (probe)."""
    probe = ffmpeg.probe(path)
    return float(probe["format"]["duration"])


@app.route("/health")
def health():
    """Health check for Docker and nginx depends_on."""
    return jsonify({"status": "ok"})


@app.route("/api/render", methods=["POST"])
def render():
    """Accept MP3 upload; return job_id and URL to output.mp4 (black 1280x720, synced audio)."""
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

    output_path = os.path.join(job_dir, "output.mp4")

    try:
        duration = mp3_duration(mp3_path)
        # Black video (lavfi) + uploaded audio → MP4
        video = ffmpeg.input(
            f"color=black:s=1280x720:r=30:d={duration}",
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
        return jsonify(
            {
                "job_id": job_id,
                "url": f"/exports/{job_id}/output.mp4",
            }
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        return jsonify({"error": f"FFmpeg failed: {stderr}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/exports/<path:filename>")
def serve_exports(filename):
    """Serve rendered files from persistent volume (also used by nginx for public URLs)."""
    return send_from_directory(EXPORTS_ROOT, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
