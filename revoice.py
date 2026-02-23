#!/usr/bin/env python3
"""
revoice.py — Re-voice a narration with ElevenLabs (via kie.ai), then render
             a Phase 5 MythForge video with real painting backgrounds.

Workflow
--------
  1. POST /api/transcribe  →  extract script text from the original MP3
  2. POST kie.ai ElevenLabs TTS  →  get a task ID
  3. Poll GET  kie.ai /api/v1/jobs/recordInfo  →  wait for audio URL
  4. Download the ElevenLabs MP3 to /tmp/
  5. POST /api/render with the new MP3  →  Phase 5 video

Usage
-----
  python3 revoice.py \\
      --audio  /opt/openclawworkspace/mythforge/hera_full_audio_combined.mp3 \\
      --title  "HERA — The Birth of War" \\
      --voice  George \\
      --kie-key YOUR_KIE_API_KEY \\
      --api-url http://51.83.154.112

ElevenLabs voices (male, recommended for mythology):
  George   — British, deep, authoritative  ← default
  Daniel   — British, mid-range
  Brian    — American, warm
  Bill     — American, very deep
  Charlie  — Australian, clear
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

KIE_BASE  = "https://api.kie.ai"
EL_MODEL  = "elevenlabs/text-to-speech-multilingual-v2"
MAX_CHARS = 4800   # safe margin under the 5 000-char ElevenLabs limit


# ---------------------------------------------------------------------------
# kie.ai helpers
# ---------------------------------------------------------------------------

def _kie_post(path: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{KIE_BASE}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _kie_get(path: str, api_key: str) -> dict:
    req = urllib.request.Request(
        f"{KIE_BASE}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Step 1 — transcribe via MythForge
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: Path, api_url: str) -> tuple[str, list[dict]]:
    """
    POST the original MP3 to /api/transcribe.
    Returns (full_text, segments_list).
    """
    print(f"[1/5] Transcribing original narration via {api_url}/api/transcribe …")
    result = subprocess.run(
        [
            "curl", "-s", "-X", "POST",
            "-F", f"mp3=@{audio_path}",
            f"{api_url}/api/transcribe",
        ],
        capture_output=True, text=True, timeout=300,
    )
    try:
        resp = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON from /api/transcribe: {result.stdout[:400]}")

    if not resp.get("success"):
        raise RuntimeError(f"/api/transcribe failed: {resp}")

    text = resp["full_text"]
    segs = resp["segments"]
    print(f"     Duration : {resp['duration_s']}s")
    print(f"     Segments : {len(segs)}")
    print(f"     Chars    : {len(text)}")
    print(f"     Preview  : {text[:160]}…")
    return text, segs


# ---------------------------------------------------------------------------
# Step 2 — split long script into ≤ MAX_CHARS chunks
# ---------------------------------------------------------------------------

def _split_text(text: str) -> list[str]:
    """Split at sentence boundaries so each chunk ≤ MAX_CHARS chars."""
    if len(text) <= MAX_CHARS:
        return [text]
    chunks = []
    while text:
        if len(text) <= MAX_CHARS:
            chunks.append(text)
            break
        cut = text.rfind(". ", 0, MAX_CHARS)
        if cut == -1:
            cut = MAX_CHARS
        else:
            cut += 1   # include the period
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return chunks


# ---------------------------------------------------------------------------
# Step 3 — generate ElevenLabs audio chunks via kie.ai
# ---------------------------------------------------------------------------

def generate_elevenlabs(
    text: str,
    voice: str,
    kie_key: str,
    *,
    stability: float = 0.50,
    similarity_boost: float = 0.75,
    style: float = 0.15,
    speed: float = 0.92,
) -> list[str]:
    """
    Submit ElevenLabs TTS tasks to kie.ai and poll until each completes.
    Returns a list of audio URLs (one per text chunk).
    """
    chunks    = _split_text(text)
    audio_urls: list[str] = []

    print(f"[2/5] Generating ElevenLabs audio — voice={voice}, {len(chunks)} chunk(s) …")

    for idx, chunk in enumerate(chunks, 1):
        print(f"     Chunk {idx}/{len(chunks)}  ({len(chunk)} chars) → submitting …")

        resp = _kie_post(
            "/api/v1/jobs/createTask",
            {
                "model": EL_MODEL,
                "input": {
                    "text":             chunk,
                    "voice":            voice,
                    "stability":        stability,
                    "similarity_boost": similarity_boost,
                    "style":            style,
                    "speed":            speed,
                },
            },
            kie_key,
        )

        if resp.get("code") != 200:
            raise RuntimeError(f"ElevenLabs task creation failed: {resp}")

        task_id = resp["data"]["taskId"]
        print(f"     Task ID : {task_id}  — polling …")

        # Poll until done (max 5 min)
        for attempt in range(60):
            time.sleep(5)
            status = _kie_get(f"/api/v1/jobs/recordInfo?taskId={task_id}", kie_key)

            if status.get("code") != 200:
                raise RuntimeError(f"Status poll failed: {status}")

            s_data       = status.get("data", {})
            success_flag = s_data.get("successFlag", 0)

            if success_flag == 1:
                # Try every plausible field name for the audio URL
                info      = s_data.get("info") or s_data.get("response") or {}
                audio_url = (
                    info.get("audioUrl")
                    or info.get("audio_url")
                    or info.get("resultUrl")
                    or info.get("resultUrls")
                    or s_data.get("resultUrls")
                    or s_data.get("audioUrl")
                )
                if isinstance(audio_url, list):
                    audio_url = audio_url[0] if audio_url else None

                if audio_url:
                    print(f"     ✓ Audio ready: {audio_url}")
                    audio_urls.append(audio_url)
                    break
                else:
                    # Dump the full response so we can debug unknown formats
                    print(f"     ✗ successFlag=1 but no audio URL found.")
                    print(f"       Full response: {json.dumps(s_data)[:500]}")
                    raise RuntimeError("Could not locate audio URL in kie.ai response")

            elif success_flag in (2, 3):
                raise RuntimeError(f"ElevenLabs generation failed: {s_data}")
            else:
                print(f"     … still processing (attempt {attempt + 1}/60)")
        else:
            raise RuntimeError("ElevenLabs generation timed out after 5 minutes")

    return audio_urls


# ---------------------------------------------------------------------------
# Step 4 — download & concatenate audio chunks
# ---------------------------------------------------------------------------

def download_audio(audio_urls: list[str], dest: Path) -> None:
    """Download one or more audio chunks; concatenate with ffmpeg if needed."""
    print(f"[3/5] Downloading audio to {dest} …")

    if len(audio_urls) == 1:
        urllib.request.urlretrieve(audio_urls[0], str(dest))
        print(f"     Downloaded: {dest.stat().st_size // 1024} KB")
        return

    # Multiple chunks — download then concat with ffmpeg
    tmp_files: list[Path] = []
    for i, url in enumerate(audio_urls):
        tmp = dest.parent / f"_chunk_{i}.mp3"
        urllib.request.urlretrieve(url, str(tmp))
        tmp_files.append(tmp)
        print(f"     Chunk {i + 1}: {tmp.stat().st_size // 1024} KB")

    # Build ffmpeg concat list
    concat_txt = dest.parent / "_concat.txt"
    concat_txt.write_text(
        "\n".join(f"file '{p}'" for p in tmp_files), encoding="utf-8"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy", str(dest),
        ],
        check=True, capture_output=True,
    )
    for f in tmp_files:
        f.unlink(missing_ok=True)
    concat_txt.unlink(missing_ok=True)
    print(f"     Concatenated: {dest.stat().st_size // 1024} KB")


# ---------------------------------------------------------------------------
# Step 5 — render Phase 5 video
# ---------------------------------------------------------------------------

def render_video(audio_path: Path, api_url: str, title: str, bgm_volume: float) -> dict:
    """POST the ElevenLabs audio to /api/render and return the JSON response."""
    print(f"[4/5] Rendering Phase 5 video via {api_url}/api/render …")
    cmd = [
        "curl", "-s", "-X", "POST",
        "-F", f"mp3=@{audio_path}",
        "-F", f"title={title}",
        "-F", f"bgm_volume={bgm_volume}",
        f"{api_url}/api/render",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=720)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON from /api/render: {result.stdout[:400]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-voice Hera narration with ElevenLabs and render Phase 5 video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--audio", required=True,
        help="Path to the original narration MP3 (e.g. hera_full_audio_combined.mp3)",
    )
    parser.add_argument("--title",      default="HERA — The Birth of War")
    parser.add_argument("--voice",      default="George",
                        help="ElevenLabs voice name (George / Daniel / Brian / Bill)")
    parser.add_argument("--kie-key",    required=True, help="kie.ai API key")
    parser.add_argument("--api-url",    default="http://51.83.154.112")
    parser.add_argument("--bgm-volume", type=float, default=0.15)
    parser.add_argument("--out-audio",  default="/tmp/hera_elevenlabs.mp3",
                        help="Where to save the downloaded ElevenLabs MP3")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    out_audio = Path(args.out_audio)
    out_audio.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  MythForge revoice — {args.title}")
    print(f"  Voice    : {args.voice}")
    print(f"  API      : {args.api_url}")
    print("=" * 60)

    try:
        # 1. Transcribe
        script_text, _ = transcribe_audio(audio_path, args.api_url)

        # 2+3. Generate ElevenLabs audio
        audio_urls = generate_elevenlabs(script_text, args.voice, args.kie_key)

        # 4. Download
        download_audio(audio_urls, out_audio)

        # 5. Render
        result = render_video(out_audio, args.api_url, args.title, args.bgm_volume)

    except Exception as exc:
        print(f"\n❌  {exc}", file=sys.stderr)
        sys.exit(1)

    print()
    if result.get("success"):
        print("=" * 60)
        print("  ✅  PHASE 5 VIDEO COMPLETE")
        print("=" * 60)
        print(f"  Video URL  : {result['url']}")
        print(f"  SRT URL    : {result['subtitles_url']}")
        print(f"  Duration   : {result['duration_s']}s")
        print(f"  Segments   : {result['segments']}")
        print(f"  Themes     : {result['themes']}")
        print(f"  Phase      : {result['phase']}")
    else:
        print(f"❌  Render failed: {result}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
