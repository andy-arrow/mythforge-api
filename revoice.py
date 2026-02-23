#!/usr/bin/env python3
"""
revoice.py — Re-voice a narration with ElevenLabs (via kie.ai), then render
             a Phase 6 MythForge video with AI-generated backgrounds.

Workflow
--------
  1. POST /api/transcribe  →  extract script text from the original MP3
  2. POST kie.ai ElevenLabs TTS  →  get a task ID
  3. Poll GET  kie.ai /api/v1/jobs/recordInfo  →  wait for audio URL
  4. Download the ElevenLabs MP3 to /tmp/
  5. POST /api/render with the new MP3  →  Phase 6 video with AI visuals

Usage
-----
  # Full AI pipeline (recommended):
  python3 revoice.py \\
      --audio  /opt/openclawworkspace/mythforge/hera_full_audio_combined.mp3 \\
      --title  "HERA — The Birth of War" \\
      --kie-key YOUR_KIE_API_KEY \\
      --api-url http://localhost \\
      --ai-visuals images

  # Legacy Phase 5 (painting backgrounds, free):
  python3 revoice.py \\
      --audio  /opt/openclawworkspace/mythforge/hera_full_audio_combined.mp3 \\
      --title  "HERA — The Birth of War" \\
      --kie-key YOUR_KIE_API_KEY \\
      --api-url http://localhost

ElevenLabs voices (Kie.ai whitelist only):
  DEEP MALE (recommended for mythology):
    Bill     — American, VERY deep, dramatic  ← DEFAULT
    George   — British, deep, authoritative
    Brian    — American, warm narrator
    Daniel   — British, mid-range
    Callum   — Scottish, clear
  
  OTHER:
    Rachel, Aria, Sarah, Laura, Charlotte, Alice, Matilda, Jessica, Lily
    Roger, Charlie, River, Liam, Will, Eric, Chris

Cost estimates:
  - ElevenLabs TTS: ~$0.01 per 1000 chars
  - AI images:      ~$0.01 per image (~$0.50 per video)
  - AI video clips: ~$0.18 per clip (~$6 per video)
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

        # Poll until done (max 10 min)
        for attempt in range(120):
            time.sleep(5)
            status = _kie_get(f"/api/v1/jobs/recordInfo?taskId={task_id}", kie_key)

            if status.get("code") != 200:
                raise RuntimeError(f"Status poll failed: {status}")

            s_data = status.get("data", {})
            state  = s_data.get("state", "")

            if state == "success":
                # resultJson is a JSON *string* — parse it
                result_json_str = s_data.get("resultJson", "{}")
                try:
                    result_obj = json.loads(result_json_str)
                except json.JSONDecodeError:
                    result_obj = {}

                # Extract audio URL from resultUrls array
                urls = result_obj.get("resultUrls") or []
                audio_url = urls[0] if urls else None

                if audio_url:
                    print(f"     ✓ Audio ready: {audio_url}")
                    audio_urls.append(audio_url)
                    break
                else:
                    print(f"     ✗ state=success but no audio URL found.")
                    print(f"       resultJson: {result_json_str[:300]}")
                    raise RuntimeError("Could not locate audio URL in kie.ai response")

            elif state == "failed":
                fail_msg = s_data.get("failMsg") or s_data.get("failCode") or "unknown"
                raise RuntimeError(f"ElevenLabs generation failed: {fail_msg}")

            else:
                # Still processing (state might be empty, "pending", "processing", etc.)
                if attempt == 0 or (attempt + 1) % 10 == 0:
                    print(f"     … attempt {attempt + 1}/120  state={state!r}  keys={list(s_data.keys())}")
                else:
                    print(f"     … still processing (attempt {attempt + 1}/120)")
        else:
            raise RuntimeError("ElevenLabs generation timed out after 10 minutes")

    return audio_urls


# ---------------------------------------------------------------------------
# Step 4 — download & concatenate audio chunks
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> None:
    """Download a file with proper headers to avoid 403 errors."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; MythForge/1.0)",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())


def download_audio(audio_urls: list[str], dest: Path) -> None:
    """Download one or more audio chunks; concatenate with ffmpeg if needed."""
    print(f"[3/5] Downloading audio to {dest} …")

    if len(audio_urls) == 1:
        _download_file(audio_urls[0], dest)
        print(f"     Downloaded: {dest.stat().st_size // 1024} KB")
        return

    # Multiple chunks — download then concat with ffmpeg
    tmp_files: list[Path] = []
    for i, url in enumerate(audio_urls):
        tmp = dest.parent / f"_chunk_{i}.mp3"
        _download_file(url, tmp)
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
# Step 5 — render Phase 6 video
# ---------------------------------------------------------------------------

def render_video(
    audio_path: Path,
    api_url: str,
    title: str,
    bgm_volume: float,
    ai_visuals: str = "none",
    kie_key: str = "",
    max_ai_segs: int = 0,
) -> dict:
    """POST the ElevenLabs audio to /api/render and return the JSON response."""
    phase = "6 (AI)" if ai_visuals != "none" else "5 (paintings)"
    print(f"[4/5] Rendering Phase {phase} video via {api_url}/api/render …")
    
    cmd = [
        "curl", "-s", "-X", "POST",
        "-F", f"mp3=@{audio_path}",
        "-F", f"title={title}",
        "-F", f"bgm_volume={bgm_volume}",
        "-F", f"ai_visuals={ai_visuals}",
    ]
    
    if ai_visuals != "none" and kie_key:
        cmd.extend(["-F", f"kie_key={kie_key}"])
    if max_ai_segs > 0:
        cmd.extend(["-F", f"max_ai_segs={max_ai_segs}"])
    
    cmd.append(f"{api_url}/api/render")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # longer for AI
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON from /api/render: {result.stdout[:400]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_task(task_id: str, kie_key: str) -> None:
    """Debug utility: check status of a Kie.ai task."""
    print(f"Checking task: {task_id}")
    status = _kie_get(f"/api/v1/jobs/recordInfo?taskId={task_id}", kie_key)
    print(json.dumps(status, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-voice Hera narration with ElevenLabs and render Phase 6 video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--audio",
        help="Path to the original narration MP3 (e.g. hera_full_audio_combined.mp3)",
    )
    parser.add_argument("--title",      default="HERA — The Birth of War")
    parser.add_argument("--voice",      default="Bill",
                        help="ElevenLabs voice: Bill (very deep), George (British), Brian (warm), Daniel")
    parser.add_argument("--kie-key",    required=True, help="kie.ai API key")
    parser.add_argument("--api-url",    default="http://51.83.154.112")
    parser.add_argument("--bgm-volume", type=float, default=0.15)
    parser.add_argument("--out-audio",  default="/tmp/hera_elevenlabs.mp3",
                        help="Where to save the downloaded ElevenLabs MP3")
    parser.add_argument("--check-task", metavar="TASK_ID",
                        help="Debug: check status of an existing Kie.ai task and exit")
    # Phase 6 options — AI video enabled by default for best quality
    parser.add_argument("--ai-visuals", choices=["none", "images", "video"],
                        default="video",
                        help="'video' (default, ~$6), 'images' (~$0.50), 'none' (paintings only)")
    parser.add_argument("--max-ai-segs", type=int, default=0,
                        help="Limit AI generation to N segments (0 = unlimited)")
    args = parser.parse_args()

    # Debug mode: just check a task and exit
    if args.check_task:
        check_task(args.check_task, args.kie_key)
        sys.exit(0)

    if not args.audio:
        parser.error("--audio is required (unless using --check-task)")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    out_audio = Path(args.out_audio)
    out_audio.parent.mkdir(parents=True, exist_ok=True)

    ai_mode = "AI images" if args.ai_visuals == "images" else (
        "AI video" if args.ai_visuals == "video" else "paintings"
    )

    print("=" * 60)
    print(f"  MythForge revoice — {args.title}")
    print(f"  Voice    : {args.voice}")
    print(f"  Visuals  : {ai_mode}")
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
        result = render_video(
            out_audio,
            args.api_url,
            args.title,
            args.bgm_volume,
            ai_visuals=args.ai_visuals,
            kie_key=args.kie_key,
            max_ai_segs=args.max_ai_segs,
        )

    except Exception as exc:
        print(f"\n❌  {exc}", file=sys.stderr)
        sys.exit(1)

    print()
    if result.get("success"):
        if args.ai_visuals == "video":
            phase_label = "PHASE 6.2 (AI VIDEO)"
        elif args.ai_visuals == "images":
            phase_label = "PHASE 6 (AI IMAGES)"
        else:
            phase_label = "PHASE 5"
        print("=" * 60)
        print(f"  ✅  {phase_label} VIDEO COMPLETE")
        print("=" * 60)
        print(f"  Video URL  : {result['url']}")
        print(f"  SRT URL    : {result['subtitles_url']}")
        print(f"  Duration   : {result['duration_s']}s")
        print(f"  Segments   : {result['segments']}")
        print(f"  AI Images  : {result.get('ai_images', 0)}")
        print(f"  AI Videos  : {result.get('ai_videos', 0)}")
        print(f"  Themes     : {result['themes']}")
        print(f"  Phase      : {result['phase']}")
    else:
        print(f"❌  Render failed: {result}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
