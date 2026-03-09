import os
import json
import subprocess
from pathlib import Path
from prosody import extract_prosody
from emotion_model import classify_emotion
from fusion.fusion import fuse
from ai_logic import respond
from mic import record

# Path to whisper.cpp binary: always resolved from candidates (no PATH fallback)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_exe = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
# Always resolve from these candidates (no PATH fallback)
WHISPER_CLI_CANDIDATES = [
    PROJECT_ROOT / "whisper.cpp" / "build" / "bin" / _exe,
    PROJECT_ROOT / "whisper.cpp" / "build" / "bin" / "Release" / _exe,
]

def _resolve_whisper_cli():
    env_val = os.environ.get("WHISPER_CLI")
    if env_val and Path(env_val).exists():
        return env_val
    for c in WHISPER_CLI_CANDIDATES:
        if c.exists():
            return str(c)
    return env_val  # may be set but path missing; subprocess will then raise with clear message

WHISPER_CLI = _resolve_whisper_cli()
# Model file (ggml-*.bin)
WHISPER_MODEL = PROJECT_ROOT / "whisper.cpp" / "models" / "ggml-small-q5_1.bin"

def whisper_transcribe(path):
    if WHISPER_CLI is None:
        tried = [str(c) for c in WHISPER_CLI_CANDIDATES if c is not None]
        raise FileNotFoundError(
            "whisper.cpp binary not found. Tried candidates: " + ", ".join(tried) + ". "
            "Build: cd whisper.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release. "
            "Or set env WHISPER_CLI=<path-to-whisper-cli.exe>"
        )
    if not WHISPER_MODEL.exists():
        raise FileNotFoundError(
            f"Whisper model not found at {WHISPER_MODEL}. "
            "Download a ggml model (e.g. ggml-small-q5_1.bin) into the models/ folder."
        )
    audio_abs = Path(path).resolve()
    args = [WHISPER_CLI, "-m", str(WHISPER_MODEL), "-f", str(audio_abs), "-l", "sv", "-oj"]
    try:
        result = subprocess.run(args, capture_output=True, text=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"whisper.cpp binary not found (tried: {WHISPER_CLI}). "
            "The executable is named whisper-cli.exe (not main.exe). "
            "Build: cd whisper.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release. "
            "Unset WHISPER_CLI to auto-detect, or set it to the path of whisper-cli.exe"
        ) from None
    if result.returncode != 0:
        raise RuntimeError(f"Whisper failed: {result.stderr or result.stdout}")

    # whisper.cpp -oj writes to <input>.json (same dir as audio), not stdout
    json_path = Path(str(path)).resolve().with_suffix(Path(path).suffix + ".json")
    if not json_path.exists():
        json_path = Path(str(path) + ".json").resolve()
    if not json_path.exists():
        return ""

    data = json.loads(json_path.read_text(encoding="utf-8"))
    # CLI structure: top-level "transcription" array, each item has "text"
    if "transcription" in data:
        return " ".join(s.get("text", "").strip() for s in data["transcription"]).strip()
    if "text" in data:
        return data["text"].strip()
    if "segments" in data:
        return " ".join(s.get("text", "").strip() for s in data["segments"]).strip()
    return ""

if __name__ == "__main__":
    while True:
        # 1. Record audio
        audio_path = record(5, "mic.wav")

        # 2. Whisper transcription (whisper.cpp)
        text = whisper_transcribe(audio_path)
        print("TEXT:", text)

        # 3. Prosody features
        features = extract_prosody(audio_path)
        print("FEATURES:", features)

        # 4. Emotion classification
        emotion = classify_emotion(features)
        print("EMOTION:", emotion)

        # 5. Fusion layer
        fused = fuse(text, emotion)
        print("FUSED:", fused)

        # 6. AI response
        reply = respond(fused)
        print("AI:", reply)
        print()