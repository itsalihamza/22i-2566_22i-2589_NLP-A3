"""Download the local speech models into models/ (one time, ~210 MB; everything runs offline after).

    python scripts/download_models.py                  # Whisper base.en + Piper en_US-lessac-medium
    python scripts/download_models.py --asr tiny.en    # a smaller/faster Whisper

- ASR: faster-whisper conversion of OpenAI Whisper (CTranslate2 format), from Hugging Face.
- TTS: Piper voice (ONNX model + its JSON config) from the rhasspy/piper-voices repository.
The embedding model for retrieval (all-MiniLM-L6-v2) is downloaded by scripts/build_index.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.voice import config  # noqa: E402

PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already there: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for block in r.iter_bytes(1 << 20):
                f.write(block)
    tmp.replace(dest)
    print(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", default=config.ASR_MODEL, help="Whisper size: tiny.en, base.en, small.en")
    parser.add_argument("--voice", default=config.TTS_VOICE, help="Piper voice, e.g. en_US-lessac-medium")
    args = parser.parse_args()

    from faster_whisper import download_model

    asr_dir = config.MODELS_DIR / f"whisper-{args.asr}"
    print(f"ASR: faster-whisper {args.asr} -> {asr_dir}")
    download_model(args.asr, output_dir=str(asr_dir))

    name, quality = args.voice.split("-")[1], args.voice.split("-")[2]
    print(f"TTS: Piper {args.voice} -> {config.MODELS_DIR / 'piper'}")
    for suffix in (".onnx", ".onnx.json"):
        download(f"{PIPER_BASE}/{name}/{quality}/{args.voice}{suffix}", config.MODELS_DIR / "piper" / f"{args.voice}{suffix}")
    print("Done. Set ASR_MODEL / TTS_VOICE if you downloaded non-default models.")


if __name__ == "__main__":
    main()
