#!/usr/bin/env python3
import argparse
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--speaker", default="")
    args = ap.parse_args()

    # Lazy import so you get a clean error if not installed
    from TTS.api import TTS

    # Load model (yes, this is heavy per call; we’ll optimize after it works)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    kwargs = dict(text=args.text, file_path=args.out, language=args.lang)
    if args.speaker:
        kwargs["speaker_wav"] = args.speaker

    tts.tts_to_file(**kwargs)

if __name__ == "__main__":
    main()

