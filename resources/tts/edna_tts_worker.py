#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--use-cuda", action="store_true")
    ap.add_argument("--language", default="")
    ap.add_argument("--speaker-wav", default="")
    args = ap.parse_args()

    # Import here so failures show up immediately in the worker output
    try:
        from TTS.api import TTS
    except Exception as e:
        print(json.dumps({"ready": False, "error": f"import TTS failed: {e}"}), flush=True)
        return 2

    try:
        # Load model once (this is the whole point)
        tts = TTS(model_name=args.model, progress_bar=False, gpu=args.use_cuda)
    except Exception as e:
        print(json.dumps({"ready": False, "error": f"load model failed: {e}"}), flush=True)
        return 2

    print(json.dumps({"ready": True, "model": args.model, "gpu": bool(args.use_cuda)}), flush=True)

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"bad json: {e}"}), flush=True)
            continue

        if req.get("cmd") == "quit":
            print(json.dumps({"ok": True, "bye": True}), flush=True)
            break

        req_id = req.get("id", 0)
        text = req.get("text", "")
        if not isinstance(text, str) or not text.strip():
            print(json.dumps({"id": req_id, "ok": False, "error": "empty text"}), flush=True)
            continue

        try:
            fd, path = tempfile.mkstemp(prefix="edna_tts_", suffix=".wav", dir=os.getenv("TMPDIR") or "/tmp")
            os.close(fd)

            # Basic synth to wav path
            if args.speaker_wav:
                tts.tts_to_file(text=text, file_path=path, speaker_wav=args.speaker_wav, language=args.language or None)
            else:
                # vits uses speaker_wav/language not needed
                tts.tts_to_file(text=text, file_path=path)

            print(json.dumps({"id": req_id, "ok": True, "wav": path}), flush=True)
        except Exception as e:
            print(json.dumps({"id": req_id, "ok": False, "error": str(e)}), flush=True)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

