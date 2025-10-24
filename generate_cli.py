#!/usr/bin/env python3
"""
generate_cli.py

Simple CLI wrapper that:
- loads CSM-1B from MODEL_DIR using the repo helper if available, and
- synthesizes text (from a text file) to a WAV at output.
"""
import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=str, default=os.environ.get("MODEL_DIR", "/workspace/csm-1b"))
parser.add_argument("--text-file", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--speaker", type=int, default=0)
parser.add_argument("--max-ms", type=int, default=10000)
args = parser.parse_args()

MODEL_DIR = args.model_dir
TEXT_FILE = args.text_file
OUTPUT = args.output

# Try to use the sesame csm generator helper (from README examples)
try:
    # Try import from local csm clone or installed package
    sys.path.insert(0, str(Path.cwd() / "csm"))
    from generator import load_csm_1b, Segment  # generator is referenced in the quickstart you provided

    # choose device
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"[generate_cli] Using device: {device}")
    gen = load_csm_1b(device=device, model_dir=MODEL_DIR) if "model_dir" in load_csm_1b.__code__.co_varnames else load_csm_1b(device=device)

    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Generate audio
    audio = gen.generate(
        text=text,
        speaker=args.speaker,
        context=[],
        max_audio_length_ms=args.max_ms,
    )
    # audio expected to be a torch tensor or numpy array: try to write via torchaudio / soundfile
    try:
        import torchaudio
        import torch
        if isinstance(audio, tuple) or isinstance(audio, list):
            # sometimes generator returns (audio_tensor, sample_rate)
            audio_arr = audio[0]
            sr = audio[1] if len(audio) > 1 else gen.sample_rate
        else:
            audio_arr = audio
            sr = getattr(gen, "sample_rate", 24000)
        # Ensure tensor on CPU
        if hasattr(audio_arr, "unsqueeze"):
            # torch tensor
            torchaudio.save(OUTPUT, audio_arr.unsqueeze(0).cpu(), sr)
        else:
            # numpy array
            import soundfile as sf
            sf.write(OUTPUT, audio_arr, sr)
        print("Wrote output to", OUTPUT)
        sys.exit(0)
    except Exception as e:
        print("Failed to write audio via torchaudio/soundfile:", e, file=sys.stderr)
        sys.exit(2)

except Exception as e:
    print("Could not use local csm generator helper:", e, file=sys.stderr)

# Fallback: try to find an example script and call it
candidates = [
    Path("csm") / "examples" / "run_csm.py",
    Path("csm") / "examples" / "generate.py",
    Path("csm") / "examples" / "synthesize.py",
    Path("csm") / "examples" / "tts.py",
]
for cand in candidates:
    if cand.exists():
        cmd = f"{sys.executable} {str(cand)} --model-dir {MODEL_DIR} --text-file {TEXT_FILE} --output {OUTPUT}"
        print("Running example script:", cmd)
        rc = os.system(cmd)
        if rc == 0:
            print("Wrote output to", OUTPUT)
            sys.exit(0)
        else:
            print("Example script failed with rc", rc, file=sys.stderr)
            sys.exit(rc)

print("No usable generation method found. Update generate_cli.py to call the proper csm TTS entrypoint.", file=sys.stderr)
sys.exit(3)
