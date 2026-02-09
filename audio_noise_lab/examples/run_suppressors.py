"""Example script demonstrating Silero and NSNet2 suppressor wrappers.

This script generates a short synthetic noisy signal and processes it
with both `silero_ns.silero_suppress` and `nsnet2.nsnet2_suppress` using
the scipy fallback (so it runs without extra model downloads).

Run:
    python -m audio_noise_lab.examples.run_suppressors

Outputs are written to `audio_noise_lab/examples/output/`.
"""
from pathlib import Path
import numpy as np

from audio_noise_lab.utils.audio_io import save_audio
from audio_noise_lab.suppressors import silero_ns, nsnet2


def make_noisy_tone(sr=16000, duration_s=2.0, freq=440.0, noise_amp=0.3):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    tone = 0.6 * np.sin(2 * np.pi * freq * t)
    noise = noise_amp * np.random.randn(len(t))
    return (tone + noise).astype(np.float32), sr


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = make_noisy_tone()

    # Save original
    save_audio(audio, sr, path=out_dir / "original.wav")

    # Process with Silero wrapper (fallback)
    try:
        out_silero = silero_ns.silero_suppress(audio, sr, use_fallback=True)
        save_audio(out_silero, sr, path=out_dir / "silero_suppressed.wav")
        print("Wrote silero_suppressed.wav")
    except Exception as e:
        print("Silero suppress failed:", e)

    # Process with NSNet2 wrapper (fallback)
    try:
        out_nsnet2 = nsnet2.nsnet2_suppress(audio, sr, use_fallback=True)
        save_audio(out_nsnet2, sr, path=out_dir / "nsnet2_suppressed.wav")
        print("Wrote nsnet2_suppressed.wav")
    except Exception as e:
        print("NSNet2 suppress failed:", e)


if __name__ == "__main__":
    main()
