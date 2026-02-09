"""Pytest tests for the new suppressor wrappers.

These tests use the scipy fallback to remain dependency-light and
verify basic output properties (shape, dtype, finite values).
"""
import numpy as np
from audio_noise_lab.suppressors import silero_ns, nsnet2


def make_noisy_tone(sr=16000, duration_s=1.0, freq=440.0, noise_amp=0.3):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    tone = 0.6 * np.sin(2 * np.pi * freq * t)
    noise = noise_amp * np.random.RandomState(0).randn(len(t))
    return (tone + noise).astype(np.float32), sr


def check_output(audio_in, audio_out):
    # Output same length
    assert audio_out.shape == audio_in.shape
    # Type
    assert audio_out.dtype == np.float32
    # Finite
    assert np.isfinite(audio_out).all()
    # Not all zeros (processing should change signal)
    assert not np.allclose(audio_out, audio_in)


def test_silero_fallback():
    audio, sr = make_noisy_tone()
    out = silero_ns.silero_suppress(audio, sr, use_fallback=True)
    check_output(audio, out)


def test_nsnet2_fallback():
    audio, sr = make_noisy_tone()
    out = nsnet2.nsnet2_suppress(audio, sr, use_fallback=True)
    check_output(audio, out)
