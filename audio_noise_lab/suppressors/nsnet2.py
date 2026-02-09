"""
NSNet2 Noise Suppression wrapper.

Provides a thin wrapper around NSNet2-style suppressors. If an
NSNet2 implementation is installed this wrapper will attempt to call
it using common API patterns. Otherwise, when `use_fallback=True` a
small scipy-based spectral gating is used.

Install hint (example):
  pip install nsnet2

"""
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _check_nsnet2_package() -> bool:
    """Check for likely nsnet2 package names without forcing installs."""
    try:
        import nsnet2  # type: ignore
        return True
    except Exception:
        pass

    try:
        import nsnet  # type: ignore
        return True
    except Exception:
        pass

    return False


def _scipy_spectral_fallback(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Fallback that mirrors `silero_ns` fallback behaviour.

    Keeps behaviour consistent between wrappers so callers can rely on
    similar fallback results when deep models are absent.
    """
    try:
        from scipy.signal import stft, istft
    except Exception as e:
        raise RuntimeError("scipy is required for fallback mode. Install with: pip install scipy") from e

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    nperseg = min(2048, max(256, int(0.025 * sample_rate)))
    noverlap = nperseg // 2
    f, t, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    n_frames = Zxx.shape[1]
    est_frames = min(n_frames, max(1, int(0.5 * sample_rate / (nperseg - noverlap))))
    noise_mag = np.mean(np.abs(Zxx[:, :est_frames]), axis=1, keepdims=True)

    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    # Slightly different subtraction factor to vary behaviour from silero fallback
    gain = np.maximum(0.0, 1.0 - 1.0 * (noise_mag / (mag + 1e-8)))
    gain = np.clip(gain, 0.0, 1.0)

    Zxx_enh = gain * mag * np.exp(1j * phase)
    _, xrec = istft(Zxx_enh, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    if len(xrec) > len(audio):
        xrec = xrec[: len(audio)]
    elif len(xrec) < len(audio):
        xrec = np.pad(xrec, (0, len(audio) - len(xrec)))

    xrec = xrec.astype(np.float32)
    maxv = np.max(np.abs(xrec))
    if maxv > 1.0:
        xrec = xrec / maxv * 0.95

    return xrec


def nsnet2_suppress(
    audio: np.ndarray,
    sample_rate: int,
    use_fallback: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """Apply NSNet2-style noise suppression.

    Attempts to call installed NSNet2-style implementations. If not
    available and `use_fallback=True`, a scipy-based fallback is used.
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if _check_nsnet2_package():
        try:
            mod = __import__("nsnet2")
        except Exception:
            try:
                mod = __import__("nsnet")
            except Exception:
                mod = None

        if mod is not None:
            try:
                # Common API: class NSNet2 or function enhance
                if hasattr(mod, "enhance"):
                    return mod.enhance(audio, sample_rate)
                if hasattr(mod, "process"):
                    return mod.process(audio, sample_rate)

                if hasattr(mod, "NSNet2"):
                    cls = getattr(mod, "NSNet2")
                    inst = cls(device=device)
                    if hasattr(inst, "forward"):
                        out = inst.forward(audio, sample_rate)
                        return out
                    if hasattr(inst, "enhance"):
                        return inst.enhance(audio, sample_rate)

            except Exception as e:
                logger.warning("NSNet2 package present but invocation failed: %s", e)

    if use_fallback:
        return _scipy_spectral_fallback(audio, sample_rate)

    raise RuntimeError(
        "NSNet2 noise suppression not available. Install an NSNet2 implementation "
        "or call with use_fallback=True to use a scipy-based fallback. "
        "Install hint: pip install nsnet2"
    )


def is_nsnet2_available(include_fallback: bool = False) -> bool:
    if _check_nsnet2_package():
        return True
    if include_fallback:
        try:
            import scipy  # type: ignore
            return True
        except Exception:
            return False
    return False
