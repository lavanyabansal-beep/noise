"""
Silero Noise Suppression wrapper.

This module provides a minimal, safe wrapper around Silero-style
noise suppression implementations. The real Silero packages are
optional; when not installed and `use_fallback=True` the function
falls back to a lightweight scipy-based spectral gating.

Install hint (if you want the true Silero models):
  pip install git+https://github.com/snakers4/silero-models

"""
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _check_silero_package() -> bool:
    """Check common silero package names without forcing downloads."""
    try:
        import silero  # type: ignore
        return True
    except Exception:
        pass

    try:
        import silero_ns  # type: ignore
        return True
    except Exception:
        pass

    return False


def _scipy_spectral_fallback(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """A minimal scipy-based spectral gating fallback.

    This is intentionally simple: it is not a drop-in replacement for
    the Silero models, but gives a reasonable reduction of stationary
    noise when no deep model is available.
    """
    try:
        from scipy.signal import stft, istft
    except Exception as e:
        raise RuntimeError("scipy is required for fallback mode. Install with: pip install scipy") from e

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Short-time Fourier transform
    nperseg = min(2048, max(256, int(0.025 * sample_rate)))
    noverlap = nperseg // 2
    f, t, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    # Estimate noise magnitude from first 0.5s or first 10 frames
    n_frames = Zxx.shape[1]
    est_frames = min(n_frames, max(1, int(0.5 * sample_rate / (nperseg - noverlap))))
    noise_mag = np.mean(np.abs(Zxx[:, :est_frames]), axis=1, keepdims=True)

    # Simple spectral subtraction with floor
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    gain = np.maximum(0.0, 1.0 - 1.2 * (noise_mag / (mag + 1e-8)))
    gain = np.clip(gain, 0.0, 1.0)

    Zxx_enh = gain * mag * np.exp(1j * phase)
    _, xrec = istft(Zxx_enh, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    # Match length
    if len(xrec) > len(audio):
        xrec = xrec[: len(audio)]
    elif len(xrec) < len(audio):
        xrec = np.pad(xrec, (0, len(audio) - len(xrec)))

    # Normalize
    xrec = xrec.astype(np.float32)
    maxv = np.max(np.abs(xrec))
    if maxv > 1.0:
        xrec = xrec / maxv * 0.95

    return xrec


def silero_suppress(
    audio: np.ndarray,
    sample_rate: int,
    use_fallback: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """Apply Silero-style noise suppression.

    Attempts to use installed Silero implementations if present. If not
    available and `use_fallback=True`, a scipy-based spectral gating is
    used instead.
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if _check_silero_package():
        try:
            # Try importing common APIs dynamically to avoid hard dependency
            mod = __import__("silero")
        except Exception:
            try:
                mod = __import__("silero_ns")
            except Exception:
                mod = None

        if mod is not None:
            # Attempt several common API patterns. If none match, raise
            # a helpful error so the user can adjust installation.
            try:
                # Pattern 1: module.apply_suppression / suppress
                if hasattr(mod, "apply_suppression"):
                    return mod.apply_suppression(audio, sample_rate)
                if hasattr(mod, "suppress"):
                    return mod.suppress(audio, sample_rate)

                # Pattern 2: torch.hub loaded model or callable
                if hasattr(mod, "load_model"):
                    model = mod.load_model(device=device)
                    if callable(model):
                        return model(audio, sample_rate)

                # Last resort: attempt attribute 'SileroNS'
                if hasattr(mod, "SileroNS"):
                    cls = getattr(mod, "SileroNS")
                    inst = cls(device=device)
                    if hasattr(inst, "suppress"):
                        return inst.suppress(audio, sample_rate)

            except Exception as e:
                logger.warning("Silero package present but invocation failed: %s", e)
                # fall through to fallback if allowed

    # If we reach here, either package not available or invocation failed
    if use_fallback:
        return _scipy_spectral_fallback(audio, sample_rate)

    raise RuntimeError(
        "Silero noise suppression not available. Install a Silero implementation "
        "or call with use_fallback=True to use a scipy-based fallback. "
        "Install hint: pip install git+https://github.com/snakers4/silero-models"
    )


def is_silero_available(include_fallback: bool = False) -> bool:
    """Return True if Silero package or fallback is available."""
    if _check_silero_package():
        return True
    if include_fallback:
        try:
            import scipy  # type: ignore
            return True
        except Exception:
            return False
    return False
