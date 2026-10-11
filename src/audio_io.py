"""Audio loading helpers. Inputs must be user-provided or real dataset WAVs."""
from __future__ import annotations

from pathlib import Path
from math import gcd
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_wav(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load a real WAV file as mono float32 audio; do not synthesize or pad data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if audio.ndim == 2:
        # Average channels to mono for this single-channel analysis pipeline.
        audio = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    elif audio.ndim != 1:
        raise ValueError(f"Unsupported WAV shape {audio.shape} in {path}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Audio contains NaN/Inf values: {path}")
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate {sample_rate} in {path}")
    return audio.astype(np.float32, copy=False), int(sample_rate)


def center_crop(audio: np.ndarray, sample_rate: int, duration_seconds: float) -> np.ndarray:
    """Return a centered real-data segment. Reject short files rather than padding."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    requested = int(round(float(duration_seconds) * int(sample_rate)))
    if requested <= 0:
        raise ValueError("duration_seconds must be positive")
    if x.size < requested:
        raise ValueError(
            f"Recording is too short: has {x.size / sample_rate:.3f}s, "
            f"but {duration_seconds:.3f}s is required. No padding is performed."
        )
    start = (x.size - requested) // 2
    return x[start:start + requested].copy()


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample an actual recording using rational polyphase resampling."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return x.copy()
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    divisor = gcd(int(source_rate), int(target_rate))
    y = resample_poly(x, target_rate // divisor, source_rate // divisor)
    return y.astype(np.float32, copy=False)


def load_segment(path: str | Path, duration_seconds: float) -> tuple[np.ndarray, int]:
    """Load a centered segment from a real WAV without inventing samples."""
    audio, sample_rate = load_wav(path)
    return center_crop(audio, sample_rate, duration_seconds), sample_rate
