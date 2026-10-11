"""FFT/STFT and statistical features computed from real recorded audio."""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

from config import FEATURE_SAMPLE_RATE
from src.audio_io import resample_audio

FEATURE_NAMES = [
    "log_rms",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
    "zero_crossing_rate",
    "signal_std",
    "peak_absolute_amplitude",
    "log_band_energy_0_250hz",
    "log_band_energy_250_500hz",
    "log_band_energy_500_1000hz",
    "log_band_energy_1000_2000hz",
    "log_band_energy_2000_4000hz",
]
BANDS_HZ = ((0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000))


def extract_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return 13 features from an actual audio segment.

    The signal is resampled to 8 kHz for a consistent feature space. This is
    a deterministic transformation of the supplied recording, not generated
    training audio.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 1024:
        raise ValueError("Audio segment is too short for the configured features (need >= 1024 samples).")
    if not np.all(np.isfinite(x)):
        raise ValueError("Audio contains NaN/Inf values")
    x = resample_audio(x, sample_rate, FEATURE_SAMPLE_RATE).astype(np.float64)
    if x.size < 1024:
        raise ValueError("Audio segment is too short after resampling")

    nperseg = min(1024, x.size)
    noverlap = min(768, nperseg // 2)
    frequencies, _, spectrum = stft(
        x,
        fs=FEATURE_SAMPLE_RATE,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    power = np.mean(np.abs(spectrum) ** 2, axis=1) + 1e-15
    total_power = float(np.sum(power))
    if total_power <= 0 or not np.isfinite(total_power):
        raise ValueError("Unable to calculate non-zero spectral power from this recording")

    non_dc = np.flatnonzero(frequencies > 0)
    peak_index = int(non_dc[np.argmax(power[non_dc])]) if non_dc.size else 0
    centroid = float(np.sum(frequencies * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * power) / total_power))
    flatness = float(np.exp(np.mean(np.log(power))) / (np.mean(power) + 1e-15))
    rms = float(np.sqrt(np.mean(x * x) + 1e-15))
    zcr = float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1])))

    values = [
        np.log10(rms + 1e-15),
        float(frequencies[peak_index]),
        centroid,
        bandwidth,
        flatness,
        zcr,
        float(np.std(x)),
        float(np.max(np.abs(x))),
    ]
    for low_hz, high_hz in BANDS_HZ:
        mask = (frequencies >= low_hz) & (frequencies < high_hz)
        values.append(float(np.log10(np.sum(power[mask]) + 1e-15)))

    features = np.asarray(values, dtype=np.float32)
    if features.size != len(FEATURE_NAMES) or not np.all(np.isfinite(features)):
        raise RuntimeError("Feature extraction returned an invalid feature vector")
    return features


def get_spectrum(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-sided magnitude spectrum for display, derived from real audio."""
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size < 8:
        raise ValueError("Audio segment is too short to calculate a spectrum")
    x = x - np.mean(x)
    window = np.hanning(x.size)
    transform = np.fft.rfft(x * window)
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    magnitude = np.abs(transform) / max(np.sum(window), 1e-12)
    return frequencies, magnitude
