"""Shared audio utilities for AI stages."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample, sosfilt


DF_SAMPLE_RATE = 48000


def ensure_channels(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples[np.newaxis, :]
    return samples


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples.astype(np.float32, copy=True)
    channels = ensure_channels(samples)
    ratio = target_sr / orig_sr
    new_length = int(round(channels.shape[1] * ratio))
    out = np.stack(
        [resample(ch, new_length).astype(np.float32) for ch in channels],
        axis=0,
    )
    return out


def mix_wet_dry(dry: np.ndarray, wet: np.ndarray, wet_amount: float) -> np.ndarray:
    wet_amount = float(np.clip(wet_amount, 0.0, 1.0))
    mixed = dry * (1.0 - wet_amount) + wet * wet_amount
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak * 0.99
    return mixed.astype(np.float32)


def apply_lowpass(samples: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Adaptive pre-filter recommended before AudioSR on MP3 sources."""
    nyquist = sr / 2
    cutoff = min(max(cutoff_hz, 2000.0), nyquist * 0.95)
    sos = butter(4, cutoff / nyquist, btype="low", output="sos")
    channels = ensure_channels(samples)
    filtered = np.stack([sosfilt(sos, ch) for ch in channels], axis=0)
    return filtered.astype(np.float32)


def estimate_cutoff_hz(high_end_rolloff_db: float, sr: int) -> float:
    """Estimate effective bandwidth cutoff from spectral roll-off."""
    if high_end_rolloff_db >= -3:
        return min(16000.0, sr / 2)
    severity = min(abs(high_end_rolloff_db), 20.0)
    return float(np.clip(16000.0 - severity * 350.0, 6000.0, 16000.0))
