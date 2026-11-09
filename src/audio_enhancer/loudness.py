"""Loudness preservation utilities."""

from __future__ import annotations

import numpy as np

from .ai.audio_utils import ensure_channels


def rms_linear(samples: np.ndarray) -> float:
    mono = ensure_channels(samples)
    if mono.shape[0] > 1:
        mono = np.mean(mono, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(mono**2)))


def rms_dbfs(samples: np.ndarray) -> float:
    rms = rms_linear(samples)
    return 20 * np.log10(max(rms, 1e-10))


def preserve_loudness(
    reference: np.ndarray,
    processed: np.ndarray,
    *,
    max_lift_db: float = 1.0,
    quiet_threshold_db: float = -18.0,
    max_excess_db: float = 0.3,
) -> tuple[np.ndarray, str]:
    """Match processed RMS to reference, with optional lift for quiet sources."""
    ref_rms = rms_linear(reference)
    out_rms = rms_linear(processed)

    if out_rms < 1e-10:
        return processed.astype(np.float32), "Loudness: unchanged (silent output)"

    ref_db = 20 * np.log10(max(ref_rms, 1e-10))
    out_db = 20 * np.log10(out_rms)

    if ref_db < quiet_threshold_db:
        target_db = min(ref_db + max_lift_db, ref_db + max(0.0, ref_db - out_db))
        target_rms = 10 ** (target_db / 20)
    else:
        max_target_rms = ref_rms * (10 ** (max_excess_db / 20))
        target_rms = min(ref_rms, max_target_rms)

    gain = target_rms / out_rms
    matched = (processed * gain).astype(np.float32)
    matched_db = 20 * np.log10(max(rms_linear(matched), 1e-10))
    return matched, f"Loudness: matched (source {ref_db:.1f} dBFS, output {matched_db:.1f} dBFS)"


def soft_ceiling(samples: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Attenuate only when peaks exceed linear ceiling."""
    peak = float(np.max(np.abs(samples)))
    if peak <= ceiling:
        return samples.astype(np.float32)
    return (samples * (ceiling / peak)).astype(np.float32)
