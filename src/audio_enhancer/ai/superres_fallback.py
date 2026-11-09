"""Fallback super-resolution when AudioSR is unavailable."""

from __future__ import annotations

import logging

import numpy as np
from pedalboard import HighShelfFilter, Pedalboard, PeakFilter

from ..analyze import AnalysisResult
from ..loudness import preserve_loudness, soft_ceiling
from .audio_utils import apply_lowpass, ensure_channels, estimate_cutoff_hz

logger = logging.getLogger(__name__)


def _apply_board(samples: np.ndarray, sr: int, board: Pedalboard) -> np.ndarray:
    channels = ensure_channels(samples)
    out_channels = []
    for ch in channels:
        out = board(ch, sr)
        out_channels.append(np.asarray(out, dtype=np.float32))
    min_len = min(len(c) for c in out_channels)
    return np.stack([c[:min_len] for c in out_channels], axis=0)


def super_resolve_fallback(
    samples: np.ndarray,
    sample_rate: int,
    analysis: AnalysisResult,
) -> np.ndarray:
    """Restore perceived highs with subtle adaptive EQ when AudioSR cannot run."""
    channels = ensure_channels(samples.astype(np.float32, copy=True))
    reference = channels
    cutoff = estimate_cutoff_hz(analysis.high_end_rolloff_db, sample_rate)
    filtered = apply_lowpass(channels, sample_rate, cutoff)

    severity = min(abs(analysis.high_end_rolloff_db), 18.0)
    shelf_gain = float(np.clip(severity * 0.18, 0.5, 2.5))
    air_gain = float(np.clip(severity * 0.1, 0.3, 1.5))

    board = Pedalboard(
        [
            HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=shelf_gain),
            PeakFilter(cutoff_frequency_hz=12000.0, gain_db=air_gain, q=0.7),
        ]
    )
    enhanced = _apply_board(filtered, sample_rate, board)
    enhanced, _ = preserve_loudness(reference, enhanced)
    enhanced = soft_ceiling(enhanced)

    logger.debug(
        "Super-res fallback (cutoff=%.0f Hz, shelf=+%.1f dB)",
        cutoff,
        shelf_gain,
    )
    return enhanced.astype(np.float32)
