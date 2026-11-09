"""Fallback artifact reduction when DeepFilterNet is unavailable."""

from __future__ import annotations

import logging

import numpy as np

from ..loudness import preserve_loudness, soft_ceiling
from .audio_utils import ensure_channels, mix_wet_dry

logger = logging.getLogger(__name__)


def reduce_artifacts_fallback(
    samples: np.ndarray,
    sample_rate: int,
    *,
    wet: float = 0.5,
) -> np.ndarray:
    """Gentle spectral denoising via noisereduce (no Rust required)."""
    import noisereduce as nr

    channels = ensure_channels(samples.astype(np.float32, copy=True))
    dry = channels
    wet_channels: list[np.ndarray] = []

    for channel in dry:
        reduced = nr.reduce_noise(
            y=channel,
            sr=sample_rate,
            prop_decrease=min(0.30, 0.15 + wet * 0.15),
            stationary=False,
        )
        wet_channels.append(np.asarray(reduced, dtype=np.float32))

    wet_audio = np.stack(wet_channels, axis=0)
    min_len = min(dry.shape[1], wet_audio.shape[1])
    result = mix_wet_dry(dry[:, :min_len], wet_audio[:, :min_len], wet)
    result, _ = preserve_loudness(dry[:, :min_len], result)
    result = soft_ceiling(result)
    logger.debug("Artifacts fallback complete (wet=%.2f)", wet)
    return result
