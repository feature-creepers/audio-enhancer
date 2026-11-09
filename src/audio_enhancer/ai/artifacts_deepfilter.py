"""Phase 1: artifact reduction via DeepFilterNet3."""

from __future__ import annotations

import logging

import numpy as np

from .audio_utils import DF_SAMPLE_RATE, ensure_channels, mix_wet_dry, resample_audio
from .model_cache import get_cache

logger = logging.getLogger(__name__)


def reduce_artifacts_deepfilter(
    samples: np.ndarray,
    sample_rate: int,
    *,
    wet: float = 0.7,
    device: str = "cpu",
) -> np.ndarray:
    """Apply conservative DeepFilterNet3 denoising with wet/dry mix."""
    import torch
    from df.enhance import enhance as df_enhance

    cache = get_cache()
    model, state = cache.ensure_deepfilter(device)

    channels = ensure_channels(samples.astype(np.float32, copy=True))
    dry = channels
    dry_48k = resample_audio(dry, sample_rate, DF_SAMPLE_RATE)

    enhanced_channels: list[np.ndarray] = []
    for channel in dry_48k:
        tensor = torch.from_numpy(channel).unsqueeze(0).float()
        with torch.no_grad():
            out = df_enhance(
                model,
                state,
                tensor,
                pad=True,
            )
        enhanced_channels.append(out.squeeze(0).cpu().numpy().astype(np.float32))

    wet_audio = np.stack(enhanced_channels, axis=0)
    wet_audio = resample_audio(wet_audio, DF_SAMPLE_RATE, sample_rate)

    min_len = min(dry.shape[1], wet_audio.shape[1])
    result = mix_wet_dry(dry[:, :min_len], wet_audio[:, :min_len], wet)
    logger.debug("DeepFilterNet artifacts phase complete (wet=%.2f)", wet)
    return result
