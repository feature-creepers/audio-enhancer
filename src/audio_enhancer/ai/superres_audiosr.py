"""AudioSR-backed super-resolution."""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np

from ..analyze import AnalysisResult
from .audio_utils import apply_lowpass, ensure_channels, estimate_cutoff_hz, resample_audio
from .model_cache import get_cache

logger = logging.getLogger(__name__)

AUDIOSR_SAMPLE_RATE = 48000


def _to_numpy(waveform) -> np.ndarray:
    import torch

    if isinstance(waveform, torch.Tensor):
        arr = waveform.detach().cpu().numpy()
    else:
        arr = np.asarray(waveform)
    if arr.ndim == 1:
        return arr[np.newaxis, :].astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr.astype(np.float32)


def super_resolve_audiosr(
    samples: np.ndarray,
    sample_rate: int,
    analysis: AnalysisResult,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Restore high frequencies with AudioSR and adaptive pre-filtering."""
    import audiosr
    import soundfile as sf

    channels = ensure_channels(samples.astype(np.float32, copy=True))
    cutoff = estimate_cutoff_hz(analysis.high_end_rolloff_db, sample_rate)
    filtered = apply_lowpass(channels, sample_rate, cutoff)
    logger.debug("AudioSR pre-filter cutoff: %.0f Hz", cutoff)

    cache = get_cache()
    model = cache.ensure_audiosr(device)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, filtered.T, sample_rate)

        waveform = audiosr.super_resolution(
            model,
            tmp_path,
            seed=42,
            guidance_scale=3.5,
            ddim_steps=50,
            device=device,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    enhanced = _to_numpy(waveform)
    if enhanced.shape[0] == 1 and channels.shape[0] == 2:
        enhanced = np.repeat(enhanced, 2, axis=0)

    if enhanced.shape[1] != channels.shape[1] or sample_rate != AUDIOSR_SAMPLE_RATE:
        enhanced = resample_audio(enhanced, AUDIOSR_SAMPLE_RATE, sample_rate)
        min_len = min(channels.shape[1], enhanced.shape[1])
        enhanced = enhanced[:, :min_len]

    peak = np.max(np.abs(enhanced))
    if peak > 1.0:
        enhanced = enhanced / peak * 0.99

    logger.debug("AudioSR super-resolution complete")
    return enhanced.astype(np.float32)
