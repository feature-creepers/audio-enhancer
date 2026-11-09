"""Phase 2: bandwidth restoration with automatic backend selection."""

from __future__ import annotations

import logging
import sys

import numpy as np

from ..analyze import AnalysisResult
from .device import audiosr_available
from .superres_audiosr import super_resolve_audiosr
from .superres_fallback import super_resolve_fallback

logger = logging.getLogger(__name__)

SUPERRES_BACKEND: str | None = None


def get_superres_backend() -> str:
    global SUPERRES_BACKEND
    if SUPERRES_BACKEND is None:
        SUPERRES_BACKEND = "audiosr" if audiosr_available() else "fallback"
    return SUPERRES_BACKEND


def superres_backend_label() -> str:
    backend = get_superres_backend()
    if backend == "audiosr":
        return "AudioSR"
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"EQ-fallback (Python {py}, no AudioSR wheel)"


def super_resolve(
    samples: np.ndarray,
    sample_rate: int,
    analysis: AnalysisResult,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Restore high frequencies with the best available backend."""
    if get_superres_backend() == "audiosr":
        return super_resolve_audiosr(
            samples,
            sample_rate,
            analysis,
            device=device,
        )
    return super_resolve_fallback(samples, sample_rate, analysis)
