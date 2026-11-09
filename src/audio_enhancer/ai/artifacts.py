"""Phase 1 artifact reduction with automatic backend selection."""

from __future__ import annotations

import logging
import sys

import numpy as np

from .artifacts_deepfilter import reduce_artifacts_deepfilter
from .artifacts_fallback import reduce_artifacts_fallback
from .device import deepfilter_available

logger = logging.getLogger(__name__)

ARTIFACTS_BACKEND: str | None = None


def get_artifacts_backend() -> str:
    """Return the active artifacts backend name."""
    global ARTIFACTS_BACKEND
    if ARTIFACTS_BACKEND is None:
        ARTIFACTS_BACKEND = "deepfilter" if deepfilter_available() else "fallback"
    return ARTIFACTS_BACKEND


def artifacts_backend_label() -> str:
    backend = get_artifacts_backend()
    if backend == "deepfilter":
        return "DeepFilterNet3"
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"spectral-fallback (Python {py}, no DeepFilterNet wheel)"


def reduce_artifacts(
    samples: np.ndarray,
    sample_rate: int,
    *,
    wet: float = 0.7,
    device: str = "cpu",
) -> np.ndarray:
    """Reduce compression artifacts using the best available backend."""
    if get_artifacts_backend() == "deepfilter":
        return reduce_artifacts_deepfilter(
            samples,
            sample_rate,
            wet=wet,
            device=device,
        )
    return reduce_artifacts_fallback(samples, sample_rate, wet=wet)
