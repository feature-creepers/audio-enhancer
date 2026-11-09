"""AI enhancement facade."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .analyze import AnalysisResult

logger = logging.getLogger(__name__)


def enhance(
    audio: np.ndarray,
    sr: int,
    *,
    mode: str = "auto",
    analysis: AnalysisResult | None = None,
    source_bitrate_kbps: int | None = None,
    device: str = "auto",
    ai_fast: bool = False,
    ai_wet: float = 0.7,
    force_superres: bool = False,
) -> tuple[np.ndarray, str]:
    """Run the AI pipeline. Requires analysis for adaptive routing."""
    from .ai.router import AiOptions, run_ai_pipeline

    if analysis is None:
        logger.warning("AI enhance called without analysis — skipping")
        return audio, "AI: skipped (no analysis)"

    options = AiOptions(
        mode=mode,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        ai_fast=ai_fast,
        ai_wet=ai_wet,
        force_superres=force_superres,
    )
    processed, _plan, summary = run_ai_pipeline(
        audio,
        sr,
        analysis=analysis,
        source_bitrate_kbps=source_bitrate_kbps,
        options=options,
    )
    return processed, summary
