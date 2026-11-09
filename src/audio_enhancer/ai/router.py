"""Adaptive AI routing and pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..analyze import AnalysisResult
from .artifacts import artifacts_backend_label, reduce_artifacts
from .device import audiosr_available, is_cuda, resolve_device
from .model_cache import get_cache
from .superres import super_resolve, superres_backend_label

logger = logging.getLogger(__name__)

AiMode = Literal["auto", "artifacts", "superres", "full"]

PHASE1_ROLLOFF_THRESHOLD = -3.0
PHASE2_ROLLOFF_THRESHOLD = -6.0
LOW_BITRATE_THRESHOLD_KBPS = 256


@dataclass
class AiOptions:
    mode: AiMode = "auto"
    device: str = "auto"
    ai_fast: bool = False
    ai_wet: float = 0.5
    force_superres: bool = False


@dataclass
class AiPlan:
    run_artifacts: bool
    run_superres: bool
    reasons: list[str]


def plan_ai_stages(
    analysis: AnalysisResult,
    options: AiOptions,
    *,
    source_bitrate_kbps: int | None,
    resolved_device: str,
) -> AiPlan:
    """Decide which AI phases to run."""
    reasons: list[str] = []
    mode = options.mode

    if mode in ("artifacts", "full"):
        run_artifacts = True
        reasons.append(f"mode={mode}")
    elif mode == "superres":
        run_artifacts = False
    else:
        low_bitrate = (
            source_bitrate_kbps is not None
            and source_bitrate_kbps < LOW_BITRATE_THRESHOLD_KBPS
        )
        run_artifacts = (
            analysis.high_end_rolloff_db < PHASE1_ROLLOFF_THRESHOLD or low_bitrate
        )
        if run_artifacts:
            reasons.append("compressed source detected")

    if mode in ("superres", "full"):
        run_superres = True
        reasons.append(f"mode={mode}")
    elif mode == "artifacts":
        run_superres = False
    else:
        severe_rolloff = analysis.high_end_rolloff_db < PHASE2_ROLLOFF_THRESHOLD
        if not severe_rolloff:
            run_superres = False
        elif options.ai_fast and not is_cuda(resolved_device) and not options.force_superres:
            run_superres = False
            reasons.append("phase2 skipped (--ai-fast on CPU)")
        elif (
            not is_cuda(resolved_device)
            and not options.force_superres
            and audiosr_available()
        ):
            run_superres = False
            reasons.append("phase2 skipped (AudioSR needs GPU, use --ai-force-superres)")
        else:
            run_superres = True
            if audiosr_available():
                reasons.append("severe high-end roll-off")
            else:
                reasons.append("severe roll-off (EQ-fallback)")

    return AiPlan(run_artifacts=run_artifacts, run_superres=run_superres, reasons=reasons)


def format_ai_plan(plan: AiPlan, device: str, options: AiOptions) -> str:
    parts: list[str] = []
    if plan.run_artifacts:
        parts.append(f"phase1 artifacts ({artifacts_backend_label()}, wet={options.ai_wet:.1f})")
    if plan.run_superres:
        parts.append(f"phase2 superres ({superres_backend_label()}, {device})")
    if not parts:
        return "AI: no phases triggered (source looks clean)"
    return "AI: " + " -> ".join(parts)


def warmup_for_plan(plan: AiPlan, device: str) -> None:
    get_cache().warmup(
        need_artifacts=plan.run_artifacts,
        need_superres=plan.run_superres,
        device=device,
    )


def warmup_for_mode(mode: AiMode, device: str, ai_fast: bool, force_superres: bool) -> None:
    """Warmup models for batch — load everything mode might need."""
    need_artifacts = mode in ("artifacts", "full", "auto")
    need_superres = mode in ("superres", "full")
    if mode == "auto" and not ai_fast:
        resolved = resolve_device(device)  # type: ignore[arg-type]
        if is_cuda(resolved) or force_superres:
            need_superres = True
    get_cache().warmup(
        need_artifacts=need_artifacts,
        need_superres=need_superres,
        device=device,
    )


def run_ai_pipeline(
    samples: np.ndarray,
    sample_rate: int,
    *,
    analysis: AnalysisResult,
    source_bitrate_kbps: int | None = None,
    options: AiOptions | None = None,
) -> tuple[np.ndarray, AiPlan, str]:
    """Run selected AI phases and return processed audio."""
    opts = options or AiOptions()
    device = resolve_device(opts.device)  # type: ignore[arg-type]
    plan = plan_ai_stages(
        analysis,
        opts,
        source_bitrate_kbps=source_bitrate_kbps,
        resolved_device=device,
    )

    audio = samples.astype(np.float32, copy=True)
    if not plan.run_artifacts and not plan.run_superres:
        return audio, plan, format_ai_plan(plan, device, opts)

    if plan.run_artifacts:
        audio = reduce_artifacts(
            audio,
            sample_rate,
            wet=opts.ai_wet,
            device=device,
        )

    if plan.run_superres:
        audio = super_resolve(
            audio,
            sample_rate,
            analysis,
            device=device,
        )

    return audio, plan, format_ai_plan(plan, device, opts)
