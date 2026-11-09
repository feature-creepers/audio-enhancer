"""Tests for AI routing and utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audio_enhancer.ai.audio_utils import estimate_cutoff_hz, mix_wet_dry
from audio_enhancer.ai.device import AI_INSTALL_HINT, AiDependencyError, resolve_device
from audio_enhancer.ai.router import AiOptions, plan_ai_stages
from audio_enhancer.analyze import AnalysisResult, BandEnergy


def _analysis(*, rolloff: float = -8.0) -> AnalysisResult:
    bands = [
        BandEnergy("sub", -50),
        BandEnergy("bass", -60),
        BandEnergy("low_mid", -65),
        BandEnergy("mid", -75),
        BandEnergy("high", -85),
        BandEnergy("air", -95),
    ]
    return AnalysisResult(
        band_energies=bands,
        peak_db=-1.0,
        rms_db=-10.0,
        high_end_rolloff_db=rolloff,
        bass_deficit_db=0.0,
        spectral_balance_score=5.0,
        boomy=False,
        muffled=False,
        eq_gains={"bass_boost": 1.5},
    )


def test_plan_auto_runs_phase1_on_rolloff() -> None:
    plan = plan_ai_stages(
        _analysis(rolloff=-4.0),
        AiOptions(mode="auto"),
        source_bitrate_kbps=320,
        resolved_device="cpu",
    )
    assert plan.run_artifacts is True
    assert plan.run_superres is False


def test_plan_auto_runs_phase2_on_cuda() -> None:
    plan = plan_ai_stages(
        _analysis(rolloff=-10.0),
        AiOptions(mode="auto"),
        source_bitrate_kbps=320,
        resolved_device="cuda",
    )
    assert plan.run_artifacts is True
    assert plan.run_superres is True


def test_plan_auto_runs_phase2_fallback_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        "audio_enhancer.ai.router.audiosr_available",
        lambda: False,
    )
    plan = plan_ai_stages(
        _analysis(rolloff=-10.0),
        AiOptions(mode="auto"),
        source_bitrate_kbps=320,
        resolved_device="cpu",
    )
    assert plan.run_superres is True
    assert any("EQ-fallback" in r for r in plan.reasons)


def test_plan_auto_skips_phase2_on_cpu_with_audiosr(monkeypatch) -> None:
    monkeypatch.setattr(
        "audio_enhancer.ai.router.audiosr_available",
        lambda: True,
    )
    plan = plan_ai_stages(
        _analysis(rolloff=-10.0),
        AiOptions(mode="auto"),
        source_bitrate_kbps=320,
        resolved_device="cpu",
    )
    assert plan.run_superres is False


def test_plan_full_runs_both_phases() -> None:
    plan = plan_ai_stages(
        _analysis(),
        AiOptions(mode="full"),
        source_bitrate_kbps=320,
        resolved_device="cpu",
    )
    assert plan.run_artifacts is True
    assert plan.run_superres is True


def test_plan_artifacts_only() -> None:
    plan = plan_ai_stages(
        _analysis(),
        AiOptions(mode="artifacts"),
        source_bitrate_kbps=320,
        resolved_device="cpu",
    )
    assert plan.run_artifacts is True
    assert plan.run_superres is False


def test_plan_low_bitrate_triggers_phase1() -> None:
    plan = plan_ai_stages(
        _analysis(rolloff=0.0),
        AiOptions(mode="auto"),
        source_bitrate_kbps=128,
        resolved_device="cpu",
    )
    assert plan.run_artifacts is True


def test_mix_wet_dry_limits_peak() -> None:
    dry = np.ones((1, 100), dtype=np.float32) * 0.9
    wet = np.ones((1, 100), dtype=np.float32) * 1.5
    mixed = mix_wet_dry(dry, wet, 0.7)
    assert np.max(np.abs(mixed)) <= 1.0


def test_estimate_cutoff_hz() -> None:
    assert estimate_cutoff_hz(-12.0, 44100) < estimate_cutoff_hz(-2.0, 44100)


def test_resolve_device_cpu() -> None:
    pytest.importorskip("torch")
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device("auto") == "cpu"
        assert resolve_device("cpu") == "cpu"


def test_ai_dependency_error_message() -> None:
    err = AiDependencyError(f"Missing AI packages: torch. {AI_INSTALL_HINT}")
    assert "requirements-ai" in str(err)


def test_artifacts_backend_without_deepfilter(monkeypatch) -> None:
    from audio_enhancer.ai import artifacts as artifacts_module

    monkeypatch.setattr(artifacts_module, "ARTIFACTS_BACKEND", None)
    monkeypatch.setattr(artifacts_module, "deepfilter_available", lambda: False)
    assert artifacts_module.get_artifacts_backend() == "fallback"
    assert "fallback" in artifacts_module.artifacts_backend_label().lower()


def test_mix_wet_dry_zero_wet_returns_dry() -> None:
    dry = np.linspace(-0.5, 0.5, 50, dtype=np.float32)[np.newaxis, :]
    wet = np.zeros_like(dry)
    mixed = mix_wet_dry(dry, wet, 0.0)
    np.testing.assert_allclose(mixed, dry, atol=1e-6)
