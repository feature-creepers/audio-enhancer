"""DSP processing chain: corrective EQ, light dynamics, loudness preservation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pedalboard import (
    Compressor,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowShelfFilter,
    PeakFilter,
    Pedalboard,
)
from scipy.signal import butter, sosfilt

from .analyze import AnalysisResult
from .loudness import preserve_loudness, soft_ceiling

PresetName = Literal["gentle", "balanced", "aggressive"]


@dataclass
class PresetConfig:
    name: PresetName
    high_shelf_gain_scale: float
    low_mid_cut_scale: float
    presence_scale: float
    bass_boost_scale: float
    bass_comp_threshold: float
    bass_comp_ratio: float
    mid_comp_threshold: float
    mid_comp_ratio: float
    high_comp_threshold: float
    high_comp_ratio: float
    limiter_threshold_db: float
    makeup_gain_db: float


PRESETS: dict[PresetName, PresetConfig] = {
    "gentle": PresetConfig(
        name="gentle",
        high_shelf_gain_scale=0.85,
        low_mid_cut_scale=0.9,
        presence_scale=0.85,
        bass_boost_scale=0.85,
        bass_comp_threshold=-16,
        bass_comp_ratio=1.5,
        mid_comp_threshold=-14,
        mid_comp_ratio=1.4,
        high_comp_threshold=-12,
        high_comp_ratio=1.3,
        limiter_threshold_db=-3.0,
        makeup_gain_db=0.0,
    ),
    "balanced": PresetConfig(
        name="balanced",
        high_shelf_gain_scale=1.0,
        low_mid_cut_scale=1.0,
        presence_scale=1.0,
        bass_boost_scale=1.0,
        bass_comp_threshold=-18,
        bass_comp_ratio=1.8,
        mid_comp_threshold=-16,
        mid_comp_ratio=1.6,
        high_comp_threshold=-14,
        high_comp_ratio=1.5,
        limiter_threshold_db=-2.0,
        makeup_gain_db=0.0,
    ),
    "aggressive": PresetConfig(
        name="aggressive",
        high_shelf_gain_scale=1.1,
        low_mid_cut_scale=1.1,
        presence_scale=1.1,
        bass_boost_scale=1.05,
        bass_comp_threshold=-20,
        bass_comp_ratio=2.2,
        mid_comp_threshold=-18,
        mid_comp_ratio=2.0,
        high_comp_threshold=-16,
        high_comp_ratio=1.8,
        limiter_threshold_db=-1.0,
        makeup_gain_db=0.5,
    ),
}


def _crossover_low(samples: np.ndarray, sr: int, cutoff: float = 250.0) -> np.ndarray:
    sos = butter(4, cutoff / (sr / 2), btype="low", output="sos")
    if samples.ndim == 1:
        return sosfilt(sos, samples).astype(np.float32)
    return np.stack([sosfilt(sos, ch) for ch in samples], axis=0).astype(np.float32)


def _crossover_high(samples: np.ndarray, sr: int, cutoff: float = 4000.0) -> np.ndarray:
    sos = butter(4, cutoff / (sr / 2), btype="high", output="sos")
    if samples.ndim == 1:
        return sosfilt(sos, samples).astype(np.float32)
    return np.stack([sosfilt(sos, ch) for ch in samples], axis=0).astype(np.float32)


def _split_three_bands(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = _crossover_low(samples, sr, 250.0)
    high = _crossover_high(samples, sr, 4000.0)
    mid = samples - low - high
    return low, mid, high


def _apply_board(samples: np.ndarray, sr: int, board: Pedalboard) -> np.ndarray:
    if samples.ndim == 1:
        out = board(samples, sr)
        return np.asarray(out, dtype=np.float32)
    channels = []
    for ch in samples:
        out = board(ch, sr)
        channels.append(np.asarray(out, dtype=np.float32))
    min_len = min(len(c) for c in channels)
    return np.stack([c[:min_len] for c in channels], axis=0)


def _build_eq_board(analysis: AnalysisResult, preset: PresetConfig) -> Pedalboard | None:
    effects = [HighpassFilter(cutoff_frequency_hz=30.0)]

    eq = analysis.eq_gains
    if "high_shelf" in eq:
        gain = eq["high_shelf"] * preset.high_shelf_gain_scale
        effects.append(HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=gain))

    if "low_mid_cut" in eq:
        gain = eq["low_mid_cut"] * preset.low_mid_cut_scale
        effects.append(PeakFilter(cutoff_frequency_hz=350.0, gain_db=gain, q=1.0))

    if "presence_boost" in eq:
        gain = eq["presence_boost"] * preset.presence_scale
        effects.append(PeakFilter(cutoff_frequency_hz=3500.0, gain_db=gain, q=0.8))

    if "bass_boost" in eq:
        gain = eq["bass_boost"] * preset.bass_boost_scale
        effects.append(LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=gain))
        if gain >= 1.2:
            effects.append(PeakFilter(cutoff_frequency_hz=60.0, gain_db=gain * 0.3, q=0.85))

    if len(effects) == 1:
        return None

    return Pedalboard(effects)


def _build_band_compressor(threshold: float, ratio: float) -> Pedalboard:
    return Pedalboard(
        [
            Compressor(
                threshold_db=threshold,
                ratio=ratio,
                attack_ms=10.0,
                release_ms=100.0,
            )
        ]
    )


def apply_dsp(
    samples: np.ndarray,
    sample_rate: int,
    analysis: AnalysisResult,
    preset_name: PresetName = "gentle",
    *,
    reference_samples: np.ndarray | None = None,
    loud_boost: bool = False,
) -> tuple[np.ndarray, str | None]:
    """Run spectral refine chain. Returns processed audio and loudness report."""
    preset = PRESETS[preset_name]
    reference = reference_samples if reference_samples is not None else samples
    audio = samples.astype(np.float32, copy=True)

    eq_board = _build_eq_board(analysis, preset)
    if eq_board is not None:
        audio = _apply_board(audio, sample_rate, eq_board)

    low, mid, high = _split_three_bands(audio, sample_rate)

    low = _apply_board(
        low,
        sample_rate,
        _build_band_compressor(preset.bass_comp_threshold, preset.bass_comp_ratio),
    )
    mid = _apply_board(
        mid,
        sample_rate,
        _build_band_compressor(preset.mid_comp_threshold, preset.mid_comp_ratio),
    )
    high = _apply_board(
        high,
        sample_rate,
        _build_band_compressor(preset.high_comp_threshold, preset.high_comp_ratio),
    )

    min_len = min(low.shape[-1], mid.shape[-1], high.shape[-1])
    audio = low[..., :min_len] + mid[..., :min_len] + high[..., :min_len]

    if loud_boost:
        legacy_makeup = {"gentle": 1.5, "balanced": 2.0, "aggressive": 2.5}[preset_name]
        audio *= 10 ** (legacy_makeup / 20)
        loudness_msg = None
    else:
        loudness_msg = None

    limiter_board = Pedalboard(
        [Limiter(threshold_db=preset.limiter_threshold_db, release_ms=50.0)]
    )
    audio = _apply_board(audio, sample_rate, limiter_board)

    if loud_boost:
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-10:
            audio = (audio * (0.99 / peak)).astype(np.float32)
    else:
        audio = soft_ceiling(audio)
        audio, loudness_msg = preserve_loudness(reference, audio)

    return audio.astype(np.float32), loudness_msg
