"""Spectral analysis for adaptive enhancement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import rfft, rfftfreq


BAND_EDGES = {
    "sub": (20, 80),
    "bass": (80, 250),
    "low_mid": (250, 1000),
    "mid": (1000, 4000),
    "high": (4000, 12000),
    "air": (12000, 20000),
}

# Relative target offsets (dB) for spectral balance scoring
BAND_TARGETS = {
    "sub": -8.0,
    "bass": -4.0,
    "low_mid": -2.0,
    "mid": 0.0,
    "high": -3.0,
    "air": -8.0,
}


@dataclass
class BandEnergy:
    name: str
    energy_db: float


@dataclass
class AnalysisResult:
    band_energies: list[BandEnergy]
    peak_db: float
    rms_db: float
    high_end_rolloff_db: float
    bass_deficit_db: float
    spectral_balance_score: float
    boomy: bool
    muffled: bool
    eq_gains: dict[str, float] = field(default_factory=dict)

    def format_report(self) -> str:
        lines = [
            f"  Peak: {self.peak_db:.1f} dBFS | RMS: {self.rms_db:.1f} dBFS",
            f"  Spectral balance score: {self.spectral_balance_score:.1f} (lower is better)",
            "  Band energies (dB):",
        ]
        for band in self.band_energies:
            lines.append(f"    {band.name:8s}: {band.energy_db:+.1f}")

        if abs(self.high_end_rolloff_db) > 1.0:
            lines.append(
                f"  High-end roll-off detected ({self.high_end_rolloff_db:+.1f} dB @ 12 kHz vs mid)"
            )
        if self.bass_deficit_db > 1.0:
            lines.append(
                f"  Bass lean vs low-mids ({self.bass_deficit_db:+.1f} dB deficit)"
            )
        if self.boomy:
            lines.append("  Low-mid buildup detected (boomy)")
        if self.muffled:
            lines.append("  Upper-mid deficiency detected (muffled)")

        if self.eq_gains:
            gains = ", ".join(f"{k} {v:+.1f} dB" for k, v in self.eq_gains.items())
            lines.append(f"  Spectral corrections: {gains}")

        return "\n".join(lines)


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return np.mean(samples, axis=0)


def _band_energy(magnitude: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return -80.0
    band_mag = magnitude[mask]
    rms = np.sqrt(np.mean(band_mag**2))
    return float(20 * np.log10(max(rms, 1e-10)))


def _spectral_balance_score(band_energies: list[BandEnergy]) -> float:
    mid_ref = next(b.energy_db for b in band_energies if b.name == "mid")
    deviations = []
    for band in band_energies:
        target = mid_ref + BAND_TARGETS.get(band.name, 0.0)
        deviations.append(abs(band.energy_db - target))
    return float(np.mean(deviations))


def analyze(samples: np.ndarray, sample_rate: int) -> AnalysisResult:
    """Analyze audio spectrum and derive adaptive EQ hints."""
    mono = _to_mono(samples)

    peak = float(np.max(np.abs(mono)))
    peak_db = 20 * np.log10(max(peak, 1e-10))
    rms = float(np.sqrt(np.mean(mono**2)))
    rms_db = 20 * np.log10(max(rms, 1e-10))

    window_size = min(len(mono), sample_rate * 30)
    if len(mono) > window_size:
        start = (len(mono) - window_size) // 2
        segment = mono[start : start + window_size]
    else:
        segment = mono

    window = np.hanning(len(segment))
    spectrum = rfft(segment * window)
    freqs = rfftfreq(len(segment), 1 / sample_rate)
    magnitude = np.abs(spectrum) / len(segment)

    band_energies = [
        BandEnergy(name=name, energy_db=_band_energy(magnitude, freqs, low, high))
        for name, (low, high) in BAND_EDGES.items()
    ]

    mid_energy = _band_energy(magnitude, freqs, 1000, 4000)
    high_energy = _band_energy(magnitude, freqs, 8000, 12000)
    high_end_rolloff_db = high_energy - mid_energy

    bass_energy = _band_energy(magnitude, freqs, 80, 250)
    low_mid_energy = _band_energy(magnitude, freqs, 250, 600)
    boomy = low_mid_energy > mid_energy + 3.0 and bass_energy > mid_energy

    upper_mid = _band_energy(magnitude, freqs, 2000, 6000)
    muffled = upper_mid < mid_energy - 4.0

    bass_deficit_db = low_mid_energy - bass_energy
    balance_score = _spectral_balance_score(band_energies)

    eq_gains = _compute_eq_gains(
        high_end_rolloff_db=high_end_rolloff_db,
        bass_deficit_db=bass_deficit_db,
        bass_energy=bass_energy,
        sub_energy=_band_energy(magnitude, freqs, 20, 80),
        mid_energy=mid_energy,
        boomy=boomy,
        muffled=muffled,
    )

    return AnalysisResult(
        band_energies=band_energies,
        peak_db=peak_db,
        rms_db=rms_db,
        high_end_rolloff_db=high_end_rolloff_db,
        bass_deficit_db=bass_deficit_db,
        spectral_balance_score=balance_score,
        boomy=boomy,
        muffled=muffled,
        eq_gains=eq_gains,
    )


def _compute_bass_boost(
    *,
    bass_deficit_db: float,
    low_mid_cut: float,
) -> float:
    """Conservative bass correction — only when genuinely lean."""
    if bass_deficit_db <= 3.0:
        return 0.0

    boost = min(1.5, (bass_deficit_db - 3.0) * 0.35)

    if low_mid_cut < 0:
        boost += min(0.5, abs(low_mid_cut) * 0.15)

    return min(boost, 1.5)


def _compute_eq_gains(
    *,
    high_end_rolloff_db: float,
    bass_deficit_db: float,
    bass_energy: float,
    sub_energy: float,
    mid_energy: float,
    boomy: bool,
    muffled: bool,
) -> dict[str, float]:
    gains: dict[str, float] = {}

    if high_end_rolloff_db < -4.0:
        gains["high_shelf"] = min(2.0, abs(high_end_rolloff_db) * 0.25)

    if boomy:
        gains["low_mid_cut"] = -2.5

    if muffled:
        gains["presence_boost"] = 1.5

    bass_boost = _compute_bass_boost(
        bass_deficit_db=bass_deficit_db,
        low_mid_cut=gains.get("low_mid_cut", 0.0),
    )
    if bass_boost >= 0.4:
        gains["bass_boost"] = bass_boost

    return gains
