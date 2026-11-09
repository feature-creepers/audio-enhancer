"""Decode MP3 (and other formats) to float32 PCM via FFmpeg."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg_utils import get_ffmpeg_path


@dataclass
class AudioData:
    samples: np.ndarray  # shape (channels, samples), float32 in [-1, 1]
    sample_rate: int
    channels: int
    source_bitrate_kbps: int | None = None

    @property
    def duration_seconds(self) -> float:
        return self.samples.shape[1] / self.sample_rate

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]


def _parse_source_bitrate(stderr: str) -> int | None:
    for line in stderr.splitlines():
        if "Audio:" not in line:
            continue
        match = re.search(r"(\d+)\s*kb/s", line, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def decode(path: str | Path) -> AudioData:
    """Decode audio file to float32 stereo/mono PCM."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ffmpeg = get_ffmpeg_path()

    # Use ffprobe-like parsing via ffmpeg stderr for channel count
    info_cmd = [
        ffmpeg,
        "-hide_banner",
        "-i",
        str(path),
    ]
    probe = subprocess.run(info_cmd, capture_output=True, text=True)
    stderr = probe.stderr

    sample_rate = 44100
    channels = 2
    for line in stderr.splitlines():
        if "Audio:" in line:
            parts = line.split(",")
            for part in parts:
                part = part.strip()
                if part.endswith(" Hz"):
                    try:
                        sample_rate = int(part.replace(" Hz", ""))
                    except ValueError:
                        pass
                if part in ("mono", "stereo"):
                    channels = 1 if part == "mono" else 2
                if " Hz" in part and "kHz" not in part:
                    tokens = part.split()
                    for token in tokens:
                        if token.isdigit() and int(token) >= 8000:
                            sample_rate = int(token)
                            break

    decode_cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(decode_cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg decode failed: {result.stderr.decode(errors='replace')}"
        )

    raw = np.frombuffer(result.stdout, dtype=np.float32)
    if raw.size == 0:
        raise RuntimeError("FFmpeg returned empty audio buffer")

    samples = raw.reshape(-1, channels).T.copy()
    source_bitrate = _parse_source_bitrate(stderr)
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        source_bitrate_kbps=source_bitrate,
    )
