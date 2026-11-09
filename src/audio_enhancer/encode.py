"""Encode float32 PCM to MP3 via FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .decode import AudioData
from .ffmpeg_utils import get_ffmpeg_path


def encode(
    audio: AudioData,
    path: str | Path,
    *,
    bitrate_kbps: int = 320,
) -> Path:
    """Write float32 PCM to MP3 at the given bitrate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if bitrate_kbps not in (192, 256, 320):
        raise ValueError(f"Unsupported bitrate: {bitrate_kbps}. Use 192, 256, or 320.")

    samples = np.ascontiguousarray(audio.samples, dtype=np.float32)
    channels = audio.channels
    interleaved = samples.T.reshape(-1)

    ffmpeg = get_ffmpeg_path()
    encode_cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(audio.sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        f"{bitrate_kbps}k",
        str(path),
    ]

    result = subprocess.run(
        encode_cmd,
        input=interleaved.tobytes(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg encode failed: {result.stderr.decode(errors='replace')}"
        )

    return path
