"""FFmpeg executable resolution."""

from __future__ import annotations

import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def get_ffmpeg_path() -> str:
    """Return path to ffmpeg binary (system PATH or imageio-ffmpeg bundle)."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg and add it to PATH, "
            "or install imageio-ffmpeg: pip install imageio-ffmpeg"
        ) from exc
