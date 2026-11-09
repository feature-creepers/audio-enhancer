"""Torch device resolution with CPU fallback."""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Literal

DevicePreference = Literal["auto", "cpu", "cuda"]

AI_INSTALL_HINT = (
    "AI dependencies are not installed. Install them with:\n"
    "  pip install -r requirements-ai.txt"
)

DEEPFILTER_INSTALL_HINT = (
    "DeepFilterNet requires Python 3.8-3.11 on Windows (prebuilt wheels).\n"
    "On Python 3.12+, phase 1 uses a spectral fallback automatically.\n"
    "For DeepFilterNet3, use Python 3.11 and run:\n"
    "  pip install -r requirements-ai-deepfilter.txt"
)


class AiDependencyError(ImportError):
    """Raised when optional AI packages are missing."""


@lru_cache(maxsize=1)
def deepfilter_available() -> bool:
    """Return True when DeepFilterNet (df) is importable."""
    try:
        import df  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def audiosr_available() -> bool:
    """Return True when AudioSR is importable."""
    try:
        import audiosr  # noqa: F401

        return True
    except ImportError:
        return False


def check_ai_dependencies() -> None:
    """Verify that core AI packages are importable."""
    missing: list[str] = []
    for package in ("torch", "noisereduce"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise AiDependencyError(
            f"Missing AI packages: {', '.join(missing)}. {AI_INSTALL_HINT}"
        )


def resolve_device(preferred: DevicePreference = "auto") -> str:
    """Resolve compute device: auto prefers CUDA when available."""
    check_ai_dependencies()
    import torch

    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_cuda(device: str) -> bool:
    return device == "cuda"


def python_supports_deepfilter() -> bool:
    """DeepFilterNet ships Windows wheels up to Python 3.11."""
    return sys.version_info < (3, 12)
