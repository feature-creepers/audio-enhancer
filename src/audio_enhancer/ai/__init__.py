"""Optional AI enhancement modules."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .router import AiMode, AiOptions, run_ai_pipeline

__all__ = ["AiMode", "AiOptions", "run_ai_pipeline"]


def __getattr__(name: str):
    if name in {"AiMode", "AiOptions", "run_ai_pipeline"}:
        from .router import AiMode, AiOptions, run_ai_pipeline

        return {"AiMode": AiMode, "AiOptions": AiOptions, "run_ai_pipeline": run_ai_pipeline}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
