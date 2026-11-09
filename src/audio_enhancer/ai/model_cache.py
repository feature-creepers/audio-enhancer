"""Lazy model loading and batch warmup."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .device import audiosr_available, check_ai_dependencies, deepfilter_available, resolve_device

logger = logging.getLogger(__name__)


@dataclass
class ModelCache:
    """Singleton cache for AI models."""

    deepfilter_model: Any | None = field(default=None, repr=False)
    deepfilter_state: Any | None = field(default=None, repr=False)
    audiosr_model: Any | None = field(default=None, repr=False)
    device: str = "cpu"

    def ensure_deepfilter(self, device: str) -> tuple[Any, Any]:
        if not deepfilter_available():
            raise RuntimeError("DeepFilterNet is not installed")
        if self.deepfilter_model is None or self.deepfilter_state is None:
            check_ai_dependencies()
            from df.enhance import init_df

            logger.info("Loading DeepFilterNet3 on %s", device)
            model, state, _ = init_df(
                model_base_dir="DeepFilterNet3",
                post_filter=False,
                atten_lim_db=12.0,
                config_allow_defaults=True,
            )
            self.deepfilter_model = model
            self.deepfilter_state = state
        return self.deepfilter_model, self.deepfilter_state

    def ensure_audiosr(self, device: str) -> Any:
        if not audiosr_available():
            raise RuntimeError("AudioSR is not installed")
        if self.audiosr_model is None:
            check_ai_dependencies()
            import audiosr

            logger.info("Loading AudioSR on %s", device)
            self.audiosr_model = audiosr.build_model(model_name="basic", device=device)
        return self.audiosr_model

    def warmup(
        self,
        *,
        need_artifacts: bool,
        need_superres: bool,
        device: str,
    ) -> None:
        """Pre-load models for batch processing."""
        self.device = resolve_device(device)  # type: ignore[arg-type]
        if need_artifacts and deepfilter_available():
            self.ensure_deepfilter(self.device)
        if need_superres and audiosr_available():
            self.ensure_audiosr(self.device)


_CACHE = ModelCache()


def get_cache() -> ModelCache:
    return _CACHE


def reset_cache() -> None:
    global _CACHE
    _CACHE = ModelCache()
