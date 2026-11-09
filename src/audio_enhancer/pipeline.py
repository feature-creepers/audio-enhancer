"""Pipeline orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .analyze import AnalysisResult, analyze
from .decode import AudioData, decode
from .dsp_chain import PresetName, apply_dsp
from .encode import encode
from .loudness import preserve_loudness
from .tags import copy_tags, format_copied_tags

if TYPE_CHECKING:
    from .ai.router import AiMode

logger = logging.getLogger(__name__)


@dataclass
class AiConfig:
    mode: str = "auto"
    device: str = "auto"
    ai_fast: bool = False
    ai_wet: float = 0.5
    force_superres: bool = False


@dataclass
class PipelineConfig:
    loud_boost: bool = False


@dataclass
class PipelineResult:
    input_path: Path
    output_path: Path | None
    analysis: AnalysisResult
    audio: AudioData
    elapsed_seconds: float
    preset: PresetName
    bitrate_kbps: int
    dry_run: bool
    ai_enabled: bool
    ai_summary: str | None = None


def _format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


@dataclass
class BatchResult:
    results: list[PipelineResult]
    failures: list[tuple[Path, str]]
    elapsed_seconds: float


def _default_output_path(input_path: Path, output_dir: Path | None = None) -> Path:
    name = f"{input_path.stem}.enhanced.mp3"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / name
    return input_path.with_name(name)


def discover_input_files(inputs_dir: Path) -> list[Path]:
    """Return sorted MP3 files from the inputs directory."""
    if not inputs_dir.is_dir():
        return []
    return sorted(
        p for p in inputs_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp3"
    )


def run_batch(
    *,
    inputs_dir: str | Path,
    outputs_dir: str | Path,
    preset: PresetName = "gentle",
    bitrate_kbps: int = 320,
    dry_run: bool = False,
    use_ai: bool = False,
    ai_config: AiConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
    verbose: bool = False,
) -> BatchResult:
    """Process every MP3 in inputs_dir and write results to outputs_dir."""
    inputs_dir = Path(inputs_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    config = ai_config or AiConfig()
    pipe_config = pipeline_config or PipelineConfig()

    files = discover_input_files(inputs_dir)
    if not files:
        raise FileNotFoundError(
            f"No MP3 files found in {inputs_dir.resolve()}. "
            "Add files to the inputs/ folder and try again."
        )

    if use_ai:
        from .ai.device import check_ai_dependencies
        from .ai.router import warmup_for_mode

        check_ai_dependencies()
        warmup_for_mode(
            config.mode,  # type: ignore[arg-type]
            config.device,
            config.ai_fast,
            config.force_superres,
        )
        print(
            f"AI models warmed up (mode={config.mode}, device={config.device})"
        )
        print()

    start = time.perf_counter()
    results: list[PipelineResult] = []
    failures: list[tuple[Path, str]] = []

    print(f"Batch mode: {len(files)} file(s) in {inputs_dir.name}/ -> {outputs_dir.name}/")
    print()

    for index, input_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {input_path.name}")
        try:
            result = run_pipeline(
                input_path,
                output_path=_default_output_path(input_path, outputs_dir),
                preset=preset,
                bitrate_kbps=bitrate_kbps,
                dry_run=dry_run,
                use_ai=use_ai,
                ai_config=config,
                pipeline_config=pipe_config,
                verbose=verbose,
            )
            results.append(result)
        except Exception as exc:
            failures.append((input_path, str(exc)))
            print(f"  Failed: {exc}")
        print()

    elapsed = time.perf_counter() - start
    print(
        f"Batch complete: {len(results)} succeeded, {len(failures)} failed "
        f"in {elapsed:.1f}s"
    )
    return BatchResult(results=results, failures=failures, elapsed_seconds=elapsed)


def run_pipeline(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    preset: PresetName = "gentle",
    bitrate_kbps: int = 320,
    dry_run: bool = False,
    use_ai: bool = False,
    ai_config: AiConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
    verbose: bool = False,
) -> PipelineResult:
    """Decode, analyze, process, and optionally encode MP3."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = _default_output_path(input_path)
    else:
        output_path = Path(output_path)

    config = ai_config or AiConfig()
    pipe_config = pipeline_config or PipelineConfig()
    start = time.perf_counter()
    ai_summary: str | None = None

    if verbose:
        logger.info("Decoding %s", input_path)

    audio = decode(input_path)
    analysis = analyze(audio.samples, audio.sample_rate)

    print(f"Analyzing: {input_path.name}")
    print(
        f"  Duration: {_format_duration(audio.duration_seconds)} | "
        f"SR: {audio.sample_rate} Hz | Channels: {audio.channels}"
    )
    if audio.source_bitrate_kbps:
        print(f"  Source bitrate: {audio.source_bitrate_kbps} kbps")
    print(analysis.format_report())

    if dry_run:
        if use_ai:
            from .ai.device import check_ai_dependencies
            from .ai.device import resolve_device
            from .ai.router import AiOptions, format_ai_plan, plan_ai_stages

            check_ai_dependencies()
            device = resolve_device(config.device)  # type: ignore[arg-type]
            plan = plan_ai_stages(
                analysis,
                AiOptions(
                    mode=config.mode,  # type: ignore[arg-type]
                    device=config.device,  # type: ignore[arg-type]
                    ai_fast=config.ai_fast,
                    ai_wet=config.ai_wet,
                    force_superres=config.force_superres,
                ),
                source_bitrate_kbps=audio.source_bitrate_kbps,
                resolved_device=device,
            )
            ai_summary = format_ai_plan(
                plan,
                device,
                AiOptions(
                    mode=config.mode,  # type: ignore[arg-type]
                    device=config.device,  # type: ignore[arg-type]
                    ai_fast=config.ai_fast,
                    ai_wet=config.ai_wet,
                    force_superres=config.force_superres,
                ),
            )
            print(f"  {ai_summary}")

        elapsed = time.perf_counter() - start
        print(f"Dry run - no output written. Done in {elapsed:.1f}s")
        return PipelineResult(
            input_path=input_path,
            output_path=None,
            analysis=analysis,
            audio=audio,
            elapsed_seconds=elapsed,
            preset=preset,
            bitrate_kbps=bitrate_kbps,
            dry_run=True,
            ai_enabled=use_ai,
            ai_summary=ai_summary,
        )

    processed = audio.samples
    if use_ai:
        from .ai.device import check_ai_dependencies

        check_ai_dependencies()
        from .ai_enhance import enhance as ai_enhance

        ai_start = time.perf_counter()
        processed, ai_summary = ai_enhance(
            processed,
            audio.sample_rate,
            mode=config.mode,
            analysis=analysis,
            source_bitrate_kbps=audio.source_bitrate_kbps,
            device=config.device,
            ai_fast=config.ai_fast,
            ai_wet=config.ai_wet,
            force_superres=config.force_superres,
        )
        print(f"  {ai_summary}")
        print(f"  AI pass done in {time.perf_counter() - ai_start:.1f}s")
        processed, ai_loudness_msg = preserve_loudness(audio.samples, processed)
        print(f"  {ai_loudness_msg}")

    print(f"Applying: {preset} preset -> spectral refine + light dynamics")
    processed, loudness_msg = apply_dsp(
        processed,
        audio.sample_rate,
        analysis,
        preset,
        reference_samples=audio.samples,
        loud_boost=pipe_config.loud_boost,
    )
    if loudness_msg:
        print(f"  {loudness_msg}")

    processed_audio = AudioData(
        samples=processed,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        source_bitrate_kbps=audio.source_bitrate_kbps,
    )

    encode(processed_audio, output_path, bitrate_kbps=bitrate_kbps)

    copied = copy_tags(input_path, output_path)
    if copied:
        print(f"  Tags copied: {format_copied_tags(copied)}")
    elif verbose:
        print("  Tags copied: none")

    elapsed = time.perf_counter() - start
    print(f"  Output: {output_path.name} ({bitrate_kbps} kbps)")
    print(f"Done in {elapsed:.1f}s")

    return PipelineResult(
        input_path=input_path,
        output_path=output_path,
        analysis=analysis,
        audio=processed_audio,
        elapsed_seconds=elapsed,
        preset=preset,
        bitrate_kbps=bitrate_kbps,
        dry_run=False,
        ai_enabled=use_ai,
        ai_summary=ai_summary,
    )
