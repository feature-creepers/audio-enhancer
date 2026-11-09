#!/usr/bin/env python3
"""MP3 Enhancer CLI — improve compressed MP3 quality via DSP."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

# Allow running from project root without install
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audio_enhancer.ai.device import AI_INSTALL_HINT, AiDependencyError
from audio_enhancer.ai.router import AiMode
from audio_enhancer.dsp_chain import PresetName
from audio_enhancer.pipeline import AiConfig, PipelineConfig, run_batch, run_pipeline

INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

app = typer.Typer(
    name="enhance",
    help="Spectral refinement for MP3 files: corrective EQ, light dynamics, loudness preserved.",
    add_completion=False,
)


def _resolve_input(path: Path) -> Path:
    """Resolve input path and emit a helpful error if missing."""
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()

    if resolved.exists():
        return resolved

    typer.echo(f"Error: input file not found: {path}", err=True)
    typer.echo(f"  Looked for: {resolved}", err=True)
    typer.echo(
        "  Copy your MP3 into this folder, or pass the full path, e.g.:\n"
        f'    python enhance.py "C:\\path\\to\\track.mp3"\n'
        "  Or place files in inputs/ and run:\n"
        "    python enhance.py --batch",
        err=True,
    )
    raise typer.Exit(code=1)


def _configure_logging(verbose: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)


def _validate_bitrate(bitrate: int) -> None:
    if bitrate not in (192, 256, 320):
        typer.echo("Error: --bitrate must be 192, 256, or 320", err=True)
        raise typer.Exit(code=1)


def _build_ai_config(
    *,
    ai_mode: AiMode,
    ai_device: str,
    ai_fast: bool,
    ai_wet: float,
    ai_force_superres: bool,
) -> AiConfig:
    if not 0.0 <= ai_wet <= 1.0:
        typer.echo("Error: --ai-wet must be between 0.0 and 1.0", err=True)
        raise typer.Exit(code=1)
    return AiConfig(
        mode=ai_mode,
        device=ai_device,
        ai_fast=ai_fast,
        ai_wet=ai_wet,
        force_superres=ai_force_superres,
    )


@app.command()
def main(
    input: Optional[Path] = typer.Argument(
        None,
        help="Path to a single MP3 file (omit with --batch to process inputs/)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path for single-file mode (default: {stem}.enhanced.mp3)",
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        "-b",
        help="Process all MP3s in inputs/ and write to outputs/",
    ),
    inputs_dir: Optional[Path] = typer.Option(
        None,
        "--inputs-dir",
        help="Input folder for batch mode (default: inputs/)",
    ),
    outputs_dir: Optional[Path] = typer.Option(
        None,
        "--outputs-dir",
        help="Output folder for batch mode (default: outputs/)",
    ),
    preset: PresetName = typer.Option(
        "gentle",
        "--preset",
        help="Processing intensity: gentle, balanced, or aggressive",
    ),
    bitrate: int = typer.Option(
        320,
        "--bitrate",
        help="Output MP3 bitrate in kbps (192, 256, or 320)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Analyze only, do not write output file",
    ),
    ai: bool = typer.Option(
        False,
        "--ai",
        help="Enable AI enhancement (artifact reduction + optional super-resolution)",
    ),
    ai_mode: AiMode = typer.Option(
        "auto",
        "--ai-mode",
        help="AI strategy: auto, artifacts, superres, or full",
    ),
    ai_device: str = typer.Option(
        "auto",
        "--ai-device",
        help="Compute device for AI: auto, cpu, or cuda",
    ),
    ai_fast: bool = typer.Option(
        False,
        "--ai-fast",
        help="Skip super-resolution on CPU for faster batch runs",
    ),
    ai_wet: float = typer.Option(
        0.5,
        "--ai-wet",
        help="Wet/dry mix for artifact reduction phase (0.0-1.0)",
    ),
    boost: bool = typer.Option(
        False,
        "--boost",
        help="Legacy loud mastering mode (makeup gain + peak normalize)",
    ),
    ai_force_superres: bool = typer.Option(
        False,
        "--ai-force-superres",
        help="Run AudioSR super-resolution even on CPU",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Verbose logging",
    ),
) -> None:
    """Enhance one MP3 file or batch-process the inputs/ folder."""
    _validate_bitrate(bitrate)
    _configure_logging(verbose)

    if ai_device not in ("auto", "cpu", "cuda"):
        typer.echo("Error: --ai-device must be auto, cpu, or cuda", err=True)
        raise typer.Exit(code=1)

    ai_config = _build_ai_config(
        ai_mode=ai_mode,
        ai_device=ai_device,
        ai_fast=ai_fast,
        ai_wet=ai_wet,
        ai_force_superres=ai_force_superres,
    )
    pipeline_config = PipelineConfig(loud_boost=boost)

    try:
        if batch or input is None:
            if input is not None:
                typer.echo("Error: use either a single INPUT file or --batch, not both.", err=True)
                raise typer.Exit(code=1)

            batch_inputs = inputs_dir or INPUTS_DIR
            batch_outputs = outputs_dir or OUTPUTS_DIR
            batch_inputs.mkdir(parents=True, exist_ok=True)

            result = run_batch(
                inputs_dir=batch_inputs,
                outputs_dir=batch_outputs,
                preset=preset,
                bitrate_kbps=bitrate,
                dry_run=dry_run,
                use_ai=ai,
                ai_config=ai_config,
                pipeline_config=pipeline_config,
                verbose=verbose,
            )
            if result.failures:
                raise typer.Exit(code=1)
            return

        input_path = _resolve_input(input)
        run_pipeline(
            input_path,
            output_path=output,
            preset=preset,
            bitrate_kbps=bitrate,
            dry_run=dry_run,
            use_ai=ai,
            ai_config=ai_config,
            pipeline_config=pipeline_config,
            verbose=verbose,
        )
    except typer.Exit:
        raise
    except AiDependencyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo(AI_INSTALL_HINT, err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        if verbose:
            raise
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
