"""Smoke tests for the MP3 enhancement pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audio_enhancer.analyze import analyze
from audio_enhancer.decode import decode
from audio_enhancer.dsp_chain import apply_dsp
from audio_enhancer.loudness import preserve_loudness, rms_dbfs
from audio_enhancer.pipeline import discover_input_files, run_batch, run_pipeline

TEST_MP3 = ROOT / "1-01 Hidden Hills.mp3"
FALLBACK_MP3 = ROOT / "tests" / "fixtures" / "sample.mp3"


@pytest.fixture(scope="module")
def test_mp3_path() -> Path:
    if TEST_MP3.exists():
        return TEST_MP3
    if FALLBACK_MP3.exists():
        return FALLBACK_MP3
    pytest.skip(f"Test file not found: {TEST_MP3} or {FALLBACK_MP3}")


def test_decode(test_mp3_path: Path) -> None:
    audio = decode(test_mp3_path)
    assert audio.samples.ndim == 2
    assert audio.channels >= 1
    assert audio.sample_rate >= 8000
    assert audio.num_samples > 0


def test_analyze(test_mp3_path: Path) -> None:
    audio = decode(test_mp3_path)
    result = analyze(audio.samples, audio.sample_rate)
    assert len(result.band_energies) == 6
    assert result.spectral_balance_score >= 0.0
    if "bass_boost" in result.eq_gains:
        assert result.eq_gains["bass_boost"] <= 1.5
    if "high_shelf" in result.eq_gains:
        assert result.eq_gains["high_shelf"] <= 2.0
    if "presence_boost" in result.eq_gains:
        assert result.eq_gains["presence_boost"] <= 1.5


def test_dsp_chain(test_mp3_path: Path) -> None:
    audio = decode(test_mp3_path)
    analysis = analyze(audio.samples, audio.sample_rate)
    processed, _ = apply_dsp(
        audio.samples,
        audio.sample_rate,
        analysis,
        "gentle",
        reference_samples=audio.samples,
    )
    assert processed.shape == audio.samples.shape
    assert processed.dtype.name == "float32"


def test_loudness_preserved(test_mp3_path: Path) -> None:
    audio = decode(test_mp3_path)
    analysis = analyze(audio.samples, audio.sample_rate)
    processed, _ = apply_dsp(
        audio.samples,
        audio.sample_rate,
        analysis,
        "gentle",
        reference_samples=audio.samples,
    )
    source_rms = rms_dbfs(audio.samples)
    output_rms = rms_dbfs(processed)
    assert abs(output_rms - source_rms) <= 1.0


def test_quiet_track_gets_slight_lift() -> None:
    sr = 44100
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    quiet = (0.01 * np.sin(2 * np.pi * 440 * t))[np.newaxis, :]
    boosted = quiet * 2.0
    matched, msg = preserve_loudness(quiet, boosted)
    assert "Loudness: matched" in msg
    source_rms = rms_dbfs(quiet)
    output_rms = rms_dbfs(matched)
    assert output_rms - source_rms <= 1.05


def test_full_pipeline(test_mp3_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "output.enhanced.mp3"
    result = run_pipeline(test_mp3_path, output_path=output, preset="gentle")
    assert result.output_path is not None
    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 1000


def test_dry_run(test_mp3_path: Path) -> None:
    result = run_pipeline(test_mp3_path, dry_run=True)
    assert result.dry_run is True
    assert result.output_path is None


def test_tags_copied(test_mp3_path: Path, tmp_path: Path) -> None:
    from mutagen.easyid3 import EasyID3

    source = test_mp3_path
    try:
        existing = EasyID3(source)
        has_easy_tags = bool(existing.get("title") or existing.get("artist"))
    except Exception:
        has_easy_tags = False

    if not has_easy_tags:
        tagged = tmp_path / "tagged.mp3"
        import shutil

        shutil.copy(source, tagged)
        EasyID3(tagged)
        tags = EasyID3(tagged)
        tags["title"] = "Test Title"
        tags["artist"] = "Test Artist"
        tags["album"] = "Test Album"
        tags.save()
        source = tagged

    output = tmp_path / "tagged.enhanced.mp3"
    run_pipeline(source, output_path=output, preset="gentle")

    result_tags = EasyID3(output)
    assert result_tags.get("title")
    assert result_tags.get("artist")
    assert result_tags.get("album")


def test_discover_input_files(tmp_path: Path) -> None:
    (tmp_path / "a.mp3").write_bytes(b"x")
    (tmp_path / "b.MP3").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_text("nope")

    found = discover_input_files(tmp_path)
    assert [p.name for p in found] == ["a.mp3", "b.MP3"]


def test_batch_pipeline(test_mp3_path: Path, tmp_path: Path) -> None:
    import shutil

    inputs_dir = tmp_path / "inputs"
    outputs_dir = tmp_path / "outputs"
    inputs_dir.mkdir()
    shutil.copy(test_mp3_path, inputs_dir / "track1.mp3")
    shutil.copy(test_mp3_path, inputs_dir / "track2.mp3")

    result = run_batch(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        preset="gentle",
    )

    assert len(result.results) == 2
    assert not result.failures
    assert (outputs_dir / "track1.enhanced.mp3").exists()
    assert (outputs_dir / "track2.enhanced.mp3").exists()


def test_batch_empty_folder(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No MP3 files found"):
        run_batch(inputs_dir=inputs_dir, outputs_dir=tmp_path / "outputs")
