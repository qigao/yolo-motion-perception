from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from yolo_motion.source_probe import (
    SourceManifestError,
    build_contact_sheet_command,
    build_ffprobe_command,
    build_trim_command,
    download_source,
    load_clip_manifest,
    load_source_manifest,
    materialize_clips,
    sha256_file,
)


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _write_source_manifest(path: Path) -> None:
    path.write_text(
        "dataset: CAVIAR\n"
        "homepage: https://example.com/caviar\n"
        "license: CC BY-SA\n"
        "attribution: EC CAVIAR project\n"
        "sources:\n"
        "  - name: walk2\n"
        "    url: https://example.com/Walk2.mpg\n"
        "    purpose: approaching\n",
        encoding="utf-8",
    )


def test_load_source_manifest_requires_license_and_sources(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    _write_source_manifest(path)

    manifest = load_source_manifest(path)

    assert manifest.dataset == "CAVIAR"
    assert manifest.license == "CC BY-SA"
    assert manifest.sources[0].name == "walk2"
    assert manifest.sources[0].purpose == "approaching"


def test_load_source_manifest_rejects_missing_license(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        "dataset: CAVIAR\n"
        "homepage: https://example.com/caviar\n"
        "attribution: EC CAVIAR project\n"
        "sources:\n"
        "  - name: walk2\n"
        "    url: https://example.com/Walk2.mpg\n"
        "    purpose: approaching\n",
        encoding="utf-8",
    )

    with pytest.raises(SourceManifestError, match="license"):
        load_source_manifest(path)


def test_download_source_writes_bytes_and_returns_sha256(tmp_path: Path) -> None:
    payload = b"benchmark-video"
    target = tmp_path / "Walk2.mpg"

    result = download_source(
        "https://example.com/Walk2.mpg",
        target,
        opener=lambda _url: _Response(payload),
    )

    assert target.read_bytes() == payload
    assert result.size_bytes == len(payload)
    assert result.sha256 == hashlib.sha256(payload).hexdigest()


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "clip.mpg"
    path.write_bytes(b"abc123")

    assert sha256_file(path) == hashlib.sha256(b"abc123").hexdigest()


def test_build_ffprobe_command_is_json_and_input_only() -> None:
    command = build_ffprobe_command(Path("source.mpg"), Path("probe.json"))

    assert command[:3] == ["ffprobe", "-v", "error"]
    assert "-show_streams" in command
    assert "-show_format" in command
    assert command[-1] == "source.mpg"


def test_build_contact_sheet_command_samples_fixed_grid() -> None:
    command = build_contact_sheet_command(
        Path("source.mpg"), Path("sheet.jpg"), columns=4, rows=3
    )

    assert command[0] == "ffmpeg"
    assert "fps=1/2" in command[command.index("-vf") + 1]
    assert "tile=4x3" in command[command.index("-vf") + 1]
    assert command[-1] == "sheet.jpg"


def test_load_clip_manifest_reads_frame_windows(tmp_path: Path) -> None:
    path = tmp_path / "clips.yaml"
    path.write_text(
        "fps: 25\n"
        "clips:\n"
        "  - name: approaching\n"
        "    source: walk2\n"
        "    start_frame: 880\n"
        "    end_frame: 1049\n"
        "    lateral: moving\n"
        "    radial: approaching\n",
        encoding="utf-8",
    )

    manifest = load_clip_manifest(path)

    assert manifest.fps == 25
    assert manifest.clips[0].source == "walk2"
    assert manifest.clips[0].start_frame == 880
    assert manifest.clips[0].radial == "approaching"


def test_build_trim_command_selects_inclusive_frame_window() -> None:
    command = build_trim_command(
        Path("Walk2.mpg"),
        Path("approaching.mp4"),
        start_frame=880,
        end_frame=1049,
        fps=25,
    )

    filter_chain = command[command.index("-vf") + 1]
    assert "between(n,880,1049)" in filter_chain
    assert "setpts=N/(25*TB)" in filter_chain
    assert command[-1] == "approaching.mp4"


def test_materialize_clips_maps_logical_source_and_runs_ffmpeg(tmp_path: Path) -> None:
    sources_yaml = tmp_path / "sources.yaml"
    _write_source_manifest(sources_yaml)
    clips_yaml = tmp_path / "clips.yaml"
    clips_yaml.write_text(
        "fps: 25\n"
        "clips:\n"
        "  - name: approaching\n"
        "    source: walk2\n"
        "    start_frame: 880\n"
        "    end_frame: 1049\n",
        encoding="utf-8",
    )
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "Walk2.mpg").write_bytes(b"video")
    commands: list[list[str]] = []

    outputs = materialize_clips(
        sources_yaml,
        clips_yaml,
        source_dir,
        tmp_path / "clips",
        runner=lambda command: commands.append(command),
    )

    assert outputs == [tmp_path / "clips" / "approaching.mp4"]
    assert commands[0][0] == "ffmpeg"
    assert str(source_dir / "Walk2.mpg") in commands[0]
    assert commands[0][-1] == str(tmp_path / "clips" / "approaching.mp4")


def test_materialize_clips_rejects_unknown_source(tmp_path: Path) -> None:
    sources_yaml = tmp_path / "sources.yaml"
    _write_source_manifest(sources_yaml)
    clips_yaml = tmp_path / "clips.yaml"
    clips_yaml.write_text(
        "fps: 25\n"
        "clips:\n"
        "  - name: approaching\n"
        "    source: missing\n"
        "    start_frame: 0\n"
        "    end_frame: 10\n",
        encoding="utf-8",
    )

    with pytest.raises(SourceManifestError, match="unknown source"):
        materialize_clips(
            sources_yaml,
            clips_yaml,
            tmp_path / "sources",
            tmp_path / "clips",
            runner=lambda _command: None,
        )
