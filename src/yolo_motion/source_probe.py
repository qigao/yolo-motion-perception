from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml


class SourceManifestError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    purpose: str
    ground_truth_url: str | None = None


@dataclass(frozen=True)
class SourceManifest:
    dataset: str
    homepage: str
    license: str
    attribution: str
    sources: tuple[SourceSpec, ...]


@dataclass(frozen=True)
class ClipSpec:
    name: str
    source: str
    start_frame: int
    end_frame: int
    gt_object_ids: tuple[int, ...] = ()
    lateral: str | None = None
    radial: str | None = None


@dataclass(frozen=True)
class ClipManifest:
    fps: int
    clips: tuple[ClipSpec, ...]


@dataclass(frozen=True)
class DownloadResult:
    size_bytes: int
    sha256: str


def _require_string(root: dict[str, Any], key: str) -> str:
    value = root.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestError(f"{key} must be a non-empty string")
    return value


def load_source_manifest(path: str | Path) -> SourceManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise SourceManifestError("manifest root must be a mapping")

    dataset = _require_string(raw, "dataset")
    homepage = _require_string(raw, "homepage")
    license_name = _require_string(raw, "license")
    attribution = _require_string(raw, "attribution")

    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise SourceManifestError("sources must be a non-empty list")

    sources: list[SourceSpec] = []
    for index, item in enumerate(raw_sources):
        if not isinstance(item, dict):
            raise SourceManifestError(f"sources[{index}] must be a mapping")
        try:
            name = _require_string(item, "name")
            url = _require_string(item, "url")
            purpose = _require_string(item, "purpose")
        except SourceManifestError as exc:
            raise SourceManifestError(f"sources[{index}]: {exc}") from exc
        ground_truth_url = item.get("ground_truth_url")
        if ground_truth_url is not None and not isinstance(ground_truth_url, str):
            raise SourceManifestError(
                f"sources[{index}]: ground_truth_url must be a string when provided"
            )
        sources.append(
            SourceSpec(
                name=name,
                url=url,
                purpose=purpose,
                ground_truth_url=ground_truth_url,
            )
        )

    return SourceManifest(
        dataset=dataset,
        homepage=homepage,
        license=license_name,
        attribution=attribution,
        sources=tuple(sources),
    )


def load_clip_manifest(path: str | Path) -> ClipManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise SourceManifestError("clip manifest root must be a mapping")

    try:
        fps = int(raw["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceManifestError("fps must be a positive integer") from exc
    if fps <= 0:
        raise SourceManifestError("fps must be a positive integer")

    raw_clips = raw.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise SourceManifestError("clips must be a non-empty list")

    clips: list[ClipSpec] = []
    for index, item in enumerate(raw_clips):
        if not isinstance(item, dict):
            raise SourceManifestError(f"clips[{index}] must be a mapping")
        try:
            name = _require_string(item, "name")
            source = _require_string(item, "source")
            start_frame = int(item["start_frame"])
            end_frame = int(item["end_frame"])
        except (KeyError, TypeError, ValueError, SourceManifestError) as exc:
            raise SourceManifestError(
                f"clips[{index}] requires name/source/start_frame/end_frame"
            ) from exc
        if start_frame < 0 or end_frame < start_frame:
            raise SourceManifestError(
                f"clips[{index}] requires 0 <= start_frame <= end_frame"
            )

        raw_gt_object_ids = item.get("gt_object_ids", [])
        if not isinstance(raw_gt_object_ids, list):
            raise SourceManifestError(f"clips[{index}]: gt_object_ids must be a list")
        try:
            gt_object_ids = tuple(int(value) for value in raw_gt_object_ids)
        except (TypeError, ValueError) as exc:
            raise SourceManifestError(
                f"clips[{index}]: gt_object_ids must contain integers"
            ) from exc

        lateral = item.get("lateral")
        radial = item.get("radial")
        if lateral is not None and not isinstance(lateral, str):
            raise SourceManifestError(f"clips[{index}]: lateral must be a string")
        if radial is not None and not isinstance(radial, str):
            raise SourceManifestError(f"clips[{index}]: radial must be a string")
        clips.append(
            ClipSpec(
                name=name,
                source=source,
                start_frame=start_frame,
                end_frame=end_frame,
                gt_object_ids=gt_object_ids,
                lateral=lateral,
                radial=radial,
            )
        )
    return ClipManifest(fps=fps, clips=tuple(clips))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_url(url: str) -> BinaryIO:
    request = Request(url, headers={"User-Agent": "yolo-motion-perception/0.1"})
    return urlopen(request)


def download_source(
    url: str,
    target: str | Path,
    *,
    opener: Callable[[str], BinaryIO] = _open_url,
) -> DownloadResult:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size_bytes = 0
    with opener(url) as response, target_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            size_bytes += len(chunk)
    return DownloadResult(size_bytes=size_bytes, sha256=digest.hexdigest())


def build_ffprobe_command(source: Path, output: Path) -> list[str]:
    del output
    return [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(source),
    ]


def build_contact_sheet_command(
    source: Path,
    output: Path,
    *,
    columns: int = 4,
    rows: int = 3,
) -> list[str]:
    if columns <= 0 or rows <= 0:
        raise ValueError("contact sheet grid dimensions must be positive")
    filter_chain = (
        "fps=1/2,scale=384:-1:force_original_aspect_ratio=decrease,"
        f"tile={columns}x{rows}"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        filter_chain,
        "-frames:v",
        "1",
        str(output),
    ]


def build_trim_command(
    source: Path,
    output: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: int,
) -> list[str]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("requires 0 <= start_frame <= end_frame")
    filter_chain = (
        f"select='between(n,{start_frame},{end_frame})',"
        f"setpts=N/({fps}*TB)"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        filter_chain,
        "-r",
        str(fps),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]


def _filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise SourceManifestError(f"source URL has no filename: {url}")
    return name


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def materialize_clips(
    source_manifest_path: str | Path,
    clip_manifest_path: str | Path,
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    runner: Callable[[list[str]], None] = _run_checked,
) -> list[Path]:
    source_manifest = load_source_manifest(source_manifest_path)
    clip_manifest = load_clip_manifest(clip_manifest_path)
    source_root = Path(source_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    source_files = {
        source.name: source_root / _filename_from_url(source.url)
        for source in source_manifest.sources
    }
    outputs: list[Path] = []
    for clip in clip_manifest.clips:
        source_path = source_files.get(clip.source)
        if source_path is None:
            raise SourceManifestError(
                f"clip {clip.name!r} references unknown source {clip.source!r}"
            )
        if not source_path.is_file():
            raise SourceManifestError(f"source file is missing: {source_path}")
        output_path = output_root / f"{clip.name}.mp4"
        runner(
            build_trim_command(
                source_path,
                output_path,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                fps=clip_manifest.fps,
            )
        )
        outputs.append(output_path)
    return outputs


def probe_sources(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_source_manifest(manifest_path)
    sources_dir = output_dir / "sources"
    ground_truth_dir = output_dir / "ground-truth"
    probes_dir = output_dir / "ffprobe"
    sheets_dir = output_dir / "contact-sheets"
    for directory in (sources_dir, ground_truth_dir, probes_dir, sheets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    sha_lines: list[str] = []
    for source in manifest.sources:
        filename = _filename_from_url(source.url)
        video_path = sources_dir / filename
        download = download_source(source.url, video_path)
        sha_lines.append(f"{download.sha256}  sources/{filename}")

        probe_path = probes_dir / f"{source.name}.json"
        probe = subprocess.run(
            build_ffprobe_command(video_path, probe_path),
            check=True,
            capture_output=True,
            text=True,
        )
        probe_path.write_text(probe.stdout, encoding="utf-8")

        sheet_path = sheets_dir / f"{source.name}.jpg"
        subprocess.run(
            build_contact_sheet_command(video_path, sheet_path),
            check=True,
        )

        record: dict[str, Any] = {
            "name": source.name,
            "purpose": source.purpose,
            "url": source.url,
            "file": f"sources/{filename}",
            **asdict(download),
        }
        if source.ground_truth_url:
            gt_filename = _filename_from_url(source.ground_truth_url)
            gt_path = ground_truth_dir / gt_filename
            gt_download = download_source(source.ground_truth_url, gt_path)
            sha_lines.append(f"{gt_download.sha256}  ground-truth/{gt_filename}")
            record["ground_truth_url"] = source.ground_truth_url
            record["ground_truth_file"] = f"ground-truth/{gt_filename}"
            record["ground_truth_sha256"] = gt_download.sha256
        records.append(record)

    metadata = {
        "dataset": manifest.dataset,
        "homepage": manifest.homepage,
        "license": manifest.license,
        "attribution": manifest.attribution,
        "sources": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sha256.txt").write_text(
        "\n".join(sha_lines) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch-benchmark-sources",
        description="Download and visually probe licensed public benchmark videos.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = probe_sources(args.manifest, args.output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0
