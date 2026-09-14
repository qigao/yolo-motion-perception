from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable
from urllib.request import urlopen

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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(
    url: str,
    target: str | Path,
    *,
    opener: Callable[[str], BinaryIO] = urlopen,
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
        f"fps=1/2,scale=384:-1:force_original_aspect_ratio=decrease,"
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
