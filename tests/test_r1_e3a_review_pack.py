import csv
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "r1_e3a_review_pack.py"
SPEC = importlib.util.spec_from_file_location("r1_e3a_review_pack", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
REVIEW_PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW_PACK)


def record(
    video: str,
    camera: str,
    split: str = "train",
    component: str = "component-A",
    anchor: str = "anchor-A",
) -> dict[str, object]:
    return {
        "source_video_id": video,
        "camera": camera,
        "split": split,
        "source_window_component_id": component,
        "anchor_capture_group_id": anchor,
        "sha256": "a" * 64,
        "byte_size": 1,
        "fps": 30.0,
        "frame_count": 9000,
        "duration_seconds": 300.0,
    }


def test_review_groups_pairs_views_and_orders_cameras() -> None:
    manifest = {"videos": [record("v2", "G330"), record("v1", "G299")]}

    groups = REVIEW_PACK.review_groups(manifest)

    assert len(groups) == 1
    assert [view["camera"] for view in groups[0]["videos"]] == ["G299", "G330"]


def test_review_groups_rejects_split_crossing() -> None:
    manifest = {
        "videos": [
            record("v1", "G299", "train"),
            record("v2", "G330", "eval"),
        ]
    }

    with pytest.raises(ValueError, match="crosses split/component"):
        REVIEW_PACK.review_groups(manifest)


def test_review_groups_rejects_component_crossing() -> None:
    manifest = {
        "videos": [
            record("v1", "G299", component="a"),
            record("v2", "G330", component="b"),
        ]
    }

    with pytest.raises(ValueError, match="crosses split/component"):
        REVIEW_PACK.review_groups(manifest)


def test_review_completion_template_covers_every_activated_video(
    tmp_path: Path,
) -> None:
    manifest = {"videos": [record("v2", "G330"), record("v1", "G299")]}

    REVIEW_PACK.write_review_completion_template(tmp_path, manifest)

    with (tmp_path / "semantic-review-completion-template.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [row["source_video_id"] for row in rows] == ["v1", "v2"]
    assert all(row["full_range_reviewed"] == "false" for row in rows)
    assert all(row["reviewed_candidate_count"] == "0" for row in rows)
    assert all(row["reviewer"] == "UNASSIGNED" for row in rows)


def test_render_pack_binds_sampling_protocol_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    sampling_path = tmp_path / "sampling.md"
    out_dir = tmp_path / "out"
    video_root = tmp_path / "videos"
    video_root.mkdir()

    manifest = {
        "schema": REVIEW_PACK.MANIFEST_SCHEMA,
        "acquisition_head_sha": "a" * 40,
        "science_head_sha": "b" * 40,
        "videos": [record("v1", "G299")],
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    sampling_path.write_text("frozen sampling protocol\n", encoding="utf-8")

    monkeypatch.setattr(REVIEW_PACK, "validate_sources", lambda *_args: None)
    monkeypatch.setattr(
        REVIEW_PACK,
        "render_overview",
        lambda _sources, output: output.write_bytes(b"overview"),
    )
    monkeypatch.setattr(
        REVIEW_PACK,
        "render_contact_sheet",
        lambda _source, output: output.write_bytes(b"contact"),
    )

    index = REVIEW_PACK.render_pack(
        video_root,
        manifest_path,
        sampling_path,
        out_dir,
    )

    assert index["sampling_protocol_sha256"] == REVIEW_PACK.sha256_file(
        sampling_path
    )
    assert (
        out_dir / "semantic-review-completion-template.csv"
    ).is_file()
    assert "full chronological source-video scan" in (
        out_dir / "README.md"
    ).read_text(encoding="utf-8")
