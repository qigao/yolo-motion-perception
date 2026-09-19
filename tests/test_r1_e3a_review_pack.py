import importlib.util
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
