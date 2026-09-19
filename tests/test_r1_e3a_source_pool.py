import pytest

from tools.r1_e3a_source_pool import (
    PLAN_SCHEMA,
    available_rows,
    build_guard_components,
    build_plan,
    validate_plan,
)


def row(video_id: str, *, pickup: int = 1, size: int = 100):
    date, start, end, site, camera = video_id.split(".")
    return {
        "source_video_id": video_id,
        "date": date,
        "hour": start[:2],
        "site": site,
        "camera": camera,
        "s3_key": f"root/{video_id}.avi",
        "annotation_sha256": "a" * 64,
        "discovery_activity_counts": {"person_picks_up_object": pickup},
        "s3": {"available": True, "byte_size": size},
    }


def parsed_rows(*items):
    payload = {
        "schema": "r1-e3a-meva-candidate-inventory-v1",
        "candidates": list(items),
    }
    return available_rows(payload)


def test_guard_merges_multiview_touching_and_one_second_offset_windows():
    rows = parsed_rows(
        row("2018-03-11.10-00-00.10-05-00.school.G299"),
        row("2018-03-11.10-00-00.10-05-00.school.G330"),
        row("2018-03-11.10-05-01.10-10-01.school.G299"),
    )

    components = build_guard_components(rows)

    assert len(components) == 1
    assert components[0]["source_window_component_id"] == (
        "2018-03-11.10-00-00.10-10-01.school.guard1s"
    )


def test_guard_keeps_windows_separate_after_more_than_one_second_gap():
    rows = parsed_rows(
        row("2018-03-11.10-00-00.10-05-00.school.G299"),
        row("2018-03-11.10-05-02.10-10-02.school.G299"),
    )

    components = build_guard_components(rows)

    assert len(components) == 2


def test_plan_is_fail_closed_for_cross_split_component_reuse():
    plan = {
        "schema": PLAN_SCHEMA,
        "components": [
            {
                "source_window_component_id": "same",
                "split": "train",
                "selected_views": [
                    {"source_video_id": "a", "expected_byte_size": 1}
                ],
            },
            {
                "source_window_component_id": "same",
                "split": "eval",
                "selected_views": [
                    {"source_video_id": "b", "expected_byte_size": 1}
                ],
            },
            {
                "source_window_component_id": "t2",
                "split": "train",
                "selected_views": [
                    {"source_video_id": "c", "expected_byte_size": 1}
                ],
            },
            {
                "source_window_component_id": "e2",
                "split": "eval",
                "selected_views": [
                    {"source_video_id": "d", "expected_byte_size": 1}
                ],
            },
        ],
    }

    with pytest.raises(ValueError, match="crosses train/eval"):
        validate_plan(plan)


def test_plan_builder_never_duplicates_selected_source_videos():
    candidates = []
    for idx in range(18):
        hour = 8 + idx // 6
        minute = (idx % 6) * 10
        start = f"{hour:02d}-{minute:02d}-00"
        end_minute = minute + 5
        end_hour = hour + end_minute // 60
        end_minute %= 60
        end = f"{end_hour:02d}-{end_minute:02d}-00"
        for camera, pickup in (("G299", 20 - idx), ("G330", 10 - idx // 2)):
            candidates.append(
                row(
                    f"2018-03-11.{start}.{end}.school.{camera}",
                    pickup=max(pickup, 1),
                )
            )

    payload = {
        "schema": "r1-e3a-meva-candidate-inventory-v1",
        "candidates": candidates,
    }
    plan = build_plan(
        payload,
        inventory_sha256="f" * 64,
        inventory_workflow_run=1,
    )

    validate_plan(plan)
    selected = [
        view["source_video_id"]
        for component in plan["components"]
        for view in component["selected_views"]
    ]
    assert len(selected) == len(set(selected))
