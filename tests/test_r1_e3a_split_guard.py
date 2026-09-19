from __future__ import annotations

import pytest

from tools.r1_e3a_split_guard import (
    assign_source_overlap_components,
    validate_source_overlap_split,
)


def test_transitive_time_overlap_forms_one_component() -> None:
    video_ids = [
        "2018-03-12.10-00-00.10-05-00.school.G299",
        "2018-03-12.10-00-01.10-05-01.school.G330",
        "2018-03-12.10-05-00.10-10-00.school.G424",
    ]

    components = assign_source_overlap_components(video_ids)

    assert len(set(components.values())) == 1


def test_nonoverlapping_windows_remain_separate() -> None:
    video_ids = [
        "2018-03-12.10-00-00.10-05-00.school.G299",
        "2018-03-12.10-05-00.10-10-00.school.G299",
    ]

    components = assign_source_overlap_components(video_ids)

    assert len(set(components.values())) == 2


def test_site_and_date_boundaries_never_merge() -> None:
    video_ids = [
        "2018-03-12.10-00-00.10-05-00.school.G299",
        "2018-03-12.10-00-01.10-05-01.hospital.G341",
        "2018-03-13.10-00-01.10-05-01.school.G299",
    ]

    components = assign_source_overlap_components(video_ids)

    assert len(set(components.values())) == 3


def test_split_validation_fails_closed_on_component_leakage() -> None:
    assignments = {
        "2018-03-12.10-00-00.10-05-00.school.G299": "train",
        "2018-03-12.10-00-01.10-05-01.school.G330": "eval",
    }

    with pytest.raises(ValueError, match="source-overlap component crosses split"):
        validate_source_overlap_split(assignments)


def test_split_validation_accepts_disjoint_components() -> None:
    assignments = {
        "2018-03-12.10-00-00.10-05-00.school.G299": "train",
        "2018-03-12.10-05-00.10-10-00.school.G299": "eval",
    }

    validate_source_overlap_split(assignments)
