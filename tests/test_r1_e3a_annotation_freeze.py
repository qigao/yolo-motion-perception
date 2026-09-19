import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "r1_e3a_annotation_freeze.py"
SPEC = importlib.util.spec_from_file_location("r1_e3a_annotation_freeze", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
FREEZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZE)

validate_review = FREEZE.validate_review


def manifest():
    return {
        "science_head_sha": "9" * 40,
        "videos": [
            {
                "source_video_id": "train-a",
                "split": "train",
                "frame_count": 300,
                "fps": 30.0,
                "source_window_component_id": "train-component",
            },
            {
                "source_video_id": "eval-a",
                "split": "eval",
                "frame_count": 240,
                "fps": 30.0,
                "source_window_component_id": "eval-component",
            },
        ],
    }


def review(records, *, train_count=0, eval_count=0):
    return {
        "schema": "r1-e3a-semantic-annotation-review-v1",
        "status": "reviewed",
        "science_head_sha": "9" * 40,
        "video_reviews": [
            {
                "video_id": "train-a",
                "full_range_reviewed": True,
                "candidate_count": train_count,
                "reviewer": "reviewer-a",
            },
            {
                "video_id": "eval-a",
                "full_range_reviewed": True,
                "candidate_count": eval_count,
                "reviewer": "reviewer-b",
            },
        ],
        "records": records,
    }


def accepted(event_id, video_id, label, start, end, context=None):
    return {
        "physical_event_id": event_id,
        "registered_video_id": video_id,
        "context_video_ids": list(context or []),
        "start_frame_inclusive": start,
        "end_frame_exclusive": end,
        "label": label,
        "actor_description": "person",
        "target_description": "object",
        "annotation_revision": 1,
        "annotator": "annotator-a",
        "review_status": "accepted",
        "reviewer": "reviewer-b",
    }


def test_valid_review_reconciles_video_candidate_counts():
    records = [
        accepted("evt-train", "train-a", "pick_up", 10, 50),
        accepted("evt-eval", "eval-a", "touch", 20, 60),
    ]

    summary = validate_review(review(records, train_count=1, eval_count=1), manifest())

    assert summary["accepted"]["train"]["pick_up"] == 1
    assert summary["accepted"]["eval"]["touch"] == 1
    assert summary["registration_minima_met"] is False


def test_duplicate_physical_event_fails_closed():
    records = [
        accepted("same", "train-a", "approach", 1, 20),
        accepted("same", "train-a", "pass_by", 30, 50),
    ]

    with pytest.raises(ValueError, match="duplicate physical_event_id"):
        validate_review(review(records, train_count=2), manifest())


def test_frame_bounds_must_fit_frozen_source():
    records = [accepted("evt", "train-a", "approach", 10, 301)]

    with pytest.raises(ValueError, match="frame bounds"):
        validate_review(review(records, train_count=1), manifest())


def test_candidate_count_includes_context_video_appearances():
    records = [
        accepted(
            "evt",
            "train-a",
            "pick_up",
            10,
            50,
            context=["eval-a"],
        )
    ]

    summary = validate_review(review(records, train_count=1, eval_count=1), manifest())

    assert summary["reviewed_candidate_appearances"]["train-a"] == 1
    assert summary["reviewed_candidate_appearances"]["eval-a"] == 1


def test_candidate_count_mismatch_fails_closed():
    records = [accepted("evt", "train-a", "pick_up", 10, 50)]

    with pytest.raises(ValueError, match="candidate_count mismatch"):
        validate_review(review(records, train_count=0), manifest())


def test_unresolved_status_is_rejected():
    record = accepted("evt", "train-a", "touch", 10, 20)
    record["review_status"] = "needs_resolution"

    with pytest.raises(ValueError, match="needs_resolution"):
        validate_review(review([record], train_count=1), manifest())


def test_rejected_record_requires_frozen_reason_code():
    record = accepted("evt", "train-a", "touch", 10, 20)
    record["review_status"] = "rejected:not-a-frozen-reason"

    with pytest.raises(ValueError, match="rejection reason"):
        validate_review(review([record], train_count=1), manifest())
