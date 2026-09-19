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
registration_minima_are_met = FREEZE.registration_minima_are_met

SOURCE_MANIFEST_SHA256 = "1" * 64
HANDBOOK_SHA256 = "2" * 64
SAMPLING_PROTOCOL_SHA256 = "3" * 64


def validate(payload, source_manifest):
    return validate_review(
        payload,
        source_manifest,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        handbook_sha256=HANDBOOK_SHA256,
        sampling_protocol_sha256=SAMPLING_PROTOCOL_SHA256,
    )


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
                "source_video_id": "train-b",
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


def review(records, *, train_count=0, train_b_count=0, eval_count=0):
    return {
        "schema": "r1-e3a-semantic-annotation-review-v1",
        "status": "reviewed",
        "science_head_sha": "9" * 40,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "handbook_sha256": HANDBOOK_SHA256,
        "sampling_protocol_sha256": SAMPLING_PROTOCOL_SHA256,
        "video_reviews": [
            {
                "video_id": "train-a",
                "full_range_reviewed": True,
                "candidate_count": train_count,
                "reviewer": "reviewer-a",
            },
            {
                "video_id": "train-b",
                "full_range_reviewed": True,
                "candidate_count": train_b_count,
                "reviewer": "reviewer-c",
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
    split = "train" if video_id.startswith("train") else "eval"
    component = "train-component" if split == "train" else "eval-component"
    return {
        "episode_id": f"episode-{event_id}",
        "physical_event_id": event_id,
        "split": split,
        "source_window_component_id": component,
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

    summary = validate(review(records, train_count=1, eval_count=1), manifest())

    assert summary["accepted"]["train"]["pick_up"] == 1
    assert summary["accepted"]["eval"]["touch"] == 1
    assert summary["registration_minima_met"] is False


def test_duplicate_physical_event_fails_closed():
    records = [
        accepted("same", "train-a", "approach", 1, 20),
        accepted("same", "train-a", "pass_by", 30, 50),
    ]

    with pytest.raises(ValueError, match="duplicate physical_event_id"):
        validate(review(records, train_count=2), manifest())


def test_frame_bounds_must_fit_frozen_source():
    records = [accepted("evt", "train-a", "approach", 10, 301)]

    with pytest.raises(ValueError, match="frame bounds"):
        validate(review(records, train_count=1), manifest())


def test_candidate_count_includes_context_video_appearances():
    records = [
        accepted(
            "evt",
            "train-a",
            "pick_up",
            10,
            50,
            context=["train-b"],
        )
    ]

    summary = validate(
        review(records, train_count=1, train_b_count=1),
        manifest(),
    )

    assert summary["reviewed_candidate_appearances"]["train-a"] == 1
    assert summary["reviewed_candidate_appearances"]["train-b"] == 1


def test_candidate_count_mismatch_fails_closed():
    records = [accepted("evt", "train-a", "pick_up", 10, 50)]

    with pytest.raises(ValueError, match="candidate_count mismatch"):
        validate(review(records, train_count=0), manifest())


def test_unresolved_status_is_rejected():
    record = accepted("evt", "train-a", "touch", 10, 20)
    record["review_status"] = "needs_resolution"

    with pytest.raises(ValueError, match="needs_resolution"):
        validate(review([record], train_count=1), manifest())


def test_rejected_record_requires_frozen_reason_code():
    record = accepted("evt", "train-a", "touch", 10, 20)
    record["review_status"] = "rejected:not-a-frozen-reason"

    with pytest.raises(ValueError, match="rejection reason"):
        validate(review([record], train_count=1), manifest())


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("source_manifest_sha256", SOURCE_MANIFEST_SHA256),
        ("handbook_sha256", HANDBOOK_SHA256),
        ("sampling_protocol_sha256", SAMPLING_PROTOCOL_SHA256),
    ],
)
def test_review_must_match_frozen_input_digests(field, expected):
    payload = review([], train_count=0, train_b_count=0, eval_count=0)
    payload[field] = "f" * 64
    assert payload[field] != expected

    with pytest.raises(ValueError, match=field):
        validate(payload, manifest())


def test_review_status_must_be_reviewed_before_freeze():
    payload = review([], train_count=0, train_b_count=0, eval_count=0)
    payload["status"] = "draft_full_video_review_pending"

    with pytest.raises(ValueError, match="review status"):
        validate(payload, manifest())


def test_duplicate_episode_id_fails_closed():
    first = accepted("event-a", "train-a", "approach", 1, 20)
    second = accepted("event-b", "train-a", "pass_by", 30, 50)
    second["episode_id"] = first["episode_id"]

    with pytest.raises(ValueError, match="duplicate episode_id"):
        validate(review([first, second], train_count=2), manifest())


def test_record_split_must_match_registered_video():
    record = accepted("evt", "train-a", "approach", 10, 20)
    record["split"] = "eval"

    with pytest.raises(ValueError, match="split mismatch"):
        validate(review([record], train_count=1), manifest())


def test_record_component_must_match_registered_video():
    record = accepted("evt", "train-a", "approach", 10, 20)
    record["source_window_component_id"] = "other-component"

    with pytest.raises(ValueError, match="source-window component mismatch"):
        validate(review([record], train_count=1), manifest())


def complete_class_counts(train_count=20, eval_count=10):
    return {
        "train": {label: train_count for label in FREEZE.LABELS},
        "eval": {label: eval_count for label in FREEZE.LABELS},
    }


def test_registration_gate_requires_two_source_videos_per_split():
    assert not registration_minima_are_met(
        complete_class_counts(),
        accepted_source_video_counts={"train": 1, "eval": 2},
        accepted_component_counts={"train": 2, "eval": 2},
    )


def test_registration_gate_requires_two_components_per_split():
    assert not registration_minima_are_met(
        complete_class_counts(),
        accepted_source_video_counts={"train": 2, "eval": 2},
        accepted_component_counts={"train": 1, "eval": 2},
    )


def test_registration_gate_accepts_class_and_structural_minima():
    assert registration_minima_are_met(
        complete_class_counts(),
        accepted_source_video_counts={"train": 2, "eval": 2},
        accepted_component_counts={"train": 2, "eval": 2},
    )
