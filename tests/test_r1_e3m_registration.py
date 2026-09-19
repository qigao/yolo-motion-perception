from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.r1_e3m_dataset import (
    MechanismDataset,
    MechanismSample,
)


def _api():
    from neural_state_machine.r1_e3m_registration import (
        MIN_EVAL_WINDOWS_PER_DELAY,
        MIN_SOURCE_VIDEOS_PER_SPLIT,
        MIN_TRAIN_WINDOWS_PER_DELAY,
        DelayRegistrationInvalid,
        build_delay_registration,
        delay_registration_from_payload,
        delay_registration_payload,
        validate_delay_registration_gate,
    )

    return (
        MIN_TRAIN_WINDOWS_PER_DELAY,
        MIN_EVAL_WINDOWS_PER_DELAY,
        MIN_SOURCE_VIDEOS_PER_SPLIT,
        DelayRegistrationInvalid,
        build_delay_registration,
        delay_registration_from_payload,
        delay_registration_payload,
        validate_delay_registration_gate,
    )


def _sample(index: int, split: str, video_index: int) -> MechanismSample:
    tensor = np.zeros((20, 6), dtype=np.float64)
    tensor[:, 0] = 0.01 * index + np.linspace(0.0, 0.19, 20)
    tensor[:, 1] = 0.005 * index + np.linspace(0.0, 0.095, 20)
    tensor[:, 2] = 0.1
    tensor[:, 3] = 0.2
    tensor[:, 4] = 0.9
    tensor[:, 5] = 1.0
    return MechanismSample(
        window_id=f"{index + 1:064x}",
        video_id=f"{split}-video-{video_index}",
        split=split,
        track_id=index,
        source_start_seconds=float(index * 2),
        source_end_seconds=float(index * 2 + 2),
        tensor=tensor,
    )


def _dataset(
    *,
    train_count: int = 24,
    eval_count: int = 12,
) -> MechanismDataset:
    training = tuple(
        _sample(index, "train", index % 3)
        for index in range(train_count)
    )
    evaluation = tuple(
        _sample(100 + index, "eval", index % 2)
        for index in range(eval_count)
    )
    return MechanismDataset(
        training=training,
        evaluation=evaluation,
        artifact_root_digest="a" * 64,
    )


def test_registration_freezes_exact_delays_and_window_ids() -> None:
    _, _, _, _, build, _, _, _ = _api()
    dataset = _dataset()

    registration = build(dataset)

    assert tuple(entry.delay for entry in registration.delays) == (
        1, 2, 5, 10, 15
    )
    assert len(registration.digest) == 64
    for entry in registration.delays:
        assert len(entry.training_window_ids) == len(dataset.training)
        assert len(entry.evaluation_window_ids) == len(dataset.evaluation)
        assert entry.training_window_ids == tuple(
            sorted(entry.training_window_ids)
        )
        assert entry.evaluation_window_ids == tuple(
            sorted(entry.evaluation_window_ids)
        )


def test_missing_target_bin_removes_only_that_delay_window() -> None:
    _, _, _, _, build, _, _, _ = _api()
    dataset = _dataset()
    first = dataset.evaluation[0]
    tensor = first.tensor.copy()
    tensor.setflags(write=True)
    target_bin = 19 - 10
    tensor[target_bin, :5] = 0.0
    tensor[target_bin, 5] = 0.0
    changed = MechanismSample(
        window_id=first.window_id,
        video_id=first.video_id,
        split=first.split,
        track_id=first.track_id,
        source_start_seconds=first.source_start_seconds,
        source_end_seconds=first.source_end_seconds,
        tensor=tensor,
    )
    changed_dataset = MechanismDataset(
        training=dataset.training,
        evaluation=(changed, *dataset.evaluation[1:]),
        artifact_root_digest=dataset.artifact_root_digest,
    )

    registration = build(changed_dataset)
    by_delay = {entry.delay: entry for entry in registration.delays}

    assert first.window_id not in by_delay[10].evaluation_window_ids
    assert first.window_id in by_delay[15].evaluation_window_ids


def test_registration_gate_requires_frozen_minima() -> None:
    train_min, eval_min, video_min, Invalid, build, _, _, validate = _api()

    assert train_min == 20
    assert eval_min == 10
    assert video_min == 2

    valid = build(_dataset())
    validate(valid)

    too_few_train = build(_dataset(train_count=19))
    with pytest.raises(Invalid, match="training windows"):
        validate(too_few_train)

    too_few_eval = build(_dataset(eval_count=9))
    with pytest.raises(Invalid, match="evaluation windows"):
        validate(too_few_eval)


def test_registration_gate_requires_two_source_videos_per_split() -> None:
    _, _, _, Invalid, build, _, _, validate = _api()
    dataset = _dataset()
    evaluation = tuple(
        MechanismSample(
            window_id=sample.window_id,
            video_id="eval-only-video",
            split=sample.split,
            track_id=sample.track_id,
            source_start_seconds=sample.source_start_seconds,
            source_end_seconds=sample.source_end_seconds,
            tensor=sample.tensor,
        )
        for sample in dataset.evaluation
    )
    single_video = MechanismDataset(
        training=dataset.training,
        evaluation=evaluation,
        artifact_root_digest=dataset.artifact_root_digest,
    )

    registration = build(single_video)

    with pytest.raises(Invalid, match="evaluation source videos"):
        validate(registration)


def test_registration_payload_roundtrip_is_self_validating() -> None:
    _, _, _, Invalid, build, from_payload, to_payload, _ = _api()
    registration = build(_dataset())

    payload = to_payload(registration)
    restored = from_payload(payload)

    assert restored == registration

    corrupted = dict(payload)
    corrupted["digest"] = "f" * 64
    with pytest.raises(Invalid, match="digest mismatch"):
        from_payload(corrupted)
