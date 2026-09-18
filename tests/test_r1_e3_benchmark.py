from __future__ import annotations

import hashlib

import numpy as np
import pytest

from neural_state_machine.r1_e2_reservoir import E2Architecture, E2ReservoirSpec
from neural_state_machine.r1_e3_dataset import RegisteredDataset, RegisteredEpisode


_LABELS = ("approach", "touch", "pick_up", "pass_by")


def _api():
    from neural_state_machine.r1_e3_benchmark import evaluate_real_track_arm

    return evaluate_real_track_arm


def _tensor(label: int, replicate: int) -> np.ndarray:
    values = np.zeros((20, 14), dtype=np.float64)
    time = np.arange(20, dtype=np.float64) / 19.0
    values[:, label] = 0.2 + 0.6 * time
    values[:, 4 + label] = 0.1 + 0.3 * (1.0 - time)
    values[:, 8 + label] = 0.15 + 0.2 * time
    values[:, 12] = 0.05 * label + 0.01 * replicate
    values[:, 13] = 0.1 + 0.02 * label
    return values


def _episode(
    *,
    split: str,
    label_index: int,
    replicate: int,
) -> RegisteredEpisode:
    tensor = _tensor(label_index, replicate)
    digest = hashlib.sha256(
        f"{split}-{label_index}-{replicate}".encode("ascii")
    ).hexdigest()
    return RegisteredEpisode(
        episode_id=f"{split}-{label_index}-{replicate}",
        video_id=f"{split}-video-{replicate}",
        split=split,
        label=_LABELS[label_index],
        tensor=tensor,
        tensor_digest=digest,
    )


def _dataset() -> RegisteredDataset:
    training = tuple(
        _episode(split="training", label_index=label, replicate=replicate)
        for replicate in range(2)
        for label in range(4)
    )
    evaluation = tuple(
        _episode(split="evaluation", label_index=label, replicate=replicate)
        for replicate in range(2)
        for label in range(4)
    )
    return RegisteredDataset(
        training=training,
        evaluation=evaluation,
        artifact_root_digest="a" * 64,
    )


def _spec(
    architecture: E2Architecture = E2Architecture.FLAT,
    seed: int = 7,
) -> E2ReservoirSpec:
    return E2ReservoirSpec(
        architecture=architecture,
        input_size=14,
        seed=seed,
    )


def test_b1_and_b2_share_exact_same_reservoir_trajectory() -> None:
    evaluate = _api()

    result = evaluate(_spec(), _dataset())

    assert result.trajectory_digest_b1 == result.trajectory_digest_b2
    assert len(result.trajectory_digest_b1) == 64


def test_history_destruction_resets_immediately_before_bin_16() -> None:
    evaluate = _api()

    result = evaluate(_spec(E2Architecture.GROUPED4, 17), _dataset())

    assert result.history_destruction.reset_before_bin == 16
    assert result.history_destruction.suffix_bins == (16, 17, 18, 19)
    assert result.history_destruction.instantaneous.metrics.total == 8
    assert result.history_destruction.temporal_mean.metrics.total == 8


def test_real_track_arm_reports_all_registered_readouts_and_digests() -> None:
    evaluate = _api()

    result = evaluate(_spec(), _dataset())

    assert result.frame_only.metrics.total == 8
    assert result.reservoir_instantaneous.metrics.total == 8
    assert result.reservoir_temporal_mean.metrics.total == 8
    assert len(result.frame_only.metrics.confusion_counts) == 4
    assert all(len(row) == 4 for row in result.frame_only.metrics.confusion_counts)
    assert len(result.frame_only.coefficient_digest) == 64
    assert len(result.reservoir_instantaneous.coefficient_digest) == 64
    assert len(result.reservoir_temporal_mean.coefficient_digest) == 64
    assert len(result.reservoir_parameter_digest) == 64
    assert len(result.training_episode_digest) == 64
    assert len(result.evaluation_episode_digest) == 64
    assert result.delta_macro_f1 == pytest.approx(
        result.reservoir_temporal_mean.metrics.macro_f1
        - result.reservoir_instantaneous.metrics.macro_f1
    )


def test_real_track_arm_is_exactly_repeatable() -> None:
    evaluate = _api()
    dataset = _dataset()
    spec = _spec(E2Architecture.HIERARCHICAL2, 29)

    first = evaluate(spec, dataset)
    second = evaluate(spec, dataset)

    assert first == second


def test_real_track_arm_rejects_wrong_reservoir_input_width() -> None:
    evaluate = _api()

    with pytest.raises(ValueError, match="input_size"):
        evaluate(
            E2ReservoirSpec(
                architecture=E2Architecture.FLAT,
                input_size=13,
                seed=7,
            ),
            _dataset(),
        )
