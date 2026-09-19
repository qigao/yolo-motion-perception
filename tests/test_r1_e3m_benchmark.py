from __future__ import annotations

import numpy as np

from neural_state_machine.r1_e2_reservoir import (
    E2Architecture,
    E2ReservoirSpec,
)
from neural_state_machine.r1_e3m_dataset import (
    MechanismDataset,
    MechanismSample,
)
from neural_state_machine.r1_e3m_pairs import build_history_pairs


def _api():
    from neural_state_machine.r1_e3m_benchmark import (
        REGISTERED_ARCHITECTURES,
        REGISTERED_SEEDS,
        classify_memory_outcome,
        evaluate_memory_arm,
        registered_specs,
    )

    return (
        REGISTERED_ARCHITECTURES,
        REGISTERED_SEEDS,
        classify_memory_outcome,
        evaluate_memory_arm,
        registered_specs,
    )


def _sample(index: int, split: str) -> MechanismSample:
    tensor = np.zeros((20, 6), dtype=np.float64)
    phase = 0.01 * index
    tensor[:, 0] = phase + np.linspace(0.0, 0.19, 20)
    tensor[:, 1] = 0.5 * phase + np.linspace(0.0, 0.095, 20)
    tensor[:, 2] = 0.1 + 0.001 * index
    tensor[:, 3] = 0.2 + 0.0015 * index
    tensor[:, 4] = 0.8 + 0.001 * (index % 10)
    tensor[:, 5] = 1.0
    return MechanismSample(
        window_id=f"{index + 1:064x}",
        video_id=f"{split}-video-{index % 2}",
        split=split,
        track_id=index,
        source_start_seconds=0.0,
        source_end_seconds=2.0,
        tensor=tensor,
    )


def _dataset() -> MechanismDataset:
    return MechanismDataset(
        training=tuple(_sample(index, "train") for index in range(12)),
        evaluation=tuple(
            _sample(index + 100, "eval") for index in range(6)
        ),
        artifact_root_digest="a" * 64,
    )


def test_registered_specs_are_exact_20_in_frozen_order() -> None:
    architectures, seeds, _, _, registered_specs = _api()

    specs = registered_specs()

    assert architectures == (
        E2Architecture.FLAT,
        E2Architecture.GROUPED4,
        E2Architecture.HIERARCHICAL2,
        E2Architecture.HIERARCHICAL4,
    )
    assert seeds == (7, 17, 29, 43, 61)
    assert len(specs) == 20
    assert specs[0] == E2ReservoirSpec(
        E2Architecture.FLAT,
        input_size=6,
        seed=7,
    )
    assert specs[4] == E2ReservoirSpec(
        E2Architecture.FLAT,
        input_size=6,
        seed=17,
    )


def test_single_arm_reports_all_delays_and_long_delay_summary() -> None:
    _, _, _, evaluate_arm, _ = _api()

    dataset = _dataset()
    pair_set = build_history_pairs(
        dataset.training,
        dataset.evaluation,
    )
    result = evaluate_arm(
        E2ReservoirSpec(
            E2Architecture.FLAT,
            input_size=6,
            seed=7,
        ),
        dataset,
        pair_set,
    )

    assert tuple(item.delay for item in result.delays) == (1, 2, 5, 10, 15)
    assert np.isfinite(result.long_delay_delta)
    assert np.isfinite(result.h1_long_delay_drop)
    assert result.artifact_root_digest == "a" * 64
    assert len(result.reservoir_parameter_digest) == 64
    assert result.pair_set_digest == pair_set.pair_digest
    assert len(result.pair_diagnostics) == len(pair_set.pairs)
    for item in result.delays:
        assert (
            item.reset_control.coefficient_digest
            == item.reservoir.coefficient_digest
        )
        assert (
            item.permuted_control.coefficient_digest
            == item.reservoir.coefficient_digest
        )


def test_outcome_a_requires_positive_robust_memory_and_reset_drop() -> None:
    _, _, classify, _, _ = _api()

    outcome = classify(
        [0.1] * 16 + [-0.01] * 4,
        [0.05] * 20,
    )

    assert outcome == "M-A"


def test_outcome_b_when_delta_positive_but_not_robust() -> None:
    _, _, classify, _, _ = _api()

    outcome = classify(
        [0.02] * 11 + [-0.01] * 9,
        [0.05] * 20,
    )

    assert outcome == "M-B"


def test_outcome_b_when_reset_control_does_not_drop() -> None:
    _, _, classify, _, _ = _api()

    outcome = classify(
        [0.1] * 20,
        [-0.01] * 20,
    )

    assert outcome == "M-B"


def test_outcome_c_when_median_delta_not_positive() -> None:
    _, _, classify, _, _ = _api()

    outcome = classify(
        [-0.01] * 20,
        [0.05] * 20,
    )

    assert outcome == "M-C"


def test_registered_benchmark_rejects_pair_set_from_other_inputs() -> None:
    import pytest

    from neural_state_machine.r1_e3m_benchmark import (
        run_registered_memory_benchmark,
    )

    dataset = _dataset()
    correct = build_history_pairs(
        dataset.training,
        dataset.evaluation,
    )

    changed_first = dataset.evaluation[0]
    changed_tensor = changed_first.tensor.copy()
    changed_tensor.setflags(write=True)
    changed_tensor[:16, 0] += 0.4
    changed = MechanismSample(
        window_id=changed_first.window_id,
        video_id=changed_first.video_id,
        split=changed_first.split,
        track_id=changed_first.track_id,
        source_start_seconds=changed_first.source_start_seconds,
        source_end_seconds=changed_first.source_end_seconds,
        tensor=changed_tensor,
    )
    wrong = build_history_pairs(
        dataset.training,
        (changed, *dataset.evaluation[1:]),
    )
    assert wrong != correct

    with pytest.raises(ValueError, match="history pair set"):
        run_registered_memory_benchmark(dataset, wrong)


def test_all_delay_arms_share_same_target_presence_mask() -> None:
    _, _, _, evaluate_arm, _ = _api()
    base = _dataset()
    first = base.evaluation[0]
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
    dataset = MechanismDataset(
        training=base.training,
        evaluation=(changed, *base.evaluation[1:]),
        artifact_root_digest=base.artifact_root_digest,
    )
    pair_set = build_history_pairs(
        dataset.training,
        dataset.evaluation,
    )

    result = evaluate_arm(
        E2ReservoirSpec(
            E2Architecture.FLAT,
            input_size=6,
            seed=7,
        ),
        dataset,
        pair_set,
    )
    delay10 = next(item for item in result.delays if item.delay == 10)

    expected = len(dataset.evaluation) - 1
    assert delay10.instantaneous.metrics.sample_count == expected
    assert delay10.reservoir.metrics.sample_count == expected
    assert delay10.reset_control.metrics.sample_count == expected
    assert delay10.permuted_control.metrics.sample_count == expected
