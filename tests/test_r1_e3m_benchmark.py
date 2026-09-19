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

    result = evaluate_arm(
        E2ReservoirSpec(
            E2Architecture.FLAT,
            input_size=6,
            seed=7,
        ),
        _dataset(),
    )

    assert tuple(item.delay for item in result.delays) == (1, 2, 5, 10, 15)
    assert np.isfinite(result.long_delay_delta)
    assert np.isfinite(result.h1_long_delay_drop)
    assert result.artifact_root_digest == "a" * 64
    assert len(result.reservoir_parameter_digest) == 64
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
