import numpy as np

from neural_state_machine.r1_e2_reservoir import E2Architecture
from neural_state_machine.r1_e3m_representation import (
    REGISTERED_ARCHITECTURES,
    REGISTERED_SEEDS,
    build_representations,
    registered_arms,
)


def _sequences() -> np.ndarray:
    values = np.zeros((3, 20, 14), dtype=np.float64)
    for sample in range(values.shape[0]):
        for step in range(values.shape[1]):
            values[sample, step, :] = (
                sample * 0.05 + step * 0.01 + np.arange(14) * 0.001
            )
    return np.clip(values, 0.0, 1.0)


def test_registered_arm_count_is_exactly_twenty():
    arms = registered_arms()

    assert REGISTERED_ARCHITECTURES == (
        E2Architecture.FLAT,
        E2Architecture.GROUPED4,
        E2Architecture.HIERARCHICAL2,
        E2Architecture.HIERARCHICAL4,
    )
    assert REGISTERED_SEEDS == (7, 17, 29, 43, 61)
    assert len(arms) == 20
    assert len({(int(arm.architecture), arm.seed) for arm in arms}) == 20


def test_representations_have_frozen_shapes_and_raw_baselines():
    sequences = _sequences()

    result = build_representations(
        sequences,
        architecture=E2Architecture.FLAT,
        seed=7,
    )

    assert result.c0.shape == (3, 14)
    assert result.s4.shape == (3, 56)
    assert result.trajectories.shape == (3, 20, 256)
    assert result.r1.shape == (3, 256)
    assert result.r4.shape == (3, 256)
    assert np.array_equal(result.c0, sequences[:, 19, :])
    assert np.array_equal(result.s4, sequences[:, 16:20, :].reshape(3, 56))
    assert np.array_equal(result.r1, result.trajectories[:, 19, :])
    assert np.allclose(
        result.r4,
        np.mean(result.trajectories[:, 16:20, :], axis=1),
    )


def test_parameter_digest_is_stable_for_same_arm():
    sequences = _sequences()[:1]

    first = build_representations(
        sequences,
        architecture=E2Architecture.GROUPED4,
        seed=17,
    )
    second = build_representations(
        sequences,
        architecture=E2Architecture.GROUPED4,
        seed=17,
    )

    assert first.parameter_digest == second.parameter_digest
    assert len(first.parameter_digest) == 64


def test_different_registered_seeds_change_reservoir_state():
    sequences = _sequences()[:1]

    first = build_representations(
        sequences,
        architecture=E2Architecture.HIERARCHICAL2,
        seed=7,
    )
    second = build_representations(
        sequences,
        architecture=E2Architecture.HIERARCHICAL2,
        seed=17,
    )

    assert not np.array_equal(first.r1, second.r1)
