from __future__ import annotations

import numpy as np

from neural_state_machine.r1_e2_reservoir import E2Architecture, E2ReservoirSpec
from neural_state_machine.r1_e3m_probe import (
    delayed_geometry_targets,
    fit_reservoir_delay_probe,
)


def _api():
    from neural_state_machine.r1_e3m_controls import (
        collect_final_states,
        collect_prefix_permuted_final_states,
        collect_reset_final_states,
        evaluate_fixed_probe,
        permute_prefix,
    )

    return (
        collect_final_states,
        collect_prefix_permuted_final_states,
        collect_reset_final_states,
        evaluate_fixed_probe,
        permute_prefix,
    )


def _spec() -> E2ReservoirSpec:
    return E2ReservoirSpec(
        architecture=E2Architecture.FLAT,
        input_size=6,
        seed=7,
    )


def _tensors(count: int = 8) -> np.ndarray:
    values = np.zeros((count, 20, 6), dtype=np.float64)
    for sample in range(count):
        values[sample, :, 0] = (
            0.01 * sample + np.linspace(0.0, 0.19, 20)
        )
        values[sample, :, 1] = (
            0.02 * sample + np.linspace(0.0, 0.095, 20)
        )
        values[sample, :, 2] = 0.1 + 0.001 * sample
        values[sample, :, 3] = 0.2 + 0.002 * sample
        values[sample, :, 4] = 0.9
        values[sample, :, 5] = 1.0
    return values


def test_reset_control_destroys_prefix_but_preserves_suffix_effect() -> None:
    normal, _, reset, _, _ = _api()
    tensors = _tensors(2)
    tensors[1, :16, 0] += 0.5
    tensors[1, :16, 1] += 0.25
    tensors[1, 16:20] = tensors[0, 16:20]

    normal_states = normal(_spec(), tensors)
    reset_states = reset(_spec(), tensors)

    assert not np.array_equal(normal_states[0], normal_states[1])
    np.testing.assert_array_equal(reset_states[0], reset_states[1])


def test_prefix_permutation_keeps_last_four_bins_byte_identical() -> None:
    _, _, _, _, permute = _api()
    tensors = _tensors(3)
    window_ids = tuple(f"{index + 1:064x}" for index in range(3))

    permuted = permute(tensors, window_ids)

    np.testing.assert_array_equal(permuted[:, 16:20], tensors[:, 16:20])
    assert permuted.flags.writeable is False


def test_prefix_permutation_is_repeatable_per_window_id() -> None:
    _, _, _, _, permute = _api()
    tensors = _tensors(4)
    window_ids = tuple(f"{index + 10:064x}" for index in range(4))

    first = permute(tensors, window_ids)
    second = permute(tensors, window_ids)

    np.testing.assert_array_equal(first, second)


def test_fixed_probe_control_never_refits_coefficients() -> None:
    normal, permuted_states, _, evaluate_fixed, _ = _api()
    training = _tensors(12)
    evaluation = _tensors(6)
    train_states = normal(_spec(), training)
    probe = fit_reservoir_delay_probe(train_states, training, 10)
    before = probe.coefficient_digest()

    control_states = permuted_states(
        _spec(),
        evaluation,
        tuple(f"{index + 100:064x}" for index in range(6)),
    )
    result = evaluate_fixed(
        probe,
        control_states,
        delayed_geometry_targets(evaluation, 10),
    )

    assert result.coefficient_digest == before
    assert probe.coefficient_digest() == before
    assert result.metrics.sample_count == 6


def test_control_final_states_are_deterministic() -> None:
    normal, permuted_states, reset, _, _ = _api()
    tensors = _tensors(5)
    ids = tuple(f"{index + 200:064x}" for index in range(5))

    np.testing.assert_array_equal(
        normal(_spec(), tensors),
        normal(_spec(), tensors),
    )
    np.testing.assert_array_equal(
        reset(_spec(), tensors),
        reset(_spec(), tensors),
    )
    np.testing.assert_array_equal(
        permuted_states(_spec(), tensors, ids),
        permuted_states(_spec(), tensors, ids),
    )
