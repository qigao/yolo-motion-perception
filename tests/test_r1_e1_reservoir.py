from __future__ import annotations

import math

import numpy as np
import pytest

from neural_state_machine.r1_e1_reservoir import (
    ReservoirArchitecture,
    ReservoirSpec,
    build_reservoir,
)


def _manual_component(
    spec: ReservoirSpec,
    *,
    component_id: int,
    input_size: int,
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    budget_id = {64: 0, 256: 1}[spec.budget]
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                spec.seed,
                0x52314531,
                budget_id,
                int(spec.architecture),
                component_id,
            ]
        )
    )
    input_weights = rng.normal(
        0.0,
        1.0 / math.sqrt(input_size),
        size=(hidden_size, input_size),
    )
    recurrent = rng.normal(
        0.0,
        1.0 / math.sqrt(hidden_size),
        size=(hidden_size, hidden_size),
    )
    radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
    recurrent_weights = recurrent * (0.9 / radius)
    return input_weights, recurrent_weights


@pytest.mark.parametrize(
    ("architecture", "expected_id"),
    [
        (ReservoirArchitecture.SHALLOW, 0),
        (ReservoirArchitecture.GROUPED2, 1),
        (ReservoirArchitecture.GROUPED4, 2),
        (ReservoirArchitecture.DEEP2, 3),
        (ReservoirArchitecture.DEEP4, 4),
    ],
)
def test_architecture_ids_are_registered(architecture: ReservoirArchitecture, expected_id: int) -> None:
    assert int(architecture) == expected_id


@pytest.mark.parametrize(
    ("architecture", "budget", "widths"),
    [
        (ReservoirArchitecture.SHALLOW, 64, (64,)),
        (ReservoirArchitecture.GROUPED2, 64, (32, 32)),
        (ReservoirArchitecture.GROUPED4, 64, (16, 16, 16, 16)),
        (ReservoirArchitecture.DEEP2, 64, (32, 32)),
        (ReservoirArchitecture.DEEP4, 64, (16, 16, 16, 16)),
        (ReservoirArchitecture.SHALLOW, 256, (256,)),
        (ReservoirArchitecture.GROUPED2, 256, (128, 128)),
        (ReservoirArchitecture.GROUPED4, 256, (64, 64, 64, 64)),
        (ReservoirArchitecture.DEEP2, 256, (128, 128)),
        (ReservoirArchitecture.DEEP4, 256, (64, 64, 64, 64)),
    ],
)
def test_registered_layouts_preserve_total_budget(
    architecture: ReservoirArchitecture,
    budget: int,
    widths: tuple[int, ...],
) -> None:
    reservoir = build_reservoir(ReservoirSpec(architecture, budget, 4, 7))

    assert reservoir.component_widths == widths
    assert reservoir.state_dim == budget


@pytest.mark.parametrize("budget", [0, 1, 63, 65, 128, 512, True, 64.0, "64"])
def test_reservoir_spec_rejects_nonregistered_budget(budget: object) -> None:
    with pytest.raises(ValueError, match="budget"):
        ReservoirSpec(ReservoirArchitecture.SHALLOW, budget, 4, 7)


@pytest.mark.parametrize("input_size", [0, -1, True, 4.0, "4", None])
def test_reservoir_spec_rejects_invalid_input_size(input_size: object) -> None:
    with pytest.raises(ValueError, match="input_size"):
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, input_size, 7)


@pytest.mark.parametrize("seed", [-1, True, 7.0, "7", None, np.int64(7)])
def test_reservoir_spec_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 4, seed)


def test_reservoir_spec_requires_registered_architecture_enum() -> None:
    with pytest.raises(ValueError, match="architecture"):
        ReservoirSpec(0, 64, 4, 7)


def test_identical_specs_reproduce_digest_and_trajectory() -> None:
    spec = ReservoirSpec(ReservoirArchitecture.GROUPED2, 64, 3, 17)
    left = build_reservoir(spec)
    right = build_reservoir(spec)
    observations = (
        np.array([0.2, -0.3, 0.5]),
        np.array([-0.4, 0.1, 0.7]),
        np.array([0.0, 0.8, -0.2]),
    )

    assert left.parameter_digest() == right.parameter_digest()
    for observation in observations:
        np.testing.assert_array_equal(left.advance(observation), right.advance(observation))


@pytest.mark.parametrize(
    "changed_spec",
    [
        ReservoirSpec(ReservoirArchitecture.GROUPED2, 64, 3, 17),
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 256, 3, 17),
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 4, 17),
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 3, 29),
    ],
)
def test_parameter_digest_binds_registered_initialization(changed_spec: ReservoirSpec) -> None:
    baseline = build_reservoir(
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 3, 17)
    ).parameter_digest()

    assert build_reservoir(changed_spec).parameter_digest() != baseline


def test_unrelated_fixture_rng_consumption_cannot_change_reservoir_parameters() -> None:
    spec = ReservoirSpec(ReservoirArchitecture.DEEP2, 64, 3, 43)
    fixture_rng = np.random.default_rng(np.random.SeedSequence([43, 0x45314154]))
    fixture_rng.normal(size=137)
    before = build_reservoir(spec).parameter_digest()
    fixture_rng.normal(size=911)
    after = build_reservoir(spec).parameter_digest()

    assert before == after


def test_parameter_digest_is_unchanged_by_state_evolution_and_reset() -> None:
    reservoir = build_reservoir(
        ReservoirSpec(ReservoirArchitecture.DEEP4, 64, 3, 61)
    )
    before = reservoir.parameter_digest()

    reservoir.advance(np.array([0.2, -0.1, 0.9]))
    reservoir.advance(np.array([-0.7, 0.4, 0.1]))
    reservoir.reset()

    assert reservoir.parameter_digest() == before


def test_reset_returns_every_component_to_canonical_zero_state() -> None:
    spec = ReservoirSpec(ReservoirArchitecture.DEEP4, 64, 3, 7)
    reset_candidate = build_reservoir(spec)
    fresh = build_reservoir(spec)
    reset_candidate.advance(np.array([0.8, -0.3, 0.1]))
    reset_candidate.advance(np.array([0.4, 0.2, -0.6]))

    reset_candidate.reset()
    observation = np.array([-0.2, 0.9, 0.3])

    np.testing.assert_array_equal(
        reset_candidate.advance(observation),
        fresh.advance(observation),
    )


def test_advance_returns_readonly_independent_c_contiguous_float64_state() -> None:
    reservoir = build_reservoir(
        ReservoirSpec(ReservoirArchitecture.GROUPED4, 64, 3, 17)
    )
    observation = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    state = reservoir.advance(observation)
    saved = state.copy()
    observation[:] = 100.0

    assert state.dtype == np.float64
    assert state.flags.c_contiguous
    assert not state.flags.writeable
    np.testing.assert_array_equal(state, saved)
    with pytest.raises(ValueError):
        state[0] = 0.0


@pytest.mark.parametrize(
    "observation",
    [
        1.0,
        np.array([1.0, 2.0]),
        np.ones((1, 3)),
        np.array([1.0, np.nan, 2.0]),
        np.array([1.0, np.inf, 2.0]),
        [1.0, 2.0, "bad"],
    ],
)
def test_advance_rejects_malformed_observations(observation: object) -> None:
    reservoir = build_reservoir(
        ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 3, 7)
    )

    with pytest.raises(ValueError, match="observation"):
        reservoir.advance(observation)


def test_shallow_matches_registered_cell_initialization_and_recurrence() -> None:
    spec = ReservoirSpec(ReservoirArchitecture.SHALLOW, 64, 3, 7)
    reservoir = build_reservoir(spec)
    input_weights, recurrent_weights = _manual_component(
        spec,
        component_id=0,
        input_size=3,
        hidden_size=64,
    )
    expected = np.zeros(64, dtype=np.float64)

    for observation in (
        np.array([0.1, -0.2, 0.3]),
        np.array([0.4, 0.0, -0.5]),
        np.array([-0.7, 0.8, 0.2]),
    ):
        expected = np.tanh(input_weights @ observation + recurrent_weights @ expected)
        np.testing.assert_array_equal(reservoir.advance(observation), expected)


def test_grouped_components_receive_the_same_external_observation() -> None:
    spec = ReservoirSpec(ReservoirArchitecture.GROUPED2, 64, 3, 29)
    observation = np.array([0.2, -0.6, 0.4])
    expected_parts = []
    for component_id in range(2):
        input_weights, recurrent_weights = _manual_component(
            spec,
            component_id=component_id,
            input_size=3,
            hidden_size=32,
        )
        expected_parts.append(
            np.tanh(input_weights @ observation + recurrent_weights @ np.zeros(32))
        )

    expected = np.concatenate(expected_parts)

    np.testing.assert_array_equal(build_reservoir(spec).advance(observation), expected)


def test_deep_layers_consume_current_prior_layer_state() -> None:
    spec = ReservoirSpec(ReservoirArchitecture.DEEP2, 64, 3, 29)
    observation = np.array([0.2, -0.6, 0.4])
    first_input, first_recurrent = _manual_component(
        spec,
        component_id=0,
        input_size=3,
        hidden_size=32,
    )
    second_input, second_recurrent = _manual_component(
        spec,
        component_id=1,
        input_size=32,
        hidden_size=32,
    )
    first_state = np.tanh(first_input @ observation + first_recurrent @ np.zeros(32))
    second_state = np.tanh(second_input @ first_state + second_recurrent @ np.zeros(32))
    expected = np.concatenate((first_state, second_state))

    np.testing.assert_array_equal(build_reservoir(spec).advance(observation), expected)
