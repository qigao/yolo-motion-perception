from __future__ import annotations

import numpy as np


def _api():
    from neural_state_machine.r1_e2_reservoir import (
        E2Architecture,
        E2ReservoirSpec,
        build_e2_reservoir,
    )

    return E2Architecture, E2ReservoirSpec, build_e2_reservoir


def test_registered_topologies_hold_fixed_neuron_budget() -> None:
    E2Architecture, E2ReservoirSpec, build = _api()
    expected_widths = {
        E2Architecture.FLAT: (256,),
        E2Architecture.GROUPED4: (64, 64, 64, 64),
        E2Architecture.HIERARCHICAL2: (128, 128),
        E2Architecture.HIERARCHICAL4: (64, 64, 64, 64),
    }

    for architecture, widths in expected_widths.items():
        reservoir = build(E2ReservoirSpec(architecture=architecture, input_size=3, seed=7))
        assert reservoir.neuron_count == 256
        assert reservoir.state_dim == 256
        assert reservoir.component_widths == widths


def test_same_seed_is_deterministic_and_different_seed_changes_parameters() -> None:
    E2Architecture, E2ReservoirSpec, build = _api()
    spec = E2ReservoirSpec(architecture=E2Architecture.HIERARCHICAL4, input_size=3, seed=7)
    first = build(spec)
    second = build(spec)
    other = build(E2ReservoirSpec(architecture=E2Architecture.HIERARCHICAL4, input_size=3, seed=17))

    assert first.parameter_digest() == second.parameter_digest()
    assert first.parameter_digest() != other.parameter_digest()


def test_parameter_accounting_is_explicit_for_each_topology() -> None:
    E2Architecture, E2ReservoirSpec, build = _api()
    expected = {
        E2Architecture.FLAT: (256 * 256, 256 * 3),
        E2Architecture.GROUPED4: (4 * 64 * 64, 4 * 64 * 3),
        E2Architecture.HIERARCHICAL2: (2 * 128 * 128, 128 * 3 + 128 * 128),
        E2Architecture.HIERARCHICAL4: (4 * 64 * 64, 64 * 3 + 3 * 64 * 64),
    }

    for architecture, (recurrent_edges, input_parameters) in expected.items():
        reservoir = build(E2ReservoirSpec(architecture=architecture, input_size=3, seed=7))
        assert reservoir.recurrent_edge_count == recurrent_edges
        assert reservoir.input_parameter_count == input_parameters
        assert reservoir.total_parameter_count == recurrent_edges + input_parameters


def test_e2_dynamics_preserve_r1_e1_tanh_radius_and_leak_contract() -> None:
    E2Architecture, E2ReservoirSpec, build = _api()
    reservoir = build(E2ReservoirSpec(architecture=E2Architecture.FLAT, input_size=2, seed=7))

    assert reservoir.activation == "tanh"
    assert reservoir.spectral_radius == 0.9
    assert reservoir.leak == 1.0

    state = reservoir.advance(np.array([0.25, -0.5], dtype=np.float64))
    assert state.shape == (256,)
    assert state.dtype == np.float64
    assert not state.flags.writeable

    reservoir.reset()
    replay = reservoir.advance(np.array([0.25, -0.5], dtype=np.float64))
    assert np.array_equal(state, replay)


def test_invalid_e2_specs_fail_closed() -> None:
    E2Architecture, E2ReservoirSpec, _ = _api()

    for input_size in (0, -1, 1.5):
        try:
            E2ReservoirSpec(architecture=E2Architecture.FLAT, input_size=input_size, seed=7)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input_size must fail")

    for seed in (-1, 1.5):
        try:
            E2ReservoirSpec(architecture=E2Architecture.FLAT, input_size=3, seed=seed)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid seed must fail")
