from __future__ import annotations

import json

import pytest


def _api():
    from neural_state_machine import r1_e2_protocol as protocol

    return protocol


def test_r1_e2_registered_manifest_is_literal_and_json_safe() -> None:
    protocol = _api()
    payload = protocol.registered_manifest_payload()

    assert payload["phase"] == "R1-E2"
    assert payload["architectures"] == {
        "flat": 0,
        "grouped4": 1,
        "hierarchical2": 2,
        "hierarchical4": 3,
    }
    assert payload["neuron_budget"] == 256
    assert payload["seeds"] == [7, 17, 29, 43, 61]
    assert payload["ridge"]["regularization"] == 1e-6
    assert payload["e2_a"]["delays"] == [1, 2, 5, 10, 20, 40, 80]
    assert payload["e2_a"]["memory_threshold"] == 0.85
    assert payload["e2_b"]["classes"] == ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]
    assert payload["e2_b"]["histories"] == [5, 10, 20, 40]
    assert payload["e2_b"]["train_per_history"] == 600
    assert payload["e2_b"]["evaluation_per_history"] == 150
    assert payload["e2_b"]["readouts"] == ["instantaneous", "temporal_mean"]
    assert payload["e2_c"]["classes"] == ["approach", "touch", "pick_up", "pass_by"]
    assert payload["e2_c"]["corruptions"] == [
        "clean",
        "drop10",
        "wrong10",
        "occlusion4",
        "jitter",
        "mixed",
    ]
    assert payload["e2_c"]["train_count"] == 800
    assert payload["e2_c"]["evaluation_count"] == 200
    assert payload["e2_c"]["temporal_window"] == 4
    assert payload["e2_c"]["readouts"] == [
        "frame_only",
        "reservoir_instantaneous",
        "reservoir_temporal_mean",
    ]
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_r1_e2_protocol_smoke_is_small_deterministic_and_never_measures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _api()

    def forbidden_measurement():
        pytest.fail("protocol smoke invoked registered measurement")

    monkeypatch.setattr(protocol, "run_registered_measurement", forbidden_measurement)
    first = protocol.protocol_smoke(seed=7)
    second = protocol.protocol_smoke(seed=7)

    assert first == second
    assert first["registered_measurement"] is False
    assert first["seed"] == 7
    assert first["arm_count"] == 4
    assert first["valid"] is True
    assert first["issues"] == []
    assert set(first["reservoir_parameter_digests"]) == {
        "flat",
        "grouped4",
        "hierarchical2",
        "hierarchical4",
    }
    assert set(first["fixture_digests"]) == {
        "e2_a",
        "e2_b_train_h5",
        "e2_b_eval_h5",
        "e2_c_train",
        "e2_c_eval",
        "e2_c_corruption",
    }
    assert all(len(value) == 64 for value in first["reservoir_parameter_digests"].values())
    assert all(len(value) == 64 for value in first["fixture_digests"].values())


def test_r1_e2_registered_runner_signature_rejects_overrides() -> None:
    protocol = _api()

    with pytest.raises(TypeError):
        protocol.run_registered_measurement(seeds=(7,))


@pytest.mark.parametrize("seed", [-1, 5, True, 7.0, "7"])
def test_r1_e2_protocol_smoke_rejects_unregistered_seed(seed: object) -> None:
    protocol = _api()

    with pytest.raises(ValueError, match="registered"):
        protocol.protocol_smoke(seed=seed)
