from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neural_state_machine.action_value_benchmark import ActionValueBenchmarkConfig
from neural_state_machine.phase3a_failure_attribution import (
    AttributionMeasurement,
    attribution_payload,
    run_failure_attribution,
)


def _small_config() -> ActionValueBenchmarkConfig:
    return ActionValueBenchmarkConfig(
        hidden_size=8,
        training_episodes=100,
        evaluation_blocks=2,
        checkpoint_interval=50,
    )


def test_attribution_has_fixed_baseline_and_single_factor_arms() -> None:
    result = run_failure_attribution(
        seed=7,
        config=_small_config(),
        reservoir_seeds=(17,),
        action_seeds=(29,),
        fixture_seeds=(31,),
    )

    assert tuple(measurement.arm for measurement in result) == (
        "baseline",
        "reservoir",
        "action_rng",
        "fixture",
    )
    baseline, reservoir, action_rng, fixture = result
    assert isinstance(baseline, AttributionMeasurement)
    assert baseline.reservoir_seed == baseline.action_seed == baseline.fixture_seed == 7
    assert reservoir.reservoir_seed == 17
    assert reservoir.action_seed == reservoir.fixture_seed == 7
    assert action_rng.action_seed == 29
    assert action_rng.reservoir_seed == action_rng.fixture_seed == 7
    assert fixture.fixture_seed == 31
    assert fixture.reservoir_seed == fixture.action_seed == 7
    assert baseline.post_training.total == 20
    assert baseline.supervised_reference.total == 20
    assert baseline.supervised_per_delay[0][1].total == 4
    assert reservoir.action_digest == baseline.action_digest
    assert fixture.action_digest == baseline.action_digest
    assert action_rng.action_digest != baseline.action_digest


def test_attribution_baseline_reuses_phase3a_fixture_and_action_lineages() -> None:
    first = run_failure_attribution(
        seed=7,
        config=_small_config(),
        reservoir_seeds=(),
        action_seeds=(),
        fixture_seeds=(),
    )
    second = run_failure_attribution(
        seed=7,
        config=_small_config(),
        reservoir_seeds=(),
        action_seeds=(),
        fixture_seeds=(),
    )
    baseline = first[0]
    assert first == second
    assert baseline.action_sequence_equal_to_baseline is True
    assert baseline.reward_sequence_equal_to_baseline is True
    assert baseline.fixture_digest != baseline.evaluation_fixture_digest


@pytest.mark.parametrize("bad", [(7, 7), (True,), (-1,)])
def test_attribution_rejects_invalid_variant_seed_sequences(bad: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        run_failure_attribution(
            seed=7,
            config=_small_config(),
            reservoir_seeds=bad,
            action_seeds=(),
            fixture_seeds=(),
        )


def test_attribution_measurements_are_immutable() -> None:
    measurement = run_failure_attribution(
        seed=7,
        config=_small_config(),
        reservoir_seeds=(),
        action_seeds=(),
        fixture_seeds=(),
    )[0]
    with pytest.raises(FrozenInstanceError):
        measurement.arm = "changed"


def test_attribution_payload_is_diagnostic_only_and_json_safe() -> None:
    measurement = run_failure_attribution(
        seed=7,
        config=_small_config(),
        reservoir_seeds=(),
        action_seeds=(),
        fixture_seeds=(),
    )[0]
    payload = attribution_payload((measurement,))
    assert payload["experiment"] == "phase-3a-failure-attribution"
    assert payload["diagnostic_only"] is True
    assert payload["schema_version"] == 1
    assert payload["arms"][0]["post_per_delay"]["5"]["total"] == 4
