from __future__ import annotations

import pytest

from neural_state_machine.r1_e1_memory import (
    MEMORY_DELAYS,
    build_memory_fixtures,
    run_memory_arm,
)
from neural_state_machine.r1_e1_reservoir import ReservoirArchitecture, ReservoirSpec
from neural_state_machine.r1_e2_reservoir import E2Architecture, E2ReservoirSpec


def _api():
    from neural_state_machine.r1_e2_memory import (
        E2_MEMORY_DELAYS,
        E2_MEMORY_SEEDS,
        build_e2_memory_fixtures,
        run_e2_memory_arm,
    )

    return (
        E2_MEMORY_DELAYS,
        E2_MEMORY_SEEDS,
        build_e2_memory_fixtures,
        run_e2_memory_arm,
    )


def test_e2a_registry_reuses_frozen_r1_e1_delay_grid_and_seed_set() -> None:
    delays, seeds, _, _ = _api()

    assert delays == MEMORY_DELAYS == (1, 2, 5, 10, 20, 40, 80)
    assert seeds == (7, 17, 29, 43, 61)


def test_e2a_fixtures_are_exactly_the_frozen_r1_e1_fixtures() -> None:
    _, _, build_fixtures, _ = _api()

    e2 = build_fixtures(17)
    reference = build_memory_fixtures(17)

    assert e2.training_digest == reference.training_digest
    assert e2.evaluation_digest == reference.evaluation_digest
    assert e2.combined_digest == reference.combined_digest
    assert len(e2.training) == len(reference.training) == 2_800
    assert len(e2.evaluation) == len(reference.evaluation) == 280


@pytest.mark.parametrize(
    ("architecture", "reference_architecture"),
    [
        (E2Architecture.FLAT, ReservoirArchitecture.SHALLOW),
        (E2Architecture.GROUPED4, ReservoirArchitecture.GROUPED4),
        (E2Architecture.HIERARCHICAL2, ReservoirArchitecture.DEEP2),
        (E2Architecture.HIERARCHICAL4, ReservoirArchitecture.DEEP4),
    ],
)
def test_e2a_arm_is_exact_compatibility_adapter(
    architecture: E2Architecture,
    reference_architecture: ReservoirArchitecture,
) -> None:
    _, _, build_fixtures, run_arm = _api()
    fixtures = build_fixtures(7)
    spec = E2ReservoirSpec(architecture=architecture, input_size=4, seed=7)

    result = run_arm(spec, fixtures)
    reference = run_memory_arm(
        ReservoirSpec(
            architecture=reference_architecture,
            budget=256,
            input_size=4,
            seed=7,
        ),
        fixtures,
    )

    assert result.architecture is architecture
    assert result.seed == 7
    assert result.core == reference
    assert result.core.fixture_digest == fixtures.combined_digest
    assert result.core.valid is True


def test_e2a_rejects_non_memory_input_shape_and_unregistered_seed() -> None:
    _, _, build_fixtures, run_arm = _api()
    fixtures = build_fixtures(7)

    with pytest.raises(ValueError, match="input_size"):
        run_arm(
            E2ReservoirSpec(
                architecture=E2Architecture.FLAT,
                input_size=3,
                seed=7,
            ),
            fixtures,
        )

    with pytest.raises(ValueError, match="registered"):
        run_arm(
            E2ReservoirSpec(
                architecture=E2Architecture.FLAT,
                input_size=4,
                seed=5,
            ),
            fixtures,
        )
