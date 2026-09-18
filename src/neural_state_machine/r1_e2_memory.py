from __future__ import annotations

from dataclasses import dataclass

from .r1_e1_memory import (
    MEMORY_DELAYS,
    MemoryArmResult,
    MemoryFixtures,
    build_memory_fixtures,
    run_memory_arm,
)
from .r1_e1_reservoir import ReservoirArchitecture, ReservoirSpec
from .r1_e2_reservoir import E2Architecture, E2ReservoirSpec


E2_MEMORY_DELAYS = MEMORY_DELAYS
E2_MEMORY_SEEDS = (7, 17, 29, 43, 61)
_E2_MEMORY_BUDGET = 256


@dataclass(frozen=True)
class E2MemoryArmResult:
    architecture: E2Architecture
    seed: int
    core: MemoryArmResult

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, E2Architecture):
            raise ValueError("architecture must be a registered E2Architecture")
        _validated_registered_seed(self.seed)
        if not isinstance(self.core, MemoryArmResult):
            raise ValueError("core must be MemoryArmResult")


def build_e2_memory_fixtures(seed: int) -> MemoryFixtures:
    return build_memory_fixtures(_validated_registered_seed(seed))


def run_e2_memory_arm(
    spec: E2ReservoirSpec,
    fixtures: MemoryFixtures,
) -> E2MemoryArmResult:
    if not isinstance(spec, E2ReservoirSpec):
        raise ValueError("spec must be E2ReservoirSpec")
    if spec.input_size != 4:
        raise ValueError("E2-A input_size must be four")
    seed = _validated_registered_seed(spec.seed)
    if not isinstance(fixtures, MemoryFixtures):
        raise ValueError("fixtures must be MemoryFixtures")

    core = run_memory_arm(
        ReservoirSpec(
            architecture=_r1_e1_architecture(spec.architecture),
            budget=_E2_MEMORY_BUDGET,
            input_size=4,
            seed=seed,
        ),
        fixtures,
    )
    return E2MemoryArmResult(
        architecture=spec.architecture,
        seed=seed,
        core=core,
    )


def _r1_e1_architecture(architecture: E2Architecture) -> ReservoirArchitecture:
    if architecture is E2Architecture.FLAT:
        return ReservoirArchitecture.SHALLOW
    if architecture is E2Architecture.GROUPED4:
        return ReservoirArchitecture.GROUPED4
    if architecture is E2Architecture.HIERARCHICAL2:
        return ReservoirArchitecture.DEEP2
    if architecture is E2Architecture.HIERARCHICAL4:
        return ReservoirArchitecture.DEEP4
    raise ValueError("architecture must be a registered E2Architecture")


def _validated_registered_seed(seed: object) -> int:
    if type(seed) is not int or seed not in E2_MEMORY_SEEDS:
        raise ValueError("seed must be one of the registered R1-E2 seeds")
    return seed
