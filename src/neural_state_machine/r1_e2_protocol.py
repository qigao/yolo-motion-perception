from __future__ import annotations

from dataclasses import dataclass

from .r1_e2_composition import (
    COMPOSITION_CLASSES,
    COMPOSITION_HISTORIES,
    E2_COMPOSITION_SEEDS,
    EVAL_GROUPS_PER_HISTORY,
    TRAIN_GROUPS_PER_HISTORY,
    build_composition_fixture_sets,
    run_composition_arm,
)
from .r1_e2_memory import (
    E2_MEMORY_DELAYS,
    E2_MEMORY_SEEDS,
    build_e2_memory_fixtures,
    run_e2_memory_arm,
)
from .r1_e2_reservoir import E2Architecture, E2ReservoirSpec, build_e2_reservoir
from .r1_e2_yolo_episode import (
    E2_EPISODE_SEEDS,
    EPISODE_CLASSES,
    EPISODE_CORRUPTIONS,
    EPISODE_TEMPORAL_WINDOW,
    build_e2_episode_corruption_plan,
    build_e2_episode_fixture_set,
    run_yolo_episode_arm,
)


REGISTERED_ARCHITECTURES = (
    E2Architecture.FLAT,
    E2Architecture.GROUPED4,
    E2Architecture.HIERARCHICAL2,
    E2Architecture.HIERARCHICAL4,
)
REGISTERED_SEEDS = (7, 17, 29, 43, 61)
NEURON_BUDGET = 256


class ProtocolInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class RegisteredArmResult:
    seed: int
    architecture: int
    memory: object
    composition: object
    yolo_episode: object


@dataclass(frozen=True)
class RegisteredMeasurementResult:
    registered_measurement: bool
    manifest: dict[str, object]
    arms: tuple[RegisteredArmResult, ...]


def registered_manifest_payload() -> dict[str, object]:
    return {
        "phase": "R1-E2",
        "architectures": {
            "flat": int(E2Architecture.FLAT),
            "grouped4": int(E2Architecture.GROUPED4),
            "hierarchical2": int(E2Architecture.HIERARCHICAL2),
            "hierarchical4": int(E2Architecture.HIERARCHICAL4),
        },
        "neuron_budget": NEURON_BUDGET,
        "seeds": list(REGISTERED_SEEDS),
        "reservoir": {
            "activation": "tanh",
            "spectral_radius": 0.9,
            "leak": 1.0,
            "dtype": "float64",
            "recurrent_bias": False,
        },
        "ridge": {
            "regularization": 1e-6,
            "bias": "enabled-unpenalized",
            "multiclass_tie": "lowest-index",
        },
        "e2_a": {
            "delays": list(E2_MEMORY_DELAYS),
            "memory_threshold": 0.85,
            "train_per_delay": 400,
            "evaluation_per_delay": 40,
        },
        "e2_b": {
            "classes": list(COMPOSITION_CLASSES),
            "histories": list(COMPOSITION_HISTORIES),
            "train_per_history": TRAIN_GROUPS_PER_HISTORY * len(COMPOSITION_CLASSES),
            "evaluation_per_history": EVAL_GROUPS_PER_HISTORY * len(COMPOSITION_CLASSES),
            "readouts": ["instantaneous", "temporal_mean"],
            "reset_accuracy": [25, 150],
        },
        "e2_c": {
            "classes": list(EPISODE_CLASSES),
            "corruptions": list(EPISODE_CORRUPTIONS),
            "train_count": 800,
            "evaluation_count": 200,
            "temporal_window": EPISODE_TEMPORAL_WINDOW,
            "readouts": [
                "frame_only",
                "reservoir_instantaneous",
                "reservoir_temporal_mean",
            ],
            "reset_accuracy": [50, 200],
        },
    }


def protocol_smoke(seed: int = 7) -> dict[str, object]:
    _validated_registered_seed(seed)

    parameter_digests: dict[str, str] = {}
    for architecture in REGISTERED_ARCHITECTURES:
        reservoir = build_e2_reservoir(
            E2ReservoirSpec(
                architecture=architecture,
                input_size=4,
                seed=seed,
            )
        )
        parameter_digests[_architecture_name(architecture)] = reservoir.parameter_digest()

    memory = build_e2_memory_fixtures(seed)
    composition_train = build_composition_fixture_sets(seed, training=True)[0]
    composition_eval = build_composition_fixture_sets(seed, training=False)[0]
    episode_train = build_e2_episode_fixture_set(seed, training=True)
    episode_eval = build_e2_episode_fixture_set(seed, training=False)
    corruption = build_e2_episode_corruption_plan(seed, episode_eval)

    fixture_digests = {
        "e2_a": memory.combined_digest,
        "e2_b_train_h5": composition_train.fixture_digest,
        "e2_b_eval_h5": composition_eval.fixture_digest,
        "e2_c_train": episode_train.fixture_digest,
        "e2_c_eval": episode_eval.fixture_digest,
        "e2_c_corruption": corruption.digest,
    }
    valid = (
        len(parameter_digests) == len(REGISTERED_ARCHITECTURES)
        and all(len(value) == 64 for value in parameter_digests.values())
        and all(len(value) == 64 for value in fixture_digests.values())
        and E2_MEMORY_SEEDS == E2_COMPOSITION_SEEDS == E2_EPISODE_SEEDS == REGISTERED_SEEDS
    )
    issues = [] if valid else [{"code": "smoke-invalid", "detail": "R1-E2 smoke invariant failed"}]
    return {
        "registered_measurement": False,
        "seed": seed,
        "arm_count": len(REGISTERED_ARCHITECTURES),
        "valid": valid,
        "issues": issues,
        "reservoir_parameter_digests": parameter_digests,
        "fixture_digests": fixture_digests,
    }


def run_registered_measurement() -> RegisteredMeasurementResult:
    arms: list[RegisteredArmResult] = []
    for seed in REGISTERED_SEEDS:
        memory_fixtures = build_e2_memory_fixtures(seed)
        composition_training = build_composition_fixture_sets(seed, training=True)
        composition_evaluation = build_composition_fixture_sets(seed, training=False)
        episode_training = build_e2_episode_fixture_set(seed, training=True)
        episode_evaluation = build_e2_episode_fixture_set(seed, training=False)
        episode_plan = build_e2_episode_corruption_plan(seed, episode_evaluation)

        for architecture in REGISTERED_ARCHITECTURES:
            memory = run_e2_memory_arm(
                E2ReservoirSpec(
                    architecture=architecture,
                    input_size=4,
                    seed=seed,
                ),
                memory_fixtures,
            )
            composition = run_composition_arm(
                E2ReservoirSpec(
                    architecture=architecture,
                    input_size=7,
                    seed=seed,
                ),
                composition_training,
                composition_evaluation,
            )
            yolo_episode = run_yolo_episode_arm(
                E2ReservoirSpec(
                    architecture=architecture,
                    input_size=9,
                    seed=seed,
                ),
                episode_training,
                episode_evaluation,
                episode_plan,
            )
            arms.append(
                RegisteredArmResult(
                    seed=seed,
                    architecture=int(architecture),
                    memory=memory,
                    composition=composition,
                    yolo_episode=yolo_episode,
                )
            )

    expected = len(REGISTERED_SEEDS) * len(REGISTERED_ARCHITECTURES)
    if len(arms) != expected:
        raise ProtocolInvalid("registered R1-E2 arm count mismatch")
    return RegisteredMeasurementResult(
        registered_measurement=True,
        manifest=registered_manifest_payload(),
        arms=tuple(arms),
    )


def _architecture_name(architecture: E2Architecture) -> str:
    if architecture is E2Architecture.FLAT:
        return "flat"
    if architecture is E2Architecture.GROUPED4:
        return "grouped4"
    if architecture is E2Architecture.HIERARCHICAL2:
        return "hierarchical2"
    if architecture is E2Architecture.HIERARCHICAL4:
        return "hierarchical4"
    raise ValueError("architecture must be registered")


def _validated_registered_seed(seed: object) -> int:
    if type(seed) is not int or seed not in REGISTERED_SEEDS:
        raise ValueError("seed must be one of the registered R1-E2 seeds")
    return seed
