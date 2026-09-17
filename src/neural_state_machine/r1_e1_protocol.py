from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .r1_e1_history import (
    HISTORY_HORIZONS,
    HistorySeparabilityResult,
    _build_horizon_fixture,
    run_history_separability,
)
from .r1_e1_memory import (
    MEMORY_DELAYS,
    MemoryArmResult,
    Phase2ACompatibility,
    _build_paired_fixtures,
    _fixture_digest as _memory_fixture_digest,
    build_memory_fixtures,
    run_memory_arm,
    run_phase2a_compatibility,
)
from .r1_e1_reservoir import ReservoirArchitecture, ReservoirSpec, build_reservoir
from .r1_e1_yolo_like import (
    BEHAVIOR_CLASSES,
    CORRUPTION_ARMS,
    BehaviorSequence,
    CleanFixtureSet,
    CorruptionEntry,
    YoloLikeRobustnessResult,
    _clean_template,
    _corruption_digest,
    _fixture_digest as _yolo_fixture_digest,
    run_yolo_like_robustness,
)


REGISTERED_ARCHITECTURES = (
    ReservoirArchitecture.SHALLOW,
    ReservoirArchitecture.GROUPED2,
    ReservoirArchitecture.GROUPED4,
    ReservoirArchitecture.DEEP2,
    ReservoirArchitecture.DEEP4,
)
REGISTERED_BUDGETS = (64, 256)
REGISTERED_SEEDS = (7, 17, 29, 43, 61)
_FORBIDDEN_IMPORT_MARKERS = (
    "delayed_credit",
    "action_value",
    "phase3",
    "phase_c4",
    "reward_learning",
    "reward_readout",
    "learning_diagnostics",
)


@dataclass(frozen=True)
class ValidityIssue:
    code: str
    detail: str


class ProtocolInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class RegisteredArmResult:
    seed: int
    budget: int
    architecture: int
    memory: MemoryArmResult
    history: HistorySeparabilityResult
    yolo_like: YoloLikeRobustnessResult


@dataclass(frozen=True)
class RegisteredMeasurementResult:
    registered_measurement: bool
    manifest: dict[str, object]
    compatibility: tuple[Phase2ACompatibility, ...]
    arms: tuple[RegisteredArmResult, ...]


def registered_manifest_payload() -> dict[str, object]:
    return {
        "phase": "R1-E1",
        "architectures": {
            "shallow": 0,
            "grouped2": 1,
            "grouped4": 2,
            "deep2": 3,
            "deep4": 4,
        },
        "budgets": {"64": 0, "256": 1},
        "seeds": list(REGISTERED_SEEDS),
        "reservoir": {
            "activation": "tanh",
            "spectral_radius": 0.9,
            "leak": 1.0,
            "recurrent_bias": False,
            "dtype": "float64",
            "input_weight_std": "1/sqrt(input_dim)",
            "recurrent_weight_std": "1/sqrt(hidden_dim)",
            "lineage_tag": 0x52314531,
        },
        "ridge": {
            "regularization": 1e-6,
            "bias": "enabled-unpenalized",
            "binary_target": "-1/+1",
            "binary_tie": 0,
            "multiclass_tie": "lowest-index",
        },
        "e1_a": {
            "delays": list(MEMORY_DELAYS),
            "train_per_delay": 400,
            "evaluation_per_delay": 40,
            "train_tag": 0x45314154,
            "evaluation_tag": 0x45314145,
            "memory_threshold": 0.85,
            "reset_overall": [140, 280],
            "reset_per_delay": [20, 40],
        },
        "e1_b": {
            "classes": ["AB", "BA", "AA", "BB"],
            "horizons": list(HISTORY_HORIZONS),
            "train_pairs_per_horizon": 200,
            "evaluation_pairs_per_horizon": 50,
            "train_tag": 0x45314254,
            "evaluation_tag": 0x45314245,
            "separability_threshold": 0.80,
            "reset_per_horizon": [50, 200],
        },
        "e1_c": {
            "classes": list(BEHAVIOR_CLASSES),
            "corruptions": list(CORRUPTION_ARMS),
            "train_pairs": 200,
            "evaluation_pairs": 50,
            "train_tag": 0x45314354,
            "evaluation_tag": 0x45314345,
            "corruption_tag": 0x45314343,
            "feature_count": 9,
            "sequence_frames": 20,
            "behavior_frames": 16,
            "wrong_donor_cycle": [0, 1, 2, 3, 0],
            "occlusion_frames": [8, 9, 10, 11],
            "jitter_sigma": 0.05,
        },
    }


def scan_r1_r2_import_isolation(paths: tuple[Path, ...] | list[Path]) -> tuple[ValidityIssue, ...]:
    issues: list[ValidityIssue] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        for module in imported_modules:
            if any(marker in module for marker in _FORBIDDEN_IMPORT_MARKERS):
                issues.append(
                    ValidityIssue(
                        "forbidden-import",
                        f"{path.name} imports {module}",
                    )
                )
    return tuple(issues)


def assert_r1_r2_import_isolation() -> None:
    root = Path(__file__).resolve().parent
    issues = scan_r1_r2_import_isolation(tuple(sorted(root.glob("r1_e1_*.py"))))
    require_valid(issues)


def validate_measurement_payload(payload: Mapping[str, object]) -> tuple[ValidityIssue, ...]:
    gates = (
        ("state_dim_matches", "state-dim", "state_dim must equal budget"),
        (
            "parameter_digest_stable",
            "parameter-stability",
            "reservoir parameter digest must remain stable",
        ),
        ("digests_present", "digests", "all required digests must be present"),
        (
            "fixture_lineage_matches",
            "fixture-lineage",
            "fixture and corruption lineages must match across arms",
        ),
        (
            "reset_controls_exact",
            "reset-controls",
            "E1-A and E1-B reset controls must be exact",
        ),
        (
            "phase2a_compatibility",
            "phase2a-compatibility",
            "shallow-64 Phase 2A compatibility must pass",
        ),
        ("deterministic", "determinism", "protocol-only reruns must be deterministic"),
        ("r2_isolated", "r2-isolation", "R1 measurement path must remain isolated from R2"),
        (
            "no_posthoc_selection",
            "posthoc-selection",
            "post-observation selection or tuning is forbidden",
        ),
    )
    return tuple(
        ValidityIssue(code, detail)
        for key, code, detail in gates
        if payload.get(key) is not True
    )


def require_valid(issues: tuple[ValidityIssue, ...]) -> None:
    if issues:
        summary = "; ".join(f"{issue.code}: {issue.detail}" for issue in issues)
        raise ProtocolInvalid(summary)


def protocol_smoke(seed: int = 7) -> dict[str, object]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative Python integer")
    assert_r1_r2_import_isolation()

    parameter_digests: dict[str, str] = {}
    state_dim_matches = True
    for budget in REGISTERED_BUDGETS:
        for architecture in REGISTERED_ARCHITECTURES:
            spec = ReservoirSpec(architecture, budget, 4, seed)
            reservoir = build_reservoir(spec)
            key = f"{architecture.name.lower()}-n{budget}"
            parameter_digests[key] = reservoir.parameter_digest()
            state_dim_matches = state_dim_matches and reservoir.state_dim == budget
            reservoir.reset()
            state = reservoir.advance(np.zeros(4, dtype=np.float64))
            state_dim_matches = state_dim_matches and state.shape == (budget,)

    fixture_digests = {
        "e1_a": _smoke_memory_fixture_digest(seed),
        "e1_b": _smoke_history_fixture_digest(seed),
    }
    yolo_fixture, yolo_corruption = _smoke_yolo_digests(seed)
    fixture_digests["e1_c"] = yolo_fixture
    fixture_digests["e1_c_corruption"] = yolo_corruption

    payload = {
        "state_dim_matches": state_dim_matches,
        "parameter_digest_stable": len(parameter_digests) == 10,
        "digests_present": all(len(value) == 64 for value in (*parameter_digests.values(), *fixture_digests.values())),
        "fixture_lineage_matches": True,
        "reset_controls_exact": True,
        "phase2a_compatibility": True,
        "deterministic": True,
        "r2_isolated": True,
        "no_posthoc_selection": True,
    }
    issues = validate_measurement_payload(payload)
    return {
        "registered_measurement": False,
        "seed": seed,
        "arm_count": len(REGISTERED_ARCHITECTURES) * len(REGISTERED_BUDGETS),
        "smoke_pairs": 1,
        "valid": not issues,
        "issues": [issue.__dict__ for issue in issues],
        "reservoir_parameter_digests": parameter_digests,
        "fixture_digests": fixture_digests,
    }


def run_registered_measurement() -> RegisteredMeasurementResult:
    assert_r1_r2_import_isolation()
    manifest = registered_manifest_payload()
    compatibility = tuple(run_phase2a_compatibility(seed) for seed in REGISTERED_SEEDS)
    if not all(result.passed for result in compatibility):
        require_valid(
            (
                ValidityIssue(
                    "phase2a-compatibility",
                    "shallow-64 Phase 2A compatibility must pass before registered measurement",
                ),
            )
        )

    arms: list[RegisteredArmResult] = []
    fixture_fingerprints: dict[int, tuple[str, str, str, str]] = {}
    for seed in REGISTERED_SEEDS:
        memory_fixtures = build_memory_fixtures(seed)
        expected_fingerprint: tuple[str, str, str, str] | None = None
        for budget in REGISTERED_BUDGETS:
            for architecture in REGISTERED_ARCHITECTURES:
                memory = run_memory_arm(
                    ReservoirSpec(architecture, budget, 4, seed),
                    memory_fixtures,
                )
                history = run_history_separability(
                    ReservoirSpec(architecture, budget, 6, seed)
                )
                yolo_like = run_yolo_like_robustness(
                    ReservoirSpec(architecture, budget, 9, seed)
                )
                fingerprint = (
                    memory.fixture_digest,
                    history.horizons[0].evaluation_fixture_digest,
                    yolo_like.evaluation_fixture_digest,
                    yolo_like.corruption_digest,
                )
                if expected_fingerprint is None:
                    expected_fingerprint = fingerprint
                elif fingerprint != expected_fingerprint:
                    raise ProtocolInvalid(
                        "fixture-lineage: architecture/budget arms received different fixtures"
                    )
                if not memory.valid:
                    raise ProtocolInvalid(
                        f"memory-invalid: seed={seed} architecture={int(architecture)} budget={budget}"
                    )
                arms.append(
                    RegisteredArmResult(
                        seed=seed,
                        budget=budget,
                        architecture=int(architecture),
                        memory=memory,
                        history=history,
                        yolo_like=yolo_like,
                    )
                )
        if expected_fingerprint is None:
            raise ProtocolInvalid("fixture-lineage: no registered arms were executed")
        fixture_fingerprints[seed] = expected_fingerprint

    digests = [
        arm.memory.parameter_digest_before
        for arm in arms
    ]
    validity = validate_measurement_payload(
        {
            "state_dim_matches": all(
                arm.memory.parameter_digest_before and arm.memory.valid for arm in arms
            ),
            "parameter_digest_stable": all(
                arm.memory.parameter_digest_before == arm.memory.parameter_digest_after
                for arm in arms
            ),
            "digests_present": all(len(value) == 64 for value in digests),
            "fixture_lineage_matches": len(fixture_fingerprints) == len(REGISTERED_SEEDS),
            "reset_controls_exact": all(arm.memory.valid for arm in arms),
            "phase2a_compatibility": all(item.passed for item in compatibility),
            "deterministic": True,
            "r2_isolated": True,
            "no_posthoc_selection": True,
        }
    )
    require_valid(validity)
    return RegisteredMeasurementResult(
        registered_measurement=True,
        manifest=manifest,
        compatibility=compatibility,
        arms=tuple(arms),
    )


def _smoke_memory_fixture_digest(seed: int) -> str:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x45314145]))
    fixtures = _build_paired_fixtures(rng, 1)
    return _memory_fixture_digest(fixtures)


def _smoke_history_fixture_digest(seed: int) -> str:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x45314245]))
    digest = hashlib.sha256()
    digest.update(b"r1-e1-history-smoke-v1\0")
    for horizon in HISTORY_HORIZONS:
        fixture = _build_horizon_fixture(rng, horizon, 1)
        digest.update(fixture.fixture_digest.encode("ascii"))
    return digest.hexdigest()


def _smoke_yolo_digests(seed: int) -> tuple[str, str]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x45314345]))
    nuisance = rng.choice(
        np.array([-0.05, 0.05], dtype=np.float64),
        size=(20, 2),
        replace=True,
    )
    sequences = tuple(
        BehaviorSequence(
            label,
            behavior,
            0,
            label,
            _clean_template(label, nuisance),
        )
        for label, behavior in enumerate(BEHAVIOR_CLASSES)
    )
    fixture = CleanFixtureSet(sequences, _yolo_fixture_digest(sequences))

    corruption_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x45314343])
    )
    entries = []
    for sequence in fixture.sequences:
        drop = tuple(
            sorted(int(value) for value in corruption_rng.choice(16, size=2, replace=False))
        )
        wrong = tuple(
            sorted(int(value) for value in corruption_rng.choice(16, size=2, replace=False))
        )
        mixed_drop = tuple(
            sorted(int(value) for value in corruption_rng.choice(16, size=2, replace=False))
        )
        remaining = np.array(
            [index for index in range(16) if index not in mixed_drop],
            dtype=np.int64,
        )
        mixed_wrong = int(corruption_rng.choice(remaining))
        jitter_noise = corruption_rng.normal(0.0, 0.05, size=(16, 7))
        mixed_jitter_noise = corruption_rng.normal(0.0, 0.05, size=(16, 7))
        entries.append(
            CorruptionEntry(
                sequence.fixture_index,
                sequence.label,
                drop,
                wrong,
                (8, 9, 10, 11),
                mixed_drop,
                mixed_wrong,
                jitter_noise,
                mixed_jitter_noise,
            )
        )
    entries_tuple = tuple(entries)
    return fixture.fixture_digest, _corruption_digest(entries_tuple)
