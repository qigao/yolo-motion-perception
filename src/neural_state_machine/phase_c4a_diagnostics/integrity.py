"""D0 integrity gate and exact registered replay for C4-A diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..action_value_benchmark import _evaluate, _new_policy
from ..memory_benchmark import AccuracyCount
from ..phase_c4_batch_probe import build_batch_design, fit_anonymous_batch_probe
from ..phase_c4_benchmark import PhaseC4Config
from ..phase_c4_delay_model import DecisionCreditRow, DelayLaw
from ..phase_c4_measurement import (
    _aggregate_stream,
    _decision_rows_and_rewards,
    _evaluate_readout,
    _permute_reward_blocks,
    _readout_from_fit,
)
from .model import FrozenC4AIdentity, freeze_float64, sha256_file


_REGISTERED_SHUFFLE_LINEAGE = 0x33534846
_EVIDENCE_RELATIVE = Path("docs/experiments/phase-c4-delay-marginalized-credit")
_EXPECTED_FROZEN_HASHES = {
    "c4a-manifest.json": "a990745cf6f0e8c5fd7ae5d9a7189737b30a40168a4db3df1a6d3043d6d5876a",
    "c4a-result.json": "7f398f474dba928d2a6aa993d0801563aa05af20faa8cd9a61ec94a6ca4697a4",
    "c4a-provenance.json": "8fb47b5e670754bc44c9ea70dd4cedce40e6f52165923142114577b4aec00351",
}
_EXPECTED_SCORES = {
    (7, "normal"): 200,
    (7, "registered_shuffled"): 147,
    (17, "normal"): 200,
    (17, "registered_shuffled"): 170,
    (29, "normal"): 200,
    (29, "registered_shuffled"): 160,
}


@dataclass(frozen=True, slots=True)
class RegisteredConditionReplay:
    seed: int
    condition: Literal["normal", "registered_shuffled"]
    design: np.ndarray
    target: np.ndarray
    rows: tuple[DecisionCreditRow, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    shuffled_rewards: tuple[float, ...] | None
    schedule_digest: str
    design_digest: str
    target_digest: str
    score: AccuracyCount

    def __post_init__(self) -> None:
        design = freeze_float64(self.design)
        target = freeze_float64(self.target)
        if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.size:
            raise ValueError("registered replay design/target shape mismatch")
        if self.condition not in {"normal", "registered_shuffled"}:
            raise ValueError("unknown registered C4-A condition")
        if len(self.actions) != len(self.rows) or len(self.rewards) != len(self.rows):
            raise ValueError("registered replay decision sequences are misaligned")
        if self.condition == "normal" and self.shuffled_rewards is not None:
            raise ValueError("normal replay must not expose shuffled_rewards")
        if self.condition == "registered_shuffled" and self.shuffled_rewards is None:
            raise ValueError("registered shuffled replay requires shuffled_rewards")
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "target", target)


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def verify_frozen_c4a_evidence(root: Path) -> FrozenC4AIdentity:
    """Require the three registered C4-A evidence files to match frozen bytes."""
    repository_root = Path(root)
    evidence_root = repository_root / _EVIDENCE_RELATIVE
    for name, expected in _EXPECTED_FROZEN_HASHES.items():
        path = evidence_root / name
        try:
            actual = sha256_file(path)
        except ValueError as exc:
            raise RuntimeError(f"frozen C4-A evidence hash mismatch: {name}") from exc
        if actual != expected:
            raise RuntimeError(f"frozen C4-A evidence hash mismatch: {name}")
    return FrozenC4AIdentity.registered()


def replay_registered_condition(
    seed: int,
    condition: Literal["normal", "registered_shuffled"],
) -> RegisteredConditionReplay:
    """Reproduce one frozen C4-A normal or registered-shuffled fit."""
    identity = FrozenC4AIdentity.registered()
    if seed not in identity.seeds:
        raise ValueError("seed must be one of the three registered C4-A seeds")
    if condition not in {"normal", "registered_shuffled"}:
        raise ValueError("condition must be normal or registered_shuffled")

    config = PhaseC4Config()
    fixtures, rows, actions, rewards = _decision_rows_and_rewards(seed, config)

    rewards_for_aggregation = rewards
    shuffled_rewards: tuple[float, ...] | None = None
    if condition == "registered_shuffled":
        shuffle_rng = np.random.default_rng(
            np.random.SeedSequence([seed, _REGISTERED_SHUFFLE_LINEAGE])
        )
        shuffled_rewards = _permute_reward_blocks(rewards, shuffle_rng, block_size=10)
        rewards_for_aggregation = shuffled_rewards

    scalars, schedule_digest, drain_count = _aggregate_stream(
        seed,
        config,
        actions,
        rewards_for_aggregation,
    )
    if drain_count != identity.drain_clocks:
        raise RuntimeError("registered C4-A replay did not use exactly five drain clocks")
    if len(scalars) != identity.design_rows:
        raise RuntimeError("registered C4-A replay scalar count differs from N+5")

    design, target = build_batch_design(rows, scalars, 2, DelayLaw.registered())
    if design.shape[0] != identity.design_rows or target.shape != (identity.design_rows,):
        raise RuntimeError("registered C4-A replay design shape differs from frozen contract")
    fit = fit_anonymous_batch_probe(
        design,
        target,
        2,
        config.hidden_size + 1,
        penalty=identity.ridge_penalty,
    )
    readout = _readout_from_fit(fit, config)

    if condition == "normal":
        post, _ = _evaluate_readout(seed, config, readout)
        score = post.overall
    else:
        evaluation = _evaluate(
            _new_policy(seed, config.action_value_config),
            readout,
            fixtures.evaluation,
            reset_before_decision=False,
        )
        score = evaluation.overall

    expected_score = _EXPECTED_SCORES[(seed, condition)]
    if score.total != 200 or score.correct != expected_score:
        raise RuntimeError(
            "registered C4-A diagnostic replay score differs from frozen evidence"
        )

    return RegisteredConditionReplay(
        seed=seed,
        condition=condition,
        design=design,
        target=target,
        rows=rows,
        actions=actions,
        rewards=rewards,
        shuffled_rewards=shuffled_rewards,
        schedule_digest=schedule_digest,
        design_digest=_array_sha256(design),
        target_digest=_array_sha256(target),
        score=score,
    )


def run_d0_integrity(root: Path) -> tuple[RegisteredConditionReplay, ...]:
    """Verify frozen evidence and reproduce all six registered C4-A conditions."""
    identity = verify_frozen_c4a_evidence(root)
    replays: list[RegisteredConditionReplay] = []
    for seed in identity.seeds:
        normal = replay_registered_condition(seed, "normal")
        shuffled = replay_registered_condition(seed, "registered_shuffled")
        if normal.schedule_digest != shuffled.schedule_digest:
            raise RuntimeError("registered shuffle changed hidden-delay schedule lineage")
        if normal.design_digest != shuffled.design_digest:
            raise RuntimeError("registered shuffle changed the anonymous design matrix")
        if normal.design.tobytes(order="C") != shuffled.design.tobytes(order="C"):
            raise RuntimeError("registered shuffle changed anonymous design bytes")
        replays.extend((normal, shuffled))
    return tuple(replays)
