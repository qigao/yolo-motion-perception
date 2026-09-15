#!/usr/bin/env python3
"""Run the corrected Phase 3B delayed-credit benchmark and optionally write evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from neural_state_machine.action_value_benchmark import run_action_value_experiment
from neural_state_machine.phase3b_delayed_benchmark import (
    DelayedCreditConfig,
    DelayedCreditResult,
    delayed_credit_payload,
    run_delayed_credit_benchmark,
)

_APPROVED = Path("docs/experiments/phase-3b-delayed-credit.json")
_DEFAULT_SEEDS = (7, 17, 29)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _frozen_phase3a_sha256(root: Path) -> str:
    path = root / "docs/experiments/phase-3a-action-value.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _continuity_matches(result: DelayedCreditResult, config: DelayedCreditConfig) -> bool:
    if result.reward_delay != 0:
        return True
    legacy = run_action_value_experiment(result.seed, config.action_value_config)
    return (
        result.actions == legacy.normal_actions
        and result.action_digest == legacy.normal_action_digest
        and result.training_reward_digest == legacy.normal_reward_digest
        and result.parameter_digest == legacy.normal_parameter_digest
        and result.post_training == legacy.post_training
        and result.per_delay == legacy.per_delay
        and result.state_reset == legacy.state_reset
        and result.reset_per_delay == legacy.reset_per_delay
        and result.training_fixture_digest == legacy.training_fixture_digest
        and result.evaluation_fixture_digest == legacy.evaluation_fixture_digest
    )


def _result_protocol_valid(result: DelayedCreditResult) -> bool:
    timelines_equal = (
        result.normal_timeline.delivery_timeline_digest
        == result.shuffled_timeline.delivery_timeline_digest
        and result.normal_timeline.lag_histogram == result.shuffled_timeline.lag_histogram
    )
    terminal_clean = (
        result.normal_timeline.queue_pending_final == 0
        and result.normal_timeline.learner_unresolved_final == 0
        and result.shuffled_timeline.queue_pending_final == 0
        and result.shuffled_timeline.learner_unresolved_final == 0
    )
    finite_counts = all(
        math.isfinite(count.accuracy)
        for count in (
            result.pre_training,
            result.post_training,
            result.state_reset,
            result.shuffled_control,
        )
    )
    return (
        result.repeatable
        and result.action_sequences_equal
        and result.reward_block_multisets_equal
        and timelines_equal
        and terminal_clean
        and finite_counts
        and result.queue_deliveries == result.normal_timeline.action_count
        and result.pending_feedback is False
    )


def _lineages_match_across_delays(results: tuple[DelayedCreditResult, ...]) -> bool:
    by_seed: dict[int, list[DelayedCreditResult]] = {}
    for result in results:
        by_seed.setdefault(result.seed, []).append(result)
    for rows in by_seed.values():
        training = {row.training_fixture_digest for row in rows}
        evaluation = {row.evaluation_fixture_digest for row in rows}
        actions = {row.normal_action_digest for row in rows}
        if len(training) != 1 or len(evaluation) != 1 or len(actions) != 1:
            return False
    return True


def _registered_behavior_passed(results: tuple[object, ...]) -> bool:
    delayed = tuple(result for result in results if result.reward_delay != 0)
    if not delayed:
        raise ValueError("registered behavior gate requires at least one non-zero reward delay")
    return all(result.behavior_passed for result in delayed)


def build_payload(
    root: Path | None = None,
    *,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    config: DelayedCreditConfig | None = None,
) -> dict[str, object]:
    resolved_root = _repository_root() if root is None else root
    resolved = DelayedCreditConfig() if config is None else config
    results = run_delayed_credit_benchmark(seeds, resolved, arm="td0")

    protocol_valid = (
        all(_result_protocol_valid(result) for result in results)
        and all(_continuity_matches(result, resolved) for result in results)
        and _lineages_match_across_delays(results)
    )
    behavior_passed = protocol_valid and _registered_behavior_passed(results)

    payload = delayed_credit_payload(results)
    payload.pop("diagnostic_only", None)
    payload["schema_version"] = 2
    payload["seeds"] = list(seeds)
    payload["reward_delays"] = list(resolved.reward_delays)
    payload["cue_to_decision_delays"] = [1, 2, 3, 4, 5]
    payload["config"] = {
        "hidden_size": resolved.hidden_size,
        "recurrent_radius": resolved.recurrent_radius,
        "step_size": resolved.step_size,
        "training_decisions": resolved.training_episodes,
        "evaluation_episodes": resolved.evaluation_blocks * 10,
        "checkpoint_interval": resolved.checkpoint_interval,
    }
    payload["frozen_phase3a_sha256"] = _frozen_phase3a_sha256(resolved_root)
    payload["protocol_valid"] = protocol_valid
    payload["behavior_passed"] = behavior_passed
    payload["all_passed"] = protocol_valid and behavior_passed
    payload["decision_status"] = (
        "protocol-valid-behavior-passed"
        if behavior_passed
        else "protocol-valid-behavior-failed"
        if protocol_valid
        else "harness-invalid"
    )
    return payload


def write_evidence(
    payload: dict[str, object], path: Path, root: Path | None = None
) -> None:
    if payload.get("protocol_valid") is not True:
        raise ValueError("protocol_valid must be true before evidence can be written")
    resolved_root = _repository_root() if root is None else root
    approved = (resolved_root / _APPROVED).resolve()
    target = path.resolve()
    if target != approved:
        raise ValueError(
            "evidence path is not approved: expected docs/experiments/phase-3b-delayed-credit.json"
        )
    if target.is_symlink():
        raise ValueError("evidence path must not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") == rendered:
        return
    fd, temporary_name = tempfile.mkstemp(prefix=".phase3b-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--require-protocol-valid", action="store_true")
    args = parser.parse_args()
    root = _repository_root()
    payload = build_payload(root)
    print(json.dumps(payload, sort_keys=True))
    if args.require_protocol_valid and payload["protocol_valid"] is not True:
        return 1
    if args.evidence is not None:
        write_evidence(payload, args.evidence, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
