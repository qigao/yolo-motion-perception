#!/usr/bin/env python3
"""Fail-closed verification for corrected Phase 3B evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from scripts.benchmark_phase3b_delayed_credit import (
        _APPROVED,
        build_payload,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from benchmark_phase3b_delayed_credit import _APPROVED, build_payload

from neural_state_machine.phase3b_delayed_benchmark import DelayedCreditConfig


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load(path: Path, root: Path) -> dict[str, object]:
    target = path.resolve()
    if target != (root / _APPROVED).resolve():
        raise RuntimeError("artifact path is not approved")
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("artifact path must be a regular file")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("artifact is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("artifact must be a JSON object")
    return value


def _validate_count(result: dict[str, object], key: str) -> None:
    count = result.get(key)
    if not isinstance(count, dict):
        raise RuntimeError(f"invalid count: {key}")
    if any(type(count.get(field)) is not int for field in ("correct", "total")):
        raise RuntimeError(f"invalid count fields: {key}")
    accuracy = count.get("accuracy")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
    ):
        raise RuntimeError(f"invalid accuracy: {key}")


def _validate_timeline(result: dict[str, object], key: str) -> None:
    timeline = result.get(key)
    if not isinstance(timeline, dict):
        raise RuntimeError(f"invalid timeline: {key}")
    for field in (
        "action_count",
        "delivery_count",
        "terminal_drain_count",
        "max_pending_before_delivery",
        "max_pending_after_delivery",
        "decisions_with_prior_feedback_pending",
        "queue_pending_final",
        "learner_unresolved_final",
    ):
        if type(timeline.get(field)) is not int or timeline[field] < 0:
            raise RuntimeError(f"invalid timeline count: {key}.{field}")
    if not _is_hex64(timeline.get("delivery_timeline_digest")):
        raise RuntimeError(f"invalid timeline digest: {key}")
    histogram = timeline.get("lag_histogram")
    if not isinstance(histogram, list) or not histogram:
        raise RuntimeError(f"invalid lag histogram: {key}")
    for row in histogram:
        if (
            not isinstance(row, dict)
            or type(row.get("lag")) is not int
            or type(row.get("count")) is not int
            or row["lag"] < 0
            or row["count"] <= 0
        ):
            raise RuntimeError(f"invalid lag histogram row: {key}")


def _validate(payload: dict[str, object]) -> None:
    if payload.get("experiment") != "phase-3b-delayed-credit":
        raise RuntimeError("wrong experiment")
    if payload.get("schema_version") != 2:
        raise RuntimeError("wrong schema version")
    if payload.get("protocol_valid") is not True:
        raise RuntimeError("protocol_valid must be true")
    if type(payload.get("behavior_passed")) is not bool:
        raise RuntimeError("invalid behavior_passed flag")
    if type(payload.get("all_passed")) is not bool:
        raise RuntimeError("invalid all_passed flag")
    if payload["all_passed"] != (
        payload["protocol_valid"] and payload["behavior_passed"]
    ):
        raise RuntimeError("inconsistent all_passed")
    if not _is_hex64(payload.get("frozen_phase3a_sha256")):
        raise RuntimeError("invalid frozen Phase 3A digest")

    seeds = payload.get("seeds")
    reward_delays = payload.get("reward_delays")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise RuntimeError("invalid seeds")
    if (
        not isinstance(reward_delays, list)
        or not reward_delays
        or reward_delays[0] != 0
        or any(type(delay) is not int or delay < 0 for delay in reward_delays)
        or len(set(reward_delays)) != len(reward_delays)
    ):
        raise RuntimeError("invalid reward delays")

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("results must be a non-empty list")
    seen: set[tuple[int, int]] = set()
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("each result must be an object")
        if result.get("arm") != "td0":
            raise RuntimeError("only td0 is allowed in corrected Phase 3B evidence")
        seed = result.get("seed")
        reward_delay = result.get("reward_delay")
        if type(seed) is not int or type(reward_delay) is not int:
            raise RuntimeError("invalid result key")
        key = (seed, reward_delay)
        if key in seen:
            raise RuntimeError("duplicate result key")
        seen.add(key)
        for count_key in (
            "pre_training",
            "post_training",
            "state_reset",
            "shuffled_control",
        ):
            _validate_count(result, count_key)
        for digest_key in (
            "normal_action_digest",
            "shuffled_action_digest",
            "normal_reward_assignment_digest",
            "shuffled_reward_assignment_digest",
            "training_fixture_digest",
            "evaluation_fixture_digest",
            "parameter_digest",
            "shuffled_parameter_digest",
        ):
            if not _is_hex64(result.get(digest_key)):
                raise RuntimeError(f"invalid digest: {digest_key}")
        _validate_timeline(result, "normal_timeline")
        _validate_timeline(result, "shuffled_timeline")
        if type(result.get("action_sequences_equal")) is not bool:
            raise RuntimeError("invalid action lineage flag")
        if type(result.get("reward_block_multisets_equal")) is not bool:
            raise RuntimeError("invalid reward multiset flag")
        if type(result.get("behavior_passed")) is not bool:
            raise RuntimeError("invalid result behavior flag")
        if type(result.get("repeatable")) is not bool:
            raise RuntimeError("invalid repeatability flag")

    expected = {(seed, delay) for seed in seeds for delay in reward_delays}
    if seen != expected:
        raise RuntimeError("result keys do not match registered seed/delay grid")


def _config_from_payload(payload: dict[str, object]) -> DelayedCreditConfig:
    config = payload.get("config")
    reward_delays = payload.get("reward_delays")
    if not isinstance(config, dict) or not isinstance(reward_delays, list):
        raise RuntimeError("invalid configuration")
    evaluation_episodes = config.get("evaluation_episodes")
    if type(evaluation_episodes) is not int or evaluation_episodes % 10:
        raise RuntimeError("evaluation_episodes must be a multiple of ten")
    try:
        return DelayedCreditConfig(
            hidden_size=config["hidden_size"],
            recurrent_radius=config["recurrent_radius"],
            step_size=config["step_size"],
            training_episodes=config["training_decisions"],
            evaluation_blocks=evaluation_episodes // 10,
            checkpoint_interval=config["checkpoint_interval"],
            reward_delays=tuple(reward_delays),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid configuration") from exc


def verify_phase3b_delayed_credit(
    path: Path | None = None,
) -> dict[str, object]:
    root = _repository_root()
    artifact = (root / _APPROVED) if path is None else path
    committed = _load(artifact, root)
    _validate(committed)
    seeds = tuple(committed["seeds"])
    config = _config_from_payload(committed)
    runtime = build_payload(root, seeds=seeds, config=config)
    if runtime != committed:
        raise RuntimeError("committed Phase 3B evidence differs from deterministic replay")
    return committed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    payload = verify_phase3b_delayed_credit(args.evidence)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
