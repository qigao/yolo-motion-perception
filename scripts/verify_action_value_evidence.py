"""Verify portable Phase 3A evidence and same-environment float integrity."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import NoReturn

from neural_state_machine import run_action_value_benchmark

_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE = _ROOT / "docs" / "experiments" / "phase-3a-action-value.json"
_FROZEN_EVIDENCE = {
    "phase_2b": _ROOT / "docs" / "experiments" / "phase-2b-failure.json",
    "phase_2c": _ROOT / "docs" / "experiments" / "phase-2c-diagnostics.json",
}
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = {
    "all_passed",
    "config",
    "evidence_schema_version",
    "frozen_evidence_sha256",
    "phase",
    "results",
    "rng_lineages",
    "seeds",
    "shuffled_pooled",
}
_RESULT_KEYS = {
    "action_sequences_equal",
    "all_reset_hidden_equal",
    "decision_hidden_digest",
    "evaluation_fixture_digest",
    "initial_parameter_digest",
    "matrix_controls",
    "normal_action_counts",
    "normal_action_digest",
    "normal_actions",
    "normal_checkpoints",
    "normal_parameter_digest",
    "normal_pending_feedback",
    "normal_reward_digest",
    "passed",
    "per_delay",
    "post_margin",
    "post_training",
    "pre_training",
    "repeatable",
    "reset_hidden_digest",
    "reset_per_delay",
    "reward_block_multisets_equal",
    "seed",
    "shuffled_action_counts",
    "shuffled_action_digest",
    "shuffled_actions",
    "shuffled_checkpoints",
    "shuffled_control",
    "shuffled_parameter_digest",
    "shuffled_pending_feedback",
    "shuffled_per_delay",
    "shuffled_reward_digest",
    "state_reset",
    "training_fixture_digest",
}
_LOCAL_RESULT_KEYS = {
    "decision_hidden_digest",
    "initial_parameter_digest",
    "matrix_controls",
    "normal_parameter_digest",
    "post_margin",
    "reset_hidden_digest",
    "shuffled_parameter_digest",
}
_CONFIG = {
    "checkpoint_interval": 100,
    "evaluation_blocks": 20,
    "hidden_size": 64,
    "recurrent_radius": 0.9,
    "step_size": 0.1,
    "training_episodes": 2_000,
}
_RNG_LINEAGES = {
    "behavior_action": ["seed", 0x33414354],
    "evaluation_fixture": ["seed", 0x4556414C],
    "reward_shuffle": ["seed", 0x33534846],
    "training_fixture": ["seed", 0x54524149],
}


def _fail(message: str) -> NoReturn:
    raise RuntimeError(f"invalid Phase 3A evidence: {message}")


def _dictionary(value: object, name: str, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        _fail(f"{name} has unexpected fields")
    if not all(type(key) is str for key in value):
        _fail(f"{name} has a non-string field")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        _fail(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _fixed_float(value: object, name: str, expected: float) -> float:
    if type(value) is not float or not math.isfinite(value) or value != expected:
        _fail(f"{name} must be the fixed finite float {expected}")
    return value


def _diagnostic_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(f"{name} must be a finite float")
    return value


def _digest(value: object, name: str, *, length: int = 64) -> str:
    pattern = _HEX_64 if length == 64 else _HEX_40
    if type(value) is not str or pattern.fullmatch(value) is None:
        _fail(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _zero_parameter_digest(hidden_size: int) -> str:
    shape = (2, hidden_size + 1)
    digest = hashlib.sha256()
    digest.update(str(shape).encode("ascii"))
    digest.update(bytes(shape[0] * shape[1] * 8))
    return digest.hexdigest()


def _validate_fixed_config(value: object) -> dict[str, object]:
    config = _dictionary(value, "config", set(_CONFIG))
    for key in (
        "checkpoint_interval",
        "evaluation_blocks",
        "hidden_size",
        "training_episodes",
    ):
        actual = _integer(config[key], f"config.{key}", minimum=1)
        if actual != _CONFIG[key]:
            _fail(f"config.{key} differs from the frozen protocol")
    for key in ("recurrent_radius", "step_size"):
        _fixed_float(config[key], f"config.{key}", _CONFIG[key])
    return config


def _validate_fixed_seeds(value: object) -> list[int]:
    if type(value) is not list or len(value) != 3:
        _fail("seeds must be a three-item list")
    seeds = [_integer(seed, "seed") for seed in value]
    if seeds != [7, 17, 29]:
        _fail("seeds must be ordered as 7, 17, 29")
    return seeds


def _validate_rng_lineages(value: object) -> dict[str, object]:
    lineages = _dictionary(value, "rng_lineages", set(_RNG_LINEAGES))
    for name, expected in _RNG_LINEAGES.items():
        lineage = lineages[name]
        if type(lineage) is not list or len(lineage) != 2:
            _fail(f"rng_lineages.{name} must be a two-item list")
        if type(lineage[0]) is not str or lineage[0] != "seed":
            _fail(f"rng_lineages.{name} must start with the string seed")
        numeric = _integer(lineage[1], f"rng_lineages.{name}[1]")
        if numeric != expected[1]:
            _fail(f"rng_lineages.{name} differs from the frozen protocol")
    return lineages


def _count(value: object, name: str, *, total: int) -> dict[str, object]:
    row = _dictionary(value, name, {"accuracy", "correct", "total"})
    actual_total = _integer(row["total"], f"{name}.total")
    correct = _integer(row["correct"], f"{name}.correct")
    accuracy = _finite(row["accuracy"], f"{name}.accuracy")
    if actual_total != total or correct > total or accuracy != correct / total:
        _fail(f"{name} has inconsistent count or accuracy")
    return row


def _per_delay(
    value: object,
    name: str,
    *,
    overall: dict[str, object],
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 5:
        _fail(f"{name} must contain five rows")
    rows: list[dict[str, object]] = []
    for expected_delay, raw in enumerate(value, start=1):
        row = _dictionary(raw, f"{name}[{expected_delay}]", {"accuracy", "correct", "delay", "total"})
        if _integer(row["delay"], f"{name}.delay", minimum=1) != expected_delay:
            _fail(f"{name} has unexpected delay order")
        _count(
            {key: row[key] for key in ("accuracy", "correct", "total")},
            f"{name}[{expected_delay}]",
            total=40,
        )
        rows.append(row)
    if sum(int(row["correct"]) for row in rows) != overall["correct"]:
        _fail(f"{name} does not sum to its overall count")
    return rows


def _action_counts(value: object, name: str, actions: list[int]) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 2:
        _fail(f"{name} must contain actions zero and one")
    rows: list[dict[str, object]] = []
    for expected_action, raw in enumerate(value):
        row = _dictionary(raw, f"{name}[{expected_action}]", {"action", "count"})
        if _integer(row["action"], f"{name}.action") != expected_action:
            _fail(f"{name} has unexpected action order")
        count = _integer(row["count"], f"{name}.count")
        if count != actions.count(expected_action):
            _fail(f"{name} does not match its action schedule")
        rows.append(row)
    if sum(int(row["count"]) for row in rows) != 2_000:
        _fail(f"{name} has an unexpected total")
    return rows


def _actions(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 2_000:
        _fail(f"{name} must contain 2000 actions")
    if any(type(action) is not int or action not in (0, 1) for action in value):
        _fail(f"{name} contains an invalid action")
    return value


def _checkpoints(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 20:
        _fail(f"{name} must contain twenty checkpoints")
    keys = {
        "accuracy",
        "episode",
        "margin_mean",
        "margin_minimum",
        "margin_p10",
        "td_error_abs_mean",
        "td_error_maximum",
        "td_error_mean",
        "td_error_p90",
    }
    rows: list[dict[str, object]] = []
    for expected_episode, raw in zip(range(100, 2_001, 100), value, strict=True):
        row = _dictionary(raw, f"{name}[{expected_episode}]", keys)
        if _integer(row["episode"], f"{name}.episode", minimum=1) != expected_episode:
            _fail(f"{name} has an unexpected episode sequence")
        _count(row["accuracy"], f"{name}.accuracy", total=100)
        for key in keys - {"accuracy", "episode"}:
            _diagnostic_float(row[key], f"{name}.{key}")
        rows.append({"accuracy": row["accuracy"], "episode": row["episode"]})
    return rows


def _expected_pass(result: dict[str, object]) -> bool:
    post = result["post_training"]
    reset = result["state_reset"]
    shuffled = result["shuffled_control"]
    assert isinstance(post, dict) and isinstance(reset, dict) and isinstance(shuffled, dict)
    return (
        result["repeatable"] is True
        and post["correct"] >= 180
        and all(row["correct"] >= 34 for row in result["per_delay"])
        and reset["correct"] == 100
        and all(row["correct"] == 20 for row in result["reset_per_delay"])
        and result["all_reset_hidden_equal"] is True
        and shuffled["correct"] < 150
        and result["action_sequences_equal"] is True
        and result["reward_block_multisets_equal"] is True
        and result["normal_pending_feedback"] is False
        and result["shuffled_pending_feedback"] is False
    )


def _portable_result(value: object, expected_seed: int) -> dict[str, object]:
    result = _dictionary(value, f"result {expected_seed}", _RESULT_KEYS)
    _validate_local_controls(result)
    if _integer(result["seed"], "result.seed") != expected_seed:
        _fail("result seeds are duplicated or out of order")
    counts = {
        key: _count(result[key], key, total=200)
        for key in ("pre_training", "post_training", "state_reset", "shuffled_control")
    }
    _per_delay(result["per_delay"], "per_delay", overall=counts["post_training"])
    _per_delay(result["reset_per_delay"], "reset_per_delay", overall=counts["state_reset"])
    _per_delay(
        result["shuffled_per_delay"],
        "shuffled_per_delay",
        overall=counts["shuffled_control"],
    )
    normal_actions = _actions(result["normal_actions"], "normal_actions")
    shuffled_actions = _actions(result["shuffled_actions"], "shuffled_actions")
    _action_counts(result["normal_action_counts"], "normal_action_counts", normal_actions)
    _action_counts(result["shuffled_action_counts"], "shuffled_action_counts", shuffled_actions)
    for key in (
        "normal_action_digest",
        "shuffled_action_digest",
        "normal_reward_digest",
        "shuffled_reward_digest",
        "training_fixture_digest",
        "evaluation_fixture_digest",
    ):
        _digest(result[key], key)
    if result["normal_action_digest"] != hashlib.sha256(bytes(normal_actions)).hexdigest():
        _fail("normal action digest does not match its schedule")
    if result["shuffled_action_digest"] != hashlib.sha256(bytes(shuffled_actions)).hexdigest():
        _fail("shuffled action digest does not match its schedule")
    for key in (
        "action_sequences_equal",
        "all_reset_hidden_equal",
        "normal_pending_feedback",
        "passed",
        "repeatable",
        "reward_block_multisets_equal",
        "shuffled_pending_feedback",
    ):
        _boolean(result[key], key)
    if result["action_sequences_equal"] is not (normal_actions == shuffled_actions):
        _fail("action sequence equality flag is inconsistent")
    if result["reward_block_multisets_equal"] is not True:
        _fail("reward-block fairness was not preserved")
    if result["normal_pending_feedback"] or result["shuffled_pending_feedback"]:
        _fail("training retained pending feedback")
    if result["repeatable"] is not True:
        _fail("per-seed execution was not repeatable")
    normal_checkpoints = _checkpoints(result["normal_checkpoints"], "normal_checkpoints")
    shuffled_checkpoints = _checkpoints(
        result["shuffled_checkpoints"], "shuffled_checkpoints"
    )
    post_margin = _dictionary(result["post_margin"], "post_margin", {"mean", "minimum", "p10"})
    for key in post_margin:
        _diagnostic_float(post_margin[key], f"post_margin.{key}")
    if result["passed"] is not _expected_pass(result):
        _fail("per-seed pass flag is inconsistent with fixed gates")
    portable = {key: result[key] for key in sorted(_RESULT_KEYS - _LOCAL_RESULT_KEYS)}
    portable["normal_checkpoints"] = normal_checkpoints
    portable["shuffled_checkpoints"] = shuffled_checkpoints
    return portable


def _portable_phase_3a_payload(payload: dict[str, object]) -> dict[str, object]:
    """Validate and return all cross-environment-stable Phase 3A fields."""
    if not isinstance(payload, dict):
        _fail("top level must be an object")
    allowed = _TOP_LEVEL_KEYS | ({"source_commit"} if "source_commit" in payload else set())
    top = _dictionary(payload, "top level", allowed)
    if "source_commit" in top:
        _digest(top["source_commit"], "source_commit", length=40)
    if top["phase"] != "3A":
        _fail("phase must be 3A")
    if type(top["evidence_schema_version"]) is not int or top["evidence_schema_version"] != 1:
        _fail("schema version must be integer one")
    _validate_fixed_config(top["config"])
    _validate_rng_lineages(top["rng_lineages"])
    _validate_fixed_seeds(top["seeds"])
    _boolean(top["all_passed"], "all_passed")

    frozen = _dictionary(
        top["frozen_evidence_sha256"],
        "frozen_evidence_sha256",
        {"phase_2b", "phase_2c"},
    )
    expected_frozen = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in _FROZEN_EVIDENCE.items()
    }
    if frozen != expected_frozen:
        _fail("frozen Phase 2 evidence hashes changed")

    raw_results = top["results"]
    if not isinstance(raw_results, list) or len(raw_results) != 3:
        _fail("results must contain exactly three seeds")
    portable_results = [
        _portable_result(raw, seed)
        for raw, seed in zip(raw_results, (7, 17, 29), strict=True)
    ]
    pooled = _count(top["shuffled_pooled"], "shuffled_pooled", total=600)
    shuffled_correct = sum(int(result["shuffled_control"]["correct"]) for result in portable_results)
    if pooled["correct"] != shuffled_correct:
        _fail("pooled shuffled count does not match per-seed counts")
    expected_all = all(result["passed"] for result in portable_results) and (
        0.40 <= float(pooled["accuracy"]) <= 0.60
    )
    if top["all_passed"] is not expected_all:
        _fail("all_passed is inconsistent with fixed gates")

    return {
        "all_passed": top["all_passed"],
        "config": top["config"],
        "evidence_schema_version": top["evidence_schema_version"],
        "frozen_evidence_sha256": top["frozen_evidence_sha256"],
        "phase": top["phase"],
        "results": portable_results,
        "rng_lineages": top["rng_lineages"],
        "seeds": top["seeds"],
        "shuffled_pooled": top["shuffled_pooled"],
    }


def _local_result(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail("local result must be an object")
    return value


def _validate_local_controls(result: dict[str, object]) -> None:
    for key in (
        "decision_hidden_digest",
        "initial_parameter_digest",
        "normal_parameter_digest",
        "reset_hidden_digest",
        "shuffled_parameter_digest",
    ):
        _digest(result.get(key), key)
    matrices = _dictionary(
        result.get("matrix_controls"),
        "matrix_controls",
        {"normal_after", "normal_before", "shuffled_after", "shuffled_before"},
    )
    for name, digests in matrices.items():
        if not isinstance(digests, list) or len(digests) != 3:
            _fail(f"matrix_controls.{name} must contain three digests")
        for digest in digests:
            _digest(digest, f"matrix_controls.{name}")
    if matrices["normal_before"] != matrices["normal_after"]:
        raise RuntimeError("Phase 3A normal reservoir matrices changed")
    if matrices["shuffled_before"] != matrices["shuffled_after"]:
        raise RuntimeError("Phase 3A shuffled reservoir matrices changed")
    if result["initial_parameter_digest"] != _zero_parameter_digest(64):
        raise RuntimeError("Phase 3A learner did not start from exact-zero parameters")
    if result["normal_parameter_digest"] == result["initial_parameter_digest"]:
        raise RuntimeError("Phase 3A normal learner did not update")
    if result["shuffled_parameter_digest"] == result["initial_parameter_digest"]:
        raise RuntimeError("Phase 3A shuffled learner did not update")


def _verify_local_float_integrity(payload: dict[str, object]) -> None:
    """Check environment-local parameter and matrix invariants."""
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 3:
        _fail("local results must contain exactly three seeds")
    for raw in results:
        result = _local_result(raw)
        _validate_local_controls(result)
        if result.get("action_sequences_equal") is not True:
            raise RuntimeError("Phase 3A action schedules differ")
        if result.get("reward_block_multisets_equal") is not True:
            raise RuntimeError("Phase 3A reward-block multisets differ")
        if result.get("normal_pending_feedback") is not False:
            raise RuntimeError("Phase 3A normal learner retained pending feedback")
        if result.get("shuffled_pending_feedback") is not False:
            raise RuntimeError("Phase 3A shuffled learner retained pending feedback")


def _load_committed_evidence(evidence_path: Path = _EVIDENCE) -> dict[str, object]:
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not load Phase 3A evidence") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("could not load Phase 3A evidence as an object")
    return payload


def _verify_source_commit_ancestor(source_commit: object) -> None:
    source = _digest(source_commit, "source_commit", length=40)
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source, "HEAD"],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError("could not verify Phase 3A source commit ancestry") from exc
    if completed.returncode != 0:
        raise RuntimeError("Phase 3A source commit is not an ancestor of HEAD")


def verify_action_value_evidence() -> dict[str, object]:
    """Reproduce Phase 3A twice and verify its committed evidence."""
    expected = _load_committed_evidence()
    if "source_commit" not in expected:
        _fail("source_commit is missing")
    _verify_source_commit_ancestor(expected["source_commit"])
    first = run_action_value_benchmark()
    second = run_action_value_benchmark()
    if first != second:
        raise RuntimeError("Phase 3A is not byte-stable in this environment")
    _verify_local_float_integrity(first)
    portable = _portable_phase_3a_payload(first)
    if portable != _portable_phase_3a_payload(expected):
        raise RuntimeError("Phase 3A portable runtime differs from evidence")
    return portable


def main() -> int:
    payload = verify_action_value_evidence()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
