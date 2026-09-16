#!/usr/bin/env python3
"""Fail-closed verification for frozen Phase 3C anonymous-credit evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from scripts.benchmark_phase3c_anonymous_credit import (
        _APPROVED,
        _DEFAULT_SEEDS,
        _PHASE3B_BASE_SHA,
        build_measurement_payload,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from benchmark_phase3c_anonymous_credit import (
        _APPROVED,
        _DEFAULT_SEEDS,
        _PHASE3B_BASE_SHA,
        build_measurement_payload,
    )

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_formal_contract import (
    APPROVED_FORMAL_COMMIT,
    REQUIRED_THEOREMS,
)

_EXPECTED_CONFIG = {
    "hidden_size": 64,
    "recurrent_radius": 0.9,
    "step_size": 0.1,
    "training_decisions": 2_000,
    "evaluation_blocks": 20,
    "checkpoint_interval": 100,
    "discount": 0.9,
    "trace_decay": 0.8,
}
_EXPECTED_ARMS = ("td0", "eligibility")
_EXPECTED_DELAYS = (1, 3, 5)
_SEQUENCE_TYPES = (list, tuple)
_CHECKPOINT_FLOAT_FIELDS = (
    "margin_mean",
    "margin_p10",
    "margin_minimum",
    "td_error_mean",
    "td_error_abs_mean",
    "td_error_p90",
    "td_error_maximum",
)
# Only non-gating checkpoint diagnostics admit bounded floating-point noise.
# Independent decimal rounding can put adjacent floats in different buckets.
# Keep an absolute bound (no relative scaling); all other evidence stays exact.
_CHECKPOINT_FLOAT_ABS_TOL = 1e-12


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _load(path: Path, root: Path) -> dict[str, object]:
    target = path.resolve()
    approved = (root / _APPROVED).resolve()
    if target != approved:
        raise RuntimeError("artifact path is not approved")
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("artifact path must be a regular file")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("artifact is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("artifact must be a JSON object")
    return payload


def _require_bool(mapping: dict[str, object], key: str) -> bool:
    value = mapping.get(key)
    if type(value) is not bool:
        raise RuntimeError(f"invalid boolean: {key}")
    return value


def _validate_count(value: object, key: str) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid count: {key}")
    correct = value.get("correct")
    total = value.get("total")
    if type(correct) is not int or type(total) is not int:
        raise RuntimeError(f"invalid count fields: {key}")
    if correct < 0 or total <= 0 or correct > total:
        raise RuntimeError(f"invalid count range: {key}")


def _validate_per_delay(value: object, key: str) -> None:
    if not isinstance(value, _SEQUENCE_TYPES) or len(value) != 5:
        raise RuntimeError(f"invalid per-delay rows: {key}")
    seen: list[int] = []
    for row in value:
        if not isinstance(row, _SEQUENCE_TYPES) or len(row) != 2 or type(row[0]) is not int:
            raise RuntimeError(f"invalid per-delay row: {key}")
        seen.append(row[0])
        _validate_count(row[1], key)
    if tuple(seen) != (1, 2, 3, 4, 5):
        raise RuntimeError(f"invalid per-delay support: {key}")


def _validate_checkpoint_rows(value: object, key: str) -> None:
    if not isinstance(value, _SEQUENCE_TYPES):
        raise RuntimeError(f"invalid checkpoints: {key}")
    for row in value:
        if not isinstance(row, dict):
            raise RuntimeError(f"invalid checkpoint row: {key}")
        episode = row.get("episode")
        if type(episode) is not int or episode <= 0:
            raise RuntimeError(f"invalid checkpoint episode: {key}")
        _validate_count(row.get("accuracy"), f"{key}.accuracy")
        for field in _CHECKPOINT_FLOAT_FIELDS:
            if not _finite_number(row.get(field)):
                raise RuntimeError(f"invalid checkpoint field: {key}.{field}")


def _validate_audit(value: object) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("invalid protocol audit")
    for key in (
        "action_count",
        "latent_record_count",
        "delivered_record_count",
        "real_feedback_count",
        "drain_feedback_count",
        "queue_pending_final",
        "inversion_count",
        "collision_step_count",
        "trace_reset_count",
    ):
        item = value.get(key)
        if type(item) is not int or item < 0:
            raise RuntimeError(f"invalid audit count: {key}")
    for key in (
        "delay_digest",
        "due_step_digest",
        "multiplicity_digest",
        "aggregate_feedback_digest",
        "learner_call_digest",
    ):
        if not _is_hex64(value.get(key)):
            raise RuntimeError(f"invalid audit digest: {key}")
    for key in ("source_relabel_invariant", "hidden_multiplicity_invariant"):
        if value.get(key) is not True:
            raise RuntimeError(f"protocol invariant must be true: {key}")
    for key in ("latent_reward_sum", "aggregate_feedback_sum"):
        if not _finite_number(value.get(key)):
            raise RuntimeError(f"invalid audit sum: {key}")
    if value["latent_reward_sum"] != value["aggregate_feedback_sum"]:
        raise RuntimeError("aggregate reward conservation failed")

    delay_histogram = value.get("delay_histogram")
    if not isinstance(delay_histogram, _SEQUENCE_TYPES):
        raise RuntimeError("invalid delay histogram")
    if tuple(
        row[0]
        for row in delay_histogram
        if isinstance(row, _SEQUENCE_TYPES) and len(row) == 2
    ) != _EXPECTED_DELAYS:
        raise RuntimeError("invalid delay support")
    if any(
        not isinstance(row, _SEQUENCE_TYPES)
        or len(row) != 2
        or type(row[0]) is not int
        or type(row[1]) is not int
        or row[1] <= 0
        for row in delay_histogram
    ):
        raise RuntimeError("invalid delay histogram row")

    multiplicity = value.get("multiplicity_histogram")
    if not isinstance(multiplicity, _SEQUENCE_TYPES) or not multiplicity:
        raise RuntimeError("invalid multiplicity histogram")
    if any(
        not isinstance(row, _SEQUENCE_TYPES)
        or len(row) != 2
        or type(row[0]) is not int
        or type(row[1]) is not int
        or row[0] <= 0
        or row[1] <= 0
        for row in multiplicity
    ):
        raise RuntimeError("invalid multiplicity histogram row")

    coefficients = value.get("trace_coefficients")
    if not isinstance(coefficients, _SEQUENCE_TYPES) or len(coefficients) != 4:
        raise RuntimeError("invalid trace coefficients")
    ages = tuple(
        row[0]
        for row in coefficients
        if isinstance(row, _SEQUENCE_TYPES) and len(row) == 2
    )
    if ages != (0, 1, 3, 5):
        raise RuntimeError("invalid trace coefficient ages")
    if any(
        not isinstance(row, _SEQUENCE_TYPES)
        or len(row) != 2
        or type(row[0]) is not int
        or not _finite_number(row[1])
        for row in coefficients
    ):
        raise RuntimeError("invalid trace coefficient row")


def _validate_protocol(value: object, *, seed: int, arm: str) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("invalid protocol result")
    if value.get("seed") != seed or value.get("arm") != arm:
        raise RuntimeError("protocol/result identity mismatch")
    for key in (
        "training_fixture_digest",
        "evaluation_fixture_digest",
        "action_digest",
        "latent_reward_digest",
        "parameter_digest",
    ):
        if not _is_hex64(value.get(key)):
            raise RuntimeError(f"invalid protocol digest: {key}")
    if value.get("immediate_continuity") is not True:
        raise RuntimeError("immediate continuity must be true")
    if value.get("repeatable") is not True:
        raise RuntimeError("protocol repeatability must be true")
    _validate_audit(value.get("audit"))


def _validate(payload: dict[str, object]) -> None:
    if payload.get("experiment") != "phase-3c-anonymous-temporal-credit":
        raise RuntimeError("wrong experiment")
    if payload.get("schema_version") != 1:
        raise RuntimeError("wrong schema version")
    if payload.get("phase3b_base_sha") != _PHASE3B_BASE_SHA:
        raise RuntimeError("wrong Phase 3B base SHA")

    formal = payload.get("formal_contract")
    if not isinstance(formal, dict):
        raise RuntimeError("missing formal contract")
    if formal.get("commit") != APPROVED_FORMAL_COMMIT:
        raise RuntimeError("wrong formal commit")
    if formal.get("theorems") != list(REQUIRED_THEOREMS):
        raise RuntimeError("wrong formal theorem list")
    if formal.get("exact_head_ci_passed") is not True:
        raise RuntimeError("formal exact-head CI must be true")
    if formal.get("axiom_audit_reviewed") is not True:
        raise RuntimeError("formal axiom audit must be reviewed")

    formal_valid = _require_bool(payload, "formal_valid")
    protocol_valid = _require_bool(payload, "protocol_valid")
    behavior_passed = _require_bool(payload, "behavior_passed")
    all_passed = _require_bool(payload, "all_passed")
    if formal_valid is not True:
        raise RuntimeError("formal_valid must be true")
    if protocol_valid is not True:
        raise RuntimeError("protocol_valid must be true")
    if all_passed != (formal_valid and protocol_valid and behavior_passed):
        raise RuntimeError("inconsistent all_passed")

    if payload.get("seeds") != list(_DEFAULT_SEEDS):
        raise RuntimeError("wrong registered seeds")
    if payload.get("delay_support") != list(_EXPECTED_DELAYS):
        raise RuntimeError("wrong registered delay support")
    if payload.get("config") != _EXPECTED_CONFIG:
        raise RuntimeError("wrong registered configuration")

    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("results must be a list")
    seen: set[tuple[int, str]] = set()
    row_flags: list[bool] = []
    for row in results:
        if not isinstance(row, dict):
            raise RuntimeError("each result must be an object")
        seed = row.get("seed")
        arm = row.get("arm")
        if type(seed) is not int or seed not in _DEFAULT_SEEDS or arm not in _EXPECTED_ARMS:
            raise RuntimeError("invalid result identity")
        key = (seed, str(arm))
        if key in seen:
            raise RuntimeError("duplicate arm/seed row")
        seen.add(key)
        _validate_protocol(row.get("protocol"), seed=seed, arm=str(arm))

        for count_key in ("post_training", "state_reset", "shuffled_control"):
            _validate_count(row.get(count_key), count_key)
        for delay_key in ("per_delay", "reset_per_delay", "shuffled_per_delay"):
            _validate_per_delay(row.get(delay_key), delay_key)
        for digest_key in (
            "shuffled_action_digest",
            "shuffled_latent_reward_digest",
            "shuffled_parameter_digest",
            "shuffled_delay_digest",
            "shuffled_due_step_digest",
            "shuffled_multiplicity_digest",
            "shuffled_aggregate_feedback_digest",
            "shuffled_learner_call_digest",
        ):
            if not _is_hex64(row.get(digest_key)):
                raise RuntimeError(f"invalid digest: {digest_key}")
        _validate_checkpoint_rows(row.get("normal_checkpoints"), "normal_checkpoints")
        _validate_checkpoint_rows(row.get("shuffled_checkpoints"), "shuffled_checkpoints")
        for flag in (
            "action_sequences_equal",
            "schedule_lineage_equal",
            "reward_block_multisets_equal",
            "behavior_passed",
        ):
            if type(row.get(flag)) is not bool:
                raise RuntimeError(f"invalid result flag: {flag}")
        if row["action_sequences_equal"] is not True:
            raise RuntimeError("action lineage changed")
        if row["schedule_lineage_equal"] is not True:
            raise RuntimeError("schedule lineage changed")
        if row["reward_block_multisets_equal"] is not True:
            raise RuntimeError("shuffled reward block multiset changed")
        row_flags.append(bool(row["behavior_passed"]))

    expected = {(seed, arm) for seed in _DEFAULT_SEEDS for arm in _EXPECTED_ARMS}
    if seen != expected:
        raise RuntimeError("result rows do not match the registered seed/arm grid")
    if behavior_passed != all(row_flags):
        raise RuntimeError("inconsistent behavior_passed")


def _portable_projection(payload: dict[str, object]) -> dict[str, object]:
    """Project deterministic evidence into a cross-version comparison surface."""
    projection = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    results = projection.get("results")
    if not isinstance(results, list):
        raise RuntimeError("portable projection requires result rows")
    for row in results:
        if not isinstance(row, dict):
            raise RuntimeError("portable projection requires object result rows")
        protocol = row.get("protocol")
        if not isinstance(protocol, dict):
            raise RuntimeError("portable projection requires protocol rows")
        protocol.pop("parameter_digest", None)
        row.pop("shuffled_parameter_digest", None)
    return projection


def _portable_difference(
    expected: object,
    actual: object,
    path: tuple[str | int, ...] = (),
) -> str | None:
    """Return the first mismatch; tolerate noise only at registered diagnostic paths."""
    location = "$" + "".join(
        f"[{part}]" if type(part) is int else f".{part}" for part in path
    )
    checkpoint_float = (
        len(path) == 5
        and path[0] == "results"
        and type(path[1]) is int
        and path[2] in ("normal_checkpoints", "shuffled_checkpoints")
        and type(path[3]) is int
        and path[4] in _CHECKPOINT_FLOAT_FIELDS
    )
    if checkpoint_float:
        if _finite_number(expected) and _finite_number(actual):
            if math.isclose(
                expected, actual, rel_tol=0.0, abs_tol=_CHECKPOINT_FLOAT_ABS_TOL
            ):
                return None
        return f"{location}: committed={expected!r}, runtime={actual!r}"
    if type(expected) is not type(actual):
        return (
            f"{location}: committed type={type(expected).__name__}, "
            f"runtime type={type(actual).__name__}"
        )
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return f"{location}: object keys differ"
        for key in sorted(expected):
            difference = _portable_difference(expected[key], actual[key], path + (key,))
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{location}: committed length={len(expected)}, runtime length={len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = _portable_difference(left, right, path + (index,))
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{location}: committed={expected!r}, runtime={actual!r}"
    return None


def verify_phase3c_anonymous_credit(path: Path | None = None) -> dict[str, object]:
    """Verify committed registered evidence against one deterministic replay."""
    root = _repository_root()
    artifact = root / _APPROVED if path is None else path
    committed = _load(artifact, root)
    _validate(committed)
    runtime = build_measurement_payload(
        seeds=_DEFAULT_SEEDS,
        config=AnonymousCreditConfig(),
    )
    _validate(runtime)
    difference = _portable_difference(
        _portable_projection(committed), _portable_projection(runtime)
    )
    if difference is not None:
        raise RuntimeError(
            "committed Phase 3C portable evidence differs from deterministic replay: "
            + difference
        )
    return committed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    payload = verify_phase3c_anonymous_credit(args.evidence)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
