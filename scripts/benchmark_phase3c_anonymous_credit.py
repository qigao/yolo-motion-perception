#!/usr/bin/env python3
"""Run Phase 3C structural gates or an explicitly requested registered measurement."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from neural_state_machine.phase3c_benchmark import (
    AnonymousCreditConfig,
    run_phase3c_measurements,
    run_phase3c_protocol_gate,
)
from neural_state_machine.phase3c_formal_contract import (
    APPROVED_FORMAL_COMMIT,
    REQUIRED_THEOREMS,
    load_phase3c_formal_contract,
)

_APPROVED = Path("docs/experiments/phase-3c-anonymous-temporal-credit.json")
_PHASE3B_BASE_SHA = "a5ab079d56fe569aded44348f9591d226ce83009"
_DEFAULT_SEEDS = (7, 17, 29)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_payload(config: AnonymousCreditConfig) -> dict[str, object]:
    return {
        "hidden_size": config.hidden_size,
        "recurrent_radius": config.recurrent_radius,
        "step_size": config.step_size,
        "training_decisions": config.training_decisions,
        "evaluation_blocks": config.evaluation_blocks,
        "checkpoint_interval": config.checkpoint_interval,
        "discount": config.discount,
        "trace_decay": config.trace_decay,
    }


def _formal_payload() -> tuple[dict[str, object], bool]:
    contract = load_phase3c_formal_contract()
    payload = {
        "commit": contract.commit,
        "theorems": list(contract.theorems),
        "exact_head_ci_passed": contract.exact_head_ci_passed,
        "axiom_audit_reviewed": contract.axiom_audit_reviewed,
    }
    valid = (
        contract.commit == APPROVED_FORMAL_COMMIT
        and contract.exact_head_ci_passed is True
        and contract.axiom_audit_reviewed is True
        and contract.theorems == REQUIRED_THEOREMS
    )
    return payload, valid


def build_payload(
    *,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    config: AnonymousCreditConfig | None = None,
) -> dict[str, object]:
    """Build structural Gate F/Gate P JSON without a behavioral surface."""
    resolved = AnonymousCreditConfig() if config is None else config
    if not isinstance(resolved, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")

    formal_contract, formal_valid = _formal_payload()
    results = run_phase3c_protocol_gate(seeds=seeds, config=resolved)
    protocol_valid = all(
        result.immediate_continuity is True and result.repeatable is True
        for result in results
    )

    return {
        "experiment": "phase-3c-anonymous-temporal-credit",
        "schema_version": 1,
        "formal_contract": formal_contract,
        "formal_valid": formal_valid,
        "protocol_valid": protocol_valid,
        "seeds": list(seeds),
        "delay_support": list(resolved.delay_support),
        "config": _config_payload(resolved),
        "results": [asdict(result) for result in results],
    }


def build_measurement_payload(
    *,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    config: AnonymousCreditConfig | None = None,
) -> dict[str, object]:
    """Build behavioral evidence only after the same-input Gate F/Gate P precondition."""
    resolved = AnonymousCreditConfig() if config is None else config
    if not isinstance(resolved, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")

    formal_contract, formal_valid = _formal_payload()
    results = run_phase3c_measurements(seeds=seeds, config=resolved)
    protocol_valid = all(
        result.protocol.immediate_continuity is True
        and result.protocol.repeatable is True
        and result.action_sequences_equal is True
        and result.schedule_lineage_equal is True
        and result.reward_block_multisets_equal is True
        for result in results
    )
    behavior_passed = (
        formal_valid
        and protocol_valid
        and bool(results)
        and all(result.behavior_passed is True for result in results)
    )

    return {
        "experiment": "phase-3c-anonymous-temporal-credit",
        "schema_version": 1,
        "phase3b_base_sha": _PHASE3B_BASE_SHA,
        "formal_contract": formal_contract,
        "formal_valid": formal_valid,
        "protocol_valid": protocol_valid,
        "behavior_passed": behavior_passed,
        "all_passed": formal_valid and protocol_valid and behavior_passed,
        "seeds": list(seeds),
        "delay_support": list(resolved.delay_support),
        "config": _config_payload(resolved),
        "results": [asdict(result) for result in results],
    }


def write_evidence(
    payload: dict[str, object],
    path: Path,
    root: Path | None = None,
) -> None:
    """Atomically write only the approved artifact after formal/protocol gates pass."""
    if payload.get("formal_valid") is not True:
        raise ValueError("formal_valid must be true before evidence can be written")
    if payload.get("protocol_valid") is not True:
        raise ValueError("protocol_valid must be true before evidence can be written")

    resolved_root = _repository_root() if root is None else root
    approved = (resolved_root / _APPROVED).resolve()
    target = path.resolve()
    if target != approved:
        raise ValueError(
            "evidence path is not approved: expected "
            "docs/experiments/phase-3c-anonymous-temporal-credit.json"
        )
    if target.is_symlink():
        raise ValueError("evidence path must not be a symlink")

    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") == rendered:
        return

    fd, temporary_name = tempfile.mkstemp(prefix=".phase3c-", dir=target.parent)
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
    parser.add_argument("--require-formal-valid", action="store_true")
    parser.add_argument("--require-protocol-valid", action="store_true")
    parser.add_argument(
        "--measure-registered",
        action="store_true",
        help="explicitly cross the behavioral measurement boundary",
    )
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()

    if args.evidence is not None and not args.measure_registered:
        parser.error("--evidence requires --measure-registered")

    try:
        payload = (
            build_measurement_payload()
            if args.measure_registered
            else build_payload()
        )
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 1

    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    if args.require_formal_valid and payload["formal_valid"] is not True:
        return 1
    if args.require_protocol_valid and payload["protocol_valid"] is not True:
        return 1
    if args.evidence is not None:
        write_evidence(payload, args.evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())