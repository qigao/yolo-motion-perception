#!/usr/bin/env python3
"""Run Phase 3C formal/protocol gates without behavioral measurement."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from neural_state_machine.phase3c_benchmark import (
    AnonymousCreditConfig,
    run_phase3c_protocol_gate,
)
from neural_state_machine.phase3c_formal_contract import (
    REQUIRED_THEOREMS,
    load_phase3c_formal_contract,
)

_DEFAULT_SEEDS = (7, 17, 29)


def build_payload(
    *,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    config: AnonymousCreditConfig | None = None,
) -> dict[str, object]:
    """Build structural Gate F/Gate P JSON without a behavioral surface."""
    resolved = AnonymousCreditConfig() if config is None else config
    if not isinstance(resolved, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")

    contract = load_phase3c_formal_contract()
    results = run_phase3c_protocol_gate(seeds=seeds, config=resolved)

    formal_valid = (
        contract.exact_head_ci_passed is True
        and contract.axiom_audit_reviewed is True
        and contract.theorems == REQUIRED_THEOREMS
    )
    protocol_valid = all(
        result.immediate_continuity is True and result.repeatable is True
        for result in results
    )

    return {
        "experiment": "phase-3c-anonymous-temporal-credit",
        "schema_version": 1,
        "formal_contract": {
            "commit": contract.commit,
            "theorems": list(contract.theorems),
            "exact_head_ci_passed": contract.exact_head_ci_passed,
            "axiom_audit_reviewed": contract.axiom_audit_reviewed,
        },
        "formal_valid": formal_valid,
        "protocol_valid": protocol_valid,
        "seeds": list(seeds),
        "delay_support": list(resolved.delay_support),
        "config": {
            "hidden_size": resolved.hidden_size,
            "recurrent_radius": resolved.recurrent_radius,
            "step_size": resolved.step_size,
            "training_decisions": resolved.training_decisions,
            "evaluation_blocks": resolved.evaluation_blocks,
            "checkpoint_interval": resolved.checkpoint_interval,
            "discount": resolved.discount,
            "trace_decay": resolved.trace_decay,
        },
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-formal-valid", action="store_true")
    parser.add_argument("--require-protocol-valid", action="store_true")
    args = parser.parse_args()

    try:
        payload = build_payload()
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 1

    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    if args.require_formal_valid and payload["formal_valid"] is not True:
        return 1
    if args.require_protocol_valid and payload["protocol_valid"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
