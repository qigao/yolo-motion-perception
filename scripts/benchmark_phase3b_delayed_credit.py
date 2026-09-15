#!/usr/bin/env python3
"""Run the Phase 3B delayed-credit benchmark and optionally write evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from neural_state_machine.phase3b_delayed_benchmark import (
    DelayedCreditConfig,
    delayed_credit_payload,
    run_delayed_credit_benchmark,
)

_APPROVED = Path("docs/experiments/phase-3b-delayed-credit.json")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _frozen_phase3a_sha256(root: Path) -> str:
    path = root / "docs/experiments/phase-3a-action-value.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_payload(
    root: Path | None = None,
    *,
    seeds: tuple[int, ...] = (7, 17, 29),
    config: DelayedCreditConfig | None = None,
) -> dict[str, object]:
    resolved_root = _repository_root() if root is None else root
    results = tuple(
        result
        for arm in ("td0", "td_lambda")
        for result in run_delayed_credit_benchmark(seeds, config, arm=arm)
    )
    payload = delayed_credit_payload(results)
    payload["frozen_phase3a_sha256"] = _frozen_phase3a_sha256(resolved_root)
    payload["all_passed"] = False
    payload["decision_status"] = "diagnostic-only-before-full-gate"
    return payload


def write_evidence(payload: dict[str, object], path: Path, root: Path | None = None) -> None:
    resolved_root = _repository_root() if root is None else root
    approved = (resolved_root / _APPROVED).resolve()
    target = path.resolve()
    if target != approved:
        raise ValueError("evidence path must be docs/experiments/phase-3b-delayed-credit.json")
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
    args = parser.parse_args()
    root = _repository_root()
    payload = build_payload(root)
    print(json.dumps(payload, sort_keys=True))
    if args.evidence is not None:
        write_evidence(payload, args.evidence, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
