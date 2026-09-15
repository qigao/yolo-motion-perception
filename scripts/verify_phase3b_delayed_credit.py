#!/usr/bin/env python3
"""Fail-closed checks for the Phase 3B diagnostic artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmark_phase3b_delayed_credit import _APPROVED, build_payload


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _validate(payload: dict[str, object]) -> None:
    if payload.get("experiment") != "phase-3b-delayed-credit":
        raise RuntimeError("wrong experiment")
    if payload.get("schema_version") != 1:
        raise RuntimeError("wrong schema version")
    if payload.get("diagnostic_only") is not True:
        raise RuntimeError("diagnostic-only marker is required before acceptance")
    if payload.get("decision_status") != "diagnostic-only-before-full-gate":
        raise RuntimeError("wrong decision status")
    frozen = payload.get("frozen_phase3a_sha256")
    if not isinstance(frozen, str) or len(frozen) != 64 or any(c not in "0123456789abcdef" for c in frozen):
        raise RuntimeError("invalid frozen Phase 3A digest")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("results must be a non-empty list")
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("each result must be an object")
        for key in ("seed", "arm", "reward_delay", "post_training", "queue_deliveries", "repeatable"):
            if key not in result:
                raise RuntimeError(f"missing result field: {key}")
        if result["arm"] not in ("td0", "td_lambda"):
            raise RuntimeError("unknown learner arm")
        if type(result["seed"]) is not int or result["seed"] < 0:
            raise RuntimeError("invalid seed")
        if type(result["reward_delay"]) is not int or result["reward_delay"] < 0:
            raise RuntimeError("invalid reward delay")
        if type(result["queue_deliveries"]) is not int or result["queue_deliveries"] < 0:
            raise RuntimeError("invalid queue count")
        if type(result["repeatable"]) is not bool:
            raise RuntimeError("invalid repeatability flag")
        for key in ("post_training", "pre_training", "state_reset"):
            count = result.get(key)
            if not isinstance(count, dict):
                raise RuntimeError(f"invalid count: {key}")
            if any(type(count.get(field)) is not int for field in ("correct", "total")):
                raise RuntimeError(f"invalid count fields: {key}")
            accuracy = count.get("accuracy")
            if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not math.isfinite(float(accuracy)):
                raise RuntimeError(f"invalid accuracy: {key}")


def verify_phase3b_delayed_credit(path: Path | None = None) -> dict[str, object]:
    root = _repository_root()
    artifact = (root / _APPROVED) if path is None else path
    committed = _load(artifact, root)
    _validate(committed)
    runtime = build_payload(root)
    if runtime["frozen_phase3a_sha256"] != committed["frozen_phase3a_sha256"]:
        raise RuntimeError("frozen Phase 3A artifact changed")
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
