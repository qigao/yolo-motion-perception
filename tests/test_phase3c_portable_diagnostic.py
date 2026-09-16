from __future__ import annotations

import json
from pathlib import Path

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from scripts.benchmark_phase3c_anonymous_credit import _APPROVED, _DEFAULT_SEEDS, build_measurement_payload
from scripts.verify_phase3c_anonymous_credit import _portable_projection


def _first_diff(left: object, right: object, path: str = "$" ) -> tuple[str, object, object] | None:
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}.<keys>", sorted(left), sorted(right)
        for key in left:
            diff = _first_diff(left[key], right[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.<len>", len(left), len(right)
        for index, (l_item, r_item) in enumerate(zip(left, right, strict=True)):
            diff = _first_diff(l_item, r_item, f"{path}[{index}]")
            if diff is not None:
                return diff
        return None
    if left != right:
        return path, left, right
    return None


def test_diagnose_committed_portable_replay_difference() -> None:
    committed = json.loads(Path(_APPROVED).read_text(encoding="utf-8"))
    runtime = build_measurement_payload(seeds=_DEFAULT_SEEDS, config=AnonymousCreditConfig())
    left = _portable_projection(committed)
    right = _portable_projection(runtime)
    diff = _first_diff(left, right)
    assert diff is None, f"first portable replay diff: path={diff[0]} committed={diff[1]!r} runtime={diff[2]!r}"
