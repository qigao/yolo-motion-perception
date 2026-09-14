"""Verify the portable Phase 2B known-failure regression contract."""

from __future__ import annotations

import json
from pathlib import Path

from neural_state_machine import run_reward_learning_benchmark

_EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "experiments"
    / "phase-2b-failure.json"
)
_LOCAL_ONLY_RESULT_KEYS = frozenset(
    {
        "initial_parameter_digest",
        "matrix_controls",
        "normal_parameter_digest",
        "shuffled_parameter_digest",
    }
)


def _portable_phase_2b_payload(payload: dict[str, object]) -> dict[str, object]:
    """Return the cross-environment-stable Phase 2B evidence projection."""
    results = payload["results"]
    if not isinstance(results, list):
        raise TypeError("Phase 2B payload has invalid results")
    if not all(isinstance(result, dict) for result in results):
        raise TypeError("Phase 2B payload has an invalid result")

    return {
        "all_passed": payload["all_passed"],
        "config": payload["config"],
        "phase": payload["phase"],
        "results": [
            {
                key: value
                for key, value in result.items()
                if key not in _LOCAL_ONLY_RESULT_KEYS
            }
            for result in results
        ],
        "seeds": payload["seeds"],
        "shuffled_pooled": payload["shuffled_pooled"],
    }


def _verify_local_float_integrity(payload: dict[str, object]) -> None:
    """Check same-environment data which is intentionally not portable."""
    results = payload["results"]
    if not isinstance(results, list):
        raise TypeError("Phase 2B payload has invalid results")

    for result in results:
        if not isinstance(result, dict):
            raise TypeError("Phase 2B payload has an invalid result")
        controls = result["matrix_controls"]
        if not isinstance(controls, dict):
            raise TypeError("Phase 2B result has invalid matrix controls")
        if controls["normal_before"] != controls["normal_after"]:
            raise RuntimeError("Phase 2B normal reservoir matrices changed")
        if controls["shuffled_before"] != controls["shuffled_after"]:
            raise RuntimeError("Phase 2B shuffled reservoir matrices changed")

        initial = result["initial_parameter_digest"]
        if result["normal_parameter_digest"] == initial:
            raise RuntimeError("Phase 2B normal learner did not update")
        if result["shuffled_parameter_digest"] == initial:
            raise RuntimeError("Phase 2B shuffled learner did not update")


def verify_reward_learning_failure() -> dict[str, object]:
    """Reproduce the frozen known failure and validate local determinism."""
    expected = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    first = run_reward_learning_benchmark()
    second = run_reward_learning_benchmark()
    if expected.get("phase") != "2B":
        raise RuntimeError("Phase 2B evidence has an unexpected phase")
    if expected.get("seeds") != [7, 17, 29]:
        raise RuntimeError("Phase 2B evidence has unexpected seeds")
    if expected.get("all_passed") is not False:
        raise RuntimeError("Phase 2B evidence must record a failed gate")
    if first != second:
        raise RuntimeError("Phase 2B is not byte-stable in this environment")
    if _portable_phase_2b_payload(first) != _portable_phase_2b_payload(expected):
        raise RuntimeError("Phase 2B portable runtime differs from frozen evidence")
    _verify_local_float_integrity(first)
    return _portable_phase_2b_payload(first)


def main() -> int:
    """Print the portable regression payload when verification succeeds."""
    payload = verify_reward_learning_failure()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
