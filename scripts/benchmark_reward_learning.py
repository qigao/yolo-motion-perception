"""Run the pre-registered Phase 2B reward-learning benchmark."""

from __future__ import annotations

import json

from neural_state_machine import run_reward_learning_benchmark


def main() -> int:
    result = run_reward_learning_benchmark()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return int(not result["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
