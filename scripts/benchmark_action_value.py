from __future__ import annotations

import json

from neural_state_machine import run_action_value_benchmark


def main() -> int:
    payload = run_action_value_benchmark()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return int(not payload["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
