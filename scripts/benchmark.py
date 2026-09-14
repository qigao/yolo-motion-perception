import json

from neural_state_machine.benchmark import run_benchmark


def main() -> int:
    print(json.dumps(run_benchmark(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
