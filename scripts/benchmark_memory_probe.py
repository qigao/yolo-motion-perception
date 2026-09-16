import json

from neural_state_machine.memory_probe import run_memory_probe_benchmark


def main() -> int:
    result = run_memory_probe_benchmark()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
