"""Run the diagnostic-only Phase 3A TD(0) versus TD(lambda) comparison."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from neural_state_machine import (
    credit_comparison_benchmark_payload,
    run_credit_comparison_benchmark,
)

_ROOT = Path(__file__).resolve().parents[1]
_APPROVED_OUTPUT = _ROOT / "docs" / "experiments" / "phase-3a-credit-comparison.json"
_FROZEN_EVIDENCE = _ROOT / "docs" / "experiments" / "phase-3a-action-value.json"


def _write_output(target: Path, payload: dict[str, object]) -> None:
    destination = Path(os.path.abspath(target))
    if destination == Path(os.path.abspath(_FROZEN_EVIDENCE)):
        raise ValueError("refusing to overwrite frozen Phase 3A evidence")
    if destination != Path(os.path.abspath(_APPROVED_OUTPUT)):
        raise ValueError("output must be the approved Phase 3A credit path")
    if destination.is_symlink():
        raise ValueError("refusing to write output through a symlink")
    rendered = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") == rendered:
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, destination)
    except OSError:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args()
    payload = credit_comparison_benchmark_payload(
        run_credit_comparison_benchmark(tuple(arguments.seeds))
    )
    if arguments.evidence is not None:
        try:
            _write_output(arguments.evidence, payload)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
