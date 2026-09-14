"""Run and persist the deterministic Phase 2C diagnostic benchmark."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from neural_state_machine import run_learning_diagnostics_benchmark

_ROOT = Path(__file__).resolve().parents[1]
_APPROVED_EVIDENCE = _ROOT / "docs" / "experiments" / "phase-2c-diagnostics.json"
_PHASE_2B_EVIDENCE = _ROOT / "docs" / "experiments" / "phase-2b-failure.json"


def _write_evidence(
    target: Path,
    payload: dict[str, object],
    *,
    approved_evidence: Path = _APPROVED_EVIDENCE,
) -> None:
    """Atomically replace only the approved Phase 2C measured evidence file."""
    destination = Path(os.path.abspath(target))
    approved = Path(os.path.abspath(approved_evidence))
    phase_2b = Path(os.path.abspath(_PHASE_2B_EVIDENCE))
    if destination == phase_2b:
        raise ValueError("refusing to overwrite frozen Phase 2B evidence")
    if destination != approved:
        raise ValueError("evidence output must be the approved Phase 2C evidence path")
    if approved.is_symlink() or destination.is_symlink():
        raise ValueError("refusing to write evidence through a symlink")

    rendered = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if approved.exists() and approved.read_text(encoding="utf-8") == rendered:
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=approved.parent,
            prefix=f".{approved.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, approved)
    except OSError:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    """Print compact benchmark evidence and optionally persist its approved form."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args()
    payload = run_learning_diagnostics_benchmark()
    if arguments.evidence is not None:
        try:
            _write_evidence(arguments.evidence, payload)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return int(not payload["all_valid"])


if __name__ == "__main__":
    raise SystemExit(main())
