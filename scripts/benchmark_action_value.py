from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from neural_state_machine import run_action_value_benchmark

_ROOT = Path(__file__).resolve().parents[1]
_APPROVED_EVIDENCE = _ROOT / "docs" / "experiments" / "phase-3a-action-value.json"
_FROZEN_EVIDENCE = (
    _ROOT / "docs" / "experiments" / "phase-2b-failure.json",
    _ROOT / "docs" / "experiments" / "phase-2c-diagnostics.json",
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _source_commit() -> str:
    """Return the exact lowercase Git commit measured by an evidence run."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError("could not determine Phase 3A source commit") from exc
    source_commit = completed.stdout.strip() if isinstance(completed.stdout, str) else ""
    if completed.returncode != 0 or _COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise RuntimeError("could not determine valid Phase 3A source commit")
    return source_commit


def _write_evidence(
    target: Path,
    payload: dict[str, object],
    *,
    approved_evidence: Path = _APPROVED_EVIDENCE,
    frozen_evidence: tuple[Path, Path] = _FROZEN_EVIDENCE,
) -> None:
    """Atomically replace only the approved Phase 3A evidence file."""
    destination = Path(os.path.abspath(target))
    approved = Path(os.path.abspath(approved_evidence))
    frozen = tuple(Path(os.path.abspath(path)) for path in frozen_evidence)
    if destination in frozen:
        raise ValueError("refusing to overwrite frozen Phase 2 evidence")
    if destination != approved:
        raise ValueError("evidence output must be the approved Phase 3A evidence path")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args()
    payload = run_action_value_benchmark()
    if arguments.evidence is not None:
        artifact = dict(payload)
        try:
            artifact["source_commit"] = _source_commit()
            _write_evidence(arguments.evidence, artifact)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return int(not payload["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
