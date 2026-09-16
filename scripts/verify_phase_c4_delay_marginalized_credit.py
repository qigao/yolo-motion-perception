#!/usr/bin/env python3
"""Strict verifier for sealed Phase C4-A and C4-B evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_state_machine.phase_c4_evidence import verify_c4_stage

_DEFAULT_ROOT = Path("docs/experiments/phase-c4-delay-marginalized-credit")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("c4a", "c4b"), required=True)
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--no-result-ok", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verify_c4_stage(args.root, args.stage, args.no_result_ok)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "no_result_ok": bool(args.no_result_ok),
                "stage": args.stage,
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
