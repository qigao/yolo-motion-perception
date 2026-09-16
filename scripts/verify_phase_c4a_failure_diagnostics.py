#!/usr/bin/env python3
"""Strict verifier for Phase C4-A failure-attribution evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_state_machine.phase_c4a_diagnostics.evidence import verify_attribution_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--no-result-ok", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_attribution_stage(args.root, allow_missing_result=args.no_result_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
