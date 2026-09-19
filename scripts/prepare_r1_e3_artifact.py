from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neural_state_machine.r1_e3_artifact_evidence import (
    freeze_artifact,
    verify_frozen_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R1 E3 artifact freeze tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--candidate", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    return parser


def _print(payload: object) -> None:
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        _print(verify_frozen_artifact(args.root))
        return 0
    if args.command == "freeze":
        _print(freeze_artifact(args.candidate, args.output))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
