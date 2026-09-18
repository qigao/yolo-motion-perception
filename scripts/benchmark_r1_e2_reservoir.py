from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from neural_state_machine.r1_e2_evidence import (
    EvidenceInvalid,
    prepare_prospective,
    verify_evidence,
    write_measurement,
)
from neural_state_machine.r1_e2_protocol import protocol_smoke, run_registered_measurement


def _current_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R1 E2 reservoir representation protocol and evidence tooling"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("protocol")

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--scientific-head", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--no-result-ok", action="store_true")

    measure = sub.add_parser("measure")
    measure.add_argument("--root", type=Path, required=True)
    measure.add_argument("--manifest-sha256", required=True)
    return parser


def _print(payload: object) -> None:
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "protocol":
        payload = protocol_smoke()
        _print(payload)
        return 0 if payload.get("valid") is True else 1

    if args.command == "prepare":
        prepared = prepare_prospective(
            args.output,
            scientific_head=args.scientific_head,
        )
        _print(prepared)
        return 0

    if args.command == "verify":
        try:
            verified = verify_evidence(
                args.root,
                no_result_ok=args.no_result_ok,
            )
        except EvidenceInvalid as exc:
            raise SystemExit(str(exc)) from exc
        _print(verified)
        return 0

    if args.command == "measure":
        manifest_path = args.root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read sealed manifest: {exc}") from exc

        sealed_head = manifest.get("scientific_head")
        current_head = _current_head()
        if current_head != sealed_head:
            raise SystemExit(
                f"scientific head mismatch: sealed={sealed_head!r} current={current_head!r}"
            )

        measurement = run_registered_measurement()
        try:
            written = write_measurement(
                args.root,
                manifest_sha256=args.manifest_sha256,
                scientific_head=current_head,
                raw_result=measurement,
            )
        except EvidenceInvalid as exc:
            raise SystemExit(str(exc)) from exc
        _print(written)
        return 0

    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
