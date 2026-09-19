from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from neural_state_machine.r1_e3_dataset import load_registered_dataset
from neural_state_machine.r1_e3_evidence import (
    EvidenceInvalid,
    prepare_prospective,
    verify_evidence,
    write_measurement,
)
from neural_state_machine.r1_e3_protocol import (
    protocol_smoke,
    run_registered_measurement,
)


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
        description="R1 E3 real-track protocol and evidence tooling"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    protocol = sub.add_parser("protocol")
    protocol.add_argument("--artifact-root", type=Path, required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--artifact-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--scientific-head", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--no-result-ok", action="store_true")

    measure = sub.add_parser("measure")
    measure.add_argument("--artifact-root", type=Path, required=True)
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


def _read_manifest(root: Path) -> tuple[dict[str, object], str]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest_sha = (root / "manifest.sha256").read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read sealed manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit("sealed manifest must be an object")
    return manifest, manifest_sha


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "protocol":
        dataset = load_registered_dataset(args.artifact_root)
        payload = protocol_smoke(dataset.artifact_root_digest)
        _print(payload)
        return 0 if payload.get("valid") is True else 1

    if args.command == "prepare":
        dataset = load_registered_dataset(args.artifact_root)
        prepared = prepare_prospective(
            args.output,
            scientific_head=args.scientific_head,
            artifact_root_digest=dataset.artifact_root_digest,
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
        manifest, sealed_manifest_sha = _read_manifest(args.root)

        if args.manifest_sha256 != sealed_manifest_sha:
            raise SystemExit(
                "manifest sha256 mismatch: "
                f"sealed={sealed_manifest_sha!r} "
                f"supplied={args.manifest_sha256!r}"
            )

        sealed_head = manifest.get("scientific_head")
        current_head = _current_head()
        if current_head != sealed_head:
            raise SystemExit(
                f"scientific head mismatch: "
                f"sealed={sealed_head!r} current={current_head!r}"
            )

        dataset = load_registered_dataset(args.artifact_root)
        sealed_artifact_digest = manifest.get("artifact_root_digest")
        if dataset.artifact_root_digest != sealed_artifact_digest:
            raise SystemExit(
                "artifact root digest mismatch: "
                f"sealed={sealed_artifact_digest!r} "
                f"current={dataset.artifact_root_digest!r}"
            )

        try:
            preflight = verify_evidence(args.root, no_result_ok=True)
        except EvidenceInvalid as exc:
            raise SystemExit(str(exc)) from exc

        if preflight.get("prospective_only") is not True:
            raise SystemExit("registered measurement result already exists")
        if preflight.get("manifest_sha256") != sealed_manifest_sha:
            raise SystemExit("prospective verifier manifest sha256 mismatch")
        if preflight.get("scientific_head") != current_head:
            raise SystemExit("prospective verifier scientific head mismatch")
        if preflight.get("artifact_root_digest") != dataset.artifact_root_digest:
            raise SystemExit("prospective verifier artifact root digest mismatch")

        if (
            manifest.get("python") != platform.python_version()
            or manifest.get("numpy") != np.__version__
        ):
            raise SystemExit("measurement runtime does not match sealed manifest")

        measurement = run_registered_measurement(dataset)
        try:
            written = write_measurement(
                args.root,
                manifest_sha256=args.manifest_sha256,
                scientific_head=current_head,
                artifact_root_digest=dataset.artifact_root_digest,
                raw_result=measurement,
            )
        except (EvidenceInvalid, FileExistsError) as exc:
            raise SystemExit(str(exc)) from exc
        _print(written)
        return 0

    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
