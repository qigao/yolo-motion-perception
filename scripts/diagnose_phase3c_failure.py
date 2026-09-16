#!/usr/bin/env python3
"""Fail-closed CLI for Phase 3C failure-attribution diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_state_machine.phase3c_diagnostics.manifest import (
    canonical_json_bytes,
    load_json_object,
    sha256_file,
)
from neural_state_machine.phase3c_diagnostics.report import (
    AttributionCandidate,
    build_report_summary,
    render_markdown,
)
from neural_state_machine.phase3c_diagnostics.runner import (
    prepare_manifest,
    run_registered_measurement,
    verify_attempt,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_new(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _cmd_prepare(args: argparse.Namespace) -> int:
    manifest, digest = prepare_manifest(_root(), args.output, profile=args.profile)
    print(
        json.dumps(
            {"manifest": str(manifest), "sha256": digest, "profile": args.profile},
            sort_keys=True,
        )
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    attempt_dir = run_registered_measurement(
        _root(),
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        attempt_id=args.attempt_id,
        approve_measurement=args.approve_measurement,
    )
    print(json.dumps({"attempt_dir": str(attempt_dir)}, sort_keys=True))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    execution = verify_attempt(args.manifest, args.attempt_dir)
    print(json.dumps(execution, sort_keys=True, allow_nan=False))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    execution = verify_attempt(args.manifest, args.attempt_dir)
    manifest = load_json_object(args.manifest)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report directory: {args.output}")
    args.output.mkdir(parents=True)

    # Causal status is intentionally conservative until a reviewed interpretation
    # maps diagnostic fields to a stronger conclusion.
    candidates = (
        AttributionCandidate("control_residual_structure", "unresolved", ("models",)),
        AttributionCandidate("representation_or_readout_limit", "unresolved", ("models",)),
        AttributionCandidate("prediction_target_mismatch", "unresolved", ("accounting",)),
        AttributionCandidate("history_interaction", "unresolved", ("accounting",)),
        AttributionCandidate("terminal_drain_contribution", "unresolved", ("accounting",)),
    )
    summary = build_report_summary(
        diagnostic_valid=bool(execution["diagnostic_valid"]),
        original_behavior_passed=False,
        candidates=candidates,
        limitations=(
            "Post-result diagnostic selected after observing the seed-17 197/200 shuffled control.",
            "Ranks and associations are descriptive and are not calibrated causal p-values.",
            "Privileged references use information unavailable to the anonymous learner.",
        ),
    )
    summary["manifest_sha256"] = sha256_file(args.manifest)
    summary["manifest_profile"] = manifest.get("profile")
    summary["attempt_id"] = execution.get("attempt_id")
    _write_new(args.output / "summary.json", canonical_json_bytes(summary))
    _write_new(args.output / "report.md", render_markdown(summary).encode("utf-8"))
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="seal a prospective manifest; no fitting")
    prepare.add_argument("--profile", choices=("registered-v1", "smoke"), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=_cmd_prepare)

    run = subparsers.add_parser("run", help="execute an explicitly approved registered measurement")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--expected-manifest-sha256", required=True)
    run.add_argument("--attempt-id", required=True)
    run.add_argument("--approve-measurement", action="store_true")
    run.set_defaults(func=_cmd_run)

    verify = subparsers.add_parser("verify", help="verify a completed attempt without retraining")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--attempt-dir", type=Path, required=True)
    verify.set_defaults(func=_cmd_verify)

    report = subparsers.add_parser("report", help="render a report from a verified attempt")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--attempt-dir", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(func=_cmd_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
