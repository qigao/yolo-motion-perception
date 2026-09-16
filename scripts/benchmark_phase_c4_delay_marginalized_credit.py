#!/usr/bin/env python3
"""Run Phase C4 protocol, sealing, and explicitly authorized measurements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_state_machine.phase_c4_benchmark import (
    PhaseC4Config,
    measure_c4a_from_protocol,
    measure_c4b_from_protocol,
    run_phase_c4_protocol_gate,
)
from neural_state_machine.phase_c4_evidence import (
    build_c4a_result_payload,
    build_c4b_result_payload,
    prepare_c4a_manifest,
    prepare_c4b_manifest,
    validate_measurement_manifest,
    verify_c4_stage,
    write_measurement_bundle,
)

_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("protocol")

    prepare_a = subparsers.add_parser("prepare-c4a")
    prepare_a.add_argument("--output", type=Path, required=True)

    measure_a = subparsers.add_parser("measure-c4a")
    measure_a.add_argument("--manifest", type=Path, required=True)
    measure_a.add_argument("--output", type=Path, required=True)

    prepare_b = subparsers.add_parser("prepare-c4b")
    prepare_b.add_argument("--c4a-result", type=Path, required=True)
    prepare_b.add_argument("--output", type=Path, required=True)

    measure_b = subparsers.add_parser("measure-c4b")
    measure_b.add_argument("--manifest", type=Path, required=True)
    measure_b.add_argument("--output", type=Path, required=True)
    return parser


def _protocol() -> int:
    results = run_phase_c4_protocol_gate()
    print(
        json.dumps(
            {
                "protocol_valid": True,
                "result_count": len(results),
                "stage": "protocol",
            },
            sort_keys=True,
        )
    )
    return 0


def _measure_c4a(manifest: Path, output: Path) -> int:
    validate_measurement_manifest(_ROOT, manifest, "c4a")
    config = PhaseC4Config()
    protocols = run_phase_c4_protocol_gate(config=config)
    results = tuple(measure_c4a_from_protocol(protocol, config) for protocol in protocols)
    payload = build_c4a_result_payload(results)
    _, result_path, provenance_path = write_measurement_bundle(
        _ROOT,
        manifest,
        output,
        payload,
    )
    verify_c4_stage(output, "c4a", False)
    print(
        json.dumps(
            {
                "operator_passed": payload["operator_passed"],
                "provenance": str(provenance_path),
                "result": str(result_path),
                "stage": "c4a",
            },
            sort_keys=True,
        )
    )
    return 0


def _measure_c4b(manifest: Path, output: Path) -> int:
    sealed = validate_measurement_manifest(_ROOT, manifest, "c4b")
    binding = sealed["c4a_binding"]
    if not isinstance(binding, dict) or binding.get("operator_passed") is not True:
        raise RuntimeError("C4-B requires sealed operator_passed=true")
    config = PhaseC4Config()
    protocols = run_phase_c4_protocol_gate(config=config)
    results = tuple(measure_c4b_from_protocol(protocol, config) for protocol in protocols)
    payload = build_c4b_result_payload(results, operator_passed=True)
    _, result_path, provenance_path = write_measurement_bundle(
        _ROOT,
        manifest,
        output,
        payload,
    )
    verify_c4_stage(output, "c4b", False)
    print(
        json.dumps(
            {
                "all_passed": payload["all_passed"],
                "behavior_passed": payload["behavior_passed"],
                "provenance": str(provenance_path),
                "result": str(result_path),
                "stage": "c4b",
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "protocol":
            return _protocol()
        if args.command == "prepare-c4a":
            path = prepare_c4a_manifest(_ROOT, args.output)
            print(json.dumps({"manifest": str(path), "stage": "c4a"}, sort_keys=True))
            return 0
        if args.command == "measure-c4a":
            return _measure_c4a(args.manifest, args.output)
        if args.command == "prepare-c4b":
            path = prepare_c4b_manifest(_ROOT, args.c4a_result, args.output)
            print(json.dumps({"manifest": str(path), "stage": "c4b"}, sort_keys=True))
            return 0
        if args.command == "measure-c4b":
            return _measure_c4b(args.manifest, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    raise RuntimeError("unreachable command dispatch")


if __name__ == "__main__":
    raise SystemExit(main())
