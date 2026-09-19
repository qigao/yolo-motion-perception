from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from neural_state_machine.r1_e3m_benchmark import (
    ARM_COUNT,
    run_registered_memory_benchmark,
)
from neural_state_machine.r1_e3m_dataset import load_mechanism_dataset
from neural_state_machine.r1_e3m_evidence import (
    EvidenceInvalid,
    prepare_prospective,
    registered_manifest_payload,
    verify_evidence,
    write_measurement,
)
from neural_state_machine.r1_e3m_pairs import build_history_pairs
from neural_state_machine.r1_e3m_probe import DELAYS
from neural_state_machine.r1_e3m_registration import (
    DelayRegistrationInvalid,
    build_delay_registration,
    validate_delay_registration_gate,
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
        description="R1 E3M label-free real-track memory protocol tooling"
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
            _jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _read_manifest(root: Path) -> tuple[dict[str, object], str]:
    try:
        manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        manifest_sha = (root / "manifest.sha256").read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read sealed manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit("sealed manifest must be an object")
    return manifest, manifest_sha


def _measurement_payload(
    result: object,
    artifact_root_digest: str,
) -> dict[str, object]:
    raw = _jsonable(result)
    if not isinstance(raw, dict):
        raise SystemExit("registered memory result must serialize to an object")
    return {
        "registered_measurement": True,
        "manifest": registered_manifest_payload(
            artifact_root_digest
        ),
        **raw,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "protocol":
        dataset = load_mechanism_dataset(args.artifact_root)
        pair_set = build_history_pairs(
            dataset.training,
            dataset.evaluation,
        )
        delay_registration = build_delay_registration(dataset)
        try:
            validate_delay_registration_gate(delay_registration)
        except DelayRegistrationInvalid as exc:
            raise SystemExit(str(exc)) from exc
        _print(
            {
                "registered_measurement": False,
                "valid": True,
                "artifact_root_digest":
                    dataset.artifact_root_digest,
                "arm_count": ARM_COUNT,
                "delays": list(DELAYS),
                "history_pair_digest": pair_set.pair_digest,
                "history_pair_count": len(pair_set.pairs),
                "history_prefix_threshold":
                    pair_set.prefix_threshold,
                "delay_registration_digest":
                    delay_registration.digest,
                "delay_registration": [
                    {
                        "delay": entry.delay,
                        "training_window_count":
                            len(entry.training_window_ids),
                        "evaluation_window_count":
                            len(entry.evaluation_window_ids),
                        "training_video_count":
                            len(entry.training_video_ids),
                        "evaluation_video_count":
                            len(entry.evaluation_video_ids),
                    }
                    for entry in delay_registration.delays
                ],
                "protocol": registered_manifest_payload(
                    dataset.artifact_root_digest
                ),
            }
        )
        return 0

    if args.command == "prepare":
        current_head = _current_head()
        if args.scientific_head != current_head:
            raise SystemExit(
                "scientific head mismatch: "
                f"declared={args.scientific_head!r} "
                f"current={current_head!r}"
            )
        dataset = load_mechanism_dataset(args.artifact_root)
        pair_set = build_history_pairs(
            dataset.training,
            dataset.evaluation,
        )
        delay_registration = build_delay_registration(dataset)
        try:
            validate_delay_registration_gate(delay_registration)
        except DelayRegistrationInvalid as exc:
            raise SystemExit(str(exc)) from exc
        prepared = prepare_prospective(
            args.output,
            scientific_head=args.scientific_head,
            artifact_root_digest=dataset.artifact_root_digest,
            history_pair_set=pair_set,
            delay_registration=delay_registration,
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
                "scientific head mismatch: "
                f"sealed={sealed_head!r} current={current_head!r}"
            )

        dataset = load_mechanism_dataset(args.artifact_root)
        pair_set = build_history_pairs(
            dataset.training,
            dataset.evaluation,
        )
        sealed_artifact_digest = manifest.get(
            "artifact_root_digest"
        )
        if dataset.artifact_root_digest != sealed_artifact_digest:
            raise SystemExit(
                "artifact root digest mismatch: "
                f"sealed={sealed_artifact_digest!r} "
                f"current={dataset.artifact_root_digest!r}"
            )

        sealed_pair_meta = manifest.get("history_pairs")
        if not isinstance(sealed_pair_meta, dict):
            raise SystemExit("sealed manifest history_pairs is missing")
        if pair_set.pair_digest != sealed_pair_meta.get("pair_digest"):
            raise SystemExit("history pair digest mismatch before measurement")
        if len(pair_set.pairs) != sealed_pair_meta.get("pair_count"):
            raise SystemExit("history pair count mismatch before measurement")
        if (
            pair_set.prefix_threshold
            != sealed_pair_meta.get("prefix_threshold")
        ):
            raise SystemExit(
                "history pair threshold mismatch before measurement"
            )

        try:
            preflight = verify_evidence(
                args.root,
                no_result_ok=True,
            )
        except EvidenceInvalid as exc:
            raise SystemExit(str(exc)) from exc

        if preflight.get("prospective_only") is not True:
            raise SystemExit(
                "registered measurement result already exists"
            )
        if (
            preflight.get("manifest_sha256")
            != sealed_manifest_sha
        ):
            raise SystemExit(
                "prospective verifier manifest sha256 mismatch"
            )
        if preflight.get("scientific_head") != current_head:
            raise SystemExit(
                "prospective verifier scientific head mismatch"
            )
        if (
            preflight.get("artifact_root_digest")
            != dataset.artifact_root_digest
        ):
            raise SystemExit(
                "prospective verifier artifact root digest mismatch"
            )
        if (
            preflight.get("history_pair_digest")
            != pair_set.pair_digest
            or preflight.get("history_pair_count")
            != len(pair_set.pairs)
            or preflight.get("history_prefix_threshold")
            != pair_set.prefix_threshold
        ):
            raise SystemExit(
                "prospective verifier history pair seal mismatch"
            )

        if (
            manifest.get("python") != platform.python_version()
            or manifest.get("numpy") != np.__version__
        ):
            raise SystemExit(
                "measurement runtime does not match sealed manifest"
            )

        result = run_registered_memory_benchmark(
            dataset,
            pair_set,
        )
        measurement = _measurement_payload(
            result,
            dataset.artifact_root_digest,
        )
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


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
