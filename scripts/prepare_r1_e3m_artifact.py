from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_state_machine.r1_e3m_artifact_evidence import (
    freeze_mechanism_artifact,
    load_mechanism_candidate,
    verify_frozen_mechanism_artifact,
)


def _payload(evidence) -> dict[str, object]:
    return {
        "valid": True,
        "root_digest": evidence.root_digest,
        "video_count": evidence.video_count,
        "window_count": evidence.window_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package or verify a frozen R1-E3M mechanism artifact."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--candidate", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "freeze":
        artifact = load_mechanism_candidate(args.candidate)
        evidence = freeze_mechanism_artifact(artifact, args.output)
    else:
        evidence = verify_frozen_mechanism_artifact(args.root)

    print(json.dumps(_payload(evidence), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
