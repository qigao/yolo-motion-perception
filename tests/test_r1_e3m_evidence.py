import hashlib
import json
from pathlib import Path

import pytest

from neural_state_machine.r1_e3m_evidence import (
    EvidenceInvalid,
    canonical_json_bytes,
    prepare_prospective,
    verify_evidence,
    write_measurement,
)


HEAD = "a" * 40
ARTIFACT = "b" * 64


def _write_approval(root: Path, manifest_sha: str) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "schema": "r1-e3m-measurement-approval-v1",
        "manifest_sha256": manifest_sha,
        "scientific_head": manifest["scientific_head"],
        "artifact_root_digest": manifest["artifact_root_digest"],
        "approved": True,
    }
    data = canonical_json_bytes(payload)
    (root / "approval.json").write_bytes(data)
    (root / "approval.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n",
        encoding="utf-8",
    )


def _arms(delta=0.06, reset=0.07):
    seeds = (7, 17, 29, 43, 61)
    return [
        {
            "architecture": index // 5,
            "seed": seeds[index % 5],
            "delta10": delta,
            "reset10": reset,
        }
        for index in range(20)
    ]


def test_prepare_and_verify_prospective_only(tmp_path: Path):
    root = tmp_path / "evidence"

    prepared = prepare_prospective(
        root,
        scientific_head=HEAD,
        artifact_root_digest=ARTIFACT,
        train_sequence_count=120,
        eval_sequence_count=40,
    )
    verified = verify_evidence(root, no_result_ok=True)

    assert len(prepared["manifest_sha256"]) == 64
    assert verified["valid"] is True
    assert verified["prospective_only"] is True
    assert verified["scientific_head"] == HEAD
    assert verified["artifact_root_digest"] == ARTIFACT
    assert verified["train_sequence_count"] == 120
    assert verified["eval_sequence_count"] == 40


def test_prepare_is_write_once(tmp_path: Path):
    root = tmp_path / "evidence"
    prepare_prospective(
        root,
        scientific_head=HEAD,
        artifact_root_digest=ARTIFACT,
        train_sequence_count=10,
        eval_sequence_count=5,
    )

    with pytest.raises(FileExistsError):
        prepare_prospective(
            root,
            scientific_head=HEAD,
            artifact_root_digest=ARTIFACT,
            train_sequence_count=10,
            eval_sequence_count=5,
        )


def test_measurement_requires_explicit_matching_approval(tmp_path: Path):
    root = tmp_path / "evidence"
    prepared = prepare_prospective(
        root,
        scientific_head=HEAD,
        artifact_root_digest=ARTIFACT,
        train_sequence_count=20,
        eval_sequence_count=10,
    )

    with pytest.raises(EvidenceInvalid, match="approval"):
        write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=HEAD,
            artifact_root_digest=ARTIFACT,
            arms=_arms(),
        )

    _write_approval(root, "f" * 64)
    with pytest.raises(EvidenceInvalid, match="approval manifest"):
        write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=HEAD,
            artifact_root_digest=ARTIFACT,
            arms=_arms(),
        )


def test_write_measurement_recomputes_outcome_and_is_write_once(tmp_path: Path):
    root = tmp_path / "evidence"
    prepared = prepare_prospective(
        root,
        scientific_head=HEAD,
        artifact_root_digest=ARTIFACT,
        train_sequence_count=20,
        eval_sequence_count=10,
    )
    _write_approval(root, prepared["manifest_sha256"])

    written = write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=HEAD,
        artifact_root_digest=ARTIFACT,
        arms=_arms(),
    )

    assert written["outcome"] == "A"
    assert written["registered_arm_count"] == 20
    verified = verify_evidence(root)
    assert verified["outcome"] == "A"
    assert verified["prospective_only"] is False

    with pytest.raises(FileExistsError, match="write-once"):
        write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=HEAD,
            artifact_root_digest=ARTIFACT,
            arms=_arms(),
        )


def test_tampered_manifest_fails_verification(tmp_path: Path):
    root = tmp_path / "evidence"
    prepare_prospective(
        root,
        scientific_head=HEAD,
        artifact_root_digest=ARTIFACT,
        train_sequence_count=20,
        eval_sequence_count=10,
    )
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload["sequence_counts"]["eval"] = 999
    (root / "manifest.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvidenceInvalid, match="sha256 mismatch"):
        verify_evidence(root, no_result_ok=True)


def test_sequence_counts_must_be_positive():
    with pytest.raises(EvidenceInvalid, match="sequence"):
        prepare_prospective(
            Path("/tmp/unused-r1-e3m-evidence"),
            scientific_head=HEAD,
            artifact_root_digest=ARTIFACT,
            train_sequence_count=0,
            eval_sequence_count=10,
        )
