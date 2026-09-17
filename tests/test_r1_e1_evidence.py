from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neural_state_machine import r1_e1_evidence as evidence


_HEAD = "9c3aa0d30b5a7966b111bb3671ede0d763924dd2"


def _prospective_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "evidence"
    prepared = evidence.prepare_prospective(root, scientific_head=_HEAD)
    return root, prepared


def _complete_fake_result() -> dict[str, object]:
    arms = []
    for seed in (7, 17, 29, 43, 61):
        for budget in (64, 256):
            for architecture in range(5):
                digest = hashlib.sha256(f"{seed}-{budget}-{architecture}".encode()).hexdigest()
                arms.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "architecture": architecture,
                        "memory": {
                            "valid": True,
                            "parameter_digest_before": digest,
                            "parameter_digest_after": digest,
                            "fixture_digest": "a" * 64,
                            "coefficient_digest": "b" * 64,
                            "prediction_digest": "c" * 64,
                            "reset_prediction_digest": "d" * 64,
                        },
                        "history": {
                            "horizons": [
                                {
                                    "coefficient_digest": "e" * 64,
                                    "prediction_digest": "f" * 64,
                                    "reset_prediction_digest": "1" * 64,
                                    "train_fixture_digest": "2" * 64,
                                    "evaluation_fixture_digest": "3" * 64,
                                    "reservoir_parameter_digest": digest,
                                }
                                for _ in (1, 5, 20, 40)
                            ]
                        },
                        "yolo_like": {
                            "corrupted": [
                                [name, {"correct": 100, "total": 200}]
                                for name in ("drop10", "wrong10", "occlusion4", "jitter", "mixed")
                            ],
                            "train_fixture_digest": "4" * 64,
                            "evaluation_fixture_digest": "5" * 64,
                            "corruption_digest": "6" * 64,
                            "coefficient_digest": "7" * 64,
                            "reservoir_parameter_digest": digest,
                            "prediction_digests": [
                                [name, "8" * 64]
                                for name in (
                                    "clean",
                                    "drop10",
                                    "wrong10",
                                    "occlusion4",
                                    "jitter",
                                    "mixed",
                                )
                            ],
                        },
                    }
                )
    return {
        "registered_measurement": True,
        "compatibility": [{"seed": seed, "passed": True} for seed in (7, 17, 29, 43, 61)],
        "arms": arms,
    }


def test_canonical_json_bytes_are_sorted_compact_utf8_with_one_newline() -> None:
    payload = {"z": "中文", "a": [2, 1]}

    encoded = evidence.canonical_json_bytes(payload)

    assert encoded == '{"a":[2,1],"z":"中文"}\n'.encode()
    assert encoded.endswith(b"\n")
    assert not encoded.endswith(b"\n\n")


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        evidence.canonical_json_bytes({"value": float("nan")})


def test_prepare_writes_only_prospective_files_and_verifies_without_measurement(
    tmp_path: Path,
) -> None:
    root, prepared = _prospective_root(tmp_path)

    assert set(path.name for path in root.iterdir()) == {
        "manifest.json",
        "manifest.sha256",
        "prospective-provenance.json",
        "prospective-provenance.sha256",
    }
    assert prepared["scientific_head"] == _HEAD
    assert prepared["manifest_sha256"] == (root / "manifest.sha256").read_text().strip()
    assert evidence.verify_evidence(root, no_result_ok=True)["valid"] is True
    assert not (root / "result.json").exists()
    assert not (root / "provenance.json").exists()
    assert not (root / "trace-index.json").exists()
    assert not (root / "report.md").exists()


def test_prepare_refuses_nonempty_destination(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "foreign.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError):
        evidence.prepare_prospective(root, scientific_head=_HEAD)


def test_prospective_verifier_fails_closed_on_manifest_tamper(tmp_path: Path) -> None:
    root, _ = _prospective_root(tmp_path)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(evidence.EvidenceInvalid, match="manifest"):
        evidence.verify_evidence(root, no_result_ok=True)


def test_measurement_bundle_requires_exact_manifest_hash_and_head(tmp_path: Path) -> None:
    root, prepared = _prospective_root(tmp_path)
    fake = _complete_fake_result()

    with pytest.raises(evidence.EvidenceInvalid, match="manifest sha256"):
        evidence.write_measurement(
            root,
            manifest_sha256="0" * 64,
            scientific_head=_HEAD,
            raw_result=fake,
        )
    with pytest.raises(evidence.EvidenceInvalid, match="scientific head"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head="0" * 40,
            raw_result=fake,
        )
    assert not (root / "result.json").exists()


def test_measurement_bundle_rejects_runtime_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, prepared = _prospective_root(tmp_path)
    monkeypatch.setattr(evidence.platform, "python_version", lambda: "0.0.0")

    with pytest.raises(evidence.EvidenceInvalid, match="runtime"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            raw_result=_complete_fake_result(),
        )

    assert not (root / "result.json").exists()


def test_measurement_bundle_is_write_once_and_links_manifest(tmp_path: Path) -> None:
    root, prepared = _prospective_root(tmp_path)
    fake = _complete_fake_result()

    written = evidence.write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=_HEAD,
        raw_result=fake,
    )

    assert written["manifest_sha256"] == prepared["manifest_sha256"]
    assert set(path.name for path in root.iterdir()) >= {
        "result.json",
        "result.sha256",
        "provenance.json",
        "provenance.sha256",
        "trace-index.json",
        "trace-index.sha256",
    }
    result = json.loads((root / "result.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    assert result["manifest_sha256"] == prepared["manifest_sha256"]
    assert result["measurement"] == fake
    assert provenance["scientific_head"] == _HEAD
    assert provenance["manifest_sha256"] == prepared["manifest_sha256"]
    assert not (root / "report.md").exists()

    with pytest.raises(FileExistsError):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            raw_result=fake,
        )


def test_full_verifier_accepts_complete_registered_bundle(tmp_path: Path) -> None:
    root, prepared = _prospective_root(tmp_path)
    evidence.write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=_HEAD,
        raw_result=_complete_fake_result(),
    )

    verified = evidence.verify_evidence(root)

    assert verified["valid"] is True
    assert verified["registered_arm_count"] == 50


def test_full_verifier_rejects_provenance_runtime_mismatch(tmp_path: Path) -> None:
    root, prepared = _prospective_root(tmp_path)
    evidence.write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=_HEAD,
        raw_result=_complete_fake_result(),
    )
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["python"] = "0.0.0"
    provenance_bytes = evidence.canonical_json_bytes(provenance)
    provenance_path.write_bytes(provenance_bytes)
    (root / "provenance.sha256").write_text(
        hashlib.sha256(provenance_bytes).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(evidence.EvidenceInvalid, match="runtime"):
        evidence.verify_evidence(root)


def test_full_verifier_rejects_unregistered_corruption_name(tmp_path: Path) -> None:
    root, prepared = _prospective_root(tmp_path)
    fake = _complete_fake_result()
    fake["arms"][0]["yolo_like"]["corrupted"][0][0] = "surprise"
    evidence.write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=_HEAD,
        raw_result=fake,
    )

    with pytest.raises(evidence.EvidenceInvalid, match="corruption"):
        evidence.verify_evidence(root)
