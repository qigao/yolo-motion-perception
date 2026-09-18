from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


_HEAD = "36f5921749815bf2e66e4199f187e6277bc20ef5"


def _api():
    from neural_state_machine import r1_e2_evidence as evidence

    return evidence


def _prospective_root(tmp_path: Path):
    evidence = _api()
    root = tmp_path / "evidence"
    prepared = evidence.prepare_prospective(root, scientific_head=_HEAD)
    return evidence, root, prepared


def _metrics(class_count: int, total: int) -> dict[str, object]:
    return {
        "correct": total,
        "total": total,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "confusion_counts": [
            [total // class_count if row == col else 0 for col in range(class_count)]
            for row in range(class_count)
        ],
    }


def _complete_fake_result() -> dict[str, object]:
    corruptions = ("clean", "drop10", "wrong10", "occlusion4", "jitter", "mixed")
    arms = []
    for seed in (7, 17, 29, 43, 61):
        for architecture in range(4):
            digest = hashlib.sha256(f"{seed}-{architecture}".encode()).hexdigest()
            memory = {
                "architecture": architecture,
                "seed": seed,
                "core": {
                    "valid": True,
                    "parameter_digest_before": digest,
                    "parameter_digest_after": digest,
                    "fixture_digest": "a" * 64,
                    "coefficient_digest": "b" * 64,
                    "prediction_digest": "c" * 64,
                    "reset_prediction_digest": "d" * 64,
                    "memory_horizon": 20,
                },
            }
            histories = []
            for history in (5, 10, 20, 40):
                readout = {
                    "metrics": _metrics(6, 150),
                    "geometry": {
                        "centroid_distances": [1.0] * 15,
                        "within_class_dispersion": [0.5] * 6,
                        "separation_ratio": 2.0,
                    },
                    "reset_correct": 25,
                    "reset_total": 150,
                    "coefficient_digest": "e" * 64,
                    "prediction_digest": "f" * 64,
                    "reset_prediction_digest": "1" * 64,
                }
                histories.append(
                    {
                        "history": history,
                        "instantaneous": readout,
                        "temporal_mean": readout,
                        "reset_groups_equal": True,
                        "train_fixture_digest": "2" * 64,
                        "evaluation_fixture_digest": "3" * 64,
                        "reservoir_parameter_digest": digest,
                    }
                )
            composition = {
                "seed": seed,
                "architecture": architecture,
                "histories": histories,
                "instantaneous_macro_accuracy": 1.0,
                "temporal_mean_macro_accuracy": 1.0,
            }
            baseline = {
                "arm_metrics": [[name, _metrics(4, 200)] for name in corruptions],
                "coefficient_digest": "4" * 64,
                "prediction_digests": [[name, "5" * 64] for name in corruptions],
                "macro_corrupted_accuracy": 1.0,
                "worst_corrupted_accuracy": 1.0,
                "reset_correct": 50,
                "reset_total": 200,
            }
            frame = dict(baseline)
            frame["reset_correct"] = 0
            frame["reset_total"] = 0
            yolo = {
                "seed": seed,
                "architecture": architecture,
                "frame_only": frame,
                "reservoir_instantaneous": baseline,
                "reservoir_temporal_mean": baseline,
                "reset_groups_equal": True,
                "train_fixture_digest": "6" * 64,
                "evaluation_fixture_digest": "7" * 64,
                "corruption_digest": "8" * 64,
                "reservoir_parameter_digest": digest,
            }
            arms.append(
                {
                    "seed": seed,
                    "architecture": architecture,
                    "memory": memory,
                    "composition": composition,
                    "yolo_episode": yolo,
                }
            )
    return {
        "registered_measurement": True,
        "manifest": {"phase": "R1-E2"},
        "arms": arms,
    }


def test_e2_prepare_writes_only_prospective_files(tmp_path: Path) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)

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


def test_e2_prepare_refuses_nonempty_destination(tmp_path: Path) -> None:
    evidence = _api()
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "foreign.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError):
        evidence.prepare_prospective(root, scientific_head=_HEAD)


def test_e2_measurement_requires_exact_manifest_hash_head_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
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

    monkeypatch.setattr(evidence.platform, "python_version", lambda: "0.0.0")
    with pytest.raises(evidence.EvidenceInvalid, match="runtime"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            raw_result=fake,
        )
    assert not (root / "result.json").exists()


def test_e2_measurement_is_write_once_and_verifier_accepts_20_registered_arms(
    tmp_path: Path,
) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    fake = _complete_fake_result()

    written = evidence.write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=_HEAD,
        raw_result=fake,
    )
    assert written["manifest_sha256"] == prepared["manifest_sha256"]

    verified = evidence.verify_evidence(root)
    assert verified["valid"] is True
    assert verified["registered_arm_count"] == 20

    with pytest.raises(FileExistsError):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            raw_result=fake,
        )


def test_e2_verifier_rejects_provenance_runtime_mismatch(tmp_path: Path) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    evidence.write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=_HEAD,
        raw_result=_complete_fake_result(),
    )
    path = root / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["python"] = "0.0.0"
    data = evidence.canonical_json_bytes(provenance)
    path.write_bytes(data)
    (root / "provenance.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(evidence.EvidenceInvalid, match="runtime"):
        evidence.verify_evidence(root)


def test_e2_verifier_rejects_unregistered_corruption_name(tmp_path: Path) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    fake = _complete_fake_result()
    fake["arms"][0]["yolo_episode"]["frame_only"]["arm_metrics"][0][0] = "surprise"
    evidence.write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=_HEAD,
        raw_result=fake,
    )

    with pytest.raises(evidence.EvidenceInvalid, match="corruption"):
        evidence.verify_evidence(root)
