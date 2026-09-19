from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neural_state_machine.r1_e3_protocol import registered_manifest_payload


_HEAD = "1" * 40
_ARTIFACT_DIGEST = "a" * 64
_CORRUPT_ARTIFACT_DIGEST = "b" * 64
_ARCHITECTURES = (0, 1, 2, 3)
_SEEDS = (7, 17, 29, 43, 61)


def _api():
    from neural_state_machine import r1_e3_evidence as evidence

    return evidence


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metrics(*, macro_f1: float = 0.60) -> dict[str, object]:
    return {
        "correct": 24,
        "total": 40,
        "accuracy": 0.60,
        "macro_f1": macro_f1,
        "precision_per_class": [0.60, 0.60, 0.60, 0.60],
        "recall_per_class": [0.60, 0.60, 0.60, 0.60],
        "f1_per_class": [0.60, 0.60, 0.60, 0.60],
        "confusion_counts": [
            [6, 1, 1, 2],
            [1, 6, 2, 1],
            [1, 2, 6, 1],
            [2, 1, 1, 6],
        ],
    }


def _readout(tag: str, *, macro_f1: float = 0.60) -> dict[str, object]:
    return {
        "metrics": _metrics(macro_f1=macro_f1),
        "coefficient_digest": _digest(f"coef-{tag}"),
        "prediction_digest": _digest(f"pred-{tag}"),
    }


def _complete_fake_result(
    *,
    artifact_digest: str = _ARTIFACT_DIGEST,
    delta: float = 0.06,
) -> dict[str, object]:
    arms = []
    for seed in _SEEDS:
        for architecture in _ARCHITECTURES:
            tag = f"{seed}-{architecture}"
            b1 = _readout(f"b1-{tag}", macro_f1=0.60)
            b2 = _readout(f"b2-{tag}", macro_f1=0.60 + delta)
            trajectory_digest = _digest(f"trajectory-{tag}")
            benchmark = {
                "seed": seed,
                "architecture": architecture,
                "frame_only": _readout(f"b0-{tag}", macro_f1=0.50),
                "reservoir_instantaneous": b1,
                "reservoir_temporal_mean": b2,
                "history_destruction": {
                    "reset_before_bin": 16,
                    "suffix_bins": [16, 17, 18, 19],
                    "instantaneous": _readout(f"reset-b1-{tag}", macro_f1=0.40),
                    "temporal_mean": _readout(f"reset-b2-{tag}", macro_f1=0.42),
                },
                "reservoir_parameter_digest": _digest(f"reservoir-{tag}"),
                "trajectory_digest_b1": trajectory_digest,
                "trajectory_digest_b2": trajectory_digest,
                "training_episode_digest": _digest("training-episodes"),
                "evaluation_episode_digest": _digest("evaluation-episodes"),
                "artifact_root_digest": artifact_digest,
                "delta_macro_f1": delta,
            }
            arms.append(
                {
                    "seed": seed,
                    "architecture": architecture,
                    "result": benchmark,
                    "delta_macro_f1": delta,
                }
            )

    return {
        "registered_measurement": True,
        "manifest": registered_manifest_payload(artifact_digest),
        "arms": arms,
        "median_delta_macro_f1": delta,
        "positive_arm_count": 20 if delta > 0.0 else 0,
        "outcome": "A" if delta >= 0.05 else ("B" if delta > 0.0 else "C"),
    }


def _prospective_root(tmp_path: Path):
    evidence = _api()
    root = tmp_path / "evidence"
    prepared = evidence.prepare_prospective(
        root,
        scientific_head=_HEAD,
        artifact_root_digest=_ARTIFACT_DIGEST,
    )
    return evidence, root, prepared


def test_canonical_json_is_sorted_compact_utf8_and_rejects_nan() -> None:
    evidence = _api()

    encoded = evidence.canonical_json_bytes({"z": "中文", "a": [2, 1]})

    assert encoded == '{"a":[2,1],"z":"中文"}\n'.encode()
    with pytest.raises(ValueError):
        evidence.canonical_json_bytes({"bad": float("nan")})


def test_prepare_writes_only_prospective_files_and_binds_artifact(
    tmp_path: Path,
) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)

    assert set(path.name for path in root.iterdir()) == {
        "manifest.json",
        "manifest.sha256",
        "prospective-provenance.json",
        "prospective-provenance.sha256",
    }
    assert prepared["scientific_head"] == _HEAD
    assert prepared["artifact_root_digest"] == _ARTIFACT_DIGEST
    assert prepared["manifest_sha256"] == (root / "manifest.sha256").read_text().strip()

    verified = evidence.verify_evidence(root, no_result_ok=True)
    assert verified == {
        "valid": True,
        "prospective_only": True,
        "scientific_head": _HEAD,
        "artifact_root_digest": _ARTIFACT_DIGEST,
        "manifest_sha256": prepared["manifest_sha256"],
        "registered_arm_count": 20,
    }
    assert not (root / "result.json").exists()


def test_prepare_is_write_once(tmp_path: Path) -> None:
    evidence, root, _ = _prospective_root(tmp_path)

    with pytest.raises(FileExistsError):
        evidence.prepare_prospective(
            root,
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
        )


@pytest.mark.parametrize(
    "artifact_digest",
    ["", "a" * 63, "A" * 64, "g" * 64, None],
)
def test_prepare_rejects_invalid_artifact_digest(
    tmp_path: Path,
    artifact_digest: object,
) -> None:
    evidence = _api()

    with pytest.raises(evidence.EvidenceInvalid, match="artifact"):
        evidence.prepare_prospective(
            tmp_path / "evidence",
            scientific_head=_HEAD,
            artifact_root_digest=artifact_digest,
        )


def test_measurement_requires_exact_manifest_head_artifact_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    raw_result = _complete_fake_result()

    with pytest.raises(evidence.EvidenceInvalid, match="manifest sha256"):
        evidence.write_measurement(
            root,
            manifest_sha256="0" * 64,
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )
    with pytest.raises(evidence.EvidenceInvalid, match="scientific head"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head="0" * 40,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )
    with pytest.raises(evidence.EvidenceInvalid, match="artifact"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            artifact_root_digest=_CORRUPT_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )

    monkeypatch.setattr(evidence.platform, "python_version", lambda: "0.0.0")
    with pytest.raises(evidence.EvidenceInvalid, match="runtime"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )
    assert not (root / "result.json").exists()


def test_measurement_is_write_once_and_full_verifier_accepts_20_arms(
    tmp_path: Path,
) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    raw_result = _complete_fake_result()

    written = evidence.write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=_HEAD,
        artifact_root_digest=_ARTIFACT_DIGEST,
        raw_result=raw_result,
    )

    assert written["manifest_sha256"] == prepared["manifest_sha256"]
    assert len(written["result_sha256"]) == 64

    verified = evidence.verify_evidence(root)
    assert verified["valid"] is True
    assert verified["prospective_only"] is False
    assert verified["registered_arm_count"] == 20
    assert verified["artifact_root_digest"] == _ARTIFACT_DIGEST
    assert verified["outcome"] == "A"

    with pytest.raises(FileExistsError):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )


def test_write_rejects_wrong_registered_arm_identity_before_files_exist(
    tmp_path: Path,
) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    raw_result = _complete_fake_result()
    raw_result["arms"][0]["seed"] = 999

    with pytest.raises(evidence.EvidenceInvalid, match="arm"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )

    assert not (root / "result.json").exists()


def test_write_rejects_b1_b2_trajectory_mismatch(tmp_path: Path) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    raw_result = _complete_fake_result()
    raw_result["arms"][0]["result"]["trajectory_digest_b2"] = "f" * 64

    with pytest.raises(evidence.EvidenceInvalid, match="trajectory"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )


def test_write_rejects_wrong_temporal_window_control(tmp_path: Path) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    raw_result = _complete_fake_result()
    raw_result["arms"][0]["result"]["history_destruction"]["suffix_bins"] = [
        15,
        16,
        17,
        18,
    ]

    with pytest.raises(evidence.EvidenceInvalid, match="suffix|temporal"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )


def test_write_rejects_malformed_confusion_matrix(tmp_path: Path) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    raw_result = _complete_fake_result()
    raw_result["arms"][0]["result"]["reservoir_instantaneous"]["metrics"][
        "confusion_counts"
    ] = [[1, 2], [3, 4]]

    with pytest.raises(evidence.EvidenceInvalid, match="confusion"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )


def test_write_rejects_artifact_digest_mismatch_inside_arm(tmp_path: Path) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    raw_result = _complete_fake_result()
    raw_result["arms"][0]["result"]["artifact_root_digest"] = _CORRUPT_ARTIFACT_DIGEST

    with pytest.raises(evidence.EvidenceInvalid, match="artifact"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )


def test_write_rejects_inconsistent_registered_outcome(tmp_path: Path) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    raw_result = _complete_fake_result()
    raw_result["outcome"] = "C"

    with pytest.raises(evidence.EvidenceInvalid, match="outcome"):
        evidence.write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=_HEAD,
            artifact_root_digest=_ARTIFACT_DIGEST,
            raw_result=raw_result,
        )


def test_full_verifier_rejects_provenance_runtime_mismatch(
    tmp_path: Path,
) -> None:
    evidence, root, prepared = _prospective_root(tmp_path)
    evidence.write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head=_HEAD,
        artifact_root_digest=_ARTIFACT_DIGEST,
        raw_result=_complete_fake_result(),
    )

    path = root / "provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["python"] = "0.0.0"
    data = evidence.canonical_json_bytes(payload)
    path.write_bytes(data)
    (root / "provenance.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(evidence.EvidenceInvalid, match="runtime|trace"):
        evidence.verify_evidence(root)
