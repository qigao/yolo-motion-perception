from __future__ import annotations

import json
from pathlib import Path

import pytest


def _api():
    from neural_state_machine.r1_e3m_evidence import (
        EvidenceInvalid,
        prepare_prospective,
        registered_manifest_payload,
        verify_evidence,
    )

    return (
        EvidenceInvalid,
        prepare_prospective,
        registered_manifest_payload,
        verify_evidence,
    )


def test_registered_manifest_freezes_mechanism_protocol() -> None:
    _, _, manifest_payload, _ = _api()

    payload = manifest_payload("a" * 64)

    assert payload["phase"] == "R1-E3M"
    assert payload["artifact_root_digest"] == "a" * 64
    assert payload["input_size"] == 6
    assert payload["neuron_budget"] == 256
    assert payload["arm_count"] == 20
    assert payload["seeds"] == [7, 17, 29, 43, 61]
    assert payload["delays"] == [1, 2, 5, 10, 15]
    assert payload["long_delays"] == [5, 10, 15]
    assert payload["ridge_regularization"] == 1e-6
    assert payload["window"] == {
        "duration_seconds": 2.0,
        "bin_count": 20,
        "min_present_bins": 16,
        "max_observation_age_seconds": 0.5,
        "strictly_causal": True,
    }
    assert payload["outcome"]["robust_positive_arm_min"] == 16
    assert payload["outcome"]["median_long_delay_delta_gt"] == 0.0
    assert payload["outcome"]["median_h1_long_delay_drop_gt"] == 0.0


def test_prepare_and_verify_prospective_only_state(tmp_path: Path) -> None:
    _, prepare, _, verify = _api()
    root = tmp_path / "evidence"

    prepared = prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
    )
    verified = verify(root, no_result_ok=True)

    assert len(prepared["manifest_sha256"]) == 64
    assert verified["valid"] is True
    assert verified["prospective_only"] is True
    assert verified["scientific_head"] == "1" * 40
    assert verified["artifact_root_digest"] == "a" * 64
    assert verified["registered_arm_count"] == 20
    assert not (root / "result.json").exists()


def test_prepare_is_write_once(tmp_path: Path) -> None:
    _, prepare, _, _ = _api()
    root = tmp_path / "evidence"

    prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
    )

    with pytest.raises(FileExistsError):
        prepare(
            root,
            scientific_head="1" * 40,
            artifact_root_digest="a" * 64,
        )


def test_verify_rejects_protocol_mutation(tmp_path: Path) -> None:
    Invalid, prepare, _, verify = _api()
    root = tmp_path / "evidence"
    prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
    )

    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["protocol"]["delays"] = [1, 2]
    manifest_path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Invalid, match="sha256"):
        verify(root, no_result_ok=True)


def test_verify_without_result_fails_when_not_explicitly_allowed(
    tmp_path: Path,
) -> None:
    Invalid, prepare, _, verify = _api()
    root = tmp_path / "evidence"
    prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
    )

    with pytest.raises(Invalid, match="result is missing"):
        verify(root)



def _metrics(mean_r2: float) -> dict[str, object]:
    return {
        "sample_count": 8,
        "mean_r2": mean_r2,
        "r2_per_channel": [mean_r2] * 4,
        "mse": 0.01,
    }


def _probe(mean_r2: float, coefficient: str) -> dict[str, object]:
    return {
        "metrics": _metrics(mean_r2),
        "coefficient_digest": coefficient,
        "prediction_digest": "d" * 64,
    }


def _measurement(artifact_digest: str) -> dict[str, object]:
    from neural_state_machine.r1_e3m_evidence import registered_manifest_payload

    arms = []
    architectures = (0, 1, 2, 3)
    seeds = (7, 17, 29, 43, 61)
    for seed in seeds:
        for architecture in architectures:
            delays = []
            for delay in (1, 2, 5, 10, 15):
                reservoir_coefficient = (
                    f"{seed:02x}{architecture:02x}{delay:02x}" + "a" * 58
                )[:64]
                delays.append(
                    {
                        "delay": delay,
                        "instantaneous": _probe(0.10, "b" * 64),
                        "reservoir": _probe(0.20, reservoir_coefficient),
                        "reset_control": _probe(0.05, reservoir_coefficient),
                        "permuted_control": _probe(0.08, reservoir_coefficient),
                        "delta_r2": 0.10,
                        "h1_drop_r2": 0.15,
                        "h2_drop_r2": 0.12,
                    }
                )
            arms.append(
                {
                    "seed": seed,
                    "architecture": architecture,
                    "delays": delays,
                    "long_delay_delta": 0.10,
                    "h1_long_delay_drop": 0.15,
                    "h2_long_delay_drop": 0.12,
                    "reservoir_parameter_digest": "e" * 64,
                    "artifact_root_digest": artifact_digest,
                }
            )
    return {
        "registered_measurement": True,
        "manifest": registered_manifest_payload(artifact_digest),
        "arms": arms,
        "median_long_delay_delta": 0.10,
        "positive_arm_count": 20,
        "median_h1_long_delay_drop": 0.15,
        "outcome": "M-A",
    }


def test_write_and_verify_registered_measurement(tmp_path: Path) -> None:
    from neural_state_machine.r1_e3m_evidence import (
        write_measurement,
    )

    _, prepare, _, verify = _api()
    root = tmp_path / "evidence"
    prepared = prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
    )

    written = write_measurement(
        root,
        manifest_sha256=prepared["manifest_sha256"],
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
        raw_result=_measurement("a" * 64),
    )
    verified = verify(root)

    assert written["registered_arm_count"] == 20
    assert written["outcome"] == "M-A"
    assert verified["valid"] is True
    assert verified["prospective_only"] is False
    assert verified["registered_arm_count"] == 20
    assert verified["outcome"] == "M-A"


def test_measurement_rejects_control_refit(tmp_path: Path) -> None:
    from neural_state_machine.r1_e3m_evidence import (
        write_measurement,
    )

    Invalid, prepare, _, _ = _api()
    root = tmp_path / "evidence"
    prepared = prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
    )
    measurement = _measurement("a" * 64)
    measurement["arms"][0]["delays"][0]["reset_control"][
        "coefficient_digest"
    ] = "f" * 64

    with pytest.raises(Invalid, match="control refit"):
        write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head="1" * 40,
            artifact_root_digest="a" * 64,
            raw_result=measurement,
        )


def test_measurement_rejects_long_delay_aggregate_mutation(
    tmp_path: Path,
) -> None:
    from neural_state_machine.r1_e3m_evidence import (
        write_measurement,
    )

    Invalid, prepare, _, _ = _api()
    root = tmp_path / "evidence"
    prepared = prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
    )
    measurement = _measurement("a" * 64)
    measurement["arms"][0]["long_delay_delta"] = 0.99

    with pytest.raises(Invalid, match="long_delay_delta"):
        write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head="1" * 40,
            artifact_root_digest="a" * 64,
            raw_result=measurement,
        )


def test_measurement_rejects_outcome_mutation(tmp_path: Path) -> None:
    from neural_state_machine.r1_e3m_evidence import (
        write_measurement,
    )

    Invalid, prepare, _, _ = _api()
    root = tmp_path / "evidence"
    prepared = prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
    )
    measurement = _measurement("a" * 64)
    measurement["outcome"] = "M-C"

    with pytest.raises(Invalid, match="outcome"):
        write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head="1" * 40,
            artifact_root_digest="a" * 64,
            raw_result=measurement,
        )
