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


def _pair_set():
    import numpy as np

    from neural_state_machine.r1_e3m_dataset import MechanismSample
    from neural_state_machine.r1_e3m_pairs import build_history_pairs

    def sample(name: str, split: str, track_id: int, prefix: float):
        tensor = np.zeros((20, 6), dtype=np.float64)
        tensor[:16, 0] = prefix
        tensor[16:20, 0] = 0.1 + 0.001 * track_id
        tensor[:, 2] = 0.1
        tensor[:, 3] = 0.2
        tensor[:, 4] = 0.9
        tensor[:, 5] = 1.0
        return MechanismSample(
            window_id=(name.encode("utf-8").hex() + "0" * 64)[:64],
            video_id=f"{split}-video-{track_id}",
            split=split,
            track_id=track_id,
            source_start_seconds=0.0,
            source_end_seconds=2.0,
            tensor=tensor,
        )

    training = (
        sample("t0", "train", 1, 0.0),
        sample("t1", "train", 2, 1.0),
        sample("t2", "train", 3, 2.0),
    )
    evaluation = (
        sample("e0", "eval", 10, 0.0),
        sample("e1", "eval", 11, 2.0),
    )
    return build_history_pairs(training, evaluation)



def _delay_registration():
    import numpy as np

    from neural_state_machine.r1_e3m_dataset import (
        MechanismDataset,
        MechanismSample,
    )
    from neural_state_machine.r1_e3m_registration import (
        build_delay_registration,
        validate_delay_registration_gate,
    )

    def sample(index: int, split: str, video_index: int):
        tensor = np.zeros((20, 6), dtype=np.float64)
        tensor[:, 0] = 0.01 * index + np.linspace(0.0, 0.19, 20)
        tensor[:, 1] = 0.005 * index + np.linspace(0.0, 0.095, 20)
        tensor[:, 2] = 0.1
        tensor[:, 3] = 0.2
        tensor[:, 4] = 0.9
        tensor[:, 5] = 1.0
        return MechanismSample(
            window_id=f"{index + 1:064x}",
            video_id=f"{split}-video-{video_index}",
            split=split,
            track_id=index,
            source_start_seconds=float(index * 2),
            source_end_seconds=float(index * 2 + 2),
            tensor=tensor,
        )

    dataset = MechanismDataset(
        training=tuple(
            sample(index, "train", index % 2)
            for index in range(20)
        ),
        evaluation=tuple(
            sample(100 + index, "eval", index % 2)
            for index in range(10)
        ),
        artifact_root_digest="a" * 64,
    )
    registration = build_delay_registration(dataset)
    validate_delay_registration_gate(registration)
    return registration

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
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
    )
    verified = verify(root, no_result_ok=True)

    assert len(prepared["manifest_sha256"]) == 64
    assert verified["valid"] is True
    assert verified["prospective_only"] is True
    assert verified["scientific_head"] == "1" * 40
    assert verified["artifact_root_digest"] == "a" * 64
    assert verified["history_pair_digest"] == _pair_set().pair_digest
    assert verified["history_pair_count"] == len(_pair_set().pairs)
    assert verified["registered_arm_count"] == 20
    assert not (root / "result.json").exists()


def test_prepare_is_write_once(tmp_path: Path) -> None:
    _, prepare, _, _ = _api()
    root = tmp_path / "evidence"

    prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
    )

    with pytest.raises(FileExistsError):
        prepare(
            root,
            scientific_head="1" * 40,
            artifact_root_digest="a" * 64,
            history_pair_set=_pair_set(),
            delay_registration=_delay_registration(),
        )


def test_verify_rejects_protocol_mutation(tmp_path: Path) -> None:
    Invalid, prepare, _, verify = _api()
    root = tmp_path / "evidence"
    prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
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
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
    )

    with pytest.raises(Invalid, match="result is missing"):
        verify(root)



def _metrics(mean_r2: float) -> dict[str, object]:
    return {
        "sample_count": 10,
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

    pair_set = _pair_set()
    delay_registration = _delay_registration()
    pair_diagnostics = [
        {
            "left_window_id": pair.left_window_id,
            "right_window_id": pair.right_window_id,
            "suffix_distance": pair.suffix_distance,
            "prefix_distance": pair.prefix_distance,
            "normal_state_distance": 0.5,
            "reset_state_distance": 0.1,
        }
        for pair in pair_set.pairs
    ]
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
                        "training_sample_count": 20,
                        "evaluation_sample_count": 10,
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
                    "delay_registration_digest":
                        delay_registration.digest,
                    "pair_set_digest": pair_set.pair_digest,
                    "pair_diagnostics": pair_diagnostics,
                }
            )
    return {
        "registered_measurement": True,
        "manifest": registered_manifest_payload(artifact_digest),
        "delay_registration_digest": delay_registration.digest,
        "arms": arms,
        "median_long_delay_delta": 0.10,
        "positive_arm_count": 20,
        "median_h1_long_delay_drop": 0.15,
        "outcome": "M-A",
        "pair_set_digest": pair_set.pair_digest,
        "history_pair_count": len(pair_set.pairs),
        "history_prefix_threshold": pair_set.prefix_threshold,
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
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
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
    assert (
        verified["delay_registration_digest"]
        == _delay_registration().digest
    )
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
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
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
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
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
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
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


def test_prepare_freezes_history_pair_payload_and_digest(tmp_path: Path) -> None:
    _, prepare, _, verify = _api()
    root = tmp_path / "evidence"
    pair_set = _pair_set()

    prepared = prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
        history_pair_set=pair_set,
        delay_registration=_delay_registration(),
    )
    payload = json.loads(
        (root / "history-pairs.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    verified = verify(root, no_result_ok=True)

    assert payload["pair_digest"] == pair_set.pair_digest
    assert payload["pair_count"] == len(pair_set.pairs)
    assert manifest["history_pairs"]["pair_digest"] == pair_set.pair_digest
    assert len(manifest["history_pairs"]["file_sha256"]) == 64
    assert prepared["history_pair_digest"] == pair_set.pair_digest
    assert verified["history_pair_digest"] == pair_set.pair_digest


def test_verify_rejects_history_pair_file_mutation(tmp_path: Path) -> None:
    Invalid, prepare, _, verify = _api()
    root = tmp_path / "evidence"
    prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
    )
    path = root / "history-pairs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prefix_threshold"] = float(payload["prefix_threshold"]) + 0.1
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Invalid, match="sha256"):
        verify(root, no_result_ok=True)


def test_measurement_rejects_history_pair_digest_mutation(
    tmp_path: Path,
) -> None:
    from neural_state_machine.r1_e3m_evidence import write_measurement

    Invalid, prepare, _, _ = _api()
    root = tmp_path / "evidence"
    prepared = prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
    )
    measurement = _measurement("a" * 64)
    measurement["pair_set_digest"] = "f" * 64

    with pytest.raises(Invalid, match="history pair digest"):
        write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head="1" * 40,
            artifact_root_digest="a" * 64,
            raw_result=measurement,
        )


def test_measurement_rejects_history_pair_diagnostic_mutation(
    tmp_path: Path,
) -> None:
    from neural_state_machine.r1_e3m_evidence import write_measurement

    Invalid, prepare, _, _ = _api()
    root = tmp_path / "evidence"
    prepared = prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
    )
    measurement = _measurement("a" * 64)
    measurement["arms"][0]["pair_diagnostics"][0][
        "prefix_distance"
    ] += 0.5

    with pytest.raises(Invalid, match="prefix_distance"):
        write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head="1" * 40,
            artifact_root_digest="a" * 64,
            raw_result=measurement,
        )


def test_verify_rejects_delay_registration_file_mutation(
    tmp_path: Path,
) -> None:
    Invalid, prepare, _, verify = _api()
    root = tmp_path / "evidence"
    prepare(
        root,
        scientific_head="1" * 40,
        artifact_root_digest="a" * 64,
        history_pair_set=_pair_set(),
        delay_registration=_delay_registration(),
    )
    path = root / "delay-registration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["delays"][0]["training_window_ids"] = (
        payload["delays"][0]["training_window_ids"][1:]
    )
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Invalid, match="sha256"):
        verify(root, no_result_ok=True)
