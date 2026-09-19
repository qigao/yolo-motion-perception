from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neural_state_machine.r1_e2_reservoir import E2Architecture
from neural_state_machine.r1_e3_dataset import (
    RegisteredDataset,
    RegisteredEpisode,
)


_ARTIFACT_DIGEST = "a" * 64


def _api():
    from neural_state_machine import r1_e3_protocol as protocol

    return protocol


def _episode(
    episode_id: str,
    *,
    split: str,
    label: str,
    video_id: str,
) -> RegisteredEpisode:
    tensor = np.zeros((20, 14), dtype=np.float64)
    tensor.flags.writeable = False
    return RegisteredEpisode(
        episode_id=episode_id,
        video_id=video_id,
        split=split,
        label=label,
        tensor=tensor,
        tensor_digest="b" * 64,
    )


def _tiny_dataset() -> RegisteredDataset:
    labels = ("approach", "touch", "pick_up", "pass_by")
    training = tuple(
        _episode(
            f"train-{label}",
            split="training",
            label=label,
            video_id="train-video",
        )
        for label in labels
    )
    evaluation = tuple(
        _episode(
            f"eval-{label}",
            split="evaluation",
            label=label,
            video_id="eval-video",
        )
        for label in labels
    )
    return RegisteredDataset(
        training=training,
        evaluation=evaluation,
        artifact_root_digest=_ARTIFACT_DIGEST,
    )


def test_registered_manifest_is_literal_and_json_safe() -> None:
    protocol = _api()

    manifest = protocol.registered_manifest_payload(_ARTIFACT_DIGEST)

    assert manifest["phase"] == "R1-E3"
    assert manifest["artifact_root_digest"] == _ARTIFACT_DIGEST
    assert manifest["architectures"] == {
        "flat": 0,
        "grouped4": 1,
        "hierarchical2": 2,
        "hierarchical4": 3,
    }
    assert manifest["seeds"] == [7, 17, 29, 43, 61]
    assert manifest["input_size"] == 14
    assert manifest["arm_count"] == 20
    assert manifest["primary_metric"] == "macro_f1"
    assert manifest["temporal_window_bins"] == [16, 17, 18, 19]
    assert manifest["ridge_regularization"] == 1e-6
    assert manifest["outcome_a"] == {
        "median_delta_macro_f1_min": 0.05,
        "positive_arm_count_min": 16,
    }


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ([0.06] * 16 + [0.0] * 4, "A"),
        ([0.02] * 20, "B"),
        ([-0.01] * 20, "C"),
        ([0.0] * 20, "C"),
    ],
)
def test_registered_outcome_classification(
    deltas: list[float],
    expected: str,
) -> None:
    protocol = _api()

    assert protocol.classify_registered_outcome(deltas) == expected


@pytest.mark.parametrize(
    "deltas",
    [
        [0.1] * 19,
        [0.1] * 21,
        [0.1] * 19 + [float("nan")],
        [0.1] * 19 + [float("inf")],
    ],
)
def test_outcome_classification_rejects_invalid_arm_vector(
    deltas: list[float],
) -> None:
    protocol = _api()

    with pytest.raises(ValueError, match="20|finite"):
        protocol.classify_registered_outcome(deltas)


def test_protocol_smoke_is_deterministic_and_never_measures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _api()

    def forbidden_measurement(*args, **kwargs):
        pytest.fail("protocol smoke invoked registered measurement")

    monkeypatch.setattr(
        protocol,
        "run_registered_measurement",
        forbidden_measurement,
    )

    first = protocol.protocol_smoke(_ARTIFACT_DIGEST, seed=7)
    second = protocol.protocol_smoke(_ARTIFACT_DIGEST, seed=7)

    assert first == second
    assert first["registered_measurement"] is False
    assert first["valid"] is True
    assert first["arm_count"] == 20
    assert first["seed"] == 7
    assert first["artifact_root_digest"] == _ARTIFACT_DIGEST
    assert set(first["reservoir_parameter_digests"]) == {
        "flat",
        "grouped4",
        "hierarchical2",
        "hierarchical4",
    }
    assert all(
        len(value) == 64
        for value in first["reservoir_parameter_digests"].values()
    )


@pytest.mark.parametrize("seed", [-1, 5, True, 7.0, "7"])
def test_protocol_smoke_rejects_unregistered_seed(seed: object) -> None:
    protocol = _api()

    with pytest.raises(ValueError, match="registered"):
        protocol.protocol_smoke(_ARTIFACT_DIGEST, seed=seed)


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "a" * 63,
        "A" * 64,
        "g" * 64,
        None,
    ],
)
def test_manifest_rejects_invalid_artifact_digest(digest: object) -> None:
    protocol = _api()

    with pytest.raises(ValueError, match="artifact_root_digest"):
        protocol.registered_manifest_payload(digest)


def test_registered_measurement_executes_exact_20_arm_identity_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _api()
    dataset = _tiny_dataset()
    calls: list[tuple[int, int, str]] = []

    def fake_evaluate(spec, received_dataset):
        assert received_dataset is dataset
        calls.append(
            (
                spec.seed,
                int(spec.architecture),
                received_dataset.artifact_root_digest,
            )
        )
        return SimpleNamespace(delta_macro_f1=0.06)

    monkeypatch.setattr(protocol, "evaluate_real_track_arm", fake_evaluate)

    result = protocol.run_registered_measurement(dataset)

    assert result.registered_measurement is True
    assert result.manifest["artifact_root_digest"] == _ARTIFACT_DIGEST
    assert len(result.arms) == 20
    assert result.outcome == "A"
    assert calls == [
        (seed, int(architecture), _ARTIFACT_DIGEST)
        for seed in (7, 17, 29, 43, 61)
        for architecture in (
            E2Architecture.FLAT,
            E2Architecture.GROUPED4,
            E2Architecture.HIERARCHICAL2,
            E2Architecture.HIERARCHICAL4,
        )
    ]


def test_registered_runner_signature_rejects_overrides() -> None:
    protocol = _api()

    with pytest.raises(TypeError):
        protocol.run_registered_measurement(
            _tiny_dataset(),
            seeds=(7,),
        )
