from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from scripts import verify_phase3c_anonymous_credit as verifier


CHECKPOINT_FIELDS = (
    "margin_mean",
    "margin_p10",
    "margin_minimum",
    "td_error_mean",
    "td_error_abs_mean",
    "td_error_p90",
    "td_error_maximum",
)
CHECKPOINT_GROUPS = ("normal_checkpoints", "shuffled_checkpoints")


@pytest.fixture
def committed() -> dict[str, Any]:
    path = verifier._repository_root() / verifier._APPROVED
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_pair(
    monkeypatch: pytest.MonkeyPatch,
    committed: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, object]:
    # Substitute only the file/replay boundaries. Exercise the real schema checks,
    # projection and comparison; do not run a training experiment per perturbation.
    monkeypatch.setattr(verifier, "_load", lambda *args: committed)
    monkeypatch.setattr(verifier, "build_measurement_payload", lambda **kwargs: runtime)
    return verifier.verify_phase3c_anonymous_credit()


@pytest.mark.parametrize(
    ("field", "expected", "actual"),
    [
        ("margin_mean", 0.0077623327093496244, 0.0077623327093496),
        ("margin_p10", -0.18245762602545, -0.18245762602544),
        # Python 3.12 diagnostic run 35051419689: adjacent rounding buckets.
        ("td_error_maximum", 2.2028465928756, 2.2028465928757),
        ("td_error_maximum", 2.9237435623309, 2.923743562331),
        ("td_error_p90", 1.4776657338109, 1.477665733811),
    ],
)
def test_verifier_accepts_observed_checkpoint_rounding_noise(
    monkeypatch: pytest.MonkeyPatch,
    committed: dict[str, Any],
    field: str,
    expected: float,
    actual: float,
) -> None:
    committed["results"][0]["normal_checkpoints"][0][field] = expected
    runtime = copy.deepcopy(committed)
    runtime["results"][0]["normal_checkpoints"][0][field] = actual

    assert _verify_pair(monkeypatch, committed, runtime) == committed


@pytest.mark.parametrize("group", CHECKPOINT_GROUPS)
@pytest.mark.parametrize("field", CHECKPOINT_FIELDS)
def test_verifier_accepts_only_bounded_checkpoint_noise_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    committed: dict[str, Any],
    group: str,
    field: str,
) -> None:
    committed["results"][0][group][0][field] = 0.0
    runtime = copy.deepcopy(committed)
    runtime["results"][0][group][0][field] = 1e-12
    before = copy.deepcopy((committed, runtime))

    assert _verify_pair(monkeypatch, committed, runtime) == committed
    assert (committed, runtime) == before


@pytest.mark.parametrize("group", CHECKPOINT_GROUPS)
@pytest.mark.parametrize("field", CHECKPOINT_FIELDS)
def test_verifier_rejects_checkpoint_difference_beyond_absolute_bound(
    monkeypatch: pytest.MonkeyPatch,
    committed: dict[str, Any],
    group: str,
    field: str,
) -> None:
    committed["results"][0][group][0][field] = 0.0
    runtime = copy.deepcopy(committed)
    runtime["results"][0][group][0][field] = math.nextafter(1e-12, math.inf)

    with pytest.raises(RuntimeError, match="differs from deterministic replay"):
        _verify_pair(monkeypatch, committed, runtime)


def test_verifier_does_not_apply_relative_tolerance(
    monkeypatch: pytest.MonkeyPatch, committed: dict[str, Any]
) -> None:
    committed["results"][0]["normal_checkpoints"][0]["margin_mean"] = 1e6
    runtime = copy.deepcopy(committed)
    runtime["results"][0]["normal_checkpoints"][0]["margin_mean"] += 1e-8

    with pytest.raises(RuntimeError, match="differs from deterministic replay"):
        _verify_pair(monkeypatch, committed, runtime)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_verifier_rejects_nonfinite_or_boolean_checkpoint_values(
    monkeypatch: pytest.MonkeyPatch, committed: dict[str, Any], value: object
) -> None:
    runtime = copy.deepcopy(committed)
    runtime["results"][0]["normal_checkpoints"][0]["margin_mean"] = value

    with pytest.raises(RuntimeError, match="invalid checkpoint field"):
        _verify_pair(monkeypatch, committed, runtime)


@pytest.mark.parametrize(
    "mutation",
    [
        "behavior_count",
        "checkpoint_count",
        "checkpoint_length",
        "checkpoint_episode",
        "protocol_digest",
        "protocol_float",
        "extra_key",
        "numeric_type",
    ],
)
def test_verifier_keeps_non_diagnostic_evidence_exact(
    monkeypatch: pytest.MonkeyPatch, committed: dict[str, Any], mutation: str
) -> None:
    runtime = copy.deepcopy(committed)
    row = runtime["results"][0]
    if mutation == "behavior_count":
        row["post_training"]["correct"] += 1
    elif mutation == "checkpoint_count":
        row["normal_checkpoints"][0]["accuracy"]["correct"] += 1
    elif mutation == "checkpoint_length":
        row["normal_checkpoints"].pop()
    elif mutation == "checkpoint_episode":
        row["normal_checkpoints"][0]["episode"] += 1
    elif mutation == "protocol_digest":
        row["protocol"]["audit"]["delay_digest"] = "f" * 64
    elif mutation == "protocol_float":
        coefficients = row["protocol"]["audit"]["trace_coefficients"]
        coefficients[1][1] = math.nextafter(coefficients[1][1], math.inf)
    elif mutation == "extra_key":
        row["normal_checkpoints"][0]["unexpected"] = 0.0
    else:
        runtime["schema_version"] = 1.0

    with pytest.raises(RuntimeError, match="differs from deterministic replay"):
        _verify_pair(monkeypatch, committed, runtime)


def test_checkpoint_field_name_outside_checkpoint_path_has_no_tolerance(
    monkeypatch: pytest.MonkeyPatch, committed: dict[str, Any]
) -> None:
    committed["margin_mean"] = 0.0
    runtime = copy.deepcopy(committed)
    runtime["margin_mean"] = 1e-14

    with pytest.raises(RuntimeError, match="differs from deterministic replay"):
        _verify_pair(monkeypatch, committed, runtime)


def test_verifier_reports_the_first_differing_field(
    monkeypatch: pytest.MonkeyPatch, committed: dict[str, Any]
) -> None:
    runtime = copy.deepcopy(committed)
    runtime["results"][0]["normal_checkpoints"][0]["margin_mean"] += 1e-8

    with pytest.raises(RuntimeError) as raised:
        _verify_pair(monkeypatch, committed, runtime)
    assert "results[0].normal_checkpoints[0].margin_mean" in str(raised.value)


def test_verifier_preserves_frozen_file_bytes(
    monkeypatch: pytest.MonkeyPatch, committed: dict[str, Any]
) -> None:
    path = verifier._repository_root() / verifier._APPROVED
    before = path.read_bytes()
    runtime = copy.deepcopy(committed)
    runtime["results"][0]["normal_checkpoints"][0]["margin_mean"] += 1e-14
    monkeypatch.setattr(verifier, "build_measurement_payload", lambda **kwargs: runtime)

    assert verifier.verify_phase3c_anonymous_credit() == committed
    assert path.read_bytes() == before
