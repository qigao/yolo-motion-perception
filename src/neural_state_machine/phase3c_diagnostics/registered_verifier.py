from __future__ import annotations

import math
from typing import Mapping

from .contracts import (
    REGISTERED_ARMS,
    REGISTERED_CONDITIONS,
    REGISTERED_SEEDS,
    ModelId,
)

_MAX_STORED_RECONSTRUCTION_RESIDUAL = 1e-10


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise RuntimeError(f"{name} must be finite" + (" and non-negative" if nonnegative else ""))
    return result


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}")
    return value


def _metric(value: object, name: str) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a metric object")
    sample_count = _integer(value.get("sample_count"), f"{name}.sample_count", minimum=0)
    metric_value = value.get("value")
    reason = value.get("reason")
    if metric_value is None:
        if sample_count < 0 or not isinstance(reason, str) or not reason:
            raise RuntimeError(f"{name} undefined metric requires a reason")
        return
    _finite(metric_value, f"{name}.value")
    if reason is not None:
        raise RuntimeError(f"{name} defined metric must not have a reason")


def _registered_training_decisions(manifest: Mapping[str, object]) -> int:
    if manifest.get("profile") != "registered-v1" or manifest.get("registered_valid_profile") is not True:
        raise RuntimeError("registered evidence validator requires registered-v1 manifest")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError("registered manifest configuration is missing")
    return _integer(configuration.get("training_decisions"), "training_decisions", minimum=1)


def _validate_d0(execution: Mapping[str, object], training_decisions: int) -> None:
    rows = execution.get("d0")
    if not isinstance(rows, list):
        raise RuntimeError("registered D0 rows are missing")
    expected = {
        (seed, arm, condition)
        for seed in REGISTERED_SEEDS
        for condition in REGISTERED_CONDITIONS
        for arm in REGISTERED_ARMS
    }
    observed: set[tuple[int, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("registered D0 row must be an object")
        seed = row.get("seed")
        arm = row.get("arm")
        condition = row.get("condition")
        if type(seed) is not int or not isinstance(arm, str) or not isinstance(condition, str):
            raise RuntimeError("registered D0 identity is malformed")
        identity = (seed, arm, condition)
        if identity in observed:
            raise RuntimeError("registered D0 grid contains a duplicate trajectory")
        observed.add(identity)
        digest = row.get("parameter_digest")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("registered D0 parameter digest is malformed")
        scalar_calls = _integer(row.get("scalar_call_count"), "D0 scalar_call_count", minimum=0)
        if scalar_calls < training_decisions:
            raise RuntimeError("registered D0 scalar-call stream is shorter than the decision stream")
    if observed != expected:
        raise RuntimeError("registered D0 trajectory grid is incomplete or unexpected")


def _validate_d1(d1: object, model: ModelId, training_decisions: int) -> None:
    if not isinstance(d1, dict):
        raise RuntimeError(f"D1 evidence is missing for {model.stable_key()}")
    provenance_rows = d1.get("provenance_rows")
    if not isinstance(provenance_rows, list) or len(provenance_rows) != training_decisions:
        raise RuntimeError(f"D1 provenance grid is incomplete for {model.stable_key()}")
    if not isinstance(d1.get("donor_summary"), dict):
        raise RuntimeError(f"D1 donor summary is missing for {model.stable_key()}")
    call_classes = d1.get("call_classes")
    if not isinstance(call_classes, dict):
        raise RuntimeError(f"D1 call classes are missing for {model.stable_key()}")
    if _integer(call_classes.get("real_decision_calls"), "D1 real_decision_calls") != training_decisions:
        raise RuntimeError(f"D1 real-decision call count differs for {model.stable_key()}")
    _integer(call_classes.get("drain_calls"), "D1 drain_calls", minimum=0)
    for field in ("no_arrival_zero", "cancellation_to_zero", "one_source", "collisions"):
        _integer(call_classes.get(field), f"D1 {field}", minimum=0)
    lags = d1.get("lags")
    if not isinstance(lags, list) or len(lags) != 11:
        raise RuntimeError(f"D1 lag grid must contain 0..10 for {model.stable_key()}")
    observed_lags: list[int] = []
    for row in lags:
        if not isinstance(row, dict):
            raise RuntimeError(f"D1 lag row is malformed for {model.stable_key()}")
        observed_lags.append(_integer(row.get("lag"), "D1 lag", minimum=0))
        _metric(row.get("feedback_reward_product"), "D1 feedback_reward_product")
        _metric(row.get("feedback_reward_correlation"), "D1 feedback_reward_correlation")
        if not isinstance(row.get("action_cells"), dict) or not isinstance(row.get("label_cells"), dict):
            raise RuntimeError(f"D1 lag contingency cells are missing for {model.stable_key()}")
    if observed_lags != list(range(11)):
        raise RuntimeError(f"D1 lag order differs from 0..10 for {model.stable_key()}")
    final_100 = d1.get("final_100")
    if not isinstance(final_100, dict):
        raise RuntimeError(f"D1 final-100 summary is missing for {model.stable_key()}")
    if _integer(final_100.get("sample_count"), "D1 final_100 sample_count") != min(100, training_decisions):
        raise RuntimeError(f"D1 final-100 sample count differs for {model.stable_key()}")
    _finite(final_100.get("feedback_sum"), "D1 final_100 feedback_sum")
    _finite(final_100.get("feedback_mean"), "D1 final_100 feedback_mean")


def _validate_model_rows(model_rows: list[dict[str, object]], training_decisions: int) -> None:
    for row in model_rows:
        if not isinstance(row, dict) or not isinstance(row.get("model_id"), dict):
            raise RuntimeError("registered model evidence row is malformed")
        model = ModelId.from_dict(row["model_id"])
        if model.family in ("original", "permutation"):
            _validate_d1(row.get("d1"), model, training_decisions)
        elif model.family == "reference":
            information_access = row.get("information_access")
            if not isinstance(information_access, str) or not information_access:
                raise RuntimeError(f"privileged information_access is missing for {model.stable_key()}")


def _validate_reconstruction_residual(value: object, name: str) -> None:
    residual = _finite(value, name, nonnegative=True)
    if residual > _MAX_STORED_RECONSTRUCTION_RESIDUAL:
        raise RuntimeError(f"{name} residual exceeds the registered accounting bound")


def _validate_accounting(execution: Mapping[str, object], training_decisions: int) -> None:
    rows = execution.get("accounting")
    if not isinstance(rows, list):
        raise RuntimeError("registered accounting rows are missing")
    expected = {
        (seed, arm, condition)
        for seed in REGISTERED_SEEDS
        for condition in REGISTERED_CONDITIONS
        for arm in REGISTERED_ARMS
    }
    observed: set[tuple[int, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("registered accounting trajectory must be an object")
        seed = row.get("seed")
        arm = row.get("arm")
        condition = row.get("condition")
        if type(seed) is not int or not isinstance(arm, str) or not isinstance(condition, str):
            raise RuntimeError("registered accounting identity is malformed")
        identity = (seed, arm, condition)
        if identity in observed:
            raise RuntimeError("registered accounting grid contains a duplicate trajectory")
        observed.add(identity)
        if arm == "td0":
            if _integer(row.get("real_steps"), "TD0 real_steps") != training_decisions:
                raise RuntimeError("TD0 accounting real-step count differs")
            if _integer(row.get("drain_weight_changes"), "TD0 drain_weight_changes") != 0:
                raise RuntimeError("TD0 drain changed parameters")
            _integer(row.get("drain_steps"), "TD0 drain_steps", minimum=0)
            continue
        if arm != "eligibility":
            raise RuntimeError("registered accounting arm is unexpected")
        _validate_reconstruction_residual(
            row.get("maximum_drain_residual"), "maximum_drain_residual"
        )
        components = row.get("history_components")
        if not isinstance(components, list) or len(components) != training_decisions:
            raise RuntimeError("eligibility accounting history grid is incomplete")
        for component in components:
            if not isinstance(component, dict):
                raise RuntimeError("eligibility accounting history component is malformed")
            _validate_reconstruction_residual(
                component.get("eligibility_reconstruction_max_residual"),
                "eligibility_reconstruction_max_residual",
            )
            _validate_reconstruction_residual(
                component.get("update_reconstruction_max_residual"),
                "update_reconstruction_max_residual",
            )
            for name in (
                "source_vs_reference_cosine",
                "other_vs_reference_cosine",
                "actual_vs_reference_cosine",
            ):
                _metric(component.get(name), name)
    if observed != expected:
        raise RuntimeError("registered accounting trajectory grid is incomplete or unexpected")


def validate_registered_evidence(
    manifest: Mapping[str, object],
    execution: Mapping[str, object],
    model_rows: list[dict[str, object]],
) -> None:
    """Fail closed on registered D0/D1/D3 evidence omitted after row/hash verification."""
    if not isinstance(manifest, Mapping) or not isinstance(execution, Mapping):
        raise RuntimeError("registered manifest/execution payload is malformed")
    if not isinstance(model_rows, list):
        raise RuntimeError("registered model evidence rows are missing")
    training_decisions = _registered_training_decisions(manifest)
    if execution.get("diagnostic_valid") is not True:
        raise RuntimeError("registered execution is not diagnostic-valid")
    _validate_d0(execution, training_decisions)
    _validate_model_rows(model_rows, training_decisions)
    _validate_accounting(execution, training_decisions)
