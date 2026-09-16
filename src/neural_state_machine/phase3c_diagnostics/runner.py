from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from neural_state_machine.action_value import NormalizedActionValue
from neural_state_machine.action_value_benchmark import (
    _build_fixture_bundle,
    _evaluate,
    _new_policy,
    _train_normal,
)
from neural_state_machine.memory_task import DelayedCueTask
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig, _execute_training
from neural_state_machine.phase3c_schedule import AnonymousRewardAggregator, build_hidden_delay_schedule
from neural_state_machine.reward_learning import _decision_hidden

from .accounting import cosine_metric, expected_drain_delta, maximum_accounting_residual, residual_terms
from .contracts import (
    REGISTERED_ARMS,
    REGISTERED_CONDITIONS,
    REGISTERED_PROFILE,
    REGISTERED_SEEDS,
    ModelId,
    profile_is_registered,
    profile_model_ids,
)
from .evaluation import EvaluationBundle, build_evaluation_bundles
from .manifest import (
    PLAN_COMMIT,
    SCIENTIFIC_REFERENCE_SHA,
    SPEC_COMMIT,
    assert_hashes,
    build_input_manifest,
    canonical_json_bytes,
    environment_snapshot,
    load_json_object,
    sha256_file,
)
from .provenance import (
    apply_permutation,
    diagnostic_permutation,
    donor_rows,
    donor_summary,
    original_block10_permutation,
)
from .references import SourceVisibleDelayedReference, fit_supervised_ridge
from .replay import ReplayResult, run_diagnostic_replay


@dataclass(frozen=True, slots=True)
class StageResult:
    name: str
    passed: bool
    payload: object = None


def run_fail_closed_stages(
    d0: Callable[[], StageResult],
    d1: Callable[[], StageResult],
    d2: Callable[[], StageResult],
    d3: Callable[[], StageResult],
) -> tuple[StageResult, ...]:
    results: list[StageResult] = []
    for stage in (d0, d1, d2, d3):
        result = stage()
        if not isinstance(result, StageResult):
            raise TypeError("diagnostic stage must return StageResult")
        results.append(result)
        if not result.passed:
            break
    return tuple(results)


def validate_complete_keys(expected: tuple[str, ...], observed: tuple[str, ...]) -> None:
    if len(expected) != len(set(expected)):
        raise ValueError("expected grid contains duplicate keys")
    if len(observed) != len(set(observed)) or set(expected) != set(observed):
        raise ValueError("observed grid is incomplete, duplicated or unexpected")


def current_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("git HEAD is not a full commit SHA")
    return value


def _registered_config() -> AnonymousCreditConfig:
    return AnonymousCreditConfig()


def _model_score_keys(models: tuple[ModelId, ...]) -> tuple[str, ...]:
    return tuple(
        f"{model.stable_key()}/evaluation={evaluation_id}"
        for model in models
        for evaluation_id in range(-1, 8)
    )


def prepare_manifest(root: Path, output: Path, *, profile: str = REGISTERED_PROFILE) -> tuple[Path, str]:
    root = root.resolve()
    implementation_sha = current_commit(root)
    input_manifest = build_input_manifest(
        root,
        implementation_sha=implementation_sha,
        scientific_reference_sha=SCIENTIFIC_REFERENCE_SHA,
    )
    models = profile_model_ids(profile)
    evaluations = build_evaluation_bundles(_registered_config())
    registered = profile_is_registered(profile)
    score_keys = _model_score_keys(models)
    if registered and len(score_keys) != 3645:
        raise RuntimeError("registered main score grid must contain 3645 rows")
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": "phase-3c-failure-attribution-v1",
        "profile": profile,
        "registered_valid_profile": registered,
        "implementation_sha": implementation_sha,
        "scientific_reference_sha": SCIENTIFIC_REFERENCE_SHA,
        "spec_commit": SPEC_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "input": input_manifest.to_payload(),
        "environment": environment_snapshot(),
        "configuration": {
            **input_manifest.registered_configuration,
            "delay_support": [1, 3, 5],
        },
        "rng": {
            "original_shuffle": "SeedSequence([seed,0x33534846])",
            "permutation": "PCG64(SeedSequence([seed,0x33434641,1,mode,replicate]))",
            "additional_evaluation": "PCG64(SeedSequence([seed,0x33434641,2,evaluation_id]))",
        },
        "model_ids": [model.to_dict() for model in models],
        "evaluation_manifest": [bundle.to_manifest_row() for bundle in evaluations],
        "main_score_keys": list(score_keys),
        "expected_model_count": len(models),
        "expected_main_score_count": len(score_keys),
        "runner": {"python_major_minor": "3.12", "timeout_minutes": 120},
    }
    if output.exists():
        raise FileExistsError("prepare output directory already exists")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(payload))
    digest = sha256_file(manifest_path)
    (output / "manifest.sha256").write_text(digest + "\n", encoding="ascii")
    return manifest_path, digest


def _accuracy_payload(count: object) -> dict[str, int]:
    return {"correct": int(count.correct), "total": int(count.total)}


def _evaluation_payload(evaluation: object, evaluation_id: int) -> dict[str, object]:
    return {
        "evaluation_id": evaluation_id,
        "overall": _accuracy_payload(evaluation.overall),
        "per_delay": [
            {"cue_delay": delay, **_accuracy_payload(count)}
            for delay, count in evaluation.per_delay
        ],
        "action_digest": evaluation.action_digest,
        "hidden_digest": evaluation.hidden_digest,
        "margin_mean": evaluation.margin_mean,
        "margin_p10": evaluation.margin_p10,
        "margin_minimum": evaluation.margin_minimum,
    }


def _score_learner(
    seed: int,
    learner: object,
    config: AnonymousCreditConfig,
    bundles: tuple[EvaluationBundle, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        if bundle.evaluation_id.seed != seed:
            continue
        policy = _new_policy(seed, config.action_value_config)
        evaluation = _evaluate(
            policy, learner, bundle.fixtures, reset_before_decision=False
        )
        rows.append(_evaluation_payload(evaluation, bundle.evaluation_id.evaluation_id))
    return rows


def _score_ridge(
    seed: int,
    reference: object,
    config: AnonymousCreditConfig,
    bundles: tuple[EvaluationBundle, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        if bundle.evaluation_id.seed != seed:
            continue
        policy = _new_policy(seed, config.action_value_config)
        hidden = np.asarray(
            [
                _decision_hidden(policy, episode, reset_before_decision=False)
                for episode in bundle.fixtures
            ],
            dtype=np.float64,
        )
        values = reference.predict(hidden)
        correct = np.asarray(
            [episode.correct_action_index for episode in bundle.fixtures], dtype=np.int64
        )
        predicted = np.argmax(values, axis=1)
        margins = values[np.arange(len(correct)), correct] - values[
            np.arange(len(correct)), 1 - correct
        ]
        per_delay = []
        for delay in range(1, 6):
            mask = np.asarray([episode.delay_steps == delay for episode in bundle.fixtures])
            per_delay.append(
                {
                    "cue_delay": delay,
                    "correct": int(np.sum(predicted[mask] == correct[mask])),
                    "total": int(np.sum(mask)),
                }
            )
        rows.append(
            {
                "evaluation_id": bundle.evaluation_id.evaluation_id,
                "overall": {
                    "correct": int(np.sum(predicted == correct)),
                    "total": len(correct),
                },
                "per_delay": per_delay,
                "margin_mean": float(np.mean(margins)),
                "margin_p10": float(np.percentile(margins, 10)),
                "margin_minimum": float(np.min(margins)),
            }
        )
    return rows


def _array_digest(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _train_ridge_reference(seed: int, config: AnonymousCreditConfig) -> object:
    fixtures = _build_fixture_bundle(seed, config.action_value_config)
    policy = _new_policy(seed, config.action_value_config)
    hidden = np.asarray(
        [
            _decision_hidden(policy, episode, reset_before_decision=False)
            for episode in fixtures.training
        ],
        dtype=np.float64,
    )
    labels = tuple(episode.correct_action_index for episode in fixtures.training)
    return fit_supervised_ridge(hidden, labels, regularization=1e-6)


def _train_immediate_reference(seed: int, config: AnonymousCreditConfig) -> NormalizedActionValue:
    av_config = config.action_value_config
    fixtures = _build_fixture_bundle(seed, av_config)
    policy = _new_policy(seed, av_config)
    learner = NormalizedActionValue(config.hidden_size, 2, config.step_size)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    _train_normal(policy, learner, DelayedCueTask(), fixtures.training, rng, av_config)
    return learner


def _train_source_visible_reference(
    seed: int, config: AnonymousCreditConfig
) -> SourceVisibleDelayedReference:
    av_config = config.action_value_config
    fixtures = _build_fixture_bundle(seed, av_config)
    policy = _new_policy(seed, av_config)
    task = DelayedCueTask()
    learner = SourceVisibleDelayedReference(config.hidden_size, step_size=config.step_size)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    schedule = build_hidden_delay_schedule(
        seed, config.training_decisions, support=config.delay_support
    )
    aggregator = AnonymousRewardAggregator(schedule)
    for step, episode in enumerate(fixtures.training):
        hidden = _decision_hidden(policy, episode, reset_before_decision=False)
        action = int(rng.integers(2))
        learner.record_decision(step, action, hidden)
        aggregator.enqueue(step, action, float(task.reward(episode, action)))
        feedback = aggregator.feedback_at(step)
        learner.deliver(
            tuple(record.source_step for record in feedback.records),
            tuple(record.reward for record in feedback.records),
        )
    for delivery_step in range(config.training_decisions, max(schedule.due_steps) + 1):
        feedback = aggregator.feedback_at(delivery_step)
        learner.deliver(
            tuple(record.source_step for record in feedback.records),
            tuple(record.reward for record in feedback.records),
        )
    if aggregator.pending_count or learner.pending_count:
        raise RuntimeError("source-visible reference ended with pending records")
    return learner


def _accounting_summary(replay: ReplayResult, arm: str, config: AnonymousCreditConfig) -> dict[str, object]:
    if arm == "td0":
        return {
            "arm": arm,
            "real_steps": len(replay.steps),
            "drain_steps": len(replay.drain_steps),
            "drain_weight_changes": sum(
                not np.array_equal(step.weights_before, step.weights_after)
                for step in replay.drain_steps
            ),
        }
    q_by_source = {
        step.step: float(step.action_values[step.action]) for step in replay.steps
    }
    residual_rows: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []
    for step in replay.steps:
        source_steps = tuple(record["source_step"] for record in step.metadata["records"])
        source_prediction = sum(q_by_source[source] for source in source_steps)
        terms = residual_terms(step.feedback, float(step.prediction), source_prediction)
        residual_rows.append(
            {
                "step": step.step,
                "feedback_minus_prediction": terms.feedback_minus_prediction,
                "feedback_minus_source_prediction": terms.feedback_minus_source_prediction,
                "source_minus_prediction": terms.source_minus_prediction,
                "squared_cross_term": terms.squared_cross_term,
            }
        )
        reference_direction = np.zeros_like(step.weights_after)
        for record in step.metadata["records"]:
            source = int(record["source_step"])
            source_capture = replay.steps[source]
            feature = np.concatenate(
                (source_capture.hidden, np.array([1.0], dtype=np.float64))
            )
            denominator = float(np.dot(feature, feature))
            action = source_capture.action
            reward = float(record["reward"])
            reference_direction[action] += (reward - q_by_source[source]) * feature / denominator
        actual_update = step.weights_after - step.weights_before
        metric = cosine_metric(actual_update, reference_direction)
        cosine_rows.append({"step": step.step, **metric.to_dict()})
    max_drain_residual = 0.0
    if replay.drain_steps:
        first = replay.drain_steps[0]
        if first.eligibility is not None and first.prediction_trace is not None:
            expected = expected_drain_delta(
                tuple(step.feedback for step in replay.drain_steps),
                first.prediction_trace,
                first.eligibility,
                config.step_size,
            )
            actual = replay.drain_steps[-1].weights_after - first.weights_before
            max_drain_residual = maximum_accounting_residual(actual, expected)
    return {
        "arm": arm,
        "residuals": residual_rows,
        "update_cosines": cosine_rows,
        "maximum_drain_residual": max_drain_residual,
    }


def _write_row(attempt_dir: Path, index: int, payload: dict[str, object]) -> str:
    path = attempt_dir / "rows" / f"{index:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite completed row {path.name}")
    path.write_bytes(canonical_json_bytes(payload))
    return sha256_file(path)


def run_registered_measurement(
    root: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    attempt_id: str,
    approve_measurement: bool,
) -> Path:
    if not approve_measurement:
        raise PermissionError("registered measurement requires explicit --approve-measurement")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("manifest digest does not match the approved digest")
    manifest = load_json_object(manifest_path)
    if manifest.get("profile") != REGISTERED_PROFILE or manifest.get("registered_valid_profile") is not True:
        raise RuntimeError("registered measurement requires registered-v1 manifest")
    if manifest.get("implementation_sha") != current_commit(root):
        raise RuntimeError("measurement checkout differs from the sealed implementation SHA")
    input_payload = manifest.get("input")
    if not isinstance(input_payload, dict) or not isinstance(input_payload.get("input_hashes"), dict):
        raise RuntimeError("manifest input hashes are missing")
    assert_hashes(root, input_payload["input_hashes"])
    if manifest.get("environment") != environment_snapshot():
        raise RuntimeError("measurement environment differs from the sealed environment")

    attempt_dir = manifest_path.parent / f"attempt-{attempt_id}"
    if attempt_dir.exists():
        raise FileExistsError("attempt directory already exists")
    attempt_dir.mkdir(parents=True)
    execution: dict[str, object] = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "manifest_sha256": expected_manifest_sha256,
        "complete": False,
        "diagnostic_valid": False,
        "d0": [],
        "models": [],
        "accounting": [],
        "first_failure": None,
    }
    execution_path = attempt_dir / "execution-manifest.json"

    config = _registered_config()
    try:
        replays: dict[tuple[int, str, str], ReplayResult] = {}
        for seed in REGISTERED_SEEDS:
            for condition in REGISTERED_CONDITIONS:
                for arm in REGISTERED_ARMS:
                    replay = run_diagnostic_replay(
                        seed, arm, config, condition, attempt_id=attempt_id
                    )
                    replays[(seed, arm, condition)] = replay
                    execution["d0"].append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "condition": condition,
                            "parameter_digest": replay.final_parameter_digest,
                            "scalar_call_count": len(replay.scalar_calls),
                        }
                    )
        execution_path.write_bytes(canonical_json_bytes(execution))

        bundles = build_evaluation_bundles(config)
        normal_rewards = {
            seed: _execute_training(seed, "td0", config, immediate_control=False).protocol.rewards
            for seed in REGISTERED_SEEDS
        }
        model_keys: list[str] = []
        main_score_keys: list[str] = []
        for index, model in enumerate(profile_model_ids(REGISTERED_PROFILE)):
            model_keys.append(model.stable_key())
            row: dict[str, object] = {"model_id": model.to_dict()}
            if model.family in ("original", "permutation"):
                reward_override = None
                if model.family == "original" and model.condition == "original_shuffle":
                    permutation = original_block10_permutation(model.seed, config.training_decisions)
                    reward_override = apply_permutation(normal_rewards[model.seed], permutation)
                    schedule = build_hidden_delay_schedule(model.seed, config.training_decisions, support=config.delay_support)
                    row["provenance"] = donor_summary(donor_rows(permutation, schedule.due_steps))
                elif model.family == "permutation":
                    permutation = diagnostic_permutation(
                        model.seed, config.training_decisions, str(model.mode), int(model.replicate)
                    )
                    reward_override = apply_permutation(normal_rewards[model.seed], permutation)
                    schedule = build_hidden_delay_schedule(model.seed, config.training_decisions, support=config.delay_support)
                    row["provenance"] = donor_summary(donor_rows(permutation, schedule.due_steps))
                execution_result = _execute_training(
                    model.seed,
                    str(model.arm),
                    config,
                    immediate_control=False,
                    reward_override=reward_override,
                )
                row["parameter_digest"] = execution_result.protocol.parameter_digest
                scores = _score_learner(model.seed, execution_result.learner, config, bundles)
            elif model.reference_kind == "supervised_ridge":
                reference = _train_ridge_reference(model.seed, config)
                row["parameter_digest"] = _array_digest(reference.weights)
                row["information_access"] = "training labels"
                scores = _score_ridge(model.seed, reference, config, bundles)
            elif model.reference_kind == "immediate_identified":
                reference = _train_immediate_reference(model.seed, config)
                row["parameter_digest"] = reference.parameter_digest()
                row["information_access"] = "immediate identified scalar reward"
                scores = _score_learner(model.seed, reference, config, bundles)
            else:
                reference = _train_source_visible_reference(model.seed, config)
                row["parameter_digest"] = _array_digest(reference.parameter_snapshot())
                row["information_access"] = "individual delayed source identities and rewards"
                scores = _score_learner(model.seed, reference, config, bundles)
            row["scores"] = scores
            for score in scores:
                main_score_keys.append(
                    f"{model.stable_key()}/evaluation={score['evaluation_id']}"
                )
            row["sha256"] = _write_row(attempt_dir, index, row)
            execution["models"].append(
                {"model_key": model.stable_key(), "row_sha256": row["sha256"]}
            )
            execution_path.write_bytes(canonical_json_bytes(execution))

        expected_models = tuple(ModelId.from_dict(item).stable_key() for item in manifest["model_ids"])
        validate_complete_keys(expected_models, tuple(model_keys))
        validate_complete_keys(tuple(manifest["main_score_keys"]), tuple(main_score_keys))
        execution["accounting"] = [
            {
                "seed": seed,
                "condition": condition,
                **_accounting_summary(replay, arm, config),
            }
            for (seed, arm, condition), replay in sorted(replays.items())
        ]
        assert_hashes(root, input_payload["input_hashes"])
        execution["complete"] = True
        execution["diagnostic_valid"] = True
        execution_path.write_bytes(canonical_json_bytes(execution))
        return attempt_dir
    except Exception as exc:
        execution["first_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        execution["complete"] = False
        execution["diagnostic_valid"] = False
        execution_path.write_bytes(canonical_json_bytes(execution))
        raise


def verify_attempt(manifest_path: Path, attempt_dir: Path) -> dict[str, object]:
    manifest = load_json_object(manifest_path)
    execution = load_json_object(attempt_dir / "execution-manifest.json")
    if execution.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("execution manifest is bound to a different input manifest")
    if execution.get("complete") is not True or execution.get("diagnostic_valid") is not True:
        raise RuntimeError("diagnostic attempt is incomplete or invalid")
    expected_models = tuple(ModelId.from_dict(item).stable_key() for item in manifest["model_ids"])
    model_rows = execution.get("models")
    if not isinstance(model_rows, list):
        raise RuntimeError("execution model rows are missing")
    validate_complete_keys(expected_models, tuple(str(row["model_key"]) for row in model_rows))
    return execution
