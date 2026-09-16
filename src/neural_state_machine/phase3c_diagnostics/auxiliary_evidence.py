from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_state_machine.action_value_benchmark import _evaluate, _new_policy
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.reward_learning import _decision_hidden

from .auxiliary import drain_weight_snapshots, score_weight_snapshot
from .contracts import ModelId
from .evaluation import EvaluationBundle
from .references import RidgeReference
from .replay import ReplayResult


@dataclass(frozen=True, slots=True)
class AuxiliaryEvidence:
    reset_scores: tuple[dict[str, object], ...]
    reverse_checks: tuple[dict[str, object], ...]
    drain_scores: tuple[dict[str, object], ...]


def _bundles_for_seed(
    bundles: tuple[EvaluationBundle, ...], seed: int
) -> tuple[EvaluationBundle, ...]:
    selected = tuple(bundle for bundle in bundles if bundle.evaluation_id.seed == seed)
    if len(selected) != 9:
        raise RuntimeError("auxiliary evidence requires exactly nine evaluation bundles per seed")
    if {bundle.evaluation_id.evaluation_id for bundle in selected} != set(range(-1, 8)):
        raise RuntimeError("auxiliary evaluation IDs differ from the sealed nine-set grid")
    return selected


def original_auxiliary_evidence(
    model: ModelId,
    replay: ReplayResult,
    bundles: tuple[EvaluationBundle, ...],
    config: AnonymousCreditConfig,
) -> AuxiliaryEvidence:
    if not isinstance(model, ModelId) or model.family != "original":
        raise ValueError("model must be an original ModelId")
    if not isinstance(replay, ReplayResult):
        raise ValueError("replay must be a ReplayResult")
    if not isinstance(config, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")
    if replay.protocol.seed != model.seed or replay.protocol.arm != model.arm:
        raise ValueError("replay identity differs from original model")
    if not replay.drain_steps:
        raise RuntimeError("original delayed replay must contain drain steps")

    selected = _bundles_for_seed(bundles, model.seed)
    final_weights = replay.drain_steps[-1].weights_after
    reset_rows: list[dict[str, object]] = []
    reverse_rows: list[dict[str, object]] = []
    for bundle in selected:
        evaluation_id = bundle.evaluation_id.evaluation_id
        score = score_weight_snapshot(
            model.seed, final_weights, bundle.fixtures, config
        )
        reset_rows.append(
            {
                "key": f"reset/{model.stable_key()}/evaluation={evaluation_id}",
                "evaluation_id": evaluation_id,
                "overall": {"correct": score.reset_correct, "total": score.total},
            }
        )
        reverse_rows.append(
            {
                "key": f"reverse/{model.stable_key()}/evaluation={evaluation_id}",
                "evaluation_id": evaluation_id,
                "order_invariant": score.canonical_actions
                == score.reversed_actions_mapped,
                "overall": {"correct": score.canonical_correct, "total": score.total},
            }
        )

    drain_rows: list[dict[str, object]] = []
    for label, weights in drain_weight_snapshots(replay, str(model.arm)):
        for bundle in selected:
            evaluation_id = bundle.evaluation_id.evaluation_id
            score = score_weight_snapshot(model.seed, weights, bundle.fixtures, config)
            drain_rows.append(
                {
                    "key": (
                        f"drain/{model.stable_key()}/{label}/evaluation={evaluation_id}"
                    ),
                    "snapshot": label,
                    "evaluation_id": evaluation_id,
                    "overall": {
                        "correct": score.canonical_correct,
                        "total": score.total,
                    },
                }
            )

    return AuxiliaryEvidence(
        reset_scores=tuple(reset_rows),
        reverse_checks=tuple(reverse_rows),
        drain_scores=tuple(drain_rows),
    )


def _ridge_reset_payload(
    seed: int,
    reference: RidgeReference,
    bundle: EvaluationBundle,
    config: AnonymousCreditConfig,
) -> dict[str, object]:
    policy = _new_policy(seed, config.action_value_config)
    hidden = np.asarray(
        [
            _decision_hidden(policy, episode, reset_before_decision=True)
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
        mask = np.asarray(
            [episode.delay_steps == delay for episode in bundle.fixtures], dtype=bool
        )
        per_delay.append(
            {
                "cue_delay": delay,
                "correct": int(np.sum(predicted[mask] == correct[mask])),
                "total": int(np.sum(mask)),
            }
        )
    return {
        "overall": {
            "correct": int(np.sum(predicted == correct)),
            "total": len(correct),
        },
        "per_delay": per_delay,
        "margin_mean": float(np.mean(margins)),
        "margin_p10": float(np.percentile(margins, 10)),
        "margin_minimum": float(np.min(margins)),
    }


def _learner_reset_payload(
    seed: int,
    reference: object,
    bundle: EvaluationBundle,
    config: AnonymousCreditConfig,
) -> dict[str, object]:
    evaluation = _evaluate(
        _new_policy(seed, config.action_value_config),
        reference,
        bundle.fixtures,
        reset_before_decision=True,
    )
    return {
        "overall": {
            "correct": evaluation.overall.correct,
            "total": evaluation.overall.total,
        },
        "per_delay": [
            {
                "cue_delay": delay,
                "correct": count.correct,
                "total": count.total,
            }
            for delay, count in evaluation.per_delay
        ],
        "margin_mean": evaluation.margin_mean,
        "margin_p10": evaluation.margin_p10,
        "margin_minimum": evaluation.margin_minimum,
    }


def reference_reset_evidence(
    model: ModelId,
    reference: object,
    bundles: tuple[EvaluationBundle, ...],
    config: AnonymousCreditConfig,
) -> tuple[dict[str, object], ...]:
    if not isinstance(model, ModelId) or model.family != "reference":
        raise ValueError("model must be a reference ModelId")
    if not isinstance(config, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")
    selected = _bundles_for_seed(bundles, model.seed)
    rows: list[dict[str, object]] = []
    for bundle in selected:
        evaluation_id = bundle.evaluation_id.evaluation_id
        if isinstance(reference, RidgeReference):
            payload = _ridge_reset_payload(model.seed, reference, bundle, config)
        else:
            payload = _learner_reset_payload(model.seed, reference, bundle, config)
        rows.append(
            {
                "key": f"reset/{model.stable_key()}/evaluation={evaluation_id}",
                "evaluation_id": evaluation_id,
                **payload,
            }
        )
    return tuple(rows)
