from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .contracts import ModelId
from .statistics import summarize_scores

_ALLOWED_STATUSES = {"supported", "not_supported", "unresolved"}


@dataclass(frozen=True, slots=True)
class AttributionCandidate:
    name: str
    status: str
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("candidate name must be non-empty")
        if self.status not in _ALLOWED_STATUSES:
            raise ValueError("candidate status must be supported, not_supported or unresolved")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, str) or not item for item in self.evidence
        ):
            raise ValueError("candidate evidence must be a tuple of non-empty field paths")

    def to_payload(self) -> dict[str, object]:
        return {"name": self.name, "status": self.status, "evidence": list(self.evidence)}


def _score_rows(row: dict[str, object], model: ModelId) -> list[dict[str, object]]:
    scores = row.get("scores")
    if not isinstance(scores, list) or len(scores) != 9:
        raise RuntimeError(f"report requires nine scores for {model.stable_key()}")
    ids: list[int] = []
    for score in scores:
        if not isinstance(score, dict):
            raise RuntimeError(f"score row must be an object for {model.stable_key()}")
        evaluation_id = score.get("evaluation_id")
        if type(evaluation_id) is not int or not -1 <= evaluation_id <= 7:
            raise RuntimeError(f"invalid evaluation_id for {model.stable_key()}")
        overall = score.get("overall")
        if not isinstance(overall, dict):
            raise RuntimeError(f"overall score is missing for {model.stable_key()}")
        correct = overall.get("correct")
        total = overall.get("total")
        if (
            type(correct) is not int
            or type(total) is not int
            or total <= 0
            or not 0 <= correct <= total
        ):
            raise RuntimeError(f"overall score is invalid for {model.stable_key()}")
        ids.append(evaluation_id)
    if set(ids) != set(range(-1, 8)) or len(ids) != len(set(ids)):
        raise RuntimeError(f"score evaluation grid is incomplete for {model.stable_key()}")
    return scores


def _original_evaluation_correct(row: dict[str, object], model: ModelId) -> int:
    for score in _score_rows(row, model):
        if score["evaluation_id"] == -1:
            return int(score["overall"]["correct"])
    raise RuntimeError(f"original evaluation score is missing for {model.stable_key()}")


def _undefined_reasons(value: object, result: Counter[str]) -> None:
    if isinstance(value, dict):
        if "value" in value and "reason" in value and value.get("value") is None:
            reason = value.get("reason")
            if isinstance(reason, str) and reason:
                result[reason] += 1
                return
        for nested in value.values():
            _undefined_reasons(nested, result)
    elif isinstance(value, list):
        for nested in value:
            _undefined_reasons(nested, result)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"{name} must be finite and non-negative")
    return result


def _accounting_evidence(execution: dict[str, object]) -> dict[str, object]:
    rows = execution.get("accounting")
    if not isinstance(rows, list):
        raise RuntimeError("execution accounting rows are missing")
    maximum_drain = 0.0
    maximum_eligibility = 0.0
    maximum_update = 0.0
    undefined = Counter()
    trajectories: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("accounting trajectory must be an object")
        drain = _finite_nonnegative(
            row.get("maximum_drain_residual", 0.0), "maximum_drain_residual"
        )
        maximum_drain = max(maximum_drain, drain)
        history = row.get("history_components", [])
        if not isinstance(history, list):
            raise RuntimeError("history_components must be a list")
        trajectory_eligibility = 0.0
        trajectory_update = 0.0
        trajectory_undefined = Counter()
        for component in history:
            if not isinstance(component, dict):
                raise RuntimeError("history component must be an object")
            eligibility = _finite_nonnegative(
                component.get("eligibility_reconstruction_max_residual", 0.0),
                "eligibility_reconstruction_max_residual",
            )
            update = _finite_nonnegative(
                component.get("update_reconstruction_max_residual", 0.0),
                "update_reconstruction_max_residual",
            )
            trajectory_eligibility = max(trajectory_eligibility, eligibility)
            trajectory_update = max(trajectory_update, update)
            _undefined_reasons(component, trajectory_undefined)
        maximum_eligibility = max(maximum_eligibility, trajectory_eligibility)
        maximum_update = max(maximum_update, trajectory_update)
        undefined.update(trajectory_undefined)
        trajectories.append(
            {
                "seed": row.get("seed"),
                "condition": row.get("condition"),
                "arm": row.get("arm"),
                "maximum_drain_residual": drain,
                "maximum_eligibility_reconstruction_residual": trajectory_eligibility,
                "maximum_update_reconstruction_residual": trajectory_update,
                "undefined_metric_reasons": dict(sorted(trajectory_undefined.items())),
            }
        )
    return {
        "trajectory_count": len(rows),
        "maximum_drain_residual": maximum_drain,
        "maximum_eligibility_reconstruction_residual": maximum_eligibility,
        "maximum_update_reconstruction_residual": maximum_update,
        "undefined_metric_reasons": dict(sorted(undefined.items())),
        "trajectories": trajectories,
        "raw_evidence_path": "execution-manifest.json#/accounting",
    }


def build_evidence_summary(
    model_rows: list[dict[str, object]],
    execution: dict[str, object],
    auxiliary: dict[str, object],
) -> dict[str, object]:
    """Index verified diagnostic evidence without fitting or rescoring any model."""
    if not isinstance(model_rows, list) or not model_rows:
        raise RuntimeError("report requires non-empty model rows")
    if not isinstance(execution, dict) or execution.get("diagnostic_valid") is not True:
        raise RuntimeError("report requires a verified diagnostic execution")
    if not isinstance(auxiliary, dict):
        raise RuntimeError("report auxiliary evidence is missing")

    parsed: list[tuple[ModelId, dict[str, object]]] = []
    keys: set[str] = set()
    main_score_count = 0
    for row in model_rows:
        if not isinstance(row, dict) or not isinstance(row.get("model_id"), dict):
            raise RuntimeError("report model row is malformed")
        model = ModelId.from_dict(row["model_id"])
        key = model.stable_key()
        if key in keys:
            raise RuntimeError(f"duplicate report model row: {key}")
        keys.add(key)
        scores = _score_rows(row, model)
        main_score_count += len(scores)
        parsed.append((model, row))

    model_scores = [
        {
            "model_key": model.stable_key(),
            "model_id": model.to_dict(),
            "parameter_digest": row.get("parameter_digest"),
            "information_access": row.get("information_access"),
            "scores": row["scores"],
        }
        for model, row in parsed
    ]

    original_shuffle: dict[tuple[int, str], int] = {}
    permutation_groups: dict[tuple[int, str, str], list[tuple[int, int]]] = defaultdict(list)
    references: list[dict[str, object]] = []
    permutation_count = 0
    for model, row in parsed:
        if model.family == "original" and model.condition == "original_shuffle":
            original_shuffle[(model.seed, str(model.arm))] = _original_evaluation_correct(
                row, model
            )
        elif model.family == "permutation":
            permutation_count += 1
            permutation_groups[(model.seed, str(model.arm), str(model.mode))].append(
                (int(model.replicate), _original_evaluation_correct(row, model))
            )
        elif model.family == "reference":
            references.append(
                {
                    "model_key": model.stable_key(),
                    "seed": model.seed,
                    "reference_kind": model.reference_kind,
                    "information_access": row.get("information_access"),
                    "scores": row["scores"],
                }
            )

    distributions: list[dict[str, object]] = []
    for (seed, arm, mode), values in sorted(permutation_groups.items()):
        ordered = sorted(values)
        replicates = tuple(replicate for replicate, _ in ordered)
        if replicates != tuple(range(32)):
            raise RuntimeError(
                f"permutation distribution must contain exactly 32 replicates: "
                f"seed={seed}, arm={arm}, mode={mode}"
            )
        scores = tuple(score for _, score in ordered)
        try:
            control = original_shuffle[(seed, arm)]
        except KeyError as exc:
            raise RuntimeError(
                f"original shuffled control is missing for seed={seed}, arm={arm}"
            ) from exc
        distributions.append(
            {
                "seed": seed,
                "arm": arm,
                "mode": mode,
                "replicate_scores": list(scores),
                "summary": summarize_scores(scores),
                "original_shuffle_score": control,
                "original_shuffle_rank_descending": 1 + sum(score > control for score in scores),
                "rank_is_descriptive_not_p_value": True,
            }
        )

    auxiliary_counts: dict[str, int] = {}
    for name in ("reset_scores", "reverse_checks", "drain_scores"):
        values = auxiliary.get(name)
        if not isinstance(values, list):
            raise RuntimeError(f"auxiliary {name} rows are missing")
        auxiliary_counts[name] = len(values)

    return {
        "counts": {
            "models": len(parsed),
            "main_scores": main_score_count,
            "permutation_models": permutation_count,
            "privileged_references": len(references),
        },
        "model_scores": model_scores,
        "permutation_distributions": distributions,
        "privileged_references": references,
        "auxiliary_counts": auxiliary_counts,
        "accounting": _accounting_evidence(execution),
        "claim_boundary": (
            "Permutation ranks and associations are descriptive; privileged-reference "
            "scores and observer-only accounting do not satisfy the original anonymous gate."
        ),
    }


def build_report_summary(
    *,
    diagnostic_valid: bool,
    original_behavior_passed: bool,
    candidates: tuple[AttributionCandidate, ...],
    limitations: tuple[str, ...] = (),
) -> dict[str, object]:
    if type(diagnostic_valid) is not bool or type(original_behavior_passed) is not bool:
        raise ValueError("diagnostic/original status flags must be booleans")
    if original_behavior_passed:
        raise ValueError("Phase 3C frozen behavior_passed must remain false")
    if not isinstance(candidates, tuple) or len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidates must be a tuple with unique names")
    if any(not isinstance(item, str) or not item for item in limitations):
        raise ValueError("limitations must contain non-empty strings")
    return {
        "diagnostic_valid": diagnostic_valid,
        "original_phase3c": {
            "formal_valid": True,
            "protocol_valid": True,
            "behavior_passed": False,
            "all_passed": False,
        },
        "candidates": [candidate.to_payload() for candidate in candidates],
        "limitations": list(limitations),
        "claim_boundary": (
            "Diagnostic validity does not change the frozen Phase 3C behavioral gate; "
            "correlation, rank and privileged-reference scores are not causal proof."
        ),
    }


def render_markdown(summary: dict[str, object]) -> str:
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    lines = ["# Phase 3C Failure-Attribution Diagnostic Report", ""]
    lines.append(f"Diagnostic valid: `{summary.get('diagnostic_valid')}`")
    evidence = summary.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("counts"), dict):
        counts = evidence["counts"]
        lines.extend(
            [
                "",
                "## Evidence index",
                "",
                f"- Models: `{counts.get('models')}`",
                f"- Main scores: `{counts.get('main_scores')}`",
                f"- Permutation models: `{counts.get('permutation_models')}`",
                f"- Privileged references: `{counts.get('privileged_references')}`",
            ]
        )
    lines.extend(["", "## Attribution candidates", ""])
    candidates = summary.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("summary candidates must be a list")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate rows must be objects")
        lines.append(f"- **{candidate.get('name')}** — `{candidate.get('status')}`")
    lines.extend(["", "## Claim boundary", "", str(summary.get("claim_boundary", "")), ""])
    return "\n".join(lines)
