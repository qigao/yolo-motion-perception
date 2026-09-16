from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .statistics import pearson_metric

_MODE_CODE = {"block10": 0, "global": 1}
_ORIGINAL_SHUFFLE_LINEAGE = 0x33534846
_DIAGNOSTIC_LINEAGE = 0x33434641


@dataclass(frozen=True, slots=True)
class DonorAssignment:
    slot: int
    donor: int
    due_step: int

    @property
    def is_fixed_point(self) -> bool:
        return self.slot == self.donor

    @property
    def is_current_decision_donor(self) -> bool:
        return self.donor == self.due_step

    @property
    def donor_relative_to_delivery(self) -> str:
        if self.donor < self.due_step:
            return "past"
        if self.donor == self.due_step:
            return "current"
        return "future"


def _positive_count(count: object) -> int:
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    return count


def _replicate(replicate: object) -> int:
    if type(replicate) is not int or not 0 <= replicate < 32:
        raise ValueError("replicate must be an integer in [0, 31]")
    return replicate


def _validate_permutation(permutation: tuple[int, ...]) -> None:
    if any(type(value) is not int for value in permutation):
        raise ValueError("permutation must contain integer indices")
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("permutation must contain every index exactly once")


def donor_rows(
    permutation: tuple[int, ...], due_steps: tuple[int, ...]
) -> tuple[DonorAssignment, ...]:
    if not isinstance(permutation, tuple) or not isinstance(due_steps, tuple):
        raise ValueError("permutation and due_steps must be tuples")
    if len(permutation) != len(due_steps):
        raise ValueError("permutation and due_steps lengths must match")
    _validate_permutation(permutation)
    if any(type(value) is not int or value < 0 for value in due_steps):
        raise ValueError("due_steps must contain non-negative integers")
    return tuple(
        DonorAssignment(slot=slot, donor=donor, due_step=due_steps[slot])
        for slot, donor in enumerate(permutation)
    )


def _block10_permutation(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    if count % 10:
        raise ValueError("block10 permutation requires count divisible by 10")
    result: list[int] = []
    for start in range(0, count, 10):
        result.extend(start + int(index) for index in rng.permutation(10))
    return tuple(result)


def diagnostic_permutation(
    seed: int, count: int, mode: str, replicate: int
) -> tuple[int, ...]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    resolved_count = _positive_count(count)
    resolved_replicate = _replicate(replicate)
    try:
        mode_code = _MODE_CODE[mode]
    except (KeyError, TypeError) as exc:
        raise ValueError("mode must be block10 or global") from exc
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [seed, _DIAGNOSTIC_LINEAGE, 1, mode_code, resolved_replicate]
            )
        )
    )
    if mode == "block10":
        result = _block10_permutation(rng, resolved_count)
    else:
        result = tuple(int(index) for index in rng.permutation(resolved_count))
    _validate_permutation(result)
    return result


def original_block10_permutation(seed: int, count: int) -> tuple[int, ...]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    resolved_count = _positive_count(count)
    rng = np.random.default_rng(np.random.SeedSequence([seed, _ORIGINAL_SHUFFLE_LINEAGE]))
    result = _block10_permutation(rng, resolved_count)
    _validate_permutation(result)
    return result


def identity_permutation(count: int) -> tuple[int, ...]:
    return tuple(range(_positive_count(count)))


def apply_permutation(values: tuple[float, ...], permutation: tuple[int, ...]) -> tuple[float, ...]:
    if len(values) != len(permutation):
        raise ValueError("values and permutation lengths must match")
    _validate_permutation(permutation)
    return tuple(float(values[index]) for index in permutation)


def expected_current_donor_matches(due_steps: tuple[int, ...], mode: str) -> float:
    if not isinstance(due_steps, tuple) or not due_steps:
        raise ValueError("due_steps must be a non-empty tuple")
    if any(type(value) is not int or value < 0 for value in due_steps):
        raise ValueError("due_steps must contain non-negative integers")
    count = len(due_steps)
    eligible = sum(
        due_step < count
        and (mode != "block10" or slot // 10 == due_step // 10)
        for slot, due_step in enumerate(due_steps)
    )
    if mode == "block10":
        return eligible / 10.0
    if mode == "global":
        return eligible / float(count)
    raise ValueError("mode must be block10 or global")


def nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    if not values:
        raise ValueError("values must not be empty")
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise ValueError("quantile must be numeric")
    q = float(quantile)
    if not math.isfinite(q) or not 0.0 < q <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return int(ordered[math.ceil(q * len(ordered)) - 1])


def donor_summary(rows: tuple[DonorAssignment, ...]) -> dict[str, object]:
    if not rows:
        raise ValueError("rows must not be empty")
    return {
        "fixed_points": sum(row.is_fixed_point for row in rows),
        "current_decision_matches": sum(row.is_current_decision_donor for row in rows),
        "relative_to_delivery": dict(Counter(row.donor_relative_to_delivery for row in rows)),
        "displacements": dict(Counter(row.donor - row.slot for row in rows)),
    }


def _finite_values(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain numeric values")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must contain finite values")
        result.append(resolved)
    return tuple(result)


def _binary_values(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError(f"{name} must contain only 0 or 1")
    return values


def _no_pairs_metric() -> dict[str, object]:
    return {"value": None, "reason": "no_valid_pairs", "sample_count": 0}


def _mean_product(left: tuple[float, ...], right: tuple[float, ...]) -> dict[str, object]:
    if len(left) != len(right):
        raise ValueError("association pair lengths must match")
    if not left:
        return _no_pairs_metric()
    return {
        "value": float(sum(a * b for a, b in zip(left, right, strict=True)) / len(left)),
        "reason": None,
        "sample_count": len(left),
    }


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> dict[str, object]:
    if len(left) != len(right):
        raise ValueError("association pair lengths must match")
    if not left:
        return _no_pairs_metric()
    return pearson_metric(left, right).to_dict()


def _binary_cells(
    feedback: tuple[float, ...], values: tuple[int, ...]
) -> dict[str, dict[str, object]]:
    if len(feedback) != len(values):
        raise ValueError("cell feedback/value lengths must match")
    rows: dict[str, dict[str, object]] = {}
    for cell in (0, 1):
        selected = tuple(
            feedback[index] for index, value in enumerate(values) if value == cell
        )
        rows[str(cell)] = {
            "count": len(selected),
            "feedback_sum": float(sum(selected)),
            "feedback_mean": None if not selected else float(sum(selected) / len(selected)),
            "signed_product_mean": (
                None
                if not selected
                else float(sum(value * (2 * cell - 1) for value in selected) / len(selected))
            ),
        }
    return rows


def provenance_association_audit(
    *,
    permutation: tuple[int, ...],
    due_steps: tuple[int, ...],
    latent_rewards: tuple[float, ...],
    actions: tuple[int, ...],
    labels: tuple[int, ...],
    actual_feedback_calls: tuple[float, ...],
) -> dict[str, object]:
    """Build observer-only D1 provenance/association evidence from fixed streams."""
    assignments = donor_rows(permutation, due_steps)
    count = len(assignments)
    rewards = _finite_values(latent_rewards, "latent_rewards")
    resolved_actions = _binary_values(actions, "actions")
    resolved_labels = _binary_values(labels, "labels")
    feedback_calls = _finite_values(actual_feedback_calls, "actual_feedback_calls")
    if len(rewards) != count or len(resolved_actions) != count or len(resolved_labels) != count:
        raise ValueError("slot, reward, action and label lengths must match")
    if any(row.due_step < row.slot for row in assignments):
        raise ValueError("due steps must not precede their source slots")

    call_count = max(count, max(due_steps) + 1)
    if len(feedback_calls) != call_count:
        raise ValueError("actual feedback call count differs from the delivery timeline")
    delivered_rewards = tuple(rewards[row.donor] for row in assignments)
    buckets: list[list[int]] = [[] for _ in range(call_count)]
    for row in assignments:
        buckets[row.due_step].append(row.slot)
    reconstructed = tuple(
        float(sum(delivered_rewards[slot] for slot in bucket)) for bucket in buckets
    )
    if any(
        expected != actual
        for expected, actual in zip(reconstructed, feedback_calls, strict=True)
    ):
        raise ValueError("actual feedback calls differ from reconstructed slot deliveries")

    provenance_rows = tuple(
        {
            "slot": row.slot,
            "donor": row.donor,
            "due_step": row.due_step,
            "cue_delay": row.due_step - row.slot,
            "slot_latent_reward": rewards[row.slot],
            "delivered_reward": delivered_rewards[row.slot],
            "slot_action": resolved_actions[row.slot],
            "slot_label": resolved_labels[row.slot],
            "donor_action": resolved_actions[row.donor],
            "donor_label": resolved_labels[row.donor],
            "delivery_multiplicity": len(buckets[row.due_step]),
            "actual_feedback_call": feedback_calls[row.due_step],
            "donor_relative_to_delivery": row.donor_relative_to_delivery,
        }
        for row in assignments
    )
    summary = donor_summary(assignments)
    summary["same_block_donors"] = sum(
        row.slot // 10 == row.donor // 10 for row in assignments
    )

    multiplicities = tuple(len(bucket) for bucket in buckets)
    call_classes = {
        "real_decision_calls": count,
        "drain_calls": call_count - count,
        "no_arrival_zero": sum(
            multiplicity == 0 and feedback == 0.0
            for multiplicity, feedback in zip(multiplicities, feedback_calls, strict=True)
        ),
        "cancellation_to_zero": sum(
            multiplicity > 0 and feedback == 0.0
            for multiplicity, feedback in zip(multiplicities, feedback_calls, strict=True)
        ),
        "one_source": sum(multiplicity == 1 for multiplicity in multiplicities),
        "collisions": sum(multiplicity > 1 for multiplicity in multiplicities),
    }

    real_feedback = feedback_calls[:count]
    lag_rows: list[dict[str, object]] = []
    for lag in range(11):
        if lag >= count:
            lag_rows.append(
                {
                    "lag": lag,
                    "feedback_reward_product": _no_pairs_metric(),
                    "feedback_reward_correlation": _no_pairs_metric(),
                    "action_cells": _binary_cells((), ()),
                    "label_cells": _binary_cells((), ()),
                }
            )
            continue
        paired_feedback = real_feedback[lag:]
        paired_rewards = rewards[: count - lag]
        paired_actions = resolved_actions[: count - lag]
        paired_labels = resolved_labels[: count - lag]
        lag_rows.append(
            {
                "lag": lag,
                "feedback_reward_product": _mean_product(paired_feedback, paired_rewards),
                "feedback_reward_correlation": _correlation(paired_feedback, paired_rewards),
                "action_cells": _binary_cells(paired_feedback, paired_actions),
                "label_cells": _binary_cells(paired_feedback, paired_labels),
            }
        )

    start = max(0, count - 100)
    tail_feedback = real_feedback[start:]
    tail_actions = resolved_actions[start:]
    tail_labels = resolved_labels[start:]
    final_100 = {
        "sample_count": len(tail_feedback),
        "feedback_sum": float(sum(tail_feedback)),
        "feedback_mean": float(sum(tail_feedback) / len(tail_feedback)),
        "action_cells": _binary_cells(tail_feedback, tail_actions),
        "label_cells": _binary_cells(tail_feedback, tail_labels),
    }
    return {
        "provenance_rows": provenance_rows,
        "donor_summary": summary,
        "call_classes": call_classes,
        "lags": tuple(lag_rows),
        "final_100": final_100,
    }
