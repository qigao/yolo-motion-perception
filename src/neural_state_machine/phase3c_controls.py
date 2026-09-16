"""Fail-closed structural controls for the Phase 3C anonymous-credit protocol."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, replace

from .phase3c_schedule import AggregateFeedback


_TRACE_ATOL = 1e-15
_REGISTERED_DELAY_SUPPORT = (1, 3, 5)
_REGISTERED_TRACE_COEFFICIENTS = (
    (0, 1.0),
    (1, 0.72),
    (3, 0.72**3),
    (5, 0.72**5),
)
_SOURCE_RELABEL_OFFSET = 1_000_000_007


@dataclass(frozen=True)
class AnonymousProtocolAudit:
    arm: str
    action_count: int
    latent_record_count: int
    delivered_record_count: int
    real_feedback_count: int
    drain_feedback_count: int
    queue_pending_final: int
    delay_histogram: tuple[tuple[int, int], ...]
    inversion_count: int
    collision_step_count: int
    multiplicity_histogram: tuple[tuple[int, int], ...]
    delay_digest: str
    due_step_digest: str
    multiplicity_digest: str
    aggregate_feedback_digest: str
    learner_call_digest: str
    latent_reward_sum: float
    aggregate_feedback_sum: float
    source_relabel_invariant: bool
    hidden_multiplicity_invariant: bool
    trace_reset_count: int
    trace_coefficients: tuple[tuple[int, float], ...]


def integer_sequence_digest(values: tuple[int, ...]) -> str:
    if not isinstance(values, tuple) or any(type(value) is not int for value in values):
        raise ValueError("integer digest requires a tuple of integers")
    digest = hashlib.sha256()
    digest.update(b"phase3c:int-sequence:v1\0")
    digest.update(len(values).to_bytes(8, "big", signed=False))
    for value in values:
        digest.update(value.to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def learner_call_digest(values: tuple[float, ...]) -> str:
    if not isinstance(values, tuple):
        raise ValueError("learner call digest requires a tuple")
    resolved: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("learner call values must be finite scalars")
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("learner call values must be finite scalars") from exc
        if not math.isfinite(scalar):
            raise ValueError("learner call values must be finite scalars")
        resolved.append(scalar)
    digest = hashlib.sha256()
    digest.update(b"phase3c:float64-call-sequence:v1\0")
    digest.update(len(resolved).to_bytes(8, "big", signed=False))
    for scalar in resolved:
        digest.update(struct.pack(">d", scalar))
    return digest.hexdigest()


def relabel_feedback_sources(
    feedbacks: tuple[AggregateFeedback, ...],
) -> tuple[AggregateFeedback, ...]:
    """Change only hidden source identities while preserving learner-visible feedback."""
    if not isinstance(feedbacks, tuple):
        raise ValueError("feedbacks must be a tuple")
    relabeled: list[AggregateFeedback] = []
    for feedback in feedbacks:
        if not isinstance(feedback, AggregateFeedback):
            raise ValueError("feedbacks must contain AggregateFeedback values")
        records = tuple(
            replace(record, source_step=record.source_step + _SOURCE_RELABEL_OFFSET)
            for record in feedback.records
        )
        relabeled.append(replace(feedback, records=records))
    return tuple(relabeled)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _validate_histogram(
    histogram: tuple[tuple[int, int], ...], name: str
) -> dict[int, int]:
    if not isinstance(histogram, tuple) or not histogram:
        raise ValueError(f"{name} must be a non-empty tuple")
    result: dict[int, int] = {}
    for key, count in histogram:
        if type(key) is not int or type(count) is not int or key < 0 or count <= 0:
            raise ValueError(f"{name} entries must be non-negative integer keys and positive counts")
        if key in result:
            raise ValueError(f"{name} must not contain duplicate keys")
        result[key] = count
    if tuple(result) != tuple(sorted(result)):
        raise ValueError(f"{name} must use canonical sorted key order")
    return result


def validate_anonymous_protocol(
    audit: AnonymousProtocolAudit, *, action_count: int
) -> None:
    if not isinstance(audit, AnonymousProtocolAudit):
        raise ValueError("audit must be an AnonymousProtocolAudit")
    if type(action_count) is not int or action_count <= 0:
        raise ValueError("action_count must be a positive integer")
    if audit.arm not in {"td0", "eligibility"}:
        raise ValueError("audit arm must be td0 or eligibility")
    if audit.action_count != action_count:
        raise ValueError("audit action count does not match the registered run")
    if audit.latent_record_count != action_count:
        raise ValueError("every real action must create exactly one latent reward")
    if audit.delivered_record_count != action_count:
        raise ValueError("every latent reward must be delivered exactly once")
    if audit.real_feedback_count != action_count:
        raise ValueError("every real decision must produce exactly one learner scalar call")
    if type(audit.drain_feedback_count) is not int or audit.drain_feedback_count < 0:
        raise ValueError("drain feedback count must be a non-negative integer")
    if audit.queue_pending_final != 0:
        raise ValueError("terminal drain must leave no latent rewards pending")

    delay_histogram = _validate_histogram(audit.delay_histogram, "delay_histogram")
    if tuple(delay_histogram) != _REGISTERED_DELAY_SUPPORT:
        raise ValueError("anonymous delay support must be exactly 1, 3, and 5")
    if sum(delay_histogram.values()) != action_count:
        raise ValueError("delay histogram must account for every real action")
    if audit.inversion_count <= 0:
        raise ValueError("registered anonymous schedule must contain an inversion")
    if audit.collision_step_count <= 0:
        raise ValueError("registered anonymous schedule must contain a collision")

    multiplicity_histogram = _validate_histogram(
        audit.multiplicity_histogram, "multiplicity_histogram"
    )
    if 0 in multiplicity_histogram:
        raise ValueError("multiplicity histogram records delivery buckets, not empty steps")
    if sum(key * count for key, count in multiplicity_histogram.items()) != action_count:
        raise ValueError("multiplicity histogram must account for every latent reward")
    if sum(count for key, count in multiplicity_histogram.items() if key >= 2) != audit.collision_step_count:
        raise ValueError("collision count and multiplicity histogram disagree")

    for name, value in (
        ("delay_digest", audit.delay_digest),
        ("due_step_digest", audit.due_step_digest),
        ("multiplicity_digest", audit.multiplicity_digest),
        ("aggregate_feedback_digest", audit.aggregate_feedback_digest),
        ("learner_call_digest", audit.learner_call_digest),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    if not math.isfinite(audit.latent_reward_sum) or not math.isfinite(
        audit.aggregate_feedback_sum
    ):
        raise ValueError("reward conservation sums must be finite")
    if audit.latent_reward_sum != audit.aggregate_feedback_sum:
        raise ValueError("aggregate feedback must conserve latent reward exactly")
    if audit.source_relabel_invariant is not True:
        raise ValueError("source relabeling must not alter learner calls")
    if audit.hidden_multiplicity_invariant is not True:
        raise ValueError("hidden multiplicity must not alter equal scalar learner calls")

    expected_reset_count = 0 if audit.arm == "td0" else 1
    if audit.trace_reset_count != expected_reset_count:
        raise ValueError("trace reset count does not match the registered arm lifecycle")

    if not isinstance(audit.trace_coefficients, tuple) or len(
        audit.trace_coefficients
    ) != len(_REGISTERED_TRACE_COEFFICIENTS):
        raise ValueError("trace coefficients must contain registered ages 0, 1, 3, and 5")
    for actual, expected in zip(
        audit.trace_coefficients, _REGISTERED_TRACE_COEFFICIENTS, strict=True
    ):
        if not isinstance(actual, tuple) or len(actual) != 2:
            raise ValueError("trace coefficient entries must be age/value pairs")
        age, value = actual
        expected_age, expected_value = expected
        if type(age) is not int or age != expected_age:
            raise ValueError("trace coefficient ages do not match the formal probes")
        if isinstance(value, bool):
            raise ValueError("trace coefficient values must be finite scalars")
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("trace coefficient values must be finite scalars") from exc
        if not math.isfinite(scalar) or not math.isclose(
            scalar, expected_value, rel_tol=0.0, abs_tol=_TRACE_ATOL
        ):
            raise ValueError("trace coefficient does not conform to the formal recurrence")
