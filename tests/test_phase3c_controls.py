from __future__ import annotations

from dataclasses import replace

import pytest

from neural_state_machine.phase3c_controls import (
    AnonymousProtocolAudit,
    integer_sequence_digest,
    learner_call_digest,
    validate_anonymous_protocol,
)


TRACE = (
    (0, 1.0),
    (1, 0.72),
    (3, 0.72**3),
    (5, 0.72**5),
)


def _valid_audit(arm: str = "eligibility") -> AnonymousProtocolAudit:
    action_count = 12
    delays = (1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5)
    due_steps = tuple(step + delay for step, delay in enumerate(delays))
    multiplicities = (1, 1, 2, 1, 2, 1, 1, 2, 1)
    feedback = (0.0, 1.0, -1.0, 0.0, 2.0, -1.0, 0.0, 1.0, -2.0, 0.0, 0.0, 0.0)
    return AnonymousProtocolAudit(
        arm=arm,
        action_count=action_count,
        latent_record_count=action_count,
        delivered_record_count=action_count,
        real_feedback_count=action_count,
        drain_feedback_count=5,
        queue_pending_final=0,
        delay_histogram=((1, 4), (3, 4), (5, 4)),
        inversion_count=3,
        collision_step_count=3,
        multiplicity_histogram=((1, 6), (2, 3)),
        delay_digest=integer_sequence_digest(delays),
        due_step_digest=integer_sequence_digest(due_steps),
        multiplicity_digest=integer_sequence_digest(multiplicities),
        aggregate_feedback_digest=learner_call_digest(feedback),
        learner_call_digest=learner_call_digest(feedback),
        latent_reward_sum=0.0,
        aggregate_feedback_sum=0.0,
        source_relabel_invariant=True,
        hidden_multiplicity_invariant=True,
        trace_reset_count=1 if arm == "eligibility" else 0,
        trace_coefficients=TRACE,
    )


@pytest.mark.parametrize("arm", ["td0", "eligibility"])
def test_valid_audit_passes(arm: str) -> None:
    audit = _valid_audit(arm)
    validate_anonymous_protocol(audit, action_count=audit.action_count)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latent_record_count", 11),
        ("delivered_record_count", 11),
        ("real_feedback_count", 11),
        ("queue_pending_final", 1),
        ("inversion_count", 0),
        ("collision_step_count", 0),
        ("latent_reward_sum", 1.0),
        ("source_relabel_invariant", False),
        ("hidden_multiplicity_invariant", False),
    ],
)
def test_gate_rejects_structural_mutations(field: str, value: object) -> None:
    audit = replace(_valid_audit(), **{field: value})
    with pytest.raises(ValueError):
        validate_anonymous_protocol(audit, action_count=12)


def test_gate_requires_exact_registered_delay_support_and_counts() -> None:
    for histogram in (
        ((1, 6), (3, 6)),
        ((1, 4), (3, 4), (5, 3)),
        ((1, 4), (3, 4), (5, 4), (7, 1)),
    ):
        with pytest.raises(ValueError):
            validate_anonymous_protocol(
                replace(_valid_audit(), delay_histogram=histogram), action_count=12
            )


def test_gate_rejects_bad_digest_shape() -> None:
    for field in (
        "delay_digest",
        "due_step_digest",
        "multiplicity_digest",
        "aggregate_feedback_digest",
        "learner_call_digest",
    ):
        with pytest.raises(ValueError):
            validate_anonymous_protocol(
                replace(_valid_audit(), **{field: "not-a-sha256"}), action_count=12
            )


def test_gate_rejects_wrong_trace_reset_for_each_arm() -> None:
    with pytest.raises(ValueError):
        validate_anonymous_protocol(
            replace(_valid_audit("eligibility"), trace_reset_count=0), action_count=12
        )
    with pytest.raises(ValueError):
        validate_anonymous_protocol(
            replace(_valid_audit("td0"), trace_reset_count=1), action_count=12
        )


def test_gate_rejects_wrong_trace_coefficient() -> None:
    wrong = ((0, 1.0), (1, 0.72), (3, 0.72**3 + 1e-10), (5, 0.72**5))
    with pytest.raises(ValueError):
        validate_anonymous_protocol(
            replace(_valid_audit(), trace_coefficients=wrong), action_count=12
        )


def test_source_relabeling_cannot_change_learner_call_digest() -> None:
    scalar_stream = (0.0, 1.0, -1.0, 2.0)
    original_sources = (0, 1, 2, 3)
    relabeled_sources = (99, 42, 7, 123)

    assert integer_sequence_digest(original_sources) != integer_sequence_digest(
        relabeled_sources
    )
    assert learner_call_digest(scalar_stream) == learner_call_digest(scalar_stream)


def test_hidden_multiplicity_with_same_scalar_has_same_learner_call_encoding() -> None:
    empty_bucket_scalar = (0.0,)
    cancelling_collision_scalar = (1.0 + -1.0,)

    assert learner_call_digest(empty_bucket_scalar) == learner_call_digest(
        cancelling_collision_scalar
    )
    assert integer_sequence_digest((0,)) != integer_sequence_digest((2,))


def test_integer_and_float_digests_are_typed_and_order_sensitive() -> None:
    assert integer_sequence_digest((1, 2, 3)) != integer_sequence_digest((3, 2, 1))
    assert learner_call_digest((1.0, -1.0)) != learner_call_digest((-1.0, 1.0))
    assert integer_sequence_digest((1,)) != learner_call_digest((1.0,))
