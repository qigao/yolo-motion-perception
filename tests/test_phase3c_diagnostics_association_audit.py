from __future__ import annotations

import pytest

from neural_state_machine.phase3c_diagnostics import provenance


PERMUTATION = (1, 0, 2, 4, 3)
DUE_STEPS = (1, 2, 2, 5, 4)
LATENT_REWARDS = (1.0, -1.0, -1.0, -1.0, 1.0)
ACTIONS = (0, 1, 0, 1, 1)
LABELS = (0, 0, 1, 1, 0)
# Slot rewards after permutation: (-1, +1, -1, +1, -1).
# t=2 is a two-source cancellation; t=5 is drain-only.
ACTUAL_FEEDBACK = (0.0, -1.0, 0.0, 0.0, -1.0, 1.0)


def _audit() -> dict[str, object]:
    audit = getattr(provenance, "provenance_association_audit", None)
    assert audit is not None, "D1 association audit must be implemented"
    return audit(
        permutation=PERMUTATION,
        due_steps=DUE_STEPS,
        latent_rewards=LATENT_REWARDS,
        actions=ACTIONS,
        labels=LABELS,
        actual_feedback_calls=ACTUAL_FEEDBACK,
    )


def test_d1_association_audit_separates_slots_donors_lags_and_call_classes() -> None:
    result = _audit()

    provenance_rows = result["provenance_rows"]
    assert len(provenance_rows) == 5
    assert provenance_rows[0]["slot"] == 0
    assert provenance_rows[0]["donor"] == 1
    assert provenance_rows[0]["due_step"] == 1
    assert provenance_rows[0]["delivered_reward"] == -1.0
    assert provenance_rows[0]["donor_relative_to_delivery"] == "current"

    summary = result["donor_summary"]
    assert summary["fixed_points"] == 1
    assert summary["current_decision_matches"] == 2
    assert summary["same_block_donors"] == 5

    call_classes = result["call_classes"]
    assert call_classes == {
        "real_decision_calls": 5,
        "drain_calls": 1,
        "no_arrival_zero": 2,
        "cancellation_to_zero": 1,
        "one_source": 3,
        "collisions": 1,
    }

    lags = result["lags"]
    assert [row["lag"] for row in lags] == list(range(11))
    assert lags[0]["feedback_reward_product"]["sample_count"] == 5
    assert lags[5]["feedback_reward_product"] == {
        "value": None,
        "reason": "no_valid_pairs",
        "sample_count": 0,
    }
    assert lags[0]["action_cells"]["0"]["count"] == 2
    assert lags[0]["action_cells"]["1"]["count"] == 3
    assert lags[0]["label_cells"]["0"]["count"] == 3
    assert lags[0]["label_cells"]["1"]["count"] == 2

    final = result["final_100"]
    assert final["sample_count"] == 5
    assert final["feedback_sum"] == -2.0


def test_d1_association_audit_checks_actual_intercepted_feedback_exactly() -> None:
    audit = getattr(provenance, "provenance_association_audit", None)
    assert audit is not None, "D1 association audit must be implemented"
    corrupted = (1.0, *ACTUAL_FEEDBACK[1:])
    with pytest.raises(ValueError, match="feedback"):
        audit(
            permutation=PERMUTATION,
            due_steps=DUE_STEPS,
            latent_rewards=LATENT_REWARDS,
            actions=ACTIONS,
            labels=LABELS,
            actual_feedback_calls=corrupted,
        )
