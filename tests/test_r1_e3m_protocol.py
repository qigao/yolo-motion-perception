import pytest

from neural_state_machine.r1_e3m_protocol import (
    ArmPrimaryResult,
    OUTCOME_A,
    OUTCOME_B,
    OUTCOME_C,
    classify_outcome,
    registered_manifest_payload,
)


def _arms(delta: float, reset: float):
    return tuple(
        ArmPrimaryResult(
            architecture=index // 5,
            seed=(7, 17, 29, 43, 61)[index % 5],
            delta10=delta,
            reset10=reset,
        )
        for index in range(20)
    )


def test_registered_manifest_freezes_primary_choices():
    payload = registered_manifest_payload()

    assert payload["phase"] == "R1-E3M"
    assert payload["window_seconds"] == 4.0
    assert payload["bin_count"] == 20
    assert payload["bin_hz"] == 5.0
    assert payload["presence_gate"] == 16
    assert payload["delays"] == [1, 2, 5, 10, 15]
    assert payload["primary_delay"] == 10
    assert payload["arm_count"] == 20
    assert payload["semantic_labels_used"] is False


def test_outcome_a_requires_effect_size_and_sixteen_positive_arms():
    assert classify_outcome(_arms(0.05, 0.05)).outcome == OUTCOME_A

    values = list(_arms(0.06, 0.06))
    for index in range(5):
        values[index] = ArmPrimaryResult(
            architecture=values[index].architecture,
            seed=values[index].seed,
            delta10=-0.01,
            reset10=0.06,
        )
    assert classify_outcome(tuple(values)).outcome == OUTCOME_B


def test_outcome_b_is_positive_but_not_robust():
    result = classify_outcome(_arms(0.01, 0.02))

    assert result.outcome == OUTCOME_B
    assert result.median_delta10 == pytest.approx(0.01)
    assert result.median_reset10 == pytest.approx(0.02)


def test_outcome_c_when_either_primary_median_is_nonpositive():
    assert classify_outcome(_arms(0.0, 0.1)).outcome == OUTCOME_C
    assert classify_outcome(_arms(0.1, 0.0)).outcome == OUTCOME_C


def test_arm_count_and_identity_fail_closed():
    with pytest.raises(ValueError, match="exactly 20"):
        classify_outcome(_arms(0.1, 0.1)[:-1])

    arms = list(_arms(0.1, 0.1))
    arms[-1] = arms[0]
    with pytest.raises(ValueError, match="duplicate"):
        classify_outcome(tuple(arms))
