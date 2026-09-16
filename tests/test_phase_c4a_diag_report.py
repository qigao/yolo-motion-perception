from __future__ import annotations

import pytest

from neural_state_machine.phase_c4a_diagnostics.report import (
    DiagnosticOutcome,
    build_attribution_table,
    render_markdown_report,
    validate_outcome_rationale,
)


_MECHANISMS = (
    "design_geometry",
    "task_relevant_target_projection",
    "bias_action_shortcut",
    "local_block_structure",
    "evaluation_fixture_sensitivity",
    "unresolved_multiple_mechanisms",
)


def _evidence() -> dict[str, tuple[str, str, str]]:
    return {
        mechanism: (
            f"observed {mechanism}",
            f"matched evidence {mechanism}",
            f"interpretation {mechanism}",
        )
        for mechanism in _MECHANISMS
    }


def test_attribution_table_has_all_required_mechanisms_and_no_outcome_classifier():
    table = build_attribution_table(_evidence(), integrity_valid=True)

    assert tuple(row.mechanism for row in table) == _MECHANISMS
    assert all(row.integrity_valid is True for row in table)
    assert all(row.observed_association.startswith("observed ") for row in table)
    assert all(row.matched_counterfactual_evidence.startswith("matched evidence ") for row in table)
    assert all(row.interpretation.startswith("interpretation ") for row in table)
    assert not hasattr(table, "outcome")


@pytest.mark.parametrize(
    "outcome",
    (
        DiagnosticOutcome.BLOCK_LOCAL_ASSOCIATION,
        DiagnosticOutcome.GLOBAL_PREDICTIVITY,
        DiagnosticOutcome.REGISTERED_OUTLIER_UNRESOLVED,
        DiagnosticOutcome.EVALUATION_FIXTURE_SENSITIVE,
        DiagnosticOutcome.MULTIPLE_MECHANISMS,
    ),
)
def test_every_valid_outcome_renders_three_separate_evidence_layers(outcome):
    table = build_attribution_table(_evidence(), integrity_valid=True)
    rationale = ("Evidence is associated with the selected prospective interpretation.",)

    validate_outcome_rationale(outcome, table, rationale)
    report = render_markdown_report(
        {"mechanism_table": table},
        outcome=outcome,
        rationale=rationale,
    )

    assert "## Observed association" in report
    assert "## Matched counterfactual evidence" in report
    assert "## Interpretation" in report
    assert f"Outcome {outcome.value}" in report
    assert "caused by" not in report.lower()
    assert "proves that" not in report.lower()
    assert "only cause" not in report.lower()


def test_integrity_invalid_forces_invalid_outcome_and_rejects_a_to_e():
    table = build_attribution_table(_evidence(), integrity_valid=False)
    rationale = ("D0 integrity failed; no D1-D5 interpretation is valid.",)

    validate_outcome_rationale(DiagnosticOutcome.INTEGRITY_INVALID, table, rationale)
    report = render_markdown_report(
        {"mechanism_table": table},
        outcome=DiagnosticOutcome.INTEGRITY_INVALID,
        rationale=rationale,
    )
    assert "Outcome invalid" in report

    with pytest.raises(ValueError, match="integrity"):
        validate_outcome_rationale(
            DiagnosticOutcome.MULTIPLE_MECHANISMS,
            table,
            rationale,
        )


def test_valid_integrity_rejects_invalid_outcome_and_empty_rationale():
    table = build_attribution_table(_evidence(), integrity_valid=True)

    with pytest.raises(ValueError, match="invalid outcome"):
        validate_outcome_rationale(
            DiagnosticOutcome.INTEGRITY_INVALID,
            table,
            ("not applicable",),
        )
    with pytest.raises(ValueError, match="rationale"):
        validate_outcome_rationale(
            DiagnosticOutcome.GLOBAL_PREDICTIVITY,
            table,
            (),
        )


def test_rationale_rejects_unregistered_strong_causal_claims():
    table = build_attribution_table(_evidence(), integrity_valid=True)

    for text in (
        "The shuffled score was caused by block structure.",
        "These diagnostics prove that anonymity is impossible.",
        "Bias is the only cause of the specificity failure.",
    ):
        with pytest.raises(ValueError, match="causal"):
            validate_outcome_rationale(
                DiagnosticOutcome.MULTIPLE_MECHANISMS,
                table,
                (text,),
            )


def test_table_rejects_missing_extra_or_malformed_mechanism_evidence():
    missing = _evidence()
    missing.pop("local_block_structure")
    with pytest.raises(ValueError, match="mechanism"):
        build_attribution_table(missing, integrity_valid=True)

    extra = _evidence()
    extra["invented_mechanism"] = ("a", "b", "c")
    with pytest.raises(ValueError, match="mechanism"):
        build_attribution_table(extra, integrity_valid=True)

    malformed = _evidence()
    malformed["design_geometry"] = ("a", "b", "")
    with pytest.raises(ValueError, match="non-empty"):
        build_attribution_table(malformed, integrity_valid=True)
