"""D5 interpretation-safe mechanism table and no-refit report model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class DiagnosticOutcome(Enum):
    """Prospective interpretation labels selected only during review/freeze."""

    BLOCK_LOCAL_ASSOCIATION = "A"
    GLOBAL_PREDICTIVITY = "B"
    REGISTERED_OUTLIER_UNRESOLVED = "C"
    EVALUATION_FIXTURE_SENSITIVE = "D"
    MULTIPLE_MECHANISMS = "E"
    INTEGRITY_INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class MechanismEvidence:
    mechanism: str
    integrity_valid: bool
    observed_association: str
    matched_counterfactual_evidence: str
    interpretation: str


_MECHANISMS = (
    "design_geometry",
    "task_relevant_target_projection",
    "bias_action_shortcut",
    "local_block_structure",
    "evaluation_fixture_sensitivity",
    "unresolved_multiple_mechanisms",
)
_STRONG_CAUSAL_PATTERNS = (
    "caused by",
    "proves that",
    "only cause",
    "demonstrates causality",
    "establishes causality",
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _reject_strong_causal_language(text: str) -> None:
    lowered = text.lower()
    if any(pattern in lowered for pattern in _STRONG_CAUSAL_PATTERNS):
        raise ValueError("rationale contains an unregistered strong causal claim")


def build_attribution_table(
    evidence_by_mechanism: Mapping[str, object],
    *,
    integrity_valid: bool,
) -> tuple[MechanismEvidence, ...]:
    """Build the fixed six-row D5 evidence table without selecting an outcome."""
    if type(integrity_valid) is not bool:
        raise ValueError("integrity_valid must be a boolean")
    if not isinstance(evidence_by_mechanism, Mapping):
        raise ValueError("mechanism evidence must be a mapping")
    if set(evidence_by_mechanism) != set(_MECHANISMS):
        raise ValueError("mechanism evidence must contain exactly the registered mechanisms")

    rows: list[MechanismEvidence] = []
    for mechanism in _MECHANISMS:
        raw = evidence_by_mechanism[mechanism]
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise ValueError("mechanism evidence must be a three-string tuple")
        observed = _text(raw[0], "observed association")
        counterfactual = _text(raw[1], "matched counterfactual evidence")
        interpretation = _text(raw[2], "interpretation")
        rows.append(
            MechanismEvidence(
                mechanism=mechanism,
                integrity_valid=integrity_valid,
                observed_association=observed,
                matched_counterfactual_evidence=counterfactual,
                interpretation=interpretation,
            )
        )
    return tuple(rows)


def _validated_table(table: object) -> tuple[MechanismEvidence, ...]:
    try:
        rows = tuple(table)
    except TypeError as exc:
        raise ValueError("mechanism table must be iterable") from exc
    if len(rows) != len(_MECHANISMS) or any(
        not isinstance(row, MechanismEvidence) for row in rows
    ):
        raise ValueError("mechanism table must contain exactly six MechanismEvidence rows")
    if tuple(row.mechanism for row in rows) != _MECHANISMS:
        raise ValueError("mechanism table order or names differ from the registered table")
    integrity = {row.integrity_valid for row in rows}
    if len(integrity) != 1:
        raise ValueError("mechanism table has inconsistent integrity state")
    for row in rows:
        _text(row.observed_association, "observed association")
        _text(row.matched_counterfactual_evidence, "matched counterfactual evidence")
        _text(row.interpretation, "interpretation")
    return rows


def validate_outcome_rationale(
    outcome: DiagnosticOutcome,
    table: object,
    rationale: tuple[str, ...],
) -> None:
    """Validate a human-selected interpretation without deriving one from evidence."""
    if not isinstance(outcome, DiagnosticOutcome):
        raise ValueError("outcome must be a DiagnosticOutcome")
    rows = _validated_table(table)
    integrity_valid = rows[0].integrity_valid
    if not integrity_valid and outcome is not DiagnosticOutcome.INTEGRITY_INVALID:
        raise ValueError("integrity failure permits only the invalid outcome")
    if integrity_valid and outcome is DiagnosticOutcome.INTEGRITY_INVALID:
        raise ValueError("invalid outcome is forbidden when integrity is valid")
    if not isinstance(rationale, tuple) or not rationale:
        raise ValueError("rationale must be a non-empty tuple")
    for line in rationale:
        clean = _text(line, "rationale")
        _reject_strong_causal_language(clean)


def render_markdown_report(
    result: Mapping[str, object],
    *,
    outcome: DiagnosticOutcome,
    rationale: tuple[str, ...],
) -> str:
    """Render separate association/counterfactual/interpretation layers."""
    if not isinstance(result, Mapping) or "mechanism_table" not in result:
        raise ValueError("result must contain mechanism_table")
    table = _validated_table(result["mechanism_table"])
    validate_outcome_rationale(outcome, table, rationale)

    for row in table:
        _reject_strong_causal_language(row.observed_association)
        _reject_strong_causal_language(row.matched_counterfactual_evidence)
        _reject_strong_causal_language(row.interpretation)

    lines = [
        "# Phase C4-A Failure Attribution",
        "",
        f"Outcome {outcome.value}",
        "",
        "This report records diagnostic associations and matched counterfactual comparisons; it does not automatically establish a causal mechanism.",
        "",
        "## Observed association",
        "",
    ]
    lines.extend(f"- `{row.mechanism}`: {row.observed_association}" for row in table)
    lines.extend(("", "## Matched counterfactual evidence", ""))
    lines.extend(
        f"- `{row.mechanism}`: {row.matched_counterfactual_evidence}" for row in table
    )
    lines.extend(("", "## Interpretation", ""))
    lines.extend(f"- `{row.mechanism}`: {row.interpretation}" for row in table)
    lines.extend(("", "### Review rationale", ""))
    lines.extend(f"- {line.strip()}" for line in rationale)
    lines.append("")
    return "\n".join(lines)
