from __future__ import annotations

from dataclasses import dataclass

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
