from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttributionCandidate:
    name: str
    status: str
    evidence: tuple[str, ...]


def build_report_summary(
    *,
    diagnostic_valid: bool,
    original_behavior_passed: bool,
    candidates: tuple[AttributionCandidate, ...],
) -> dict[str, object]:
    return {}
