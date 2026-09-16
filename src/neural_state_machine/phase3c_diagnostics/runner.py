from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class StageResult:
    name: str
    passed: bool
    payload: object = None


def run_fail_closed_stages(
    d0: Callable[[], StageResult],
    d1: Callable[[], StageResult],
    d2: Callable[[], StageResult],
    d3: Callable[[], StageResult],
) -> tuple[StageResult, ...]:
    return ()


def validate_complete_keys(expected: tuple[str, ...], observed: tuple[str, ...]) -> None:
    return None
