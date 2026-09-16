from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DonorAssignment:
    slot: int
    donor: int
    due_step: int


def donor_rows(permutation: tuple[int, ...], due_steps: tuple[int, ...]) -> tuple[DonorAssignment, ...]:
    return ()


def diagnostic_permutation(seed: int, count: int, mode: str, replicate: int) -> tuple[int, ...]:
    return tuple(range(count))


def original_block10_permutation(seed: int, count: int) -> tuple[int, ...]:
    return tuple(range(count))


def expected_current_donor_matches(due_steps: tuple[int, ...], mode: str) -> float:
    return 0.0


def nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    return 0
