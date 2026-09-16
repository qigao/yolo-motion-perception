from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

_MODE_CODE = {"block10": 0, "global": 1}
_ORIGINAL_SHUFFLE_LINEAGE = 0x33534846
_DIAGNOSTIC_LINEAGE = 0x33434641


@dataclass(frozen=True, slots=True)
class DonorAssignment:
    slot: int
    donor: int
    due_step: int

    @property
    def is_fixed_point(self) -> bool:
        return self.slot == self.donor

    @property
    def is_current_decision_donor(self) -> bool:
        return self.donor == self.due_step

    @property
    def donor_relative_to_delivery(self) -> str:
        if self.donor < self.due_step:
            return "past"
        if self.donor == self.due_step:
            return "current"
        return "future"


def _positive_count(count: object) -> int:
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    return count


def _replicate(replicate: object) -> int:
    if type(replicate) is not int or not 0 <= replicate < 32:
        raise ValueError("replicate must be an integer in [0, 31]")
    return replicate


def _validate_permutation(permutation: tuple[int, ...]) -> None:
    if any(type(value) is not int for value in permutation):
        raise ValueError("permutation must contain integer indices")
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("permutation must contain every index exactly once")


def donor_rows(
    permutation: tuple[int, ...], due_steps: tuple[int, ...]
) -> tuple[DonorAssignment, ...]:
    if not isinstance(permutation, tuple) or not isinstance(due_steps, tuple):
        raise ValueError("permutation and due_steps must be tuples")
    if len(permutation) != len(due_steps):
        raise ValueError("permutation and due_steps lengths must match")
    _validate_permutation(permutation)
    if any(type(value) is not int or value < 0 for value in due_steps):
        raise ValueError("due_steps must contain non-negative integers")
    return tuple(
        DonorAssignment(slot=slot, donor=donor, due_step=due_steps[slot])
        for slot, donor in enumerate(permutation)
    )


def _block10_permutation(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    if count % 10:
        raise ValueError("block10 permutation requires count divisible by 10")
    result: list[int] = []
    for start in range(0, count, 10):
        result.extend(start + int(index) for index in rng.permutation(10))
    return tuple(result)


def diagnostic_permutation(
    seed: int, count: int, mode: str, replicate: int
) -> tuple[int, ...]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    resolved_count = _positive_count(count)
    resolved_replicate = _replicate(replicate)
    try:
        mode_code = _MODE_CODE[mode]
    except (KeyError, TypeError) as exc:
        raise ValueError("mode must be block10 or global") from exc
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [seed, _DIAGNOSTIC_LINEAGE, 1, mode_code, resolved_replicate]
            )
        )
    )
    if mode == "block10":
        result = _block10_permutation(rng, resolved_count)
    else:
        result = tuple(int(index) for index in rng.permutation(resolved_count))
    _validate_permutation(result)
    return result


def original_block10_permutation(seed: int, count: int) -> tuple[int, ...]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    resolved_count = _positive_count(count)
    rng = np.random.default_rng(np.random.SeedSequence([seed, _ORIGINAL_SHUFFLE_LINEAGE]))
    result = _block10_permutation(rng, resolved_count)
    _validate_permutation(result)
    return result


def identity_permutation(count: int) -> tuple[int, ...]:
    return tuple(range(_positive_count(count)))


def apply_permutation(values: tuple[float, ...], permutation: tuple[int, ...]) -> tuple[float, ...]:
    if len(values) != len(permutation):
        raise ValueError("values and permutation lengths must match")
    _validate_permutation(permutation)
    return tuple(float(values[index]) for index in permutation)


def expected_current_donor_matches(due_steps: tuple[int, ...], mode: str) -> float:
    if not isinstance(due_steps, tuple) or not due_steps:
        raise ValueError("due_steps must be a non-empty tuple")
    if any(type(value) is not int or value < 0 for value in due_steps):
        raise ValueError("due_steps must contain non-negative integers")
    count = len(due_steps)
    eligible = sum(
        due_step < count
        and (mode != "block10" or slot // 10 == due_step // 10)
        for slot, due_step in enumerate(due_steps)
    )
    if mode == "block10":
        return eligible / 10.0
    if mode == "global":
        return eligible / float(count)
    raise ValueError("mode must be block10 or global")


def nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    if not values:
        raise ValueError("values must not be empty")
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise ValueError("quantile must be numeric")
    q = float(quantile)
    if not math.isfinite(q) or not 0.0 < q <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return int(ordered[math.ceil(q * len(ordered)) - 1])


def donor_summary(rows: tuple[DonorAssignment, ...]) -> dict[str, object]:
    if not rows:
        raise ValueError("rows must not be empty")
    return {
        "fixed_points": sum(row.is_fixed_point for row in rows),
        "current_decision_matches": sum(row.is_current_decision_donor for row in rows),
        "relative_to_delivery": dict(Counter(row.donor_relative_to_delivery for row in rows)),
        "displacements": dict(Counter(row.donor - row.slot for row in rows)),
    }
