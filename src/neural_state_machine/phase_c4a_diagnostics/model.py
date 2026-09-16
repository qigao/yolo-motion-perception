"""Frozen identity and immutable data helpers for C4-A failure attribution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class FrozenC4AIdentity:
    scientific_head: str
    formal_head: str
    manifest_sha256: str
    result_sha256: str
    provenance_sha256: str
    seeds: tuple[int, ...]
    hidden_size: int
    recurrent_radius: float
    training_decisions: int
    delay_support: tuple[int, ...]
    drain_clocks: int
    design_rows: int
    ridge_penalty: float

    @classmethod
    def registered(cls) -> "FrozenC4AIdentity":
        return cls(
            scientific_head="8ae3154950ed53c4d0a0f555463042ff72674d31",
            formal_head="8b2180ed24b6ff03db4b927ec29cdd9903b0ccac",
            manifest_sha256=(
                "a990745cf6f0e8c5fd7ae5d9a7189737b30a40168a4db3df1a6d3043d6d5876a"
            ),
            result_sha256=(
                "7f398f474dba928d2a6aa993d0801563aa05af20faa8cd9a61ec94a6ca4697a4"
            ),
            provenance_sha256=(
                "8fb47b5e670754bc44c9ea70dd4cedce40e6f52165923142114577b4aec00351"
            ),
            seeds=(7, 17, 29),
            hidden_size=64,
            recurrent_radius=0.9,
            training_decisions=2_000,
            delay_support=(1, 3, 5),
            drain_clocks=5,
            design_rows=2_005,
            ridge_penalty=1e-6,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticConfig:
    permutation_replicates: int
    permutation_lineage: int = 0x43344144
    modes: tuple[str, ...] = ("block10", "global")
    near_zero_margin: float = 1e-9
    registered: bool = False

    def __post_init__(self) -> None:
        if type(self.permutation_replicates) is not int or self.permutation_replicates <= 0:
            raise ValueError("permutation_replicates must be a positive integer")
        if self.registered and self.permutation_replicates != 32:
            raise ValueError("registered diagnostics require exactly 32 replicates")
        if not self.registered and self.permutation_replicates >= 32:
            raise ValueError("testing diagnostics must use fewer than 32 replicates")
        if self.permutation_lineage != 0x43344144:
            raise ValueError("permutation_lineage must match the registered diagnostic lineage")
        if self.modes != ("block10", "global"):
            raise ValueError("modes must be exactly ('block10', 'global')")
        if self.near_zero_margin != 1e-9:
            raise ValueError("near_zero_margin must be exactly 1e-9")

    @classmethod
    def registered(cls) -> "DiagnosticConfig":
        return cls(permutation_replicates=32, registered=True)

    @classmethod
    def testing(cls, replicates: int) -> "DiagnosticConfig":
        return cls(permutation_replicates=replicates, registered=False)


def freeze_float64(array: object) -> np.ndarray:
    """Return a detached C-contiguous read-only float64 copy."""
    frozen = np.ascontiguousarray(array, dtype=np.float64).copy()
    frozen.flags.writeable = False
    return frozen


def sha256_file(path: Path) -> str:
    """Hash a regular file while rejecting symlinks and missing paths."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"required regular file is missing or unsafe: {candidate}")
    return hashlib.sha256(candidate.read_bytes()).hexdigest()
