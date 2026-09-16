"""Public surface for Phase C4-A failure-attribution diagnostics."""

from .model import (
    DiagnosticConfig,
    FrozenC4AIdentity,
    freeze_float64,
    sha256_file,
)

__all__ = [
    "DiagnosticConfig",
    "FrozenC4AIdentity",
    "freeze_float64",
    "sha256_file",
]
