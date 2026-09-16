from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from neural_state_machine.phase_c4a_diagnostics.model import (
    DiagnosticConfig,
    FrozenC4AIdentity,
    freeze_float64,
    sha256_file,
)


def test_identity_freezes_registered_c4a_inputs():
    identity = FrozenC4AIdentity.registered()

    assert identity.scientific_head == "8ae3154950ed53c4d0a0f555463042ff72674d31"
    assert identity.formal_head == "8b2180ed24b6ff03db4b927ec29cdd9903b0ccac"
    assert identity.manifest_sha256 == (
        "a990745cf6f0e8c5fd7ae5d9a7189737b30a40168a4db3df1a6d3043d6d5876a"
    )
    assert identity.result_sha256 == (
        "7f398f474dba928d2a6aa993d0801563aa05af20faa8cd9a61ec94a6ca4697a4"
    )
    assert identity.provenance_sha256 == (
        "8fb47b5e670754bc44c9ea70dd4cedce40e6f52165923142114577b4aec00351"
    )
    assert identity.seeds == (7, 17, 29)
    assert identity.hidden_size == 64
    assert identity.recurrent_radius == 0.9
    assert identity.training_decisions == 2_000
    assert identity.delay_support == (1, 3, 5)
    assert identity.drain_clocks == 5
    assert identity.design_rows == 2_005
    assert identity.ridge_penalty == 1e-6


def test_registered_diagnostic_config_is_exact():
    config = DiagnosticConfig.registered()

    assert config.permutation_replicates == 32
    assert config.permutation_lineage == 0x43344144
    assert config.modes == ("block10", "global")
    assert config.near_zero_margin == 1e-9
    assert config.registered is True


def test_testing_config_is_explicitly_unregistered():
    config = DiagnosticConfig.testing(2)

    assert config.permutation_replicates == 2
    assert config.registered is False
    with pytest.raises(ValueError, match="positive"):
        DiagnosticConfig.testing(0)


def test_frozen_arrays_are_detached_c_contiguous_float64_and_read_only():
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    frozen = freeze_float64(source)

    source[:] = -1
    assert frozen.dtype == np.float64
    assert frozen.flags.c_contiguous is True
    assert frozen.flags.writeable is False
    np.testing.assert_array_equal(
        frozen,
        np.arange(6, dtype=np.float64).reshape(2, 3),
    )
    with pytest.raises(ValueError):
        frozen[0, 0] = 99.0


def test_sha256_file_rejects_symlinks_and_hashes_regular_files(tmp_path: Path):
    regular = tmp_path / "payload.bin"
    regular.write_bytes(b"c4a-diagnostics\n")
    expected = hashlib.sha256(regular.read_bytes()).hexdigest()

    assert sha256_file(regular) == expected

    link = tmp_path / "payload-link.bin"
    try:
        link.symlink_to(regular)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="regular file"):
        sha256_file(link)
