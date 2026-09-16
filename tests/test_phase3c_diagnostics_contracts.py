from __future__ import annotations

import pytest

from neural_state_machine.phase3c_diagnostics.contracts import ModelId, registered_model_ids


def test_registered_grid_is_complete_and_unique() -> None:
    rows = registered_model_ids()
    assert len(rows) == len(set(rows)) == 405
    assert sum(row.family == "permutation" for row in rows) == 384
    assert sum(row.family == "original" for row in rows) == 12
    assert sum(row.family == "reference" for row in rows) == 9


def test_model_id_rejects_bool_as_integer() -> None:
    with pytest.raises(ValueError, match="seed"):
        ModelId(family="original", seed=True, arm="td0", condition="normal")


def test_original_shuffle_is_not_permutation_replicate_zero() -> None:
    rows = registered_model_ids()
    original = {
        row for row in rows if row.family == "original" and row.condition == "original_shuffle"
    }
    replicate_zero = {
        row for row in rows if row.family == "permutation" and row.replicate == 0
    }
    assert original
    assert replicate_zero
    assert original.isdisjoint(replicate_zero)
