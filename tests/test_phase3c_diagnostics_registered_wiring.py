from __future__ import annotations

import inspect

from neural_state_machine.phase3c_diagnostics import runner_core


def test_registered_measurement_overrides_preserved_base_runner() -> None:
    assert runner_core.run_registered_measurement is not runner_core._base.run_registered_measurement


def test_registered_measurement_wires_replay_bound_d1_and_d3_helpers() -> None:
    source = inspect.getsource(runner_core.run_registered_measurement)
    assert "_anonymous_model_row" in source
    assert "_accounting_summary" in source
    assert "profile_model_ids" in source
    assert "validate_complete_keys" in source
