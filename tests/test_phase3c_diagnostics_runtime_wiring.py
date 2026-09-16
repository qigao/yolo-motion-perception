from __future__ import annotations

import inspect

from neural_state_machine.phase3c_diagnostics import runner


def test_registered_measurement_is_wrapped_not_direct_core_export() -> None:
    assert runner.run_registered_measurement is not runner._core.run_registered_measurement


def test_registered_measurement_wrapper_mentions_staging_and_auxiliary_seal() -> None:
    source = inspect.getsource(runner.run_registered_measurement)
    assert "staging" in source
    assert "seal_auxiliary_artifact" in source
    assert "verify_attempt" in source
