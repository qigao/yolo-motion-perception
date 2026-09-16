from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.phase3c_diagnostics.replay import D0Failure, assert_array_isolation


def test_observer_copy_must_not_share_writable_storage() -> None:
    learner = np.arange(4.0)
    with pytest.raises(D0Failure, match="array_alias"):
        assert_array_isolation(learner, learner, attempt_id="alias")


def test_independent_readonly_observer_copy_is_accepted() -> None:
    learner = np.arange(4.0)
    observer = learner.copy()
    observer.flags.writeable = False
    assert_array_isolation(learner, observer, attempt_id="copy")
    assert not np.shares_memory(learner, observer)
