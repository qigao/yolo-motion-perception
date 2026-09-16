from __future__ import annotations

import numpy as np

from neural_state_machine.phase3c_diagnostics.references import (
    SourceVisibleDelayedReference,
    fit_supervised_ridge,
)


def test_supervised_ridge_learns_separable_hidden_labels() -> None:
    hidden = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
    labels = (0, 0, 1, 1)
    reference = fit_supervised_ridge(hidden, labels)
    predictions = reference.predict(hidden)
    assert tuple(np.argmax(predictions, axis=1)) == labels


def test_source_visible_reference_uses_decision_time_prediction_and_source_order() -> None:
    learner = SourceVisibleDelayedReference(1, step_size=0.1)
    q1 = learner.record_decision(1, 1, np.array([2.0], dtype=np.float64))
    q0 = learner.record_decision(0, 0, np.array([-2.0], dtype=np.float64))
    assert q0 == q1 == 0.0
    learner.deliver((1, 0), (1.0, -1.0))
    weights = learner.parameter_snapshot()
    # Delivery is normalized and source-sorted, with one update on each action row.
    assert not np.array_equal(weights, np.zeros_like(weights))
    assert weights[0, 0] > 0.0
    assert weights[1, 0] > 0.0
