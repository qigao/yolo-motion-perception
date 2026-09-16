from __future__ import annotations

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_diagnostics.auxiliary_evidence import reference_reset_evidence
from neural_state_machine.phase3c_diagnostics.contracts import ModelId
from neural_state_machine.phase3c_diagnostics.evaluation import build_evaluation_bundles
from neural_state_machine.phase3c_diagnostics.runner_core import (
    _train_immediate_reference,
    _train_ridge_reference,
    _train_source_visible_reference,
)


SMALL = AnonymousCreditConfig(
    hidden_size=8,
    recurrent_radius=0.9,
    step_size=0.1,
    training_decisions=10,
    evaluation_blocks=1,
    checkpoint_interval=10,
    delay_support=(1, 3, 5),
    discount=0.9,
    trace_decay=0.8,
)


def test_all_three_reference_kinds_emit_nine_reset_scores_for_one_seed() -> None:
    bundles = build_evaluation_bundles(SMALL)
    cases = (
        ("supervised_ridge", _train_ridge_reference(7, SMALL)),
        ("immediate_identified", _train_immediate_reference(7, SMALL)),
        ("source_visible_delayed", _train_source_visible_reference(7, SMALL)),
    )
    all_keys: set[str] = set()
    for kind, reference in cases:
        model = ModelId("reference", 7, reference_kind=kind)
        rows = reference_reset_evidence(model, reference, bundles, SMALL)
        assert len(rows) == 9
        assert all(row["overall"]["total"] == 10 for row in rows)
        keys = {row["key"] for row in rows}
        assert len(keys) == 9
        assert not (all_keys & keys)
        all_keys.update(keys)
    assert len(all_keys) == 27
