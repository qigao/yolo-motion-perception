from __future__ import annotations

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_diagnostics.auxiliary_evidence import original_auxiliary_evidence
from neural_state_machine.phase3c_diagnostics.contracts import ModelId
from neural_state_machine.phase3c_diagnostics.evaluation import build_evaluation_bundles
from neural_state_machine.phase3c_diagnostics.replay import run_diagnostic_replay


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


def test_original_auxiliary_evidence_matches_registered_small_lifecycle() -> None:
    model = ModelId("original", 7, arm="eligibility", condition="normal")
    replay = run_diagnostic_replay(7, "eligibility", SMALL)
    bundles = build_evaluation_bundles(SMALL)

    evidence = original_auxiliary_evidence(model, replay, bundles, SMALL)

    assert len(evidence.reset_scores) == 9
    assert len(evidence.reverse_checks) == 9
    assert len(evidence.drain_scores) == (1 + len(replay.drain_steps) + 1) * 9
    assert len({row["key"] for row in evidence.reset_scores}) == 9
    assert len({row["key"] for row in evidence.reverse_checks}) == 9
    assert len({row["key"] for row in evidence.drain_scores}) == len(evidence.drain_scores)
    assert all(row["order_invariant"] is True for row in evidence.reverse_checks)
    assert all(row["overall"]["total"] == 10 for row in evidence.reset_scores)
    assert all(row["overall"]["total"] == 10 for row in evidence.drain_scores)
