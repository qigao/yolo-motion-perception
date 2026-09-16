from __future__ import annotations

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_diagnostics import runner_core
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


def test_eligibility_accounting_records_source_and_other_history_components() -> None:
    replay = run_diagnostic_replay(7, "eligibility", SMALL, "normal")
    summary = runner_core._accounting_summary(replay, "eligibility", SMALL)

    rows = summary.get("history_components")
    assert isinstance(rows, list)
    assert len(rows) == SMALL.training_decisions
    assert [row["step"] for row in rows] == list(range(SMALL.training_decisions))
    for row in rows:
        assert row["eligibility_reconstruction_max_residual"] <= 1e-10
        assert row["update_reconstruction_max_residual"] <= 1e-10
        assert row["source_update_norm"] >= 0.0
        assert row["other_update_norm"] >= 0.0
        assert "source_vs_reference_cosine" in row
        assert "other_vs_reference_cosine" in row
        assert "actual_vs_reference_cosine" in row


def test_td0_accounting_does_not_invent_persistent_history_components() -> None:
    replay = run_diagnostic_replay(7, "td0", SMALL, "normal")
    summary = runner_core._accounting_summary(replay, "td0", SMALL)
    assert "history_components" not in summary
    assert summary["drain_weight_changes"] == 0
