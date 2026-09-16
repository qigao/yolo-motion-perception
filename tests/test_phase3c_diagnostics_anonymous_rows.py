from __future__ import annotations

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_diagnostics import runner_core
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


def _helper():
    helper = getattr(runner_core, "_anonymous_model_row", None)
    assert helper is not None, "registered anonymous row helper must be implemented"
    return helper


def _inputs():
    replays = {
        (7, "td0", "normal"): run_diagnostic_replay(7, "td0", SMALL, "normal"),
        (7, "td0", "original_shuffle"): run_diagnostic_replay(
            7, "td0", SMALL, "original_shuffle"
        ),
    }
    normal_rewards = replays[(7, "td0", "normal")].protocol.rewards
    return replays, normal_rewards, build_evaluation_bundles(SMALL)


def test_original_row_contains_complete_replay_bound_d1_and_nine_scores() -> None:
    replays, normal_rewards, bundles = _inputs()
    model = ModelId("original", 7, arm="td0", condition="normal")
    row = _helper()(
        model,
        normal_rewards=normal_rewards,
        replays=replays,
        config=SMALL,
        bundles=bundles,
        attempt_id="row-original",
    )

    assert len(row["scores"]) == 9
    assert len(row["d1"]["provenance_rows"]) == SMALL.training_decisions
    assert row["d1"]["donor_summary"]["fixed_points"] == SMALL.training_decisions
    assert len(row["parameter_digest"]) == 64


def test_permutation_row_uses_single_pass_capture_and_complete_d1() -> None:
    replays, normal_rewards, bundles = _inputs()
    model = ModelId("permutation", 7, arm="td0", mode="block10", replicate=0)
    row = _helper()(
        model,
        normal_rewards=normal_rewards,
        replays=replays,
        config=SMALL,
        bundles=bundles,
        attempt_id="row-permutation",
    )

    assert len(row["scores"]) == 9
    assert len(row["d1"]["provenance_rows"]) == SMALL.training_decisions
    assert row["d1"]["donor_summary"]["same_block_donors"] == SMALL.training_decisions
    assert len(row["parameter_digest"]) == 64
