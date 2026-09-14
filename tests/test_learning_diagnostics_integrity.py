"""Fast, independent evidence-boundary and public-schema regressions."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import neural_state_machine.learning_diagnostics as diagnostics

_ROOT = Path(__file__).resolve().parents[1]


def _measured_result() -> diagnostics.LearningDiagnosticsResult:
    """Rehydrate measured summaries, with explicit private raw-vector sentinels."""
    payload = json.loads((_ROOT / "docs/experiments/phase-2c-diagnostics.json").read_text())
    entry = payload["results"][0]

    def counts(value: dict[str, int]) -> diagnostics.AccuracyCount:
        return diagnostics.AccuracyCount(**value)

    def per_delay(value: list[list[object]]) -> tuple:
        return tuple((delay, counts(count)) for delay, count in value)

    geometry = entry.pop("geometry")
    geometry["training"] = counts(geometry["training"])
    geometry["evaluation"] = counts(geometry["evaluation"])
    geometry["per_delay"] = per_delay(geometry["per_delay"])
    for key in ("training_margins", "evaluation_margins"):
        geometry[key] = diagnostics.MarginSummary(**geometry[key])
    geometry["per_delay_margins"] = tuple(
        (delay, diagnostics.MarginSummary(**margin))
        for delay, margin in geometry["per_delay_margins"]
    )
    supervised = entry.pop("supervised")
    reward = entry.pop("reward_trajectory")
    for branch in (supervised, reward):
        branch["overall"] = counts(branch["overall"])
        branch["per_delay"] = per_delay(branch["per_delay"])
        for checkpoint in branch["checkpoints"]:
            checkpoint["overall"] = counts(checkpoint["overall"])
            checkpoint["per_delay"] = per_delay(checkpoint["per_delay"])
    supervised["checkpoints"] = tuple(
        diagnostics.DiagnosticCheckpoint(**checkpoint) for checkpoint in supervised["checkpoints"]
    )
    reward["gradient_blocks"] = tuple(
        diagnostics.GradientBlockDiagnostic(
            **block,
            sampled_episode_gradients=((0.125,),),
            supervised_episode_gradients=((0.25,),),
            expected_bandit_to_supervised_norm_ratios=(0.5,),
            sampled_gradient=(0.125,),
            supervised_gradient=(0.25,),
        )
        for block in reward["gradient_blocks"]
    )
    reward["checkpoints"] = tuple(
        diagnostics.RewardCheckpoint(
            **checkpoint,
            correct_action_probabilities=(0.25, 0.75),
            gradient_blocks=reward["gradient_blocks"][: checkpoint["episode"] // 10],
        )
        for checkpoint in reward["checkpoints"]
    )
    reward["final_block"] = counts(reward["final_block"])
    for key in ("matrix_digests_before", "matrix_digests_after"):
        reward[key] = tuple(reward[key])
        entry[key] = tuple(entry[key])
    return diagnostics.LearningDiagnosticsResult(
        **{**entry, "config": diagnostics.LearningDiagnosticsConfig(**entry["config"])},
        geometry=diagnostics.GeometryDiagnostic(**geometry),
        supervised=diagnostics.SupervisedDiagnostic(**supervised),
        reward_trajectory=diagnostics.RewardTrajectoryDiagnostic(**reward),
    )


@pytest.fixture
def evidence_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[dict, dict, Path]:
    """Replace only costly scientific measurements; retain real evidence assembly."""
    committed = json.loads((_ROOT / "docs/experiments/phase-2b-failure.json").read_text())
    runtime = {
        "phase": "2B",
        "config": {
            "hidden_size": 64,
            "recurrent_radius": 0.9,
            "learning_rate": 0.05,
            "temperature": 1.0,
            "training_episodes": 2000,
            "evaluation_blocks": 20,
        },
        "seeds": [7],
        "all_passed": False,
        "results": [deepcopy(committed["results"][0])],
        "shuffled_pooled": {"accuracy": 0.5, "correct": 100, "total": 200},
    }
    evidence_path = tmp_path / "phase-2b-failure.json"
    evidence_path.write_text(json.dumps(committed))
    monkeypatch.setattr(diagnostics, "_PHASE_2B_EVIDENCE", evidence_path)
    measured = _measured_result()

    def dataset(rows: int) -> diagnostics._HiddenDataset:
        states = np.zeros((rows, 64), dtype=np.float64)
        return diagnostics._HiddenDataset(
            states=states,
            labels=np.tile(np.array([0, 1], dtype=np.int64), rows // 2),
            delays=np.tile(np.arange(1, 6, dtype=np.int64), rows // 5),
            fixture_digest="a" * 64,
            state_digest=diagnostics._array_digest(states),
        )

    fixtures = diagnostics._DiagnosticFixtures((), (), dataset(2000), dataset(200))
    monkeypatch.setattr(diagnostics, "_build_diagnostic_fixtures", lambda *_: fixtures)
    monkeypatch.setattr(diagnostics, "_run_geometry", lambda *_: measured.geometry)
    monkeypatch.setattr(diagnostics, "_run_supervised_diagnostic", lambda *_: measured.supervised)
    monkeypatch.setattr(diagnostics, "_run_reward_trajectory", lambda *_: measured.reward_trajectory)
    monkeypatch.setattr(diagnostics, "run_reward_learning_benchmark", lambda *_: deepcopy(runtime))
    return committed, runtime, evidence_path


def test_evidence_boundary_accepts_the_unmodified_envelopes(evidence_boundary: tuple) -> None:
    result = diagnostics.run_learning_diagnostics(7)
    assert result.phase_2b_portable_evidence_match is True
    assert result.diagnostic_valid is True
    assert result.repeatable is True
    assert result.classification == "REWARD_CREDIT_FAILURE"


@pytest.mark.parametrize("boundary", ["committed", "runtime"])
@pytest.mark.parametrize(
    "mutation",
    [
        "unexpected_phase",
        "changed_config",
        "unexpected_ordered_seeds",
        "all_passed_true",
        "altered_pooled_control",
        "duplicate_result_seed",
        "missing_result_seed",
        "missing_phase",
        "malformed_results",
    ],
)
def test_malformed_phase_two_b_envelope_invalidates_public_diagnosis(
    evidence_boundary: tuple,
    boundary: str,
    mutation: str,
) -> None:
    committed, runtime, evidence_path = evidence_boundary
    payload = committed if boundary == "committed" else runtime
    if mutation == "unexpected_phase":
        payload["phase"] = "unexpected"
    elif mutation == "changed_config":
        payload["config"]["learning_rate"] = 0.1
    elif mutation == "unexpected_ordered_seeds":
        payload["seeds"] = [29, 17, 7] if boundary == "committed" else [29]
    elif mutation == "all_passed_true":
        payload["all_passed"] = True
    elif mutation == "altered_pooled_control":
        payload["shuffled_pooled"]["correct"] += 1
    elif mutation == "duplicate_result_seed":
        payload["results"].append(deepcopy(payload["results"][0]))
    elif mutation == "missing_result_seed":
        payload["results"].pop()
    elif mutation == "missing_phase":
        del payload["phase"]
    else:
        payload["results"] = {"seed": 7}
    evidence_path.write_text(json.dumps(committed))

    result = diagnostics.run_learning_diagnostics(7)

    assert result.phase_2b_portable_evidence_match is False
    assert result.diagnostic_valid is False
    assert result.classification == "PROTOCOL_MISMATCH"


@pytest.mark.parametrize("target", ["selected", "unselected", "runtime", "both"])
@pytest.mark.parametrize("mutation", ["missing", "malformed"])
@pytest.mark.parametrize(
    "key",
    [
        "all_reset_hidden_equal", "final_block", "normal_training_choice_digest",
        "normal_training_reward_digest", "passed", "per_delay", "post_training",
        "pre_training", "recurrent_choice_digest", "repeatable", "reset_choice_digest",
        "reset_per_delay", "seed", "shuffled_choice_digest", "shuffled_control",
        "shuffled_final_block", "shuffled_per_delay", "shuffled_total_training_reward",
        "shuffled_training_choice_digest", "shuffled_training_reward_digest", "state_reset",
        "total_training_reward",
    ],
)
def test_required_phase_two_b_result_fields_fail_closed(
    evidence_boundary: tuple,
    target: str,
    mutation: str,
    key: str,
) -> None:
    committed, runtime, evidence_path = evidence_boundary
    entries = [committed["results"][2 if target == "unselected" else 0]]
    if target == "runtime":
        entries = [runtime["results"][0]]
    elif target == "both":
        entries.append(runtime["results"][0])
    for entry in entries:
        if mutation == "missing":
            del entry[key]
        else:
            entry[key] = None
    evidence_path.write_text(json.dumps(committed))

    assert diagnostics._match_phase_2b_evidence(7, runtime["results"][0]) is False
    result = diagnostics.run_learning_diagnostics(7)
    assert result.phase_2b_portable_evidence_match is False
    assert result.diagnostic_valid is False
    assert result.classification == "PROTOCOL_MISMATCH"


@pytest.mark.parametrize("boundary", ["committed", "runtime"])
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("post_training", "correct"), 169.0),
        (("post_training", "total"), "200"),
        (("post_training", "accuracy"), None),
        (("post_training", "accuracy"), 0.85),
        (("per_delay", 0, "correct"), True),
        (("per_delay", 0, "accuracy"), True),
        (("per_delay", 0, "delay"), True),
        (("per_delay", 0, "total"), 20),
        (("normal_training_choice_digest",), "not-a-digest"),
        (("repeatable",), 1),
        (("passed",), 0),
        (("total_training_reward",), True),
        (("total_training_reward",), -1),
        (("shuffled_total_training_reward",), 2001),
    ],
)
def test_nested_phase_two_b_fields_reject_malformed_types_and_counts(
    evidence_boundary: tuple,
    boundary: str,
    path: tuple,
    value: object,
) -> None:
    committed, runtime, evidence_path = evidence_boundary
    payload = committed if boundary == "committed" else runtime
    node = payload["results"][0]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    evidence_path.write_text(json.dumps(committed))

    result = diagnostics.run_learning_diagnostics(7)
    assert result.phase_2b_portable_evidence_match is False
    assert result.diagnostic_valid is False
    assert result.classification == "PROTOCOL_MISMATCH"


@pytest.mark.parametrize("malformation", ["missing", "invalid_json", "wrong_root"])
def test_unreadable_phase_two_b_evidence_fails_closed(
    evidence_boundary: tuple, malformation: str,
) -> None:
    _, _, evidence_path = evidence_boundary
    if malformation == "missing":
        evidence_path.unlink()
    else:
        evidence_path.write_text("{" if malformation == "invalid_json" else "[]")
    result = diagnostics.run_learning_diagnostics(7)
    assert result.phase_2b_portable_evidence_match is False
    assert result.diagnostic_valid is False
    assert result.classification == "PROTOCOL_MISMATCH"


def test_runtime_envelope_accepts_the_passing_singleton_seed(evidence_boundary: tuple) -> None:
    committed, runtime, _ = evidence_boundary
    runtime["seeds"] = [17]
    runtime["results"] = [deepcopy(committed["results"][1])]
    runtime["all_passed"] = True
    entry = diagnostics._phase_2b_runtime_entry(17, diagnostics.LearningDiagnosticsConfig())
    assert entry is not None
    assert diagnostics._match_phase_2b_evidence(17, entry) is True


@pytest.mark.parametrize("boundary", ["committed", "runtime"])
def test_second_run_envelope_failure_clears_public_portable_match(
    evidence_boundary: tuple,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    committed, runtime, evidence_path = evidence_boundary
    calls = 0

    def second_run_has_malformed_evidence(*_: object) -> dict:
        nonlocal calls
        calls += 1
        if calls == 2:
            if boundary == "runtime":
                runtime["phase"] = "unexpected"
            else:
                committed["phase"] = "unexpected"
                evidence_path.write_text(json.dumps(committed))
        return deepcopy(runtime)

    monkeypatch.setattr(diagnostics, "run_reward_learning_benchmark", second_run_has_malformed_evidence)
    result = diagnostics.run_learning_diagnostics(7)
    assert result.phase_2b_portable_evidence_match is False
    assert result.repeatable is False
    assert result.diagnostic_valid is False
    assert result.classification == "PROTOCOL_MISMATCH"


def test_unchanged_shuffled_parameters_invalidate_only_local_integrity(
    evidence_boundary: tuple,
) -> None:
    _, runtime, _ = evidence_boundary
    entry = runtime["results"][0]
    entry["shuffled_parameter_digest"] = entry["initial_parameter_digest"]

    assert diagnostics._phase_2b_runtime_local_evidence(entry) is None
    result = diagnostics.run_learning_diagnostics(7)
    assert result.phase_2b_portable_evidence_match is True
    assert result.repeatable is True
    assert result.diagnostic_valid is False
    assert result.classification == "PROTOCOL_MISMATCH"


def test_benchmark_public_key_sets_are_exhaustively_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accidental public additions, omissions, and raw-vector leaks must fail."""
    measured = _measured_result()
    monkeypatch.setattr(
        diagnostics, "run_learning_diagnostics",
        lambda seed, config: replace(measured, seed=seed, config=config),
    )
    payload = diagnostics.run_learning_diagnostics_benchmark()
    assert set(payload) == {
        "all_valid", "classification_counts", "config", "phase", "phase_2b_evidence_digest",
        "results", "seeds",
    }
    config_keys = {
        "hidden_size", "recurrent_radius", "learning_rate", "temperature",
        "training_episodes", "evaluation_blocks", "checkpoint_interval",
    }
    assert set(payload["config"]) == config_keys
    assert set(payload["classification_counts"]) == {"REWARD_CREDIT_FAILURE"}
    for result in payload["results"]:
        assert set(result) == {
            "seed", "config", "geometry", "supervised", "reward_trajectory", "classification",
            "matrix_digests_before", "matrix_digests_after", "phase_2b_portable_evidence_match",
            "diagnostic_valid", "repeatable", "training_fixture_digest", "training_state_digest",
            "evaluation_fixture_digest", "evaluation_state_digest",
        }
        assert set(result["config"]) == config_keys
        geometry = result["geometry"]
        assert set(geometry) == {
            "training", "evaluation", "per_delay", "training_margins", "evaluation_margins",
            "per_delay_margins", "delay_five_to_one_median_ratio", "probe_digest", "geometry_passed",
        }
        for margin in (
            geometry["training_margins"], geometry["evaluation_margins"],
            *(summary for _, summary in geometry["per_delay_margins"]),
        ):
            assert set(margin) == {"minimum", "percentile_10", "median"}
        supervised = result["supervised"]
        assert set(supervised) == {
            "overall", "per_delay", "checkpoints", "parameter_digest", "supervised_passed",
        }
        reward = result["reward_trajectory"]
        assert set(reward) == {
            "overall", "per_delay", "checkpoints", "gradient_blocks", "zero_norm_block_count",
            "total_training_reward", "final_block", "parameter_digest_before",
            "parameter_digest_after", "matrix_digests_before", "matrix_digests_after",
            "training_choice_digest", "training_reward_digest", "reward_passed",
        }
        for checkpoint in supervised["checkpoints"]:
            assert set(checkpoint) == {"episode", "overall", "per_delay", "parameter_digest"}
        for checkpoint in reward["checkpoints"]:
            assert set(checkpoint) == {
                "episode", "overall", "per_delay", "mean_correct_action_probability",
                "percentile_10_correct_action_probability",
                "mean_expected_bandit_to_supervised_norm_ratio", "zero_norm_block_count",
                "parameter_digest",
            }
            assert "gradient_blocks" not in checkpoint
        for block in reward["gradient_blocks"]:
            assert set(block) == {
                "episode", "mean_expected_bandit_to_supervised_norm_ratio", "cosine",
            }
        for count in (geometry["training"], geometry["evaluation"], reward["final_block"]):
            assert set(count) == {"correct", "total"}
        for summary in (
            geometry, supervised, reward, *supervised["checkpoints"], *reward["checkpoints"],
        ):
            if "overall" in summary:
                assert set(summary["overall"]) == {"correct", "total"}
            for _, count in summary["per_delay"]:
                assert set(count) == {"correct", "total"}

    def no_raw_vectors(value: object) -> None:
        if isinstance(value, dict):
            assert {
                "sampled_episode_gradients", "supervised_episode_gradients",
                "expected_bandit_to_supervised_norm_ratios", "sampled_gradient",
                "supervised_gradient", "correct_action_probabilities",
            }.isdisjoint(value)
            for nested in value.values():
                no_raw_vectors(nested)
        elif isinstance(value, list):
            for nested in value:
                no_raw_vectors(nested)

    no_raw_vectors(payload)
