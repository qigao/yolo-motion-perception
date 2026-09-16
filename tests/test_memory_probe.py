import hashlib
import json
import runpy
import subprocess
import sys
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from neural_state_machine import Cue, DelayedCueTask, RecurrentPolicy
from neural_state_machine import memory_probe as probe_module
from neural_state_machine.memory_probe import (
    FittedLinearProbe,
    MemoryProbeConfig,
    fit_linear_probe,
)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "7", None, np.int64(7)])
def test_run_memory_probe_rejects_invalid_seed_before_running(
    seed: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(*args):
        pytest.fail("invalid input reached the probe protocol")

    monkeypatch.setattr(probe_module, "_run_probe_once", unexpected_run)
    with pytest.raises(ValueError, match="non-negative Python integer"):
        probe_module.run_memory_probe(seed, MemoryProbeConfig(training_blocks=1, evaluation_blocks=1))


def test_run_memory_probe_rejects_invalid_config_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(*args):
        pytest.fail("invalid config reached the probe protocol")

    monkeypatch.setattr(probe_module, "_run_probe_once", unexpected_run)
    with pytest.raises(ValueError, match="config must be a MemoryProbeConfig"):
        probe_module.run_memory_probe(7, {})


@pytest.mark.parametrize(
    "changed",
    [None, "seed", "config", "training", "recurrent", "per_delay", "state_reset",
     "all_reset_hidden_equal", "output_weight_digest_before", "output_weight_digest_after",
     "probe_digest", "recurrent_choice_digest", "reset_choice_digest"],
)
def test_run_memory_probe_compares_two_complete_independent_runs(
    monkeypatch: pytest.MonkeyPatch, changed: str | None,
) -> None:
    config = MemoryProbeConfig(training_blocks=1, evaluation_blocks=1)
    first = probe_module._ProbeRun(
        seed=7,
        config=config,
        training=probe_module.ProbeAccuracy(10, 10),
        recurrent=probe_module.ProbeAccuracy(10, 10),
        per_delay=tuple((delay, probe_module.ProbeAccuracy(2, 2)) for delay in range(1, 6)),
        state_reset=probe_module.ProbeAccuracy(5, 10),
        all_reset_hidden_equal=True,
        output_weight_digest_before="output",
        output_weight_digest_after="output",
        probe_digest="probe",
        recurrent_choice_digest="recurrent",
        reset_choice_digest="reset",
    )
    mutations = {
        "seed": 17,
        "config": replace(config, hidden_size=8),
        "training": probe_module.ProbeAccuracy(9, 10),
        "recurrent": probe_module.ProbeAccuracy(9, 10),
        "per_delay": ((1, probe_module.ProbeAccuracy(1, 2)), *first.per_delay[1:]),
        "state_reset": probe_module.ProbeAccuracy(6, 10),
        "all_reset_hidden_equal": False,
        "output_weight_digest_before": "changed",
        "output_weight_digest_after": "changed",
        "probe_digest": "changed",
        "recurrent_choice_digest": "changed",
        "reset_choice_digest": "changed",
    }
    second = replace(first) if changed is None else replace(first, **{changed: mutations[changed]})
    runs = iter((first, second))

    def fake_run(seed: int, received: MemoryProbeConfig):
        if seed != 7 or received is not config:
            pytest.fail("wrapper changed the requested seed or config")
        return next(runs)

    monkeypatch.setattr(probe_module, "_run_probe_once", fake_run)
    result = probe_module.run_memory_probe(7, config)

    assert result.repeatable is (changed is None)
    for field in first.__dataclass_fields__:
        assert getattr(result, field) == getattr(first, field)


def test_real_small_probe_run_is_exactly_repeatable() -> None:
    config = MemoryProbeConfig(
        hidden_size=8, recurrent_radius=0.7, training_blocks=2,
        evaluation_blocks=2, regularization=1e-4,
    )

    first = probe_module.run_memory_probe(11, config)
    second = probe_module.run_memory_probe(11, config)

    assert first == second
    assert first.repeatable is True


def _passing_result(seed: int = 7) -> probe_module.MemoryProbeResult:
    return probe_module.MemoryProbeResult(
        seed=seed,
        config=MemoryProbeConfig(),
        training=probe_module.ProbeAccuracy(2000, 2000),
        recurrent=probe_module.ProbeAccuracy(180, 200),
        per_delay=tuple(
            (delay, probe_module.ProbeAccuracy(correct, 40))
            for delay, correct in enumerate((34, 36, 36, 37, 37), start=1)
        ),
        state_reset=probe_module.ProbeAccuracy(100, 200),
        all_reset_hidden_equal=True,
        output_weight_digest_before="same",
        output_weight_digest_after="same",
        probe_digest="probe",
        recurrent_choice_digest="recurrent",
        reset_choice_digest="reset",
        repeatable=True,
    )


@pytest.mark.parametrize(
    ("counts", "passed"),
    [((34, 36, 36, 37, 37), True), ((35, 36, 36, 36, 36), False),
     ((33, 36, 37, 37, 37), False)],
    ids=["exact-overall-and-delay-boundary", "overall-below-boundary", "delay-below-boundary"],
)
def test_acceptance_accuracy_boundaries_with_coherent_counts(counts, passed: bool) -> None:
    result = replace(
        _passing_result(),
        recurrent=probe_module.ProbeAccuracy(sum(counts), 200),
        per_delay=tuple(
            (delay, probe_module.ProbeAccuracy(correct, 40))
            for delay, correct in enumerate(counts, start=1)
        ),
    )

    assert probe_module._passes_acceptance(result) is passed


@pytest.mark.parametrize(
    ("field", "value"),
    [("training", probe_module.ProbeAccuracy(1999, 1999)),
     ("state_reset", probe_module.ProbeAccuracy(99, 200)),
     ("state_reset", probe_module.ProbeAccuracy(101, 200)),
     ("all_reset_hidden_equal", False), ("output_weight_digest_after", "changed"),
     ("repeatable", False)],
)
def test_acceptance_requires_each_locked_control(field: str, value: object) -> None:
    assert not probe_module._passes_acceptance(replace(_passing_result(), **{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [("hidden_size", 8), ("recurrent_radius", 0.7), ("training_blocks", 1),
     ("evaluation_blocks", 1), ("regularization", 1e-4)],
)
def test_acceptance_rejects_non_protocol_config(field: str, value: object) -> None:
    result = _passing_result()
    assert not probe_module._passes_acceptance(
        replace(result, config=replace(result.config, **{field: value}))
    )


@pytest.mark.parametrize("malformation", ["missing", "duplicate", "order", "total", "correct"])
def test_acceptance_rejects_malformed_delay_accounting(malformation: str) -> None:
    result = _passing_result()
    malformed = {
        "missing": result.per_delay[:-1],
        "duplicate": ((2, result.per_delay[0][1]), *result.per_delay[1:]),
        "order": tuple(reversed(result.per_delay)),
        "total": ((1, probe_module.ProbeAccuracy(35, 41)), *result.per_delay[1:]),
        "correct": ((1, probe_module.ProbeAccuracy(35, 40)), *result.per_delay[1:]),
    }
    assert not probe_module._passes_acceptance(replace(result, per_delay=malformed[malformation]))


@pytest.mark.parametrize(
    "seeds", [None, 7, [], [7, 7], [True], [-1], [1.5], ["7"], "7", [7, False]],
)
def test_benchmark_rejects_invalid_seeds_before_running(
    seeds: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(*args):
        pytest.fail("invalid seed sequence reached the probe protocol")

    monkeypatch.setattr(probe_module, "_run_probe_once", unexpected_run)
    with pytest.raises(ValueError):
        probe_module.run_memory_probe_benchmark(seeds)


@pytest.mark.parametrize("failing_seed", [None, 7, 17, 29])
def test_benchmark_payload_is_stable_json_data(
    monkeypatch: pytest.MonkeyPatch, failing_seed: int | None,
) -> None:
    results = {seed: _passing_result(seed) for seed in (7, 17, 29)}
    if failing_seed is not None:
        results[failing_seed] = replace(results[failing_seed], repeatable=False)
    monkeypatch.setattr(probe_module, "run_memory_probe", lambda seed, config=None: results[seed])

    payload = probe_module.run_memory_probe_benchmark()

    assert payload["phase"] == "2A"
    assert payload["all_passed"] is (failing_seed is None)
    assert payload["reward_baseline_gates_phase_2a"] is False
    assert [entry["seed"] for entry in payload["results"]] == [7, 17, 29]
    assert payload["results"][0] == {
        "seed": 7,
        "config": {"hidden_size": 64, "recurrent_radius": 0.9, "training_blocks": 200,
                   "evaluation_blocks": 20, "regularization": 1e-6},
        "passed": failing_seed != 7,
        "training": {"correct": 2000, "total": 2000, "accuracy": 1.0},
        "recurrent": {"correct": 180, "total": 200, "accuracy": 0.9},
        "per_delay": {
            "1": {"correct": 34, "total": 40, "accuracy": 0.85},
            "2": {"correct": 36, "total": 40, "accuracy": 0.9},
            "3": {"correct": 36, "total": 40, "accuracy": 0.9},
            "4": {"correct": 37, "total": 40, "accuracy": 0.925},
            "5": {"correct": 37, "total": 40, "accuracy": 0.925},
        },
        "state_reset": {"correct": 100, "total": 200, "accuracy": 0.5},
        "all_reset_hidden_equal": True,
        "output_weight_digest_before": "same", "output_weight_digest_after": "same",
        "probe_digest": "probe", "recurrent_choice_digest": "recurrent",
        "reset_choice_digest": "reset", "repeatable": failing_seed != 7,
    }
    assert [entry["passed"] for entry in payload["results"]] == [
        seed != failing_seed for seed in (7, 17, 29)
    ]
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_phase_2a_public_api_is_exported() -> None:
    import neural_state_machine as package

    for name in (
        "FittedLinearProbe", "MemoryProbeConfig", "MemoryProbeResult", "ProbeAccuracy",
        "fit_linear_probe", "run_memory_probe", "run_memory_probe_benchmark",
    ):
        assert name in package.__all__
        assert getattr(package, name) is getattr(probe_module, name)


@pytest.mark.parametrize("passed", [False, True])
def test_probe_benchmark_cli_exit_status_tracks_acceptance(
    passed: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"phase": "2A", "all_passed": passed, "results": []}
    monkeypatch.setattr(probe_module, "run_memory_probe_benchmark", lambda: payload)

    with pytest.raises(SystemExit) as caught:
        runpy.run_path("scripts/benchmark_memory_probe.py", run_name="__main__")

    assert caught.value.code == (0 if passed else 1)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def test_probe_benchmark_cli_emits_one_compact_sorted_json_object() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/benchmark_memory_probe.py"],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)

    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    assert completed.stdout.rstrip() == json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert payload["all_passed"] is True
    assert [entry["seed"] for entry in payload["results"]] == [7, 17, 29]


@pytest.mark.parametrize(
    ("name", "value"),
    [("hidden_size", 0), ("hidden_size", True), ("training_blocks", -1),
     ("training_blocks", 1.5), ("evaluation_blocks", 0), ("evaluation_blocks", False)],
)
def test_memory_probe_config_rejects_invalid_counts(name: str, value: object) -> None:
    values = {
        "hidden_size": 64,
        "recurrent_radius": 0.9,
        "training_blocks": 200,
        "evaluation_blocks": 20,
        "regularization": 1e-6,
    }
    values[name] = value
    with pytest.raises(ValueError):
        MemoryProbeConfig(**values)


@pytest.mark.parametrize("radius", [-0.1, 1.0, np.inf, np.nan, True])
def test_memory_probe_config_rejects_invalid_radius(radius: object) -> None:
    with pytest.raises(ValueError):
        MemoryProbeConfig(recurrent_radius=radius)


@pytest.mark.parametrize("strength", [0.0, -1.0, np.inf, np.nan, True])
def test_memory_probe_config_rejects_invalid_regularization(strength: object) -> None:
    with pytest.raises(ValueError):
        MemoryProbeConfig(regularization=strength)


def test_memory_probe_config_defaults_are_the_locked_protocol() -> None:
    assert MemoryProbeConfig() == MemoryProbeConfig(hidden_size=64, recurrent_radius=0.9,
                                                     training_blocks=200, evaluation_blocks=20,
                                                     regularization=1e-6)


def test_probe_fixtures_are_shuffled_balanced_fresh_blocks() -> None:
    fixtures = probe_module._build_fixtures(DelayedCueTask(), np.random.default_rng(21), 3)
    assert type(fixtures) is tuple
    assert len(fixtures) == 30
    for start in range(0, 30, 10):
        block = fixtures[start : start + 10]
        assert Counter(episode.correct_action_index for episode in block) == {0: 5, 1: 5}
        assert Counter(episode.delay_steps for episode in block) == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
        assert len({(episode.cue, episode.delay_steps) for episode in block}) == 10
    assert len({episode.delay_stimuli[0][2] for episode in fixtures}) == 30


def _manual_hidden(policy: RecurrentPolicy, episode, *, reset_before_decision: bool) -> np.ndarray:
    policy.reset_state()
    policy.advance(episode.cue_stimulus)
    for stimulus in episode.delay_stimuli:
        policy.advance(stimulus)
    if reset_before_decision:
        policy.reset_state()
    return policy.advance(episode.decision_stimulus)


@pytest.mark.parametrize("reset", [False, True])
def test_collect_hidden_matches_the_literal_advance_only_protocol(reset: bool) -> None:
    episode = DelayedCueTask().build_episode(Cue.RIGHT, 4, np.random.default_rng(4))
    actual_policy = RecurrentPolicy(4, 2, hidden_size=8, seed=7)
    reference_policy = RecurrentPolicy(4, 2, hidden_size=8, seed=7)
    actual = probe_module._collect_hidden(actual_policy, episode, reset_before_decision=reset)
    expected = _manual_hidden(reference_policy, episode, reset_before_decision=reset)
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    with pytest.raises(RuntimeError, match="preceding decision"):
        actual_policy.learn(1.0)


def test_reset_dataset_has_identical_states_and_literal_side_labels() -> None:
    fixtures = probe_module._build_fixtures(DelayedCueTask(), np.random.default_rng(3), 2)
    policy = RecurrentPolicy(4, 2, hidden_size=8, seed=9)
    dataset = probe_module._collect_dataset(policy, fixtures, reset_before_decision=True)
    assert dataset.states.shape == (20, 8)
    assert dataset.labels.shape == (20,)
    assert dataset.delays.shape == (20,)
    assert not dataset.states.flags.writeable
    assert not dataset.labels.flags.writeable
    assert not dataset.delays.flags.writeable
    assert np.array_equal(dataset.states, np.repeat(dataset.states[:1], 20, axis=0))
    np.testing.assert_array_equal(dataset.labels, [episode.correct_action_index for episode in fixtures])
    np.testing.assert_array_equal(dataset.delays, [episode.delay_steps for episode in fixtures])


def test_dataset_collection_never_changes_policy_parameters_or_creates_eligibility() -> None:
    policy = RecurrentPolicy(4, 2, hidden_size=8, seed=12)
    fixtures = probe_module._build_fixtures(DelayedCueTask(), np.random.default_rng(8), 2)
    input_before, recurrent_before, output_before = (policy._input_weights.copy(),
                                                      policy._recurrent_weights.copy(),
                                                      policy._output_weights.copy())
    digest_before = policy.output_weight_digest()
    probe_module._collect_dataset(policy, fixtures, reset_before_decision=False)
    probe_module._collect_dataset(policy, fixtures, reset_before_decision=True)
    np.testing.assert_array_equal(policy._input_weights, input_before)
    np.testing.assert_array_equal(policy._recurrent_weights, recurrent_before)
    np.testing.assert_array_equal(policy._output_weights, output_before)
    assert policy.output_weight_digest() == digest_before
    with pytest.raises(RuntimeError, match="preceding decision"):
        policy.learn(1.0)


def test_probe_scoring_uses_literal_labels_and_groups_all_five_delays() -> None:
    dataset = probe_module._StateDataset(
        states=np.arange(20, dtype=np.float64).reshape(10, 2),
        labels=np.array([0, 1] * 5),
        delays=np.repeat(np.arange(1, 6), 2),
    )
    choices = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0], dtype=np.int64)

    overall, per_delay = probe_module._score_predictions(dataset, choices)

    assert overall == probe_module.ProbeAccuracy(7, 10)
    assert overall.accuracy == 0.7
    assert per_delay == (
        (1, probe_module.ProbeAccuracy(1, 2)),
        (2, probe_module.ProbeAccuracy(2, 2)),
        (3, probe_module.ProbeAccuracy(1, 2)),
        (4, probe_module.ProbeAccuracy(2, 2)),
        (5, probe_module.ProbeAccuracy(1, 2)),
    )
    expected = hashlib.sha256(np.asarray(choices, dtype=np.uint8).tobytes()).hexdigest()
    assert probe_module._choice_digest(choices) == expected


def test_run_probe_once_matches_independent_protocol() -> None:
    seed = 7
    config = MemoryProbeConfig(
        hidden_size=9,
        recurrent_radius=0.7,
        training_blocks=3,
        evaluation_blocks=2,
        regularization=1e-4,
    )
    policy = RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=seed,
        recurrent_radius=config.recurrent_radius,
    )
    task = DelayedCueTask()
    train_fixtures = probe_module._build_fixtures(
        task,
        np.random.default_rng(np.random.SeedSequence([seed, 0x50524F42])),
        config.training_blocks,
    )
    evaluation_fixtures = probe_module._build_fixtures(
        task,
        np.random.default_rng(np.random.SeedSequence([seed, 0x4556414C])),
        config.evaluation_blocks,
    )
    assert (
        train_fixtures[0].delay_stimuli[0].tobytes()
        != evaluation_fixtures[0].delay_stimuli[0].tobytes()
    )
    before = policy.output_weight_digest()
    training_data = probe_module._collect_dataset(
        policy, train_fixtures, reset_before_decision=False
    )
    recurrent_data = probe_module._collect_dataset(
        policy, evaluation_fixtures, reset_before_decision=False
    )
    reset_data = probe_module._collect_dataset(
        policy, evaluation_fixtures, reset_before_decision=True
    )
    fitted = fit_linear_probe(
        training_data.states,
        training_data.labels,
        regularization=config.regularization,
    )
    training_choices = fitted.predict(training_data.states)
    recurrent_choices = fitted.predict(recurrent_data.states)
    reset_choices = fitted.predict(reset_data.states)
    training, _ = probe_module._score_predictions(training_data, training_choices)
    recurrent, per_delay = probe_module._score_predictions(
        recurrent_data, recurrent_choices
    )
    state_reset, _ = probe_module._score_predictions(reset_data, reset_choices)

    actual = probe_module._run_probe_once(seed, config)

    assert actual.seed == seed
    assert actual.config == config
    assert actual.training == training
    assert actual.recurrent == recurrent
    assert actual.per_delay == per_delay
    assert actual.state_reset == state_reset
    assert actual.all_reset_hidden_equal is np.array_equal(
        reset_data.states,
        np.repeat(reset_data.states[:1], reset_data.states.shape[0], axis=0),
    )
    assert actual.output_weight_digest_before == before
    assert actual.output_weight_digest_after == before
    assert actual.probe_digest == fitted.digest()
    assert actual.recurrent_choice_digest == probe_module._choice_digest(
        recurrent_choices
    )
    assert actual.reset_choice_digest == probe_module._choice_digest(reset_choices)


def test_probe_protocol_has_one_shared_decision_vector_for_both_cues() -> None:
    task = DelayedCueTask()
    left = task.build_episode(Cue.LEFT, 1, np.random.default_rng(1))
    right = task.build_episode(Cue.RIGHT, 5, np.random.default_rng(2))

    np.testing.assert_array_equal(left.decision_stimulus, right.decision_stimulus)
    np.testing.assert_array_equal(left.decision_stimulus, [0.0, 0.0, 0.0, 1.0])


def test_fitted_probe_defensively_copies_readonly_float64_weights() -> None:
    source = np.array([1, -2], dtype=np.int64)
    probe = FittedLinearProbe(source, 0)
    source[:] = 99

    np.testing.assert_array_equal(probe.weights, [1.0, -2.0])
    assert probe.weights.dtype == np.float64
    assert not probe.weights.flags.writeable
    assert not np.shares_memory(probe.weights, source)


def test_prediction_is_readonly_independent_and_uses_left_on_exact_tie() -> None:
    states = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 8.0]])
    probe = FittedLinearProbe(np.array([1.0, 0.0]), 0.0)
    prediction = probe.predict(states)
    states[:] = 100.0

    np.testing.assert_array_equal(prediction, [1, 0, 0])
    assert prediction.dtype == np.int64
    assert not prediction.flags.writeable


def test_probe_digest_covers_shape_weights_and_float64_bias() -> None:
    probe = FittedLinearProbe(np.array([1.5, -2.0]), 0.25)
    expected = hashlib.sha256()
    expected.update(str((2,)).encode("ascii"))
    expected.update(np.array([1.5, -2.0], dtype=np.float64).tobytes(order="C"))
    expected.update(np.asarray(0.25, dtype=np.float64).tobytes())

    assert probe.digest() == expected.hexdigest()


@pytest.mark.parametrize(
    ("weights", "bias"),
    [
        (1.0, 0.0),
        (np.ones((1, 2)), 0.0),
        (np.array([]), 0.0),
        (np.array([np.nan]), 0.0),
        (np.array([1.0]), np.inf),
    ],
)
def test_fitted_probe_rejects_invalid_parameters(weights: object, bias: object) -> None:
    with pytest.raises(ValueError):
        FittedLinearProbe(weights, bias)


@pytest.mark.parametrize(
    "states",
    [1.0, np.ones(2), np.ones((1, 1, 2)), [[np.nan, 0.0]], [[1.0]]],
)
def test_predict_rejects_invalid_state_matrices(states: object) -> None:
    with pytest.raises(ValueError):
        FittedLinearProbe(np.ones(2), 0.0).predict(states)


def test_fit_linear_probe_uses_closed_form_ridge_without_bias_penalty() -> None:
    states = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    probe = fit_linear_probe(states, labels, regularization=1.0)

    np.testing.assert_allclose(probe.weights, [6.0 / 11.0], rtol=0.0, atol=1e-15)
    assert probe.bias == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_array_equal(probe.predict(states), labels)


@pytest.mark.parametrize(
    ("states", "labels"),
    [
        (np.empty((0, 2)), np.empty(0, dtype=np.int64)),
        (np.empty((2, 0)), np.array([0, 1])),
        (np.ones(2), np.array([0, 1])),
        (np.array([[0.0], [np.inf]]), np.array([0, 1])),
        (np.ones((2, 1)), np.array([[0], [1]])),
        (np.ones((2, 1)), np.array([0])),
        (np.ones((2, 1)), np.array([0.0, 1.0])),
        (np.ones((2, 1)), np.array([False, True])),
        (np.ones((2, 1)), np.array([0, 2])),
        (np.ones((2, 1)), np.array([0, 0])),
    ],
)
def test_fit_linear_probe_rejects_invalid_datasets(states: object, labels: object) -> None:
    with pytest.raises(ValueError):
        fit_linear_probe(states, labels, regularization=1e-6)


@pytest.mark.parametrize("regularization", [0.0, -1.0, np.inf, np.nan, True])
def test_fit_linear_probe_rejects_invalid_regularization(regularization: object) -> None:
    with pytest.raises(ValueError):
        fit_linear_probe(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            regularization=regularization,
        )


def test_fit_wraps_numpy_solve_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = np.linalg.LinAlgError("singular")

    def fail_solve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raise failure

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    with pytest.raises(RuntimeError, match="linear probe solve failed") as caught:
        fit_linear_probe(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            regularization=1e-6,
        )
    assert caught.value.__cause__ is failure
