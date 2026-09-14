import hashlib
from collections import Counter

import numpy as np
import pytest

from neural_state_machine import Cue, DelayedCueTask, RecurrentPolicy
from neural_state_machine import memory_probe as probe_module
from neural_state_machine.memory_probe import (
    FittedLinearProbe,
    MemoryProbeConfig,
    fit_linear_probe,
)


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
