import hashlib
from collections import Counter
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import neural_state_machine.memory_benchmark as benchmark
from neural_state_machine.memory_benchmark import AccuracyCount, MemoryExperimentConfig
from neural_state_machine.memory_task import Cue, DelayedCueTask
from neural_state_machine.policy import RecurrentPolicy


@pytest.mark.parametrize("field", ["hidden_size", "evaluation_blocks"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_config_rejects_nonpositive_or_noninteger_sizes(field, value) -> None:
    with pytest.raises(ValueError):
        MemoryExperimentConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -10, 1, 19, True, 20.0, "20", None])
def test_config_requires_complete_positive_training_blocks(value) -> None:
    with pytest.raises(ValueError):
        MemoryExperimentConfig(training_episodes=value)


@pytest.mark.parametrize("value", [0.0, -0.1, np.inf, -np.inf, np.nan, True, "0.1", None])
def test_config_rejects_invalid_learning_rate(value) -> None:
    with pytest.raises(ValueError):
        MemoryExperimentConfig(learning_rate=value)


@pytest.mark.parametrize("value", [-0.1, 1.0, np.inf, np.nan, True, "0.9", None])
def test_config_rejects_invalid_recurrent_radius(value) -> None:
    with pytest.raises(ValueError):
        MemoryExperimentConfig(recurrent_radius=value)


@pytest.mark.parametrize("field", ["epsilon_start", "epsilon_end"])
@pytest.mark.parametrize("value", [-0.01, 1.01, np.inf, np.nan, True, None])
def test_config_rejects_invalid_exploration_probabilities(field, value) -> None:
    with pytest.raises(ValueError):
        MemoryExperimentConfig(**{field: value})


def test_config_rejects_increasing_exploration() -> None:
    with pytest.raises(ValueError):
        MemoryExperimentConfig(epsilon_start=0.1, epsilon_end=0.2)


def test_valid_boundary_config_is_frozen() -> None:
    config = MemoryExperimentConfig(
        hidden_size=1,
        evaluation_blocks=1,
        training_episodes=10,
        learning_rate=1.0,
        recurrent_radius=0.0,
        epsilon_start=1.0,
        epsilon_end=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        config.training_episodes = 20
    MemoryExperimentConfig(epsilon_start=0.0, epsilon_end=0.0)
    MemoryExperimentConfig(epsilon_start=1.0, epsilon_end=1.0)


@pytest.mark.parametrize(
    ("correct", "total"),
    [
        (-1, 10),
        (11, 10),
        (0, 0),
        (0, -1),
        (True, 10),
        (1, True),
        (1.0, 10),
        (1, 10.0),
        (np.int64(1), 10),
        (1, None),
    ],
)
def test_accuracy_count_rejects_invalid_counts(correct, total) -> None:
    with pytest.raises(ValueError):
        AccuracyCount(correct, total)


@pytest.mark.parametrize(
    ("correct", "total", "expected"), [(0, 10, 0.0), (3, 4, 0.75), (7, 7, 1.0)]
)
def test_accuracy_is_literal_ratio_and_counts_are_frozen(correct, total, expected) -> None:
    count = AccuracyCount(correct, total)
    assert count.accuracy == expected
    with pytest.raises(FrozenInstanceError):
        count.correct = 1


def test_every_shuffled_block_contains_each_cue_delay_pair_once() -> None:
    expected = Counter(
        {
            (Cue.LEFT, 1): 1,
            (Cue.LEFT, 2): 1,
            (Cue.LEFT, 3): 1,
            (Cue.LEFT, 4): 1,
            (Cue.LEFT, 5): 1,
            (Cue.RIGHT, 1): 1,
            (Cue.RIGHT, 2): 1,
            (Cue.RIGHT, 3): 1,
            (Cue.RIGHT, 4): 1,
            (Cue.RIGHT, 5): 1,
        }
    )
    left_rng = np.random.default_rng(8)
    right_rng = np.random.default_rng(8)
    orders = []
    for _ in range(4):
        block = benchmark._balanced_cases(left_rng)
        assert Counter(block) == expected
        assert block == benchmark._balanced_cases(right_rng)
        orders.append(tuple(block))
    assert len(set(orders)) > 1


@pytest.mark.parametrize(("index", "expected"), [(0, 0.8), (9, 0.5), (19, 1 / 6)])
def test_epsilon_includes_endpoints_and_interpolates_each_episode(index, expected) -> None:
    config = MemoryExperimentConfig(training_episodes=20, epsilon_start=0.8, epsilon_end=1 / 6)
    assert benchmark._epsilon(index, config) == pytest.approx(expected)


class _RecordingPolicy(RecurrentPolicy):
    """Record boundary arguments without replacing neural or learning behavior."""

    def __init__(self):
        super().__init__(input_size=4, action_count=2, hidden_size=8, seed=7)
        self.calls = []

    def reset_state(self):
        self.calls.append(("reset",))
        return super().reset_state()

    def advance(self, stimulus):
        self.calls.append(("advance", stimulus))
        return super().advance(stimulus)

    def decide(self, stimulus, legal_action_indices, *, explore_probability=0.0, rng=None):
        self.calls.append(("decide", stimulus, legal_action_indices, explore_probability, rng))
        return super().decide(
            stimulus,
            legal_action_indices,
            explore_probability=explore_probability,
            rng=rng,
        )

    def learn(self, reward):
        self.calls.append(("learn", reward))
        return super().learn(reward)


@pytest.mark.parametrize("cue", [Cue.LEFT, Cue.RIGHT])
@pytest.mark.parametrize("delay", [1, 3, 5])
@pytest.mark.parametrize("learn", [False, True])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_episode_passes_only_stimuli_and_legal_actions_in_protocol_order(
    cue,
    delay,
    learn,
    reset,
    epsilon,
) -> None:
    task = DelayedCueTask()
    rng = np.random.default_rng(31)
    episode = task.build_episode(cue, delay, rng)
    policy = _RecordingPolicy()
    before = policy.output_weight_digest()

    action, reward = benchmark._run_episode(
        policy,
        task,
        episode,
        explore_probability=epsilon,
        rng=rng,
        learn=learn,
        reset_before_decision=reset,
    )

    expected_names = ["reset"] + ["advance"] * (delay + 1)
    expected_names += (["reset"] if reset else []) + ["decide"]
    expected_names += ["learn"] if learn else []
    assert [call[0] for call in policy.calls] == expected_names
    stimuli = [call[1] for call in policy.calls if call[0] == "advance"]
    assert stimuli[0] is episode.cue_stimulus
    np.testing.assert_array_equal(
        stimuli[0], [1.0, 0.0, 0.0, 0.0] if cue == Cue.LEFT else [0.0, 1.0, 0.0, 0.0]
    )
    for actual, expected in zip(stimuli[1:], episode.delay_stimuli, strict=True):
        assert actual is expected
        assert actual.shape == (4,)
        np.testing.assert_array_equal(actual[[0, 1, 3]], [0.0, 0.0, 0.0])
    decision_call = next(call for call in policy.calls if call[0] == "decide")
    assert decision_call[1] is episode.decision_stimulus
    np.testing.assert_array_equal(decision_call[1], [0.0, 0.0, 0.0, 1.0])
    assert decision_call[2] == (0, 1)
    assert all(type(index) is int for index in decision_call[2])
    assert decision_call[3] == epsilon
    assert decision_call[4] is (rng if epsilon > 0 else None)
    assert action in (0, 1)
    assert reward == (1.0 if action == int(cue) else -1.0)
    if learn:
        assert policy.calls[-1] == ("learn", reward)
        assert policy.output_weight_digest() != before
        with pytest.raises(RuntimeError):
            policy.learn(reward)
    else:
        assert policy.output_weight_digest() == before


def _literal_fixtures():
    task = DelayedCueTask()
    rng = np.random.default_rng(18)
    return tuple(
        task.build_episode(cue, delay, rng)
        for _ in range(20)
        for cue in (Cue.LEFT, Cue.RIGHT)
        for delay in range(1, 6)
    )


def test_evaluation_fixtures_are_immutable_fresh_balanced_blocks() -> None:
    fixtures = benchmark._evaluation_fixtures(DelayedCueTask(), np.random.default_rng(21), 20)
    assert type(fixtures) is tuple
    assert len(fixtures) == 200
    for start in range(0, 200, 10):
        block = fixtures[start : start + 10]
        assert Counter(episode.correct_action_index for episode in block) == {0: 5, 1: 5}
        assert Counter(episode.delay_steps for episode in block) == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
        assert len({(episode.cue, episode.delay_steps) for episode in block}) == 10
    # Fresh draws, not one ten-case block replayed twenty times.
    assert len({episode.delay_stimuli[0][2] for episode in fixtures}) == 200
    for episode in fixtures:
        with pytest.raises(FrozenInstanceError):
            episode.delay_steps = 1
        for stimulus in (episode.cue_stimulus, *episode.delay_stimuli, episode.decision_stimulus):
            assert not stimulus.flags.writeable


@pytest.mark.parametrize("reset", [False, True])
def test_evaluation_scores_literal_labels_with_balanced_integer_counts(reset) -> None:
    # A real policy with a synthetic zero readout always chooses index zero.
    policy = RecurrentPolicy(4, 2, hidden_size=8, seed=3)
    policy._output_weights.fill(0.0)
    fixtures = _literal_fixtures()
    before = policy.output_weight_digest()
    result = benchmark._evaluate(policy, DelayedCueTask(), fixtures, reset_before_decision=reset)

    assert result.overall == AccuracyCount(100, 200)
    assert result.choices == (0,) * 200
    assert result.per_delay == tuple((delay, AccuracyCount(20, 40)) for delay in (1, 2, 3, 4, 5))
    assert type(result.overall.correct) is int
    assert type(result.overall.total) is int
    assert all(
        type(score.correct) is int and type(score.total) is int for _, score in result.per_delay
    )
    assert Counter(episode.correct_action_index for episode in fixtures) == {0: 100, 1: 100}
    assert policy.output_weight_digest() == before


@pytest.mark.parametrize("trained", [False, True])
def test_reset_ablation_gives_one_constant_action_without_changing_learned_weights(trained) -> None:
    policy = RecurrentPolicy(4, 2, hidden_size=8, seed=14)
    fixtures = _literal_fixtures()
    if trained:
        for episode in fixtures[:10]:
            benchmark._run_episode(
                policy, DelayedCueTask(), episode, explore_probability=0.0, rng=None, learn=True
            )
    before_weights = policy.output_weight_digest()
    before_stimuli = tuple(
        tuple(
            stimulus.tobytes()
            for stimulus in (
                episode.cue_stimulus,
                *episode.delay_stimuli,
                episode.decision_stimulus,
            )
        )
        for episode in fixtures
    )
    recurrent = benchmark._evaluate(policy, DelayedCueTask(), fixtures, reset_before_decision=False)
    assert policy.output_weight_digest() == before_weights
    reset = benchmark._evaluate(policy, DelayedCueTask(), fixtures, reset_before_decision=True)
    assert policy.output_weight_digest() == before_weights
    assert len(set(reset.choices)) == 1
    assert reset.overall == AccuracyCount(100, 200)
    assert reset.overall.accuracy == 0.5
    assert all(score == AccuracyCount(20, 40) for _, score in reset.per_delay)
    assert recurrent == benchmark._evaluate(
        policy,
        DelayedCueTask(),
        fixtures,
        reset_before_decision=False,
    )
    assert policy.output_weight_digest() == before_weights
    after_stimuli = tuple(
        tuple(
            stimulus.tobytes()
            for stimulus in (
                episode.cue_stimulus,
                *episode.delay_stimuli,
                episode.decision_stimulus,
            )
        )
        for episode in fixtures
    )
    assert after_stimuli == before_stimuli


def _reference_train(policy, task, rng, config):
    """Independent, explicit schedule and policy calls; no benchmark helpers."""
    rewards = []
    correct = []
    probabilities = np.linspace(config.epsilon_start, config.epsilon_end, config.training_episodes)
    for _ in range(config.training_episodes // 10):
        cases = [(cue, delay) for cue in (Cue.LEFT, Cue.RIGHT) for delay in (1, 2, 3, 4, 5)]
        rng.shuffle(cases)
        for cue, delay in cases:
            episode = task.build_episode(cue, delay, rng)
            policy.reset_state()
            policy.advance(episode.cue_stimulus)
            for stimulus in episode.delay_stimuli:
                policy.advance(stimulus)
            probability = float(probabilities[len(rewards)])
            decision = policy.decide(
                episode.decision_stimulus,
                (0, 1),
                explore_probability=probability,
                rng=rng if probability > 0 else None,
            )
            matched = decision.action_index == int(cue)
            reward = 1 if matched else -1
            policy.learn(reward)
            rewards.append(reward)
            correct.append(int(matched))
    return rewards, correct


@pytest.mark.parametrize("episodes", [20, 2000])
@pytest.mark.parametrize("exploration", [(0.0, 0.0), (0.6, 0.1)])
def test_training_matches_fresh_balanced_schedule_and_updates_only_readout(episodes, exploration):
    config = MemoryExperimentConfig(
        hidden_size=8,
        training_episodes=episodes,
        epsilon_start=exploration[0],
        epsilon_end=exploration[1],
    )
    task = DelayedCueTask()
    policy = RecurrentPolicy(4, 2, hidden_size=8, seed=9)
    reference = RecurrentPolicy(4, 2, hidden_size=8, seed=9)
    input_before = policy._input_weights.copy()
    recurrent_before = policy._recurrent_weights.copy()
    output_before = policy.output_weight_digest()

    reward, final_block = benchmark._train(policy, task, np.random.default_rng(13), config)
    expected_rewards, expected_correct = _reference_train(
        reference,
        task,
        np.random.default_rng(13),
        config,
    )

    assert type(reward) is int
    assert reward == sum(expected_rewards)
    assert len(expected_rewards) == episodes
    assert final_block == AccuracyCount(sum(expected_correct[episodes - 10 : episodes]), 10)
    assert np.array_equal(policy._input_weights, input_before)
    assert np.array_equal(policy._recurrent_weights, recurrent_before)
    assert policy.output_weight_digest() != output_before
    assert policy.output_weight_digest() == reference.output_weight_digest()
    # The final decision was rewarded exactly once, consuming its eligibility.
    with pytest.raises(RuntimeError):
        policy.learn(1.0)


def _reference_choices(policy, fixtures, *, reset):
    choices = []
    for episode in fixtures:
        policy.reset_state()
        policy.advance(episode.cue_stimulus)
        for stimulus in episode.delay_stimuli:
            policy.advance(stimulus)
        if reset:
            policy.reset_state()
        choices.append(policy.decide(episode.decision_stimulus, (0, 1)).action_index)
    return choices


@pytest.mark.parametrize(
    "config",
    [
        MemoryExperimentConfig(),
        MemoryExperimentConfig(
            hidden_size=9,
            learning_rate=0.13,
            recurrent_radius=0.7,
            training_episodes=30,
            evaluation_blocks=3,
            epsilon_start=0.4,
            epsilon_end=0.0,
        ),
    ],
)
def test_run_once_matches_independent_matched_evaluation_and_training(config) -> None:
    # This reference constructs the required four-input/two-action policy and
    # distinct RNG lineages directly; using a training RNG for fixtures changes
    # the expected choices/readout rather than silently changing both sides.
    seed = 7
    policy = RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=seed,
        learning_rate=config.learning_rate,
        recurrent_radius=config.recurrent_radius,
    )
    task = DelayedCueTask()
    train_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x54524149]))
    evaluation_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4556414C]))
    fixtures = []
    for _ in range(config.evaluation_blocks):
        cases = [(cue, delay) for cue in (Cue.LEFT, Cue.RIGHT) for delay in (1, 2, 3, 4, 5)]
        evaluation_rng.shuffle(cases)
        fixtures.extend(task.build_episode(cue, delay, evaluation_rng) for cue, delay in cases)
    fixtures = tuple(fixtures)
    pre_choices = _reference_choices(policy, fixtures, reset=False)
    rewards, correct = _reference_train(policy, task, train_rng, config)
    post_choices = _reference_choices(policy, fixtures, reset=False)
    reset_choices = _reference_choices(policy, fixtures, reset=True)

    result = benchmark._run_once(seed, config)

    assert result.pre_training == AccuracyCount(
        sum(
            action == episode.correct_action_index
            for action, episode in zip(pre_choices, fixtures, strict=True)
        ),
        len(fixtures),
    )
    assert result.post_training == AccuracyCount(
        sum(
            action == episode.correct_action_index
            for action, episode in zip(post_choices, fixtures, strict=True)
        ),
        len(fixtures),
    )
    assert result.state_reset == AccuracyCount(
        config.evaluation_blocks * 5, config.evaluation_blocks * 10
    )
    for delay, score in result.per_delay:
        assert score == AccuracyCount(
            sum(
                action == episode.correct_action_index
                for action, episode in zip(post_choices, fixtures, strict=True)
                if episode.delay_steps == delay
            ),
            config.evaluation_blocks * 2,
        )
    assert tuple(delay for delay, _ in result.per_delay) == (1, 2, 3, 4, 5)
    assert result.total_training_reward == sum(rewards)
    assert result.final_block == AccuracyCount(sum(correct[-10:]), 10)
    assert result.output_weight_digest == policy.output_weight_digest()
    assert result.recurrent_choice_digest == hashlib.sha256(bytes(post_choices)).hexdigest()
    assert result.reset_choice_digest == hashlib.sha256(bytes(reset_choices)).hexdigest()
    with pytest.raises(FrozenInstanceError):
        result.total_training_reward = 0


def test_package_exposes_validated_benchmark_config_and_counts() -> None:
    from neural_state_machine import AccuracyCount as PublicCount
    from neural_state_machine import MemoryExperimentConfig as PublicConfig

    assert PublicCount(3, 4).accuracy == 0.75
    with pytest.raises(ValueError):
        PublicConfig(training_episodes=11)
