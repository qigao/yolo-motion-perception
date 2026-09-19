from __future__ import annotations

from inspect import signature

import numpy as np

from neural_state_machine.r1_e3m_dataset import MechanismSample


def _api():
    from neural_state_machine.r1_e3m_pairs import (
        build_history_pairs,
        score_history_pairs,
    )

    return build_history_pairs, score_history_pairs


def _sample(
    name: str,
    split: str,
    track_id: int,
    prefix_value: float,
    suffix_value: float,
) -> MechanismSample:
    tensor = np.zeros((20, 6), dtype=np.float64)
    tensor[:16, 0] = prefix_value
    tensor[16:20, 0] = suffix_value
    tensor[:, 5] = 1.0
    return MechanismSample(
        window_id=(name.encode("utf-8").hex() + "0" * 64)[:64],
        video_id=f"{split}-video-{track_id}",
        split=split,
        track_id=track_id,
        source_start_seconds=0.0,
        source_end_seconds=2.0,
        tensor=tensor,
    )


def test_pair_builder_has_no_reservoir_state_parameter() -> None:
    build, _ = _api()

    names = set(signature(build).parameters)

    assert "states" not in names
    assert "reservoir_states" not in names
    assert "scores" not in names


def test_training_threshold_and_eval_pairs_are_deterministic() -> None:
    build, _ = _api()
    training = (
        _sample("t0", "train", 1, 0.0, 0.000),
        _sample("t1", "train", 2, 0.2, 0.001),
        _sample("t2", "train", 3, 0.4, 0.002),
    )
    evaluation = (
        _sample("e0", "eval", 10, 0.0, 0.000),
        _sample("e1", "eval", 11, 0.5, 0.001),
    )

    first = build(training, evaluation)
    second = build(training, evaluation)

    assert first == second
    assert first.prefix_threshold > 0.0
    assert len(first.pairs) == 1
    assert first.pairs[0].prefix_distance > first.prefix_threshold
    assert len(first.pair_digest) == 64


def test_pairs_never_join_same_track_identity() -> None:
    build, _ = _api()
    training = (
        _sample("t0", "train", 1, 0.0, 0.000),
        _sample("t1", "train", 2, 0.2, 0.001),
    )
    evaluation = (
        _sample("e0", "eval", 10, 0.0, 0.000),
        _sample("e1", "eval", 11, 0.6, 0.001),
        _sample("e2", "eval", 10, 0.7, 0.0005),
    )

    pair_set = build(training, evaluation)
    by_id = {sample.window_id: sample for sample in evaluation}

    for pair in pair_set.pairs:
        left = by_id[pair.left_window_id]
        right = by_id[pair.right_window_id]
        assert (left.video_id, left.track_id) != (
            right.video_id,
            right.track_id,
        )


def test_state_scoring_cannot_change_frozen_pair_selection() -> None:
    build, score = _api()
    training = (
        _sample("t0", "train", 1, 0.0, 0.000),
        _sample("t1", "train", 2, 0.2, 0.001),
    )
    evaluation = (
        _sample("e0", "eval", 10, 0.0, 0.000),
        _sample("e1", "eval", 11, 0.6, 0.001),
    )
    pair_set = build(training, evaluation)
    before = pair_set.pair_digest

    normal = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    reset = np.zeros((2, 2), dtype=np.float64)
    diagnostics = score(pair_set, evaluation, normal, reset)

    assert pair_set.pair_digest == before
    assert len(diagnostics) == 1
    assert diagnostics[0].normal_state_distance > 0.0
    assert diagnostics[0].reset_state_distance == 0.0


def test_prefix_threshold_uses_all_unordered_training_pairs() -> None:
    build, _ = _api()
    training = (
        _sample("u0", "train", 1, 0.0, 0.0),
        _sample("u1", "train", 2, 0.2, 0.0),
        _sample("u2", "train", 3, 1.0, 0.0),
        _sample("u3", "train", 4, 1.4, 0.0),
    )
    evaluation = (
        _sample("v0", "eval", 10, 0.0, 0.0),
        _sample("v1", "eval", 11, 2.0, 0.001),
    )

    result = build(training, evaluation)

    prefix_vectors = [sample.tensor[:16].reshape(-1) for sample in training]
    distances = []
    for left_index in range(len(training)):
        for right_index in range(left_index + 1, len(training)):
            left = prefix_vectors[left_index]
            right = prefix_vectors[right_index]
            distances.append(
                float(np.sqrt(np.mean(np.square(left - right), dtype=np.float64)))
            )
    expected = float(np.median(np.asarray(distances, dtype=np.float64)))

    assert np.isclose(result.prefix_threshold, expected)


def test_eval_filters_by_prefix_before_selecting_suffix_nearest_neighbor() -> None:
    build, _ = _api()
    training = (
        _sample("w0", "train", 1, 0.0, 0.0),
        _sample("w1", "train", 2, 0.1, 0.0),
        _sample("w2", "train", 3, 1.0, 0.0),
    )
    anchor = _sample("x0", "eval", 10, 0.0, 0.000)
    too_similar_history = _sample("x1", "eval", 11, 0.01, 0.001)
    valid_history = _sample("x2", "eval", 12, 2.0, 0.010)

    result = build(
        training,
        (anchor, too_similar_history, valid_history),
    )
    pair_ids = {
        (pair.left_window_id, pair.right_window_id)
        for pair in result.pairs
    }

    anchor_valid = tuple(
        sorted((anchor.window_id, valid_history.window_id))
    )
    anchor_too_similar = tuple(
        sorted((anchor.window_id, too_similar_history.window_id))
    )
    assert anchor_valid in pair_ids
    assert anchor_too_similar not in pair_ids
