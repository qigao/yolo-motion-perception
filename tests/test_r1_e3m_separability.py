import numpy as np

from neural_state_machine.r1_e2_reservoir import E2Architecture
from neural_state_machine.r1_e3m_separability import (
    fit_raw_standardizer,
    run_suffix_matched_separability,
)
from neural_state_machine.r1_e3m_windows import UnlabeledWindow


def _window(
    sequence_id: str,
    video_id: str,
    split: str,
    prefix_value: float,
    suffix_value: float,
) -> UnlabeledWindow:
    tensor = np.zeros((20, 14), dtype=np.float64)
    tensor[:16, 0] = prefix_value
    tensor[16:20, 0] = suffix_value
    tensor[:, 5] = 1.0
    tensor[:, 11] = 1.0
    return UnlabeledWindow(
        sequence_id=sequence_id,
        video_id=video_id,
        split=split,
        source_window_component_id=f"{video_id}-component",
        start_seconds=0.0,
        end_seconds=4.0,
        actor_track_id=1,
        target_track_id=2,
        tensor=tensor,
        tensor_digest="c" * 64,
    )


def test_raw_standardizer_is_fit_from_training_only():
    train = (
        _window("t1", "train-a", "train", 0.1, 0.2),
        _window("t2", "train-b", "train", 0.3, 0.4),
    )
    first = fit_raw_standardizer(train)

    changed_eval = train + (
        _window("e", "eval-a", "eval", 1.0, 1.0),
    )
    second = fit_raw_standardizer(changed_eval)

    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.std, second.std)


def test_suffix_matching_ignores_same_video_neighbor():
    windows = (
        _window("t1", "train-a", "train", 0.1, 0.2),
        _window("t2", "train-b", "train", 0.3, 0.4),
        _window("a", "eval-a", "eval", 0.1, 0.50),
        _window("same", "eval-a", "eval", 0.9, 0.50),
        _window("other", "eval-b", "eval", 0.8, 0.51),
    )

    result = run_suffix_matched_separability(
        windows,
        architecture=E2Architecture.FLAT,
        seed=7,
    )

    pair_ids = {
        frozenset((pair.sequence_a, pair.sequence_b))
        for pair in result.pairs
    }
    assert frozenset(("a", "same")) not in pair_ids
    assert frozenset(("a", "other")) in pair_ids


def test_suffix_tie_break_uses_lexical_sequence_id():
    windows = (
        _window("t1", "train-a", "train", 0.1, 0.2),
        _window("t2", "train-b", "train", 0.3, 0.4),
        _window("anchor", "eval-a", "eval", 0.2, 0.5),
        _window("b-choice", "eval-b", "eval", 0.7, 0.6),
        _window("c-choice", "eval-c", "eval", 0.8, 0.6),
    )

    result = run_suffix_matched_separability(
        windows,
        architecture=E2Architecture.FLAT,
        seed=7,
    )

    anchor_pairs = [
        pair
        for pair in result.pairs
        if "anchor" in (pair.sequence_a, pair.sequence_b)
    ]
    assert anchor_pairs
    selected = anchor_pairs[0]
    assert "b-choice" in (selected.sequence_a, selected.sequence_b)


def test_suffix_matching_is_deterministic():
    windows = (
        _window("t1", "train-a", "train", 0.1, 0.2),
        _window("t2", "train-b", "train", 0.3, 0.4),
        _window("a", "eval-a", "eval", 0.1, 0.5),
        _window("b", "eval-b", "eval", 0.8, 0.51),
        _window("c", "eval-c", "eval", 0.4, 0.7),
    )

    first = run_suffix_matched_separability(
        windows,
        architecture=E2Architecture.FLAT,
        seed=17,
    )
    second = run_suffix_matched_separability(
        windows,
        architecture=E2Architecture.FLAT,
        seed=17,
    )

    assert first == second
