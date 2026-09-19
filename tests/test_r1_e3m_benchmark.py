import numpy as np
import pytest

from neural_state_machine.r1_e2_reservoir import E2Architecture
from neural_state_machine.r1_e3m_benchmark import (
    REGISTERED_DELAYS,
    evaluate_arm_memory,
    fit_arm_memory,
    target_for_delay,
)
from neural_state_machine.r1_e3m_windows import UnlabeledWindow


def _window(sequence_id: str, split: str, tensor: np.ndarray) -> UnlabeledWindow:
    return UnlabeledWindow(
        sequence_id=sequence_id,
        video_id=f"{split}-video",
        split=split,
        source_window_component_id=f"{split}-component",
        start_seconds=0.0,
        end_seconds=4.0,
        actor_track_id=1,
        target_track_id=2,
        tensor=tensor,
        tensor_digest="a" * 64,
    )


def test_registered_delays_and_target_bins_are_exact():
    tensors = np.arange(2 * 20 * 14, dtype=np.float64).reshape(2, 20, 14)

    assert REGISTERED_DELAYS == (1, 2, 5, 10, 15)
    for delay in REGISTERED_DELAYS:
        assert np.array_equal(
            target_for_delay(tensors, delay),
            tensors[:, 19 - delay, :],
        )

    with pytest.raises(ValueError, match="registered delay"):
        target_for_delay(tensors, 3)


def test_eval_data_does_not_change_fitted_probe_digests():
    rng = np.random.default_rng(7)
    train = [
        _window(f"train-{index}", "train", rng.random((20, 14)))
        for index in range(90)
    ]
    eval_a = [
        _window(f"eval-{index}", "eval", rng.random((20, 14)))
        for index in range(20)
    ]
    eval_b = [
        _window(
            window.sequence_id,
            "eval",
            np.clip(window.tensor * 0.0 + 0.75, 0.0, 1.0),
        )
        for window in eval_a
    ]

    first = fit_arm_memory(
        tuple(train + eval_a),
        architecture=E2Architecture.FLAT,
        seed=7,
    )
    second = fit_arm_memory(
        tuple(train + eval_b),
        architecture=E2Architecture.FLAT,
        seed=7,
    )

    assert first.probe_digests == second.probe_digests


def test_suffix_baseline_directly_recovers_delay_one_and_two():
    rng = np.random.default_rng(11)
    windows = tuple(
        [
            _window(f"train-{index}", "train", rng.random((20, 14)))
            for index in range(120)
        ]
        + [
            _window(f"eval-{index}", "eval", rng.random((20, 14)))
            for index in range(40)
        ]
    )

    fitted = fit_arm_memory(
        windows,
        architecture=E2Architecture.FLAT,
        seed=7,
    )
    result = evaluate_arm_memory(fitted, windows)

    assert result.delays[1].s4.macro_r2 > 0.999
    assert result.delays[2].s4.macro_r2 > 0.999


def test_reservoir_recovers_longer_history_when_suffix_is_uninformative():
    train = []
    evaluation = []
    values = np.linspace(0.05, 0.95, 100)
    for index, value in enumerate(values):
        tensor = np.zeros((20, 14), dtype=np.float64)
        tensor[:16, 0] = value
        tensor[:16, 5] = 1.0
        nuisance = ((index * 37) % 101) / 101.0
        tensor[16:20, 1] = nuisance
        tensor[:, 11] = 1.0
        tensor[:, 6] = 0.5
        window = _window(
            f"seq-{index}",
            "train" if index < 70 else "eval",
            tensor,
        )
        (train if index < 70 else evaluation).append(window)

    windows = tuple(train + evaluation)
    fitted = fit_arm_memory(
        windows,
        architecture=E2Architecture.FLAT,
        seed=7,
    )
    result = evaluate_arm_memory(fitted, windows)

    delay10 = result.delays[10]
    assert delay10.r1.macro_r2 > delay10.s4.macro_r2 + 0.05
