import numpy as np

from neural_state_machine.r1_e2_reservoir import E2Architecture
from neural_state_machine.r1_e3m_benchmark import fit_arm_memory
from neural_state_machine.r1_e3m_history import (
    evaluate_history_destruction,
    transform_prefix,
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
        tensor_digest="b" * 64,
    )


def test_reverse_and_shuffle_preserve_final_suffix_byte_identically():
    rng = np.random.default_rng(7)
    tensors = rng.random((3, 20, 14))
    sequence_ids = tuple(f"seq-{index}" for index in range(3))

    reversed_values = transform_prefix(tensors, sequence_ids, mode="reverse")
    shuffled_values = transform_prefix(tensors, sequence_ids, mode="shuffle")

    assert np.array_equal(reversed_values[:, 16:20], tensors[:, 16:20])
    assert np.array_equal(shuffled_values[:, 16:20], tensors[:, 16:20])
    assert np.array_equal(reversed_values[:, :16], tensors[:, 15::-1])


def test_hash_shuffle_is_stable_and_sequence_specific():
    prefix = np.arange(20 * 14, dtype=np.float64).reshape(1, 20, 14)
    tensors = np.concatenate((prefix, prefix), axis=0)

    first = transform_prefix(
        tensors,
        ("sequence-a", "sequence-b"),
        mode="shuffle",
    )
    second = transform_prefix(
        tensors,
        ("sequence-a", "sequence-b"),
        mode="shuffle",
    )

    assert np.array_equal(first, second)
    assert not np.array_equal(first[0, :16], first[1, :16])


def test_history_controls_reuse_fitted_probe_digest():
    rng = np.random.default_rng(19)
    windows = tuple(
        [
            _window(f"train-{index}", "train", rng.random((20, 14)))
            for index in range(80)
        ]
        + [
            _window(f"eval-{index}", "eval", rng.random((20, 14)))
            for index in range(20)
        ]
    )
    fitted = fit_arm_memory(
        windows,
        architecture=E2Architecture.FLAT,
        seed=7,
    )

    result = evaluate_history_destruction(fitted, windows)

    expected = fitted.probes[10].r1.coefficient_digest()
    assert result.delays[10].full_r1.coefficient_digest == expected
    assert result.delays[10].reset_r1.coefficient_digest == expected
    assert result.delays[10].reverse_r1.coefficient_digest == expected
    assert result.delays[10].shuffle_r1.coefficient_digest == expected


def test_reset_degrades_known_long_history_fixture():
    windows = []
    values = np.linspace(0.05, 0.95, 100)
    for index, value in enumerate(values):
        tensor = np.zeros((20, 14), dtype=np.float64)
        tensor[:16, 0] = value
        tensor[:16, 5] = 1.0
        nuisance = ((index * 37) % 101) / 101.0
        tensor[16:20, 1] = nuisance
        tensor[:, 6] = 0.5
        tensor[:, 11] = 1.0
        windows.append(
            _window(
                f"seq-{index}",
                "train" if index < 70 else "eval",
                tensor,
            )
        )

    fitted = fit_arm_memory(
        tuple(windows),
        architecture=E2Architecture.FLAT,
        seed=7,
    )
    result = evaluate_history_destruction(fitted, tuple(windows))

    delay10 = result.delays[10]
    assert delay10.reset_drop_r1 > 0.05
