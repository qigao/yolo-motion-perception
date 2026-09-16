from __future__ import annotations

import copy

from scripts.verify_phase3c_anonymous_credit import _portable_projection


def _payload(margin_mean: float) -> dict[str, object]:
    return {
        "results": [
            {
                "protocol": {"parameter_digest": "a" * 64},
                "shuffled_parameter_digest": "b" * 64,
                "normal_checkpoints": [{"margin_mean": margin_mean}],
                "shuffled_checkpoints": [{"margin_mean": margin_mean}],
                "post_training": {"correct": 83, "total": 200},
            }
        ]
    }


def test_portable_projection_normalizes_cross_version_checkpoint_float_noise() -> None:
    committed = _payload(0.0077623327093496244)
    runtime = _payload(0.0077623327093496)

    assert _portable_projection(committed) == _portable_projection(runtime)


def test_portable_projection_keeps_scientifically_material_checkpoint_change() -> None:
    committed = _payload(0.0077623327093496244)
    changed = copy.deepcopy(committed)
    changed["results"][0]["normal_checkpoints"][0]["margin_mean"] = 0.0077623328

    assert _portable_projection(committed) != _portable_projection(changed)
