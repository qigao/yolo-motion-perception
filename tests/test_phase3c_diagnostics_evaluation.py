from __future__ import annotations

from collections import Counter

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_diagnostics.evaluation import build_evaluation_bundles


def test_evaluation_manifest_has_nine_bundles_per_seed() -> None:
    bundles = build_evaluation_bundles(AnonymousCreditConfig())
    assert len(bundles) == 27
    assert len({bundle.evaluation_id for bundle in bundles}) == 27
    for bundle in bundles:
        assert len(bundle.fixtures) == 200
        assert Counter(episode.delay_steps for episode in bundle.fixtures) == {
            1: 40,
            2: 40,
            3: 40,
            4: 40,
            5: 40,
        }
        assert len(bundle.digest) == 64


def test_additional_evaluation_bundles_are_repeatable() -> None:
    first = build_evaluation_bundles(AnonymousCreditConfig())
    second = build_evaluation_bundles(AnonymousCreditConfig())
    assert [(row.evaluation_id, row.digest) for row in first] == [
        (row.evaluation_id, row.digest) for row in second
    ]
