from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_state_machine.action_value_benchmark import _build_fixture_bundle, _episode_digest
from neural_state_machine.memory_task import DelayedCueEpisode, DelayedCueTask
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.reward_learning import _build_fixtures

from .contracts import EvaluationId, REGISTERED_SEEDS

_EVALUATION_LINEAGE = 0x33434641


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    evaluation_id: EvaluationId
    fixtures: tuple[DelayedCueEpisode, ...]
    digest: str

    def to_manifest_row(self) -> dict[str, object]:
        return {
            "seed": self.evaluation_id.seed,
            "evaluation_id": self.evaluation_id.evaluation_id,
            "kind": "original" if self.evaluation_id.is_original else "additional",
            "fixture_count": len(self.fixtures),
            "digest": self.digest,
        }


def build_evaluation_bundles(
    config: AnonymousCreditConfig,
) -> tuple[EvaluationBundle, ...]:
    if not isinstance(config, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")
    result: list[EvaluationBundle] = []
    task = DelayedCueTask()
    for seed in REGISTERED_SEEDS:
        original = _build_fixture_bundle(seed, config.action_value_config)
        result.append(
            EvaluationBundle(
                evaluation_id=EvaluationId(seed, -1),
                fixtures=original.evaluation,
                digest=original.evaluation_fixture_digest,
            )
        )
        for evaluation_id in range(8):
            rng = np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence(
                        [seed, _EVALUATION_LINEAGE, 2, evaluation_id]
                    )
                )
            )
            fixtures = _build_fixtures(task, rng, config.evaluation_blocks)
            result.append(
                EvaluationBundle(
                    evaluation_id=EvaluationId(seed, evaluation_id),
                    fixtures=fixtures,
                    digest=_episode_digest(fixtures),
                )
            )
    rows = tuple(result)
    if len(rows) != 27 or len({row.evaluation_id for row in rows}) != 27:
        raise RuntimeError("evaluation manifest is incomplete or contains duplicate IDs")
    return rows


def evaluation_manifest_rows(
    config: AnonymousCreditConfig,
) -> tuple[dict[str, object], ...]:
    return tuple(bundle.to_manifest_row() for bundle in build_evaluation_bundles(config))
