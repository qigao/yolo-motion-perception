"""D4 fixed block-local/global permutation diagnostics for C4-A."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..action_value_benchmark import _build_fixture_bundle, _evaluate, _new_policy
from ..phase_c4_batch_probe import build_batch_design, fit_anonymous_batch_probe
from ..phase_c4_benchmark import PhaseC4Config
from ..phase_c4_delay_model import DelayLaw
from ..phase_c4_measurement import _aggregate_stream, _readout_from_fit, _secondary_scores
from .attribution import compare_weight_alignment, fit_supervised_reference, project_target
from .integrity import RegisteredConditionReplay
from .model import DiagnosticConfig


PermutationMode = Literal["block10", "global"]
_REGISTERED_LINEAGE = 0x43344144
_REGISTERED_SHUFFLED_SCORES = {7: 147, 17: 170, 29: 160}


@dataclass(frozen=True, slots=True)
class PermutationRow:
    seed: int
    mode: PermutationMode
    replicate: int
    permutation_digest: str
    original_score: int
    secondary_scores: tuple[int, ...]
    target_parallel_ratio: float
    target_residual_ratio: float
    target_fit_residual_norm: float = 0.0
    target_parallel_correlation: float | None = None
    supervised_global_cosine: float | None = None
    supervised_action_cosines: tuple[float | None, float | None] = (None, None)
    supervised_norm_ratio: float | None = None
    supervised_projection_magnitude: float | None = None


@dataclass(frozen=True, slots=True)
class PermutationSummary:
    seed: int
    mode: PermutationMode
    count: int
    mean: float
    median: float
    minimum: int
    maximum: int
    p10: int
    p90: int
    count_lt_150: int
    count_ge_150: int
    count_ge_170: int
    registered_shuffled_score: int
    registered_reference_is_p_value: bool = False


def _mode_id(mode: PermutationMode) -> int:
    if mode == "block10":
        return 0
    if mode == "global":
        return 1
    raise ValueError("mode must be block10 or global")


def permutation_indices(
    seed: int,
    mode: PermutationMode,
    replicate: int,
    n: int,
) -> np.ndarray:
    """Return the exact prospective permutation without rejection or resampling."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(replicate) is not int or not (0 <= replicate < 32):
        raise ValueError("replicate must be an integer in [0, 31]")
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    mode_id = _mode_id(mode)
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([seed, _REGISTERED_LINEAGE, mode_id, replicate])
        )
    )
    if mode == "global":
        indices = np.asarray(rng.permutation(n), dtype=np.int64)
    else:
        indices = np.arange(n, dtype=np.int64)
        for start in range(0, n, 10):
            stop = min(start + 10, n)
            indices[start:stop] = rng.permutation(np.arange(start, stop, dtype=np.int64))
    frozen = np.ascontiguousarray(indices, dtype=np.int64)
    frozen.flags.writeable = False
    return frozen


def permute_rewards(rewards: tuple[float, ...], indices: np.ndarray) -> tuple[float, ...]:
    """Reassign only latent reward values according to one fixed permutation."""
    if not isinstance(rewards, tuple) or not rewards:
        raise ValueError("rewards must be a non-empty tuple")
    try:
        permutation = np.asarray(indices)
    except (TypeError, ValueError) as exc:
        raise ValueError("indices must be a permutation") from exc
    if (
        permutation.ndim != 1
        or permutation.size != len(rewards)
        or not np.issubdtype(permutation.dtype, np.integer)
    ):
        raise ValueError("indices must be a permutation of the reward indices")
    canonical = np.asarray(permutation, dtype=np.int64)
    if not np.array_equal(np.sort(canonical), np.arange(len(rewards), dtype=np.int64)):
        raise ValueError("indices must be a permutation of the reward indices")
    return tuple(float(rewards[int(index)]) for index in canonical)


def _permutation_digest(indices: np.ndarray) -> str:
    canonical = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def run_permutation_grid_for_seed(
    replay: RegisteredConditionReplay,
    phase_config: PhaseC4Config,
    diagnostic_config: DiagnosticConfig,
) -> tuple[PermutationRow, ...]:
    """Run the fixed D4 grid for one seed using D0-frozen rows/actions/design."""
    if not isinstance(replay, RegisteredConditionReplay) or replay.condition != "normal":
        raise ValueError("replay must be the D0 normal registered condition")
    if not isinstance(phase_config, PhaseC4Config):
        raise ValueError("phase_config must be PhaseC4Config")
    if not isinstance(diagnostic_config, DiagnosticConfig):
        raise ValueError("diagnostic_config must be DiagnosticConfig")
    if diagnostic_config.permutation_lineage != _REGISTERED_LINEAGE:
        raise ValueError("diagnostic permutation lineage mismatch")

    fixtures = _build_fixture_bundle(replay.seed, phase_config.action_value_config)
    features = np.vstack(tuple(row.feature for row in replay.rows))
    correct_actions = np.asarray(
        [episode.correct_action_index for episode in fixtures.training],
        dtype=np.int64,
    )
    supervised = fit_supervised_reference(features, correct_actions)

    output: list[PermutationRow] = []
    for mode in diagnostic_config.modes:
        mode_value: PermutationMode = "block10" if mode == "block10" else "global"
        for replicate in range(diagnostic_config.permutation_replicates):
            indices = permutation_indices(
                replay.seed,
                mode_value,
                replicate,
                len(replay.rewards),
            )
            permuted_rewards = permute_rewards(replay.rewards, indices)
            scalars, schedule_digest, drain_count = _aggregate_stream(
                replay.seed,
                phase_config,
                replay.actions,
                permuted_rewards,
            )
            if schedule_digest != replay.schedule_digest:
                raise RuntimeError("diagnostic permutation changed hidden-delay schedule")
            if drain_count != max(phase_config.delay_support):
                raise RuntimeError("diagnostic permutation changed public drain horizon")

            design, target = build_batch_design(
                replay.rows,
                scalars,
                2,
                DelayLaw.registered(),
            )
            if design.tobytes(order="C") != replay.design.tobytes(order="C"):
                raise RuntimeError("diagnostic permutation changed the D0 design matrix")
            fit = fit_anonymous_batch_probe(
                design,
                target,
                2,
                phase_config.hidden_size + 1,
                penalty=phase_config.ridge_penalty,
            )
            readout = _readout_from_fit(fit, phase_config)
            evaluation = _evaluate(
                _new_policy(replay.seed, phase_config.action_value_config),
                readout,
                fixtures.evaluation,
                reset_before_decision=False,
            )
            secondary = _secondary_scores(replay.seed, phase_config, readout)
            projection = project_target(design, target)
            alignment = compare_weight_alignment(fit.weights, supervised.weights)
            output.append(
                PermutationRow(
                    seed=replay.seed,
                    mode=mode_value,
                    replicate=replicate,
                    permutation_digest=_permutation_digest(indices),
                    original_score=evaluation.overall.correct,
                    secondary_scores=tuple(row.overall.correct for row in secondary),
                    target_parallel_ratio=projection.parallel_ratio,
                    target_residual_ratio=projection.residual_ratio,
                    target_fit_residual_norm=projection.fit_residual_norm,
                    target_parallel_correlation=projection.parallel_target_correlation,
                    supervised_global_cosine=alignment.global_cosine,
                    supervised_action_cosines=alignment.action_cosines,
                    supervised_norm_ratio=alignment.norm_ratio,
                    supervised_projection_magnitude=alignment.supervised_projection_magnitude,
                )
            )
    return tuple(output)


def _nearest_rank(values: np.ndarray, quantile: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=np.int64))
    index = math.ceil(quantile * ordered.size) - 1
    index = max(0, min(index, ordered.size - 1))
    return int(ordered[index])


def summarize_permutation_rows(
    rows: tuple[PermutationRow, ...],
    *,
    registered_shuffled_score: int | None = None,
) -> PermutationSummary:
    """Summarize one seed/mode without treating the registered score as a p-value."""
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("rows must be a non-empty tuple")
    if any(not isinstance(row, PermutationRow) for row in rows):
        raise ValueError("rows must contain only PermutationRow values")
    seed = rows[0].seed
    mode = rows[0].mode
    if any(row.seed != seed or row.mode != mode for row in rows):
        raise ValueError("rows must share one seed and permutation mode")
    scores = np.asarray([row.original_score for row in rows], dtype=np.int64)
    if registered_shuffled_score is None:
        try:
            registered_shuffled_score = _REGISTERED_SHUFFLED_SCORES[seed]
        except KeyError as exc:
            raise ValueError("registered shuffled score is unknown for this seed") from exc
    if type(registered_shuffled_score) is not int:
        raise ValueError("registered_shuffled_score must be an integer")
    return PermutationSummary(
        seed=seed,
        mode=mode,
        count=len(rows),
        mean=float(np.mean(scores)),
        median=float(np.median(scores)),
        minimum=int(np.min(scores)),
        maximum=int(np.max(scores)),
        p10=_nearest_rank(scores, 0.10),
        p90=_nearest_rank(scores, 0.90),
        count_lt_150=int(np.count_nonzero(scores < 150)),
        count_ge_150=int(np.count_nonzero(scores >= 150)),
        count_ge_170=int(np.count_nonzero(scores >= 170)),
        registered_shuffled_score=registered_shuffled_score,
        registered_reference_is_p_value=False,
    )
