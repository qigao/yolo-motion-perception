#!/usr/bin/env python3
"""Fail-closed CLI for Phase C4-A failure-attribution diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from neural_state_machine.action_value_benchmark import _build_fixture_bundle, _new_policy
from neural_state_machine.phase3c_schedule import build_hidden_delay_schedule
from neural_state_machine.phase_c4_batch_probe import fit_anonymous_batch_probe
from neural_state_machine.phase_c4_benchmark import PhaseC4Config
from neural_state_machine.phase_c4_measurement import _readout_from_fit, _secondary_scores
from neural_state_machine.phase_c4a_diagnostics.attribution import (
    associate_target,
    compare_weight_alignment,
    fit_supervised_reference,
    project_target,
)
from neural_state_machine.phase_c4a_diagnostics.decomposition import (
    compare_weights,
    decompose_readout_margin,
    summarize_margin_components,
)
from neural_state_machine.phase_c4a_diagnostics.evidence import (
    prepare_attribution_manifest,
    validate_attribution_manifest,
    write_attribution_bundle,
)
from neural_state_machine.phase_c4a_diagnostics.geometry import analyze_design_geometry
from neural_state_machine.phase_c4a_diagnostics.integrity import run_d0_integrity
from neural_state_machine.phase_c4a_diagnostics.model import DiagnosticConfig
from neural_state_machine.phase_c4a_diagnostics.permutations import (
    run_permutation_grid_for_seed,
    summarize_permutation_rows,
)
from neural_state_machine.phase_c4a_diagnostics.report import build_attribution_table
from neural_state_machine.reward_learning import _decision_hidden


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(nested) for nested in value]
    return value


def _fit_registered(replay, config: PhaseC4Config):
    return fit_anonymous_batch_probe(
        replay.design,
        replay.target,
        2,
        config.hidden_size + 1,
        penalty=config.ridge_penalty,
    )


def _arrival_counts(seed: int, config: PhaseC4Config) -> np.ndarray:
    schedule = build_hidden_delay_schedule(
        seed,
        config.training_decisions,
        support=config.delay_support,
    )
    counts = Counter(schedule.due_steps)
    return np.asarray(
        [counts.get(clock, 0) for clock in range(config.training_decisions + 5)],
        dtype=np.int64,
    )


def _margin_summary(seed: int, config: PhaseC4Config, weights: np.ndarray) -> dict[str, object]:
    fixtures = _build_fixture_bundle(seed, config.action_value_config)
    policy = _new_policy(seed, config.action_value_config)
    components = []
    delays = []
    for episode in fixtures.evaluation:
        hidden = _decision_hidden(policy, episode, reset_before_decision=False)
        feature = np.concatenate((hidden, np.asarray([1.0], dtype=np.float64)))
        components.append(
            decompose_readout_margin(weights, feature, episode.correct_action_index)
        )
        delays.append(episode.delay_steps)
    summary = summarize_margin_components(tuple(components), tuple(delays))
    return _jsonable(summary)  # type: ignore[return-value]


def _registered_attribution(seed: int, normal, shuffled, config: PhaseC4Config) -> dict[str, object]:
    fixtures = _build_fixture_bundle(seed, config.action_value_config)
    features = np.vstack(tuple(row.feature for row in normal.rows))
    correct_actions = np.asarray(
        [episode.correct_action_index for episode in fixtures.training],
        dtype=np.int64,
    )
    cue_delays = np.asarray(
        [episode.delay_steps for episode in fixtures.training],
        dtype=np.int64,
    )
    schedule = build_hidden_delay_schedule(
        seed,
        config.training_decisions,
        support=config.delay_support,
    )
    hidden_delays = np.asarray(schedule.delays, dtype=np.int64)
    arrivals = _arrival_counts(seed, config)
    donor = shuffled.shuffled_rewards
    if donor is None:
        raise RuntimeError("registered shuffled replay is missing donor rewards")

    normal_fit = _fit_registered(normal, config)
    shuffled_fit = _fit_registered(shuffled, config)
    supervised = fit_supervised_reference(features, correct_actions)
    normal_projection = project_target(normal.design, normal.target)
    shuffled_projection = project_target(shuffled.design, shuffled.target)
    normal_association = associate_target(
        normal.target,
        real_decision_count=config.training_decisions,
        actions=np.asarray(normal.actions, dtype=np.int64),
        correct_actions=correct_actions,
        cue_delays=cue_delays,
        hidden_delays=hidden_delays,
        normal_rewards=np.asarray(normal.rewards, dtype=np.float64),
        donor_rewards=np.asarray(normal.rewards, dtype=np.float64),
        arrival_counts=arrivals,
    )
    shuffled_association = associate_target(
        shuffled.target,
        real_decision_count=config.training_decisions,
        actions=np.asarray(shuffled.actions, dtype=np.int64),
        correct_actions=correct_actions,
        cue_delays=cue_delays,
        hidden_delays=hidden_delays,
        normal_rewards=np.asarray(normal.rewards, dtype=np.float64),
        donor_rewards=np.asarray(donor, dtype=np.float64),
        arrival_counts=arrivals,
    )
    normal_readout = _readout_from_fit(normal_fit, config)
    shuffled_readout = _readout_from_fit(shuffled_fit, config)
    normal_secondary = _secondary_scores(seed, config, normal_readout)
    shuffled_secondary = _secondary_scores(seed, config, shuffled_readout)

    return {
        "seed": seed,
        "normal_score": normal.score.correct,
        "shuffled_score": shuffled.score.correct,
        "normal_projection": _jsonable(normal_projection),
        "shuffled_projection": _jsonable(shuffled_projection),
        "normal_association": _jsonable(normal_association),
        "shuffled_association": _jsonable(shuffled_association),
        "normal_supervised_alignment": _jsonable(
            compare_weight_alignment(normal_fit.weights, supervised.weights)
        ),
        "shuffled_supervised_alignment": _jsonable(
            compare_weight_alignment(shuffled_fit.weights, supervised.weights)
        ),
        "weight_comparison": _jsonable(compare_weights(normal_fit.weights, shuffled_fit.weights)),
        "normal_margin_decomposition": _margin_summary(seed, config, normal_fit.weights),
        "shuffled_margin_decomposition": _margin_summary(seed, config, shuffled_fit.weights),
        "normal_secondary_scores": [row.overall.correct for row in normal_secondary],
        "shuffled_secondary_scores": [row.overall.correct for row in shuffled_secondary],
    }


def _mechanism_evidence(
    geometry: list[dict[str, object]],
    registered: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> dict[str, tuple[str, str, str]]:
    ranks = ", ".join(f"seed {row['seed']}: rank {row['rank']}" for row in geometry)
    projection = ", ".join(
        "seed {}: shuffled parallel ratio {:.6f}".format(
            row["seed"],
            float(row["shuffled_projection"]["parallel_ratio"]),  # type: ignore[index]
        )
        for row in registered
    )
    block_rows = [row for row in summaries if row["mode"] == "block10"]
    global_rows = [row for row in summaries if row["mode"] == "global"]
    block_text = ", ".join(
        f"seed {row['seed']}: mean {float(row['mean']):.3f}" for row in block_rows
    )
    global_text = ", ".join(
        f"seed {row['seed']}: mean {float(row['mean']):.3f}" for row in global_rows
    )
    secondary = ", ".join(
        "seed {}: registered shuffled secondary {}".format(
            row["seed"], row["shuffled_secondary_scores"]
        )
        for row in registered
    )
    return {
        "design_geometry": (
            ranks,
            "Normal and shuffled registered conditions use the same D0 design bytes.",
            "Design geometry is explanatory context and is not sufficient by itself.",
        ),
        "task_relevant_target_projection": (
            projection,
            "The matched permutation grid records target projection and supervised-direction alignment for every replicate.",
            "Observed alignment indicates task-relevant structure under the frozen design without establishing a causal source.",
        ),
        "bias_action_shortcut": (
            "Registered normal and shuffled readouts were decomposed into hidden, bias, and action-block contributions.",
            "The decompositions reuse the fitted readouts and do not retrain ablated models.",
            "Component differences are post-hoc associations rather than independent interventions.",
        ),
        "local_block_structure": (
            f"block10 summaries: {block_text}",
            f"global summaries: {global_text}",
            "Block10/global differences are matched diagnostic associations, not a pure causal intervention.",
        ),
        "evaluation_fixture_sensitivity": (
            secondary,
            "Every permutation and registered readout is evaluated on the original plus eight fixed secondary sets.",
            "Cross-fixture variation describes evaluation-realization sensitivity without changing the frozen C4-A verdict.",
        ),
        "unresolved_multiple_mechanisms": (
            "D1-D4 evidence is retained jointly for all three registered seeds.",
            "No diagnostic row is discarded because its score or association is inconvenient.",
            "Multiple or unresolved mechanisms remain valid interpretations for later no-refit review.",
        ),
    }


def _run_registered_measurement(root: Path) -> dict[str, object]:
    replays = run_d0_integrity(root)
    config = PhaseC4Config()
    diagnostic = DiagnosticConfig.registered()
    geometry: list[dict[str, object]] = []
    registered: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    permutation_summaries: list[dict[str, object]] = []

    for seed in (7, 17, 29):
        normal = next(
            replay for replay in replays if replay.seed == seed and replay.condition == "normal"
        )
        shuffled = next(
            replay
            for replay in replays
            if replay.seed == seed and replay.condition == "registered_shuffled"
        )
        metrics = analyze_design_geometry(normal.design)
        geometry_row = _jsonable(metrics)
        assert isinstance(geometry_row, dict)
        geometry_row["seed"] = seed
        geometry.append(geometry_row)
        registered.append(_registered_attribution(seed, normal, shuffled, config))

        rows = run_permutation_grid_for_seed(normal, config, diagnostic)
        permutation_rows.extend(_jsonable(row) for row in rows)  # type: ignore[arg-type]
        for mode in ("block10", "global"):
            mode_rows = tuple(row for row in rows if row.mode == mode)
            summary = _jsonable(summarize_permutation_rows(mode_rows))
            assert isinstance(summary, dict)
            permutation_summaries.append(summary)

    table = build_attribution_table(
        _mechanism_evidence(geometry, registered, permutation_summaries),
        integrity_valid=True,
    )
    return {
        "schema_version": 1,
        "stage": "c4a-failure-attribution",
        "integrity_valid": True,
        "geometry": geometry,
        "registered_attribution": registered,
        "permutation_rows": permutation_rows,
        "permutation_summaries": permutation_summaries,
        "mechanism_table": [_jsonable(row) for row in table],
    }


def _cmd_protocol(_: argparse.Namespace) -> int:
    replays = run_d0_integrity(_root())
    print(
        json.dumps(
            {
                "condition_count": len(replays),
                "integrity_valid": True,
                "stage": "protocol",
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    path = prepare_attribution_manifest(_root(), args.output)
    print(json.dumps({"manifest": str(path), "stage": "prepare"}, sort_keys=True))
    return 0


def _cmd_measure(args: argparse.Namespace) -> int:
    root = _root()
    manifest = Path(args.manifest)
    # This preflight must complete before D0 replay or any permutation fitting.
    validate_attribution_manifest(root, manifest)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite measurement output: {output}")

    result = _run_registered_measurement(root)
    output.mkdir(parents=True)
    output_manifest = output / "manifest.json"
    shutil.copyfile(manifest, output_manifest)
    write_attribution_bundle(root, output_manifest, output, result, {})
    print(json.dumps({"output": str(output), "stage": "measure"}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    protocol = commands.add_parser("protocol", help="reproduce the full registered D0 gate only")
    protocol.set_defaults(func=_cmd_protocol)

    prepare = commands.add_parser("prepare", help="write the prospective registered manifest only")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=_cmd_prepare)

    measure = commands.add_parser("measure", help="run the sealed registered D0-D5 diagnostic measurement")
    measure.add_argument("--manifest", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.set_defaults(func=_cmd_measure)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
