from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import median
from typing import Mapping

import numpy as np

from neural_state_machine.r1_e2_reservoir import E2Architecture
from neural_state_machine.r1_e3m_benchmark import (
    ARM_COUNT,
    LONG_DELAYS,
    REGISTERED_ARCHITECTURES,
    REGISTERED_SEEDS,
    classify_memory_outcome,
)
from neural_state_machine.r1_e3m_pairs import (
    HistoryPairSet,
    history_pair_set_from_payload,
    history_pair_set_payload,
)
from neural_state_machine.r1_e3m_probe import DELAYS, RIDGE_REGULARIZATION


class EvidenceInvalid(RuntimeError):
    pass


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def registered_manifest_payload(
    artifact_root_digest: object,
) -> dict[str, object]:
    digest = _validated_digest(
        artifact_root_digest,
        "artifact_root_digest",
    )
    return {
        "phase": "R1-E3M",
        "artifact_root_digest": digest,
        "architectures": {
            "flat": int(E2Architecture.FLAT),
            "grouped4": int(E2Architecture.GROUPED4),
            "hierarchical2": int(E2Architecture.HIERARCHICAL2),
            "hierarchical4": int(E2Architecture.HIERARCHICAL4),
        },
        "seeds": list(REGISTERED_SEEDS),
        "input_size": 6,
        "neuron_budget": 256,
        "arm_count": ARM_COUNT,
        "delays": list(DELAYS),
        "long_delays": list(LONG_DELAYS),
        "ridge_regularization": RIDGE_REGULARIZATION,
        "window": {
            "duration_seconds": 2.0,
            "bin_count": 20,
            "min_present_bins": 16,
            "max_observation_age_seconds": 0.5,
            "strictly_causal": True,
        },
        "controls": {
            "h1_reset_before_bin": 16,
            "h1_suffix_bins": [16, 17, 18, 19],
            "h2_permuted_prefix_bins": list(range(16)),
            "h2_unchanged_suffix_bins": [16, 17, 18, 19],
            "control_refit_allowed": False,
        },
        "outcome": {
            "robust_positive_arm_min": 16,
            "median_long_delay_delta_gt": 0.0,
            "median_h1_long_delay_drop_gt": 0.0,
            "m_b_requires_positive_median_delta": True,
            "m_c_median_delta_lte": 0.0,
        },
    }


def prepare_prospective(
    root: Path | str,
    *,
    scientific_head: str,
    artifact_root_digest: object,
    history_pair_set: HistoryPairSet,
) -> dict[str, object]:
    root_path = Path(root)
    head = _validated_head(scientific_head)
    artifact_digest = _validated_digest(
        artifact_root_digest,
        "artifact_root_digest",
    )
    if root_path.exists() and any(root_path.iterdir()):
        raise FileExistsError(
            f"destination is not empty: {root_path}"
        )
    root_path.mkdir(parents=True, exist_ok=True)
    if not isinstance(history_pair_set, HistoryPairSet):
        raise EvidenceInvalid(
            "history_pair_set must be a validated HistoryPairSet"
        )
    history_pair_sha = _write_pair(
        root_path,
        "history-pairs.json",
        history_pair_set_payload(history_pair_set),
    )
    history_pair_meta = {
        "pair_digest": history_pair_set.pair_digest,
        "prefix_threshold": history_pair_set.prefix_threshold,
        "pair_count": len(history_pair_set.pairs),
        "file_sha256": history_pair_sha,
    }

    manifest = {
        "schema": "r1-e3m-manifest-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "history_pairs": history_pair_meta,
        "protocol": registered_manifest_payload(artifact_digest),
    }
    manifest_sha = _write_pair(
        root_path,
        "manifest.json",
        manifest,
    )
    prospective = {
        "schema": "r1-e3m-prospective-provenance-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
        "python": manifest["python"],
        "numpy": manifest["numpy"],
        "history_pair_digest": history_pair_set.pair_digest,
        "history_pair_file_sha256": history_pair_sha,
        "registered_measurement": False,
    }
    _write_pair(
        root_path,
        "prospective-provenance.json",
        prospective,
    )
    return {
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
        "history_pair_digest": history_pair_set.pair_digest,
        "history_pair_count": len(history_pair_set.pairs),
        "history_prefix_threshold": history_pair_set.prefix_threshold,
    }


def write_measurement(
    root: Path | str,
    *,
    manifest_sha256: str,
    scientific_head: str,
    artifact_root_digest: object,
    raw_result: object,
) -> dict[str, object]:
    root_path = Path(root)
    head = _validated_head(scientific_head)
    artifact_digest = _validated_digest(
        artifact_root_digest,
        "artifact_root_digest",
    )
    sealed = _load_prospective(root_path)

    if manifest_sha256 != sealed["manifest_sha256"]:
        raise EvidenceInvalid(
            "manifest sha256 does not match sealed manifest"
        )
    if head != sealed["scientific_head"]:
        raise EvidenceInvalid(
            "scientific head does not match sealed manifest"
        )
    if artifact_digest != sealed["artifact_root_digest"]:
        raise EvidenceInvalid(
            "artifact root digest does not match sealed manifest"
        )
    _require_runtime_match(
        sealed["manifest"],
        python_version=platform.python_version(),
        numpy_version=np.__version__,
    )

    for name in (
        "result.json",
        "result.sha256",
        "provenance.json",
        "provenance.sha256",
        "trace-index.json",
        "trace-index.sha256",
    ):
        if (root_path / name).exists():
            raise FileExistsError(
                "registered measurement evidence is write-once"
            )

    measurement = _jsonable(raw_result)
    validation = _validate_registered_measurement(
        measurement,
        expected_artifact_digest=artifact_digest,
        expected_pair_set=sealed["history_pair_set"],
    )

    result_payload = {
        "schema": "r1-e3m-result-v1",
        "manifest_sha256": sealed["manifest_sha256"],
        "artifact_root_digest": artifact_digest,
        "history_pair_digest": sealed["history_pair_digest"],
        "measurement": measurement,
    }
    result_sha = _write_pair(
        root_path,
        "result.json",
        result_payload,
    )
    provenance = {
        "schema": "r1-e3m-provenance-v1",
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "history_pair_digest": sealed["history_pair_digest"],
        "manifest_sha256": sealed["manifest_sha256"],
        "result_sha256": result_sha,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "registered_measurement": True,
        "registered_arm_count": validation["registered_arm_count"],
        "history_pair_digest": sealed["history_pair_digest"],
        "history_pair_count": sealed["history_pair_count"],
        "outcome": validation["outcome"],
    }
    provenance_sha = _write_pair(
        root_path,
        "provenance.json",
        provenance,
    )
    trace = {
        "schema": "r1-e3m-trace-index-v1",
        "manifest_sha256": sealed["manifest_sha256"],
        "result_sha256": result_sha,
        "provenance_sha256": provenance_sha,
        "artifact_root_digest": artifact_digest,
        "history_pair_digest": sealed["history_pair_digest"],
    }
    trace_sha = _write_pair(
        root_path,
        "trace-index.json",
        trace,
    )
    return {
        "manifest_sha256": sealed["manifest_sha256"],
        "result_sha256": result_sha,
        "provenance_sha256": provenance_sha,
        "trace_index_sha256": trace_sha,
        "registered_arm_count": validation["registered_arm_count"],
        "outcome": validation["outcome"],
    }


def verify_evidence(
    root: Path | str,
    *,
    no_result_ok: bool = False,
) -> dict[str, object]:
    root_path = Path(root)
    sealed = _load_prospective(root_path)

    if not (root_path / "result.json").exists():
        if no_result_ok:
            return {
                "valid": True,
                "prospective_only": True,
                "scientific_head": sealed["scientific_head"],
                "artifact_root_digest": sealed[
                    "artifact_root_digest"
                ],
                "manifest_sha256": sealed["manifest_sha256"],
                "history_pair_digest": sealed["history_pair_digest"],
                "history_pair_count": sealed["history_pair_count"],
                "history_prefix_threshold": sealed[
                    "history_prefix_threshold"
                ],
                "registered_arm_count": ARM_COUNT,
                "delays": list(DELAYS),
            }
        raise EvidenceInvalid(
            "registered result is missing"
        )

    result = _mapping(
        _read_verified(root_path, "result.json"),
        "result",
    )
    provenance = _mapping(
        _read_verified(root_path, "provenance.json"),
        "provenance",
    )
    trace = _mapping(
        _read_verified(root_path, "trace-index.json"),
        "trace index",
    )

    if result.get("schema") != "r1-e3m-result-v1":
        raise EvidenceInvalid("result schema mismatch")
    if (
        result.get("manifest_sha256")
        != sealed["manifest_sha256"]
    ):
        raise EvidenceInvalid(
            "result manifest sha256 mismatch"
        )
    if (
        result.get("artifact_root_digest")
        != sealed["artifact_root_digest"]
    ):
        raise EvidenceInvalid(
            "result artifact root digest mismatch"
        )
    if (
        result.get("history_pair_digest")
        != sealed["history_pair_digest"]
    ):
        raise EvidenceInvalid(
            "result history pair digest mismatch"
        )

    result_sha = _read_digest(
        root_path,
        "result.sha256",
    )
    provenance_sha = _read_digest(
        root_path,
        "provenance.sha256",
    )

    if provenance.get("schema") != "r1-e3m-provenance-v1":
        raise EvidenceInvalid("provenance schema mismatch")
    if (
        provenance.get("scientific_head")
        != sealed["scientific_head"]
    ):
        raise EvidenceInvalid(
            "provenance scientific head mismatch"
        )
    if (
        provenance.get("artifact_root_digest")
        != sealed["artifact_root_digest"]
    ):
        raise EvidenceInvalid(
            "provenance artifact root digest mismatch"
        )
    if (
        provenance.get("history_pair_digest")
        != sealed["history_pair_digest"]
    ):
        raise EvidenceInvalid(
            "provenance history pair digest mismatch"
        )
    if (
        provenance.get("manifest_sha256")
        != sealed["manifest_sha256"]
    ):
        raise EvidenceInvalid(
            "provenance manifest sha256 mismatch"
        )
    if provenance.get("result_sha256") != result_sha:
        raise EvidenceInvalid(
            "provenance result sha256 mismatch"
        )
    if provenance.get("registered_measurement") is not True:
        raise EvidenceInvalid(
            "provenance measurement flag mismatch"
        )
    _require_runtime_match(
        sealed["manifest"],
        python_version=provenance.get("python"),
        numpy_version=provenance.get("numpy"),
    )

    if trace.get("schema") != "r1-e3m-trace-index-v1":
        raise EvidenceInvalid("trace schema mismatch")
    if (
        trace.get("manifest_sha256")
        != sealed["manifest_sha256"]
    ):
        raise EvidenceInvalid(
            "trace manifest sha256 mismatch"
        )
    if trace.get("result_sha256") != result_sha:
        raise EvidenceInvalid(
            "trace result sha256 mismatch"
        )
    if trace.get("provenance_sha256") != provenance_sha:
        raise EvidenceInvalid(
            "trace provenance sha256 mismatch"
        )
    if (
        trace.get("artifact_root_digest")
        != sealed["artifact_root_digest"]
    ):
        raise EvidenceInvalid(
            "trace artifact root digest mismatch"
        )
    if (
        trace.get("history_pair_digest")
        != sealed["history_pair_digest"]
    ):
        raise EvidenceInvalid(
            "trace history pair digest mismatch"
        )

    validation = _validate_registered_measurement(
        result.get("measurement"),
        expected_artifact_digest=sealed[
            "artifact_root_digest"
        ],
        expected_pair_set=sealed["history_pair_set"],
    )
    if (
        provenance.get("registered_arm_count")
        != validation["registered_arm_count"]
    ):
        raise EvidenceInvalid(
            "provenance registered arm count mismatch"
        )
    if provenance.get("outcome") != validation["outcome"]:
        raise EvidenceInvalid(
            "provenance outcome mismatch"
        )

    return {
        "valid": True,
        "prospective_only": False,
        "scientific_head": sealed["scientific_head"],
        "artifact_root_digest": sealed[
            "artifact_root_digest"
        ],
        "manifest_sha256": sealed["manifest_sha256"],
        "history_pair_digest": sealed["history_pair_digest"],
        "history_pair_count": sealed["history_pair_count"],
        "history_prefix_threshold": sealed[
            "history_prefix_threshold"
        ],
        "registered_arm_count": validation[
            "registered_arm_count"
        ],
        "outcome": validation["outcome"],
        "median_long_delay_delta": validation[
            "median_long_delay_delta"
        ],
        "positive_arm_count": validation[
            "positive_arm_count"
        ],
        "median_h1_long_delay_drop": validation[
            "median_h1_long_delay_drop"
        ],
    }


def _load_prospective(
    root_path: Path,
) -> dict[str, object]:
    manifest = _mapping(
        _read_verified(root_path, "manifest.json"),
        "manifest",
    )
    prospective = _mapping(
        _read_verified(
            root_path,
            "prospective-provenance.json",
        ),
        "prospective provenance",
    )
    manifest_sha = _read_digest(
        root_path,
        "manifest.sha256",
    )
    history_pair_file_sha = _read_digest(
        root_path,
        "history-pairs.sha256",
    )
    try:
        history_pair_set = history_pair_set_from_payload(
            _read_verified(root_path, "history-pairs.json")
        )
    except ValueError as exc:
        raise EvidenceInvalid(
            f"history pair seal invalid: {exc}"
        ) from exc

    if manifest.get("schema") != "r1-e3m-manifest-v1":
        raise EvidenceInvalid("manifest schema mismatch")
    head = _validated_head(
        manifest.get("scientific_head")
    )
    artifact_digest = _validated_digest(
        manifest.get("artifact_root_digest"),
        "artifact_root_digest",
    )
    history_meta = _mapping(
        manifest.get("history_pairs"),
        "manifest history_pairs",
    )
    expected_history_meta = {
        "pair_digest": history_pair_set.pair_digest,
        "prefix_threshold": history_pair_set.prefix_threshold,
        "pair_count": len(history_pair_set.pairs),
        "file_sha256": history_pair_file_sha,
    }
    if history_meta != expected_history_meta:
        raise EvidenceInvalid(
            "manifest history pair seal mismatch"
        )
    if (
        manifest.get("protocol")
        != registered_manifest_payload(artifact_digest)
    ):
        raise EvidenceInvalid(
            "manifest protocol does not match R1-E3M registration"
        )

    if (
        prospective.get("schema")
        != "r1-e3m-prospective-provenance-v1"
    ):
        raise EvidenceInvalid(
            "prospective provenance schema mismatch"
        )
    if prospective.get("scientific_head") != head:
        raise EvidenceInvalid(
            "prospective scientific head mismatch"
        )
    if (
        prospective.get("artifact_root_digest")
        != artifact_digest
    ):
        raise EvidenceInvalid(
            "prospective artifact root digest mismatch"
        )
    if prospective.get("manifest_sha256") != manifest_sha:
        raise EvidenceInvalid(
            "prospective manifest sha256 mismatch"
        )
    if (
        prospective.get("history_pair_digest")
        != history_pair_set.pair_digest
        or prospective.get("history_pair_file_sha256")
        != history_pair_file_sha
    ):
        raise EvidenceInvalid(
            "prospective history pair seal mismatch"
        )
    if prospective.get("registered_measurement") is not False:
        raise EvidenceInvalid(
            "prospective measurement flag mismatch"
        )
    _require_runtime_match(
        manifest,
        python_version=prospective.get("python"),
        numpy_version=prospective.get("numpy"),
    )
    return {
        "manifest": manifest,
        "prospective": prospective,
        "scientific_head": head,
        "artifact_root_digest": artifact_digest,
        "manifest_sha256": manifest_sha,
        "history_pair_set": history_pair_set,
        "history_pair_digest": history_pair_set.pair_digest,
        "history_pair_count": len(history_pair_set.pairs),
        "history_prefix_threshold": history_pair_set.prefix_threshold,
        "history_pair_file_sha256": history_pair_file_sha,
    }


def _validate_registered_measurement(
    value: object,
    *,
    expected_artifact_digest: str,
    expected_pair_set: HistoryPairSet,
) -> dict[str, object]:
    measurement = _mapping(
        value,
        "registered measurement",
    )
    if measurement.get("registered_measurement") is not True:
        raise EvidenceInvalid(
            "registered measurement flag is missing"
        )
    if (
        measurement.get("manifest")
        != registered_manifest_payload(
            expected_artifact_digest
        )
    ):
        raise EvidenceInvalid(
            "registered measurement manifest mismatch"
        )

    if (
        measurement.get("pair_set_digest")
        != expected_pair_set.pair_digest
    ):
        raise EvidenceInvalid(
            "registered measurement history pair digest mismatch"
        )
    if (
        measurement.get("history_pair_count")
        != len(expected_pair_set.pairs)
    ):
        raise EvidenceInvalid(
            "registered measurement history pair count mismatch"
        )
    _require_close(
        measurement.get("history_prefix_threshold"),
        expected_pair_set.prefix_threshold,
        "history_prefix_threshold",
    )

    arms = measurement.get("arms")
    if not isinstance(arms, list) or len(arms) != ARM_COUNT:
        raise EvidenceInvalid(
            "registered arm count mismatch"
        )
    expected_keys = {
        (seed, int(architecture))
        for seed in REGISTERED_SEEDS
        for architecture in REGISTERED_ARCHITECTURES
    }
    seen: set[tuple[int, int]] = set()
    long_deltas: list[float] = []
    h1_long_drops: list[float] = []

    for arm in arms:
        arm_map = _mapping(arm, "registered arm")
        seed = arm_map.get("seed")
        architecture = arm_map.get("architecture")
        if type(seed) is not int or type(architecture) is not int:
            raise EvidenceInvalid(
                "registered arm identity must be integer"
            )
        key = (seed, architecture)
        if key not in expected_keys or key in seen:
            raise EvidenceInvalid(
                "registered arm identity is duplicated or unregistered"
            )
        seen.add(key)

        if (
            arm_map.get("artifact_root_digest")
            != expected_artifact_digest
        ):
            raise EvidenceInvalid(
                "registered arm artifact root digest mismatch"
            )
        _validated_digest(
            arm_map.get("reservoir_parameter_digest"),
            "reservoir_parameter_digest",
        )
        if (
            arm_map.get("pair_set_digest")
            != expected_pair_set.pair_digest
        ):
            raise EvidenceInvalid(
                "registered arm history pair digest mismatch"
            )
        pair_rows = arm_map.get("pair_diagnostics")
        if (
            not isinstance(pair_rows, list)
            or len(pair_rows) != len(expected_pair_set.pairs)
        ):
            raise EvidenceInvalid(
                "registered arm history pair diagnostics mismatch"
            )
        for expected_pair, pair_value in zip(
            expected_pair_set.pairs,
            pair_rows,
            strict=True,
        ):
            pair_map = _mapping(
                pair_value,
                "history pair diagnostic",
            )
            if (
                pair_map.get("left_window_id")
                != expected_pair.left_window_id
                or pair_map.get("right_window_id")
                != expected_pair.right_window_id
            ):
                raise EvidenceInvalid(
                    "history pair diagnostic identity mismatch"
                )
            _require_close(
                pair_map.get("suffix_distance"),
                expected_pair.suffix_distance,
                "history pair suffix_distance",
            )
            _require_close(
                pair_map.get("prefix_distance"),
                expected_pair.prefix_distance,
                "history pair prefix_distance",
            )
            normal_distance = _finite_float(
                pair_map.get("normal_state_distance"),
                "history pair normal_state_distance",
            )
            reset_distance = _finite_float(
                pair_map.get("reset_state_distance"),
                "history pair reset_state_distance",
            )
            if normal_distance < 0.0 or reset_distance < 0.0:
                raise EvidenceInvalid(
                    "history pair state distances must be non-negative"
                )

        delay_rows = arm_map.get("delays")
        if (
            not isinstance(delay_rows, list)
            or len(delay_rows) != len(DELAYS)
        ):
            raise EvidenceInvalid(
                "registered arm delays mismatch"
            )

        by_delay: dict[int, dict[str, float]] = {}
        for expected_delay, delay_value in zip(
            DELAYS,
            delay_rows,
            strict=True,
        ):
            delay_map = _mapping(
                delay_value,
                "delay result",
            )
            if delay_map.get("delay") != expected_delay:
                raise EvidenceInvalid(
                    "registered delay order/identity mismatch"
                )

            instantaneous = _validate_probe_result(
                delay_map.get("instantaneous"),
                "instantaneous",
            )
            reservoir = _validate_probe_result(
                delay_map.get("reservoir"),
                "reservoir",
            )
            reset = _validate_probe_result(
                delay_map.get("reset_control"),
                "reset_control",
            )
            permuted = _validate_probe_result(
                delay_map.get("permuted_control"),
                "permuted_control",
            )
            sample_counts = {
                instantaneous["sample_count"],
                reservoir["sample_count"],
                reset["sample_count"],
                permuted["sample_count"],
            }
            if len(sample_counts) != 1:
                raise EvidenceInvalid(
                    "delay result sample-count mismatch"
                )
            if (
                reset["coefficient_digest"]
                != reservoir["coefficient_digest"]
                or permuted["coefficient_digest"]
                != reservoir["coefficient_digest"]
            ):
                raise EvidenceInvalid(
                    "control refit detected"
                )

            delta = (
                reservoir["mean_r2"]
                - instantaneous["mean_r2"]
            )
            h1_drop = (
                reservoir["mean_r2"]
                - reset["mean_r2"]
            )
            h2_drop = (
                reservoir["mean_r2"]
                - permuted["mean_r2"]
            )
            _require_close(
                delay_map.get("delta_r2"),
                delta,
                "delta_r2",
            )
            _require_close(
                delay_map.get("h1_drop_r2"),
                h1_drop,
                "h1_drop_r2",
            )
            _require_close(
                delay_map.get("h2_drop_r2"),
                h2_drop,
                "h2_drop_r2",
            )
            by_delay[expected_delay] = {
                "delta": delta,
                "h1": h1_drop,
                "h2": h2_drop,
            }

        expected_long_delta = _mean(
            by_delay[delay]["delta"]
            for delay in LONG_DELAYS
        )
        expected_h1_long = _mean(
            by_delay[delay]["h1"]
            for delay in LONG_DELAYS
        )
        expected_h2_long = _mean(
            by_delay[delay]["h2"]
            for delay in LONG_DELAYS
        )
        _require_close(
            arm_map.get("long_delay_delta"),
            expected_long_delta,
            "long_delay_delta",
        )
        _require_close(
            arm_map.get("h1_long_delay_drop"),
            expected_h1_long,
            "h1_long_delay_drop",
        )
        _require_close(
            arm_map.get("h2_long_delay_drop"),
            expected_h2_long,
            "h2_long_delay_drop",
        )
        long_deltas.append(expected_long_delta)
        h1_long_drops.append(expected_h1_long)

    if seen != expected_keys:
        raise EvidenceInvalid(
            "registered arm set does not match registration"
        )

    expected_median_delta = float(median(long_deltas))
    expected_positive_count = sum(
        value > 0.0 for value in long_deltas
    )
    expected_median_h1 = float(median(h1_long_drops))
    expected_outcome = classify_memory_outcome(
        long_deltas,
        h1_long_drops,
    )

    _require_close(
        measurement.get("median_long_delay_delta"),
        expected_median_delta,
        "median_long_delay_delta",
    )
    if (
        measurement.get("positive_arm_count")
        != expected_positive_count
    ):
        raise EvidenceInvalid(
            "positive_arm_count mismatch"
        )
    _require_close(
        measurement.get("median_h1_long_delay_drop"),
        expected_median_h1,
        "median_h1_long_delay_drop",
    )
    if measurement.get("outcome") != expected_outcome:
        raise EvidenceInvalid(
            "registered outcome mismatch"
        )

    return {
        "registered_arm_count": len(arms),
        "median_long_delay_delta": expected_median_delta,
        "positive_arm_count": expected_positive_count,
        "median_h1_long_delay_drop": expected_median_h1,
        "outcome": expected_outcome,
        "history_pair_digest": expected_pair_set.pair_digest,
        "history_pair_count": len(expected_pair_set.pairs),
    }


def _validate_probe_result(
    value: object,
    name: str,
) -> dict[str, object]:
    probe = _mapping(value, name)
    coefficient_digest = _validated_digest(
        probe.get("coefficient_digest"),
        f"{name} coefficient_digest",
    )
    prediction_digest = _validated_digest(
        probe.get("prediction_digest"),
        f"{name} prediction_digest",
    )
    metrics = _mapping(
        probe.get("metrics"),
        f"{name} metrics",
    )
    sample_count = metrics.get("sample_count")
    if type(sample_count) is not int or sample_count <= 0:
        raise EvidenceInvalid(
            f"{name} sample_count must be positive"
        )
    mean_r2 = _finite_float(
        metrics.get("mean_r2"),
        f"{name} mean_r2",
    )
    mse = _finite_float(
        metrics.get("mse"),
        f"{name} mse",
    )
    if mse < 0.0:
        raise EvidenceInvalid(
            f"{name} mse must be non-negative"
        )
    per_channel = metrics.get("r2_per_channel")
    if not isinstance(per_channel, list) or len(per_channel) != 4:
        raise EvidenceInvalid(
            f"{name} r2_per_channel must contain four values"
        )
    for index, metric in enumerate(per_channel):
        _finite_float(
            metric,
            f"{name} r2_per_channel[{index}]",
        )
    return {
        "coefficient_digest": coefficient_digest,
        "prediction_digest": prediction_digest,
        "sample_count": sample_count,
        "mean_r2": mean_r2,
    }


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_pair(
    root: Path,
    name: str,
    payload: object,
) -> str:
    data = canonical_json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    (root / name).write_bytes(data)
    (root / f"{name.removesuffix('.json')}.sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )
    return digest


def _read_verified(root: Path, name: str) -> object:
    data_path = root / name
    sha_path = root / f"{name.removesuffix('.json')}.sha256"
    if not data_path.is_file() or not sha_path.is_file():
        raise EvidenceInvalid(
            f"missing {name} or checksum"
        )
    data = data_path.read_bytes()
    expected = sha_path.read_text(
        encoding="utf-8"
    ).strip()
    _validated_digest(expected, f"{name} sha256")
    actual = hashlib.sha256(data).hexdigest()
    if expected != actual:
        raise EvidenceInvalid(
            f"{name} sha256 mismatch"
        )
    try:
        return json.loads(data)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise EvidenceInvalid(
            f"invalid {name}"
        ) from exc


def _read_digest(root: Path, name: str) -> str:
    path = root / name
    if not path.is_file():
        raise EvidenceInvalid(f"missing {name}")
    return _validated_digest(
        path.read_text(encoding="utf-8").strip(),
        name,
    )


def _validated_head(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(
            char not in "0123456789abcdef"
            for char in value
        )
    ):
        raise EvidenceInvalid(
            "scientific head must be a 40-character lowercase git sha"
        )
    return value


def _validated_digest(
    value: object,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            char not in "0123456789abcdef"
            for char in value
        )
    ):
        raise EvidenceInvalid(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _finite_float(
    value: object,
    name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise EvidenceInvalid(
            f"{name} must be finite"
        )
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceInvalid(
            f"{name} must be finite"
        )
    return result


def _mapping(
    value: object,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceInvalid(
            f"{name} must be an object"
        )
    return value


def _require_runtime_match(
    manifest: Mapping[str, object],
    *,
    python_version: object,
    numpy_version: object,
) -> None:
    if (
        manifest.get("python") != python_version
        or manifest.get("numpy") != numpy_version
    ):
        raise EvidenceInvalid(
            "runtime does not match sealed manifest"
        )


def _require_close(
    actual: object,
    expected: float,
    name: str,
) -> None:
    observed = _finite_float(actual, name)
    if not math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise EvidenceInvalid(
            f"{name} mismatch"
        )


def _mean(values: object) -> float:
    array = np.asarray(
        tuple(values),
        dtype=np.float64,
    )
    if array.size == 0 or not np.isfinite(array).all():
        raise EvidenceInvalid(
            "aggregate values must be finite and non-empty"
        )
    return float(
        np.mean(array, dtype=np.float64)
    )
