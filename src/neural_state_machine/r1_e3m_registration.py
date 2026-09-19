from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from neural_state_machine.r1_e3m_dataset import MechanismDataset, MechanismSample
from neural_state_machine.r1_e3m_probe import DELAYS, delay_valid_mask


MIN_TRAIN_WINDOWS_PER_DELAY = 20
MIN_EVAL_WINDOWS_PER_DELAY = 10
MIN_SOURCE_VIDEOS_PER_SPLIT = 2
_DIGEST_VERSION = b"r1-e3m-delay-registration-v1\0"


class DelayRegistrationInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class DelayEligibility:
    delay: int
    training_window_ids: tuple[str, ...]
    evaluation_window_ids: tuple[str, ...]
    training_video_ids: tuple[str, ...]
    evaluation_video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.delay) is not int or self.delay not in DELAYS:
            raise DelayRegistrationInvalid(
                "delay eligibility uses an unregistered delay"
            )
        _validated_ids(self.training_window_ids, "training_window_ids")
        _validated_ids(self.evaluation_window_ids, "evaluation_window_ids")
        _validated_text_ids(self.training_video_ids, "training_video_ids")
        _validated_text_ids(self.evaluation_video_ids, "evaluation_video_ids")


@dataclass(frozen=True)
class DelayRegistration:
    artifact_root_digest: str
    delays: tuple[DelayEligibility, ...]
    digest: str

    def __post_init__(self) -> None:
        _validated_digest(
            self.artifact_root_digest,
            "artifact_root_digest",
        )
        if tuple(item.delay for item in self.delays) != DELAYS:
            raise DelayRegistrationInvalid(
                "delay registration must contain exact registered delays"
            )
        _validated_digest(self.digest, "delay registration digest")
        expected = _registration_digest(
            self.artifact_root_digest,
            self.delays,
        )
        if self.digest != expected:
            raise DelayRegistrationInvalid(
                "delay registration digest mismatch"
            )


def build_delay_registration(
    dataset: MechanismDataset,
) -> DelayRegistration:
    if not isinstance(dataset, MechanismDataset):
        raise ValueError("dataset must be MechanismDataset")

    training_tensors = _tensor_batch(dataset.training)
    evaluation_tensors = _tensor_batch(dataset.evaluation)

    entries = []
    for delay in DELAYS:
        train_mask = delay_valid_mask(training_tensors, delay)
        eval_mask = delay_valid_mask(evaluation_tensors, delay)
        train_samples = tuple(
            sample
            for sample, valid in zip(
                dataset.training,
                train_mask,
                strict=True,
            )
            if bool(valid)
        )
        eval_samples = tuple(
            sample
            for sample, valid in zip(
                dataset.evaluation,
                eval_mask,
                strict=True,
            )
            if bool(valid)
        )
        entries.append(
            DelayEligibility(
                delay=delay,
                training_window_ids=tuple(
                    sorted(sample.window_id for sample in train_samples)
                ),
                evaluation_window_ids=tuple(
                    sorted(sample.window_id for sample in eval_samples)
                ),
                training_video_ids=tuple(
                    sorted({sample.video_id for sample in train_samples})
                ),
                evaluation_video_ids=tuple(
                    sorted({sample.video_id for sample in eval_samples})
                ),
            )
        )

    delays = tuple(entries)
    digest = _registration_digest(
        dataset.artifact_root_digest,
        delays,
    )
    return DelayRegistration(
        artifact_root_digest=dataset.artifact_root_digest,
        delays=delays,
        digest=digest,
    )


def validate_delay_registration_gate(
    registration: DelayRegistration,
) -> DelayRegistration:
    if not isinstance(registration, DelayRegistration):
        raise ValueError("registration must be DelayRegistration")

    for entry in registration.delays:
        if len(entry.training_window_ids) < MIN_TRAIN_WINDOWS_PER_DELAY:
            raise DelayRegistrationInvalid(
                f"delay {entry.delay} has fewer than "
                f"{MIN_TRAIN_WINDOWS_PER_DELAY} training windows"
            )
        if len(entry.evaluation_window_ids) < MIN_EVAL_WINDOWS_PER_DELAY:
            raise DelayRegistrationInvalid(
                f"delay {entry.delay} has fewer than "
                f"{MIN_EVAL_WINDOWS_PER_DELAY} evaluation windows"
            )
        if len(entry.training_video_ids) < MIN_SOURCE_VIDEOS_PER_SPLIT:
            raise DelayRegistrationInvalid(
                f"delay {entry.delay} has fewer than "
                f"{MIN_SOURCE_VIDEOS_PER_SPLIT} training source videos"
            )
        if len(entry.evaluation_video_ids) < MIN_SOURCE_VIDEOS_PER_SPLIT:
            raise DelayRegistrationInvalid(
                f"delay {entry.delay} has fewer than "
                f"{MIN_SOURCE_VIDEOS_PER_SPLIT} evaluation source videos"
            )
    return registration


def delay_registration_payload(
    registration: DelayRegistration,
) -> dict[str, object]:
    if not isinstance(registration, DelayRegistration):
        raise ValueError("registration must be DelayRegistration")
    return {
        "schema": "r1-e3m-delay-registration-v1",
        "artifact_root_digest": registration.artifact_root_digest,
        "digest": registration.digest,
        "min_train_windows_per_delay": MIN_TRAIN_WINDOWS_PER_DELAY,
        "min_eval_windows_per_delay": MIN_EVAL_WINDOWS_PER_DELAY,
        "min_source_videos_per_split": MIN_SOURCE_VIDEOS_PER_SPLIT,
        "delays": [
            {
                "delay": entry.delay,
                "training_window_ids": list(entry.training_window_ids),
                "evaluation_window_ids": list(entry.evaluation_window_ids),
                "training_video_ids": list(entry.training_video_ids),
                "evaluation_video_ids": list(entry.evaluation_video_ids),
            }
            for entry in registration.delays
        ],
    }


def delay_registration_from_payload(
    payload: object,
) -> DelayRegistration:
    if not isinstance(payload, dict):
        raise DelayRegistrationInvalid(
            "delay registration payload must be an object"
        )
    if payload.get("schema") != "r1-e3m-delay-registration-v1":
        raise DelayRegistrationInvalid(
            "delay registration schema mismatch"
        )
    if (
        payload.get("min_train_windows_per_delay")
        != MIN_TRAIN_WINDOWS_PER_DELAY
        or payload.get("min_eval_windows_per_delay")
        != MIN_EVAL_WINDOWS_PER_DELAY
        or payload.get("min_source_videos_per_split")
        != MIN_SOURCE_VIDEOS_PER_SPLIT
    ):
        raise DelayRegistrationInvalid(
            "delay registration minima mismatch"
        )
    raw_delays = payload.get("delays")
    if not isinstance(raw_delays, list):
        raise DelayRegistrationInvalid(
            "delay registration delays must be an array"
        )
    delays = []
    for raw in raw_delays:
        if not isinstance(raw, dict):
            raise DelayRegistrationInvalid(
                "delay registration row must be an object"
            )
        delays.append(
            DelayEligibility(
                delay=_integer(raw.get("delay"), "delay"),
                training_window_ids=_tuple_strings(
                    raw.get("training_window_ids"),
                    "training_window_ids",
                ),
                evaluation_window_ids=_tuple_strings(
                    raw.get("evaluation_window_ids"),
                    "evaluation_window_ids",
                ),
                training_video_ids=_tuple_strings(
                    raw.get("training_video_ids"),
                    "training_video_ids",
                ),
                evaluation_video_ids=_tuple_strings(
                    raw.get("evaluation_video_ids"),
                    "evaluation_video_ids",
                ),
            )
        )
    artifact_digest = _validated_digest(
        payload.get("artifact_root_digest"),
        "artifact_root_digest",
    )
    digest = _validated_digest(
        payload.get("digest"),
        "delay registration digest",
    )
    return DelayRegistration(
        artifact_root_digest=artifact_digest,
        delays=tuple(delays),
        digest=digest,
    )


def _registration_digest(
    artifact_root_digest: str,
    delays: tuple[DelayEligibility, ...],
) -> str:
    payload = {
        "artifact_root_digest": artifact_root_digest,
        "delays": [
            {
                "delay": entry.delay,
                "training_window_ids": list(entry.training_window_ids),
                "evaluation_window_ids": list(entry.evaluation_window_ids),
                "training_video_ids": list(entry.training_video_ids),
                "evaluation_video_ids": list(entry.evaluation_video_ids),
            }
            for entry in delays
        ],
        "min_train_windows_per_delay": MIN_TRAIN_WINDOWS_PER_DELAY,
        "min_eval_windows_per_delay": MIN_EVAL_WINDOWS_PER_DELAY,
        "min_source_videos_per_split": MIN_SOURCE_VIDEOS_PER_SPLIT,
    }
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_DIGEST_VERSION)
    digest.update(encoded)
    return digest.hexdigest()


def _tensor_batch(
    samples: tuple[MechanismSample, ...],
) -> np.ndarray:
    if not samples:
        raise DelayRegistrationInvalid(
            "delay registration requires non-empty samples"
        )
    return np.stack([sample.tensor for sample in samples], axis=0)


def _validated_ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise DelayRegistrationInvalid(f"{name} must be a tuple")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise DelayRegistrationInvalid(
            f"{name} must be sorted and unique"
        )
    for value in values:
        _validated_digest(value, name)


def _validated_text_ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise DelayRegistrationInvalid(f"{name} must be a tuple")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise DelayRegistrationInvalid(
            f"{name} must be sorted and unique"
        )
    if any(not isinstance(value, str) or not value for value in values):
        raise DelayRegistrationInvalid(
            f"{name} must contain non-empty strings"
        )


def _tuple_strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DelayRegistrationInvalid(f"{name} must be an array")
    if any(not isinstance(item, str) for item in value):
        raise DelayRegistrationInvalid(
            f"{name} must contain strings"
        )
    return tuple(value)


def _validated_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise DelayRegistrationInvalid(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise DelayRegistrationInvalid(f"{name} must be an integer")
    return value
