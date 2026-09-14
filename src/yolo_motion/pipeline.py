from __future__ import annotations

from dataclasses import dataclass

from .history import TrackHistory
from .motion import MotionConfig, estimate_motion
from .state import classify_motion
from .types import MotionEvidence, MotionState, TrackObservation


@dataclass(frozen=True)
class TrackMotionResult:
    track_id: int
    observation: TrackObservation
    evidence: MotionEvidence
    state: MotionState


class MotionPipeline:
    def __init__(self, config: MotionConfig):
        self.config = config
        self._histories: dict[int, TrackHistory] = {}

    def update(self, observation: TrackObservation) -> TrackMotionResult | None:
        history = self._histories.get(observation.track_id)
        if history is None:
            history = TrackHistory(observation.track_id, self.config.history_seconds)
            self._histories[observation.track_id] = history
        history.add(observation)
        evidence = estimate_motion(history.observations(), self.config)
        if (
            evidence.sample_count < self.config.min_samples
            or evidence.duration < self.config.min_duration
        ):
            return None
        return TrackMotionResult(
            track_id=observation.track_id,
            observation=observation,
            evidence=evidence,
            state=classify_motion(evidence, self.config),
        )

    def drop_stale(self, active_track_ids: set[int]) -> set[int]:
        removed = set(self._histories) - set(active_track_ids)
        for track_id in removed:
            del self._histories[track_id]
        return removed

    def track_ids(self) -> set[int]:
        return set(self._histories)
