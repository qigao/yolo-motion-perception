from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .flow_types import ArticulatedFlowEvidence
from .gait import GaitConfig, GaitEvidence, estimate_gait
from .gait_state import LocomotionState, classify_gait


@dataclass(frozen=True)
class OpticalGaitResult:
    track_id: int
    evidence: GaitEvidence
    state: LocomotionState


class _ArticulatedFlowHistory:
    def __init__(self, track_id: int, history_seconds: float):
        self.track_id = track_id
        self.history_seconds = float(history_seconds)
        self._items: deque[ArticulatedFlowEvidence] = deque()

    def add(self, observation: ArticulatedFlowEvidence) -> None:
        if observation.track_id != self.track_id:
            raise ValueError("observation track_id does not match gait history track_id")
        if self._items and observation.start_timestamp < self._items[-1].end_timestamp - 1e-9:
            raise ValueError("gait observation intervals must not overlap")
        self._items.append(observation)
        cutoff = observation.end_timestamp - self.history_seconds
        while self._items and self._items[0].end_timestamp <= cutoff:
            self._items.popleft()

    def observations(self) -> tuple[ArticulatedFlowEvidence, ...]:
        return tuple(self._items)


class OpticalGaitPipeline:
    def __init__(self, config: GaitConfig):
        self.config = config
        self._histories: dict[int, _ArticulatedFlowHistory] = {}

    def update(self, observation: ArticulatedFlowEvidence) -> OpticalGaitResult | None:
        history = self._histories.get(observation.track_id)
        if history is None:
            history = _ArticulatedFlowHistory(observation.track_id, self.config.history_seconds)
            self._histories[observation.track_id] = history
        history.add(observation)

        evidence = estimate_gait(history.observations(), self.config)
        if evidence.sample_count < self.config.min_samples or evidence.duration < self.config.min_duration:
            return None

        return OpticalGaitResult(
            track_id=observation.track_id,
            evidence=evidence,
            state=classify_gait(evidence, self.config),
        )

    def drop_stale(self, active_track_ids: set[int]) -> set[int]:
        removed = set(self._histories) - set(active_track_ids)
        for track_id in removed:
            del self._histories[track_id]
        return removed

    def track_ids(self) -> set[int]:
        return set(self._histories)
