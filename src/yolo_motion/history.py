from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from .types import TrackObservation


class TrackHistory:
    def __init__(self, track_id: int, history_seconds: float):
        if history_seconds <= 0.0:
            raise ValueError("history_seconds must be positive")
        self.track_id = track_id
        self.history_seconds = float(history_seconds)
        self._items: deque[TrackObservation] = deque()

    def add(self, observation: TrackObservation) -> None:
        if observation.track_id != self.track_id:
            raise ValueError("observation track_id does not match history track_id")
        if self._items and observation.timestamp <= self._items[-1].timestamp:
            raise ValueError("timestamps must be strictly increasing")
        self._items.append(observation)
        cutoff = observation.timestamp - self.history_seconds
        while self._items and self._items[0].timestamp < cutoff:
            self._items.popleft()

    def observations(self) -> Sequence[TrackObservation]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)
