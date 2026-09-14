"""Deterministic neural state machine game experiment."""

from .controller import RecurrentController
from .encoding import encode_observation
from .environment import ToyGame
from .experiment import EpisodeTrace, context_separation, replay_equal, run_episode
from .memory_benchmark import AccuracyCount, MemoryExperimentConfig
from .memory_probe import (
    FittedLinearProbe,
    MemoryProbeConfig,
    MemoryProbeResult,
    ProbeAccuracy,
    fit_linear_probe,
    run_memory_probe,
    run_memory_probe_benchmark,
)
from .memory_task import Cue, DelayedCueEpisode, DelayedCueTask
from .policy import PolicyDecision, RecurrentPolicy
from .reward_readout import RewardModulatedReadout, RewardReadoutDecision
from .types import (
    Action,
    GameConfig,
    GameObservation,
    GameSnapshot,
    NeuralDecision,
    StepResult,
)

__all__ = [
    "AccuracyCount",
    "Action",
    "Cue",
    "DelayedCueEpisode",
    "DelayedCueTask",
    "EpisodeTrace",
    "FittedLinearProbe",
    "GameConfig",
    "GameObservation",
    "GameSnapshot",
    "MemoryExperimentConfig",
    "MemoryProbeConfig",
    "MemoryProbeResult",
    "NeuralDecision",
    "PolicyDecision",
    "ProbeAccuracy",
    "RecurrentController",
    "RecurrentPolicy",
    "RewardModulatedReadout",
    "RewardReadoutDecision",
    "StepResult",
    "ToyGame",
    "context_separation",
    "encode_observation",
    "fit_linear_probe",
    "replay_equal",
    "run_episode",
    "run_memory_probe",
    "run_memory_probe_benchmark",
]
