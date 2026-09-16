"""Deterministic neural state machine game experiment."""

from .action_value import ActionValueDecision, ActionValueUpdate, NormalizedActionValue
from .action_value_benchmark import (
    ActionValueBenchmarkConfig,
    ActionValueCheckpoint,
    ActionValueExperimentResult,
    run_action_value_benchmark,
    run_action_value_experiment,
)
from .controller import RecurrentController
from .delayed_credit import DelayedRewardQueue, PendingReward, RewardDelivery
from .encoding import encode_observation
from .environment import ToyGame
from .experiment import EpisodeTrace, context_separation, replay_equal, run_episode
from .learning_diagnostics import (
    LearningDiagnosticsConfig,
    LearningDiagnosticsResult,
    run_learning_diagnostics,
    run_learning_diagnostics_benchmark,
)
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
from .phase3a_failure_attribution import (
    AttributionMeasurement,
    attribution_payload,
    run_failure_attribution,
)
from .phase3a_credit_compare import (
    CreditComparison,
    EligibilityTraceActionValue,
    credit_comparison_benchmark_payload,
    credit_comparison_payload,
    run_credit_comparison,
    run_credit_comparison_benchmark,
)
from .phase3c_benchmark import (
    AnonymousCreditConfig,
    Phase3CMeasurementResult,
    Phase3CProtocolResult,
    registered_behavior_passed,
    run_phase3c_measurement,
    run_phase3c_measurements,
    run_phase3c_protocol_gate,
)
from .policy import PolicyDecision, RecurrentPolicy
from .reward_learning import (
    RewardLearningConfig,
    RewardLearningResult,
    run_reward_learning_benchmark,
    run_reward_learning_experiment,
)
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
    "ActionValueDecision",
    "ActionValueBenchmarkConfig",
    "ActionValueCheckpoint",
    "ActionValueExperimentResult",
    "ActionValueUpdate",
    "AnonymousCreditConfig",
    "AttributionMeasurement",
    "CreditComparison",
    "Cue",
    "DelayedCueEpisode",
    "DelayedCueTask",
    "DelayedRewardQueue",
    "EpisodeTrace",
    "FittedLinearProbe",
    "GameConfig",
    "GameObservation",
    "GameSnapshot",
    "LearningDiagnosticsConfig",
    "LearningDiagnosticsResult",
    "MemoryExperimentConfig",
    "MemoryProbeConfig",
    "MemoryProbeResult",
    "NeuralDecision",
    "NormalizedActionValue",
    "EligibilityTraceActionValue",
    "PendingReward",
    "Phase3CMeasurementResult",
    "Phase3CProtocolResult",
    "PolicyDecision",
    "ProbeAccuracy",
    "RecurrentController",
    "RecurrentPolicy",
    "RewardDelivery",
    "RewardLearningConfig",
    "RewardLearningResult",
    "RewardModulatedReadout",
    "RewardReadoutDecision",
    "StepResult",
    "ToyGame",
    "context_separation",
    "attribution_payload",
    "credit_comparison_payload",
    "credit_comparison_benchmark_payload",
    "encode_observation",
    "fit_linear_probe",
    "registered_behavior_passed",
    "replay_equal",
    "run_episode",
    "run_action_value_benchmark",
    "run_action_value_experiment",
    "run_failure_attribution",
    "run_credit_comparison",
    "run_credit_comparison_benchmark",
    "run_learning_diagnostics",
    "run_learning_diagnostics_benchmark",
    "run_memory_probe",
    "run_memory_probe_benchmark",
    "run_phase3c_measurement",
    "run_phase3c_measurements",
    "run_phase3c_protocol_gate",
    "run_reward_learning_benchmark",
    "run_reward_learning_experiment",
]
