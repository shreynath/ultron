"""ULTRON-owned, randomly initialized foundation-model training path."""

from app.foundation.checkpoints import CheckpointManifest, CheckpointStore
from app.foundation.data import (
    CorpusDocument,
    CorpusPipeline,
    DatasetRelease,
    Provenance,
    TokenPacker,
)
from app.foundation.diagnostics import (
    corpus_diagnostics,
    initialization_diagnostics,
    tokenizer_ab_release,
)
from app.foundation.evaluation import FoundationEvaluator, PromotionPolicy
from app.foundation.experiments import ExperimentRecord, ExperimentRegistry
from app.foundation.generation import (
    GenerationConfig,
    GenerationResult,
    UltronGenerator,
)
from app.foundation.model import UltronDecoder, UltronModelConfig
from app.foundation.phase_k import (
    PhaseKExperimentRecord,
    PhaseKRegistry,
    corpus_balance_report,
    parameter_breakdown,
    phase_k_preflight,
)
from app.foundation.phase_l import (
    CorpusDeficitPlan,
    PhaseLCorpusGate,
    PhaseLSustainedSummary,
    checkpoint_portability,
    corpus_deficit_plan,
    phase_l_corpus_gate,
    phase_l_experiment_report,
    summarize_sustained_run,
)
from app.foundation.scaling import ScalingHarness, ScalingRecommendation
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import FoundationTrainer, TrainingConfig, TrainingResult

__all__ = [
    "CheckpointManifest",
    "CheckpointStore",
    "CorpusDeficitPlan",
    "CorpusDocument",
    "CorpusPipeline",
    "DatasetRelease",
    "ExperimentRecord",
    "ExperimentRegistry",
    "FoundationEvaluator",
    "FoundationTrainer",
    "GenerationConfig",
    "GenerationResult",
    "PhaseKExperimentRecord",
    "PhaseKRegistry",
    "PhaseLCorpusGate",
    "PhaseLSustainedSummary",
    "PromotionPolicy",
    "Provenance",
    "ScalingHarness",
    "ScalingRecommendation",
    "TokenPacker",
    "TokenizerConfig",
    "TrainingConfig",
    "TrainingResult",
    "UltronDecoder",
    "UltronGenerator",
    "UltronModelConfig",
    "UltronTokenizer",
    "checkpoint_portability",
    "corpus_balance_report",
    "corpus_deficit_plan",
    "corpus_diagnostics",
    "initialization_diagnostics",
    "parameter_breakdown",
    "phase_k_preflight",
    "phase_l_corpus_gate",
    "phase_l_experiment_report",
    "summarize_sustained_run",
    "tokenizer_ab_release",
]
