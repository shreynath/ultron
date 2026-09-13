"""Phase O: persistent, bounded execution of ULTRON-owned training jobs.

This is an adapter over the existing corpus, MLX trainer, checkpoint store,
model-only evaluator, Study Queue, and Phase N policy ledger.  It deliberately
does not manufacture a capability metric, use a teacher at runtime, or turn a
memory event into weight updates without a separate immutable dataset release.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.capabilities import Capability, CapabilityReport
from app.foundation.checkpoints import (
    CheckpointError,
    CheckpointManifest,
    CheckpointStore,
    canonical_hash,
    sha256_file,
)
from app.foundation.data import DatasetRelease, ProcessedDocument
from app.foundation.development import (
    CandidatePlan,
    CandidateState,
    DevelopmentOrchestrator,
    EvaluationEvidence,
    EvaluationMode,
    TrainingCompletion,
)
from app.foundation.evaluation import (
    EvaluationComparison,
    EvaluationReport,
    EvaluationSuite,
    FoundationEvaluator,
)
from app.foundation.model import UltronModelConfig
from app.foundation.phase_p import (
    CapabilityComparison,
    CapabilityRun,
    DevelopmentTrace,
    ExperimentHypothesis,
    LearningOpportunity,
    PhasePEngine,
    compare_capability_runs,
    phase_n_evidence,
)
from app.foundation.phase_p import (
    CapabilitySuite as ModelCapabilitySuite,
)
from app.foundation.phase_r import (
    LearningAttribution,
    NextExperimentRecommendation,
    PhaseRStore,
)
from app.foundation.phase_s import (
    NextTrainingRecommendation,
    PhaseSStore,
    TrainingIntervention,
)
from app.foundation.tokenizer import UltronTokenizer
from app.foundation.training import FoundationTrainer, TrainingConfig, TrainingResult
from app.local_intelligence.study_queue import LearningDecision, StudyQueueStore


class AutonomyLevel(int, Enum):
    MANUAL = 0
    PROPOSALS = 1
    DATASET_PREPARATION = 2
    BOUNDED_TRAINING = 3
    EVALUATION_AND_REJECTION = 4
    POLICY_CONTROLLED_PROMOTION = 5


class TrainingJobStatus(str, Enum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    CHECKPOINTING = "CHECKPOINTING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RESUMABLE = "RESUMABLE"


class DatasetEligibility(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    BLOCKED = "BLOCKED"


class FailureCategory(str, Enum):
    MISSING_KNOWLEDGE = "MISSING_KNOWLEDGE"
    WRONG_KNOWLEDGE = "WRONG_KNOWLEDGE"
    REASONING_FAILURE = "REASONING_FAILURE"
    LANGUAGE_FAILURE = "LANGUAGE_FAILURE"
    INSTRUCTION_FAILURE = "INSTRUCTION_FAILURE"
    CONTEXT_FAILURE = "CONTEXT_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"
    GENERALIZATION_FAILURE = "GENERALIZATION_FAILURE"
    EVALUATION_FAILURE = "EVALUATION_FAILURE"


class DatasetRegistryRecord(BaseModel):
    """Read-only facts derived from one immutable CorpusPipeline release."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    release_id: str
    release_path: str
    manifest_file_sha256: str
    release_manifest_sha256: str
    registered_at: datetime
    processing_version: str
    source_types: dict[str, int]
    license_distribution: dict[str, int]
    document_count: int
    token_count: int
    languages: dict[str, int]
    domains: dict[str, int]
    quality_score: float | None
    duplicate_rate: float
    near_duplicate_rate: float
    contamination_status: str
    synthetic_percentage: float
    teacher_generated_percentage: float
    provenance_coverage: float
    validation_status: str
    curriculum_roles: tuple[str, ...]
    eligibility: DatasetEligibility
    eligibility_reasons: tuple[str, ...]


class ModelArtifact(BaseModel):
    """Immutable identity of one complete, locally inspectable checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    model_id: str
    model_version: str
    parameter_count: int
    architecture_config_sha256: str
    tokenizer_id: str
    tokenizer_version: str
    tokenizer_sha256: str
    initialization: str
    pretrained_weights: str
    parent_checkpoint: str | None
    training_run_id: str
    dataset_release_id: str
    dataset_manifest_sha256: str
    tokens_consumed: int
    global_step: int
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    training_config: dict[str, Any]
    hardware: dict[str, Any]
    runtime: dict[str, Any]
    created_at: datetime
    lifecycle_status: str


class TrainingJobEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: TrainingJobStatus
    detail: str


class TrainingJob(BaseModel):
    """A durable request to execute an existing Phase N candidate plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    candidate_id: str
    experiment_id: str
    parent_model_id: str | None = None
    dataset_id: str
    dataset_release_id: str
    release_path: str
    tokenizer_path: str
    architecture_config: dict[str, Any]
    training_config: dict[str, Any]
    curriculum_stage: str
    hypothesis: str
    expected_token_budget: int = Field(gt=0)
    maximum_wall_seconds: int = Field(default=86_400, ge=1)
    checkpoint_interval: int = Field(ge=1)
    evaluation_schedule: str = "MODEL_ONLY_REQUIRED"
    status: TrainingJobStatus = TrainingJobStatus.QUEUED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    ended_at: datetime | None = None
    current_step: int = Field(default=0, ge=0)
    current_tokens: int = Field(default=0, ge=0)
    checkpoint_path: str | None = None
    resume_checkpoint_path: str | None = None
    initial_weights_path: str | None = None
    failure_reason: str | None = None
    events: tuple[TrainingJobEvent, ...] = ()


class ModelOnlyEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evaluation_id: str
    job_id: str
    candidate_id: str
    mode: EvaluationMode = EvaluationMode.BASE_MODEL
    checkpoint_path: str
    report: EvaluationReport
    report_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FailureAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str
    candidate_id: str
    source_report_sha256: str
    categories: dict[FailureCategory, tuple[str, ...]]
    summary: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InterventionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    proposal_id: str
    source_analysis_id: str
    category: FailureCategory
    curriculum_stage: str
    dataset_ids: tuple[str, ...]
    rationale: str
    value_score: float
    state: str = "PROPOSED"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StudyQueueTrainingRequirement(BaseModel):
    """A verified Study Queue event that still needs a license-valid release."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    event_id: str
    subject: str
    source_url: str
    snapshot_path: str | None
    source_sha256: str | None
    training_eligible: bool = False
    reason: str


class DevelopmentMonitor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    champion_id: str | None
    current_candidate_id: str | None
    current_job: TrainingJob | None
    current_dataset_id: str | None
    current_curriculum: str | None
    latest_evaluation: ModelOnlyEvaluation | None
    latest_capability_run: CapabilityRun | None = None
    latest_capability_comparison: CapabilityComparison | None = None
    target_weakness: str | None = None
    next_experiment: ExperimentHypothesis | None = None
    latest_development_trace: DevelopmentTrace | None = None
    latest_learning_attribution: LearningAttribution | None = None
    next_learning_recommendation: NextExperimentRecommendation | None = None
    latest_training_intervention: TrainingIntervention | None = None
    next_training_recommendation: NextTrainingRecommendation | None = None
    resource_state: dict[str, int | str | None] = Field(default_factory=dict)
    training_tokens: int
    next_action: str
    reason: str
    jobs_by_status: dict[str, int]


def _atomic_json(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class PhaseOStore:
    """Persistent state for Phase O; job transitions are retained in each job."""

    def __init__(self, root: Path) -> None:
        self.root = root / "phase_o"
        for name in (
            "datasets",
            "models",
            "jobs",
            "evaluations",
            "analyses",
            "proposals",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _immutable(self, path: Path, value: BaseModel) -> None:
        payload = value.model_dump_json(indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError(f"Immutable record already exists: {path.name}")
            return
        _atomic_json(path, value)

    def dataset(self, record: DatasetRegistryRecord) -> DatasetRegistryRecord:
        self._immutable(self.root / "datasets" / f"{record.dataset_id}.json", record)
        return record

    def datasets(self) -> tuple[DatasetRegistryRecord, ...]:
        return tuple(
            DatasetRegistryRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted((self.root / "datasets").glob("*.json"))
        )

    def model(self, artifact: ModelArtifact) -> ModelArtifact:
        self._immutable(self.root / "models" / f"{artifact.artifact_id}.json", artifact)
        return artifact

    def job(self, job: TrainingJob) -> TrainingJob:
        path = self.root / "jobs" / f"{job.job_id}.json"
        if path.exists():
            raise ValueError(
                "Training job IDs are immutable; use update_job for state transitions."
            )
        _atomic_json(path, job)
        return job

    def get_job(self, job_id: str) -> TrainingJob:
        path = self.root / "jobs" / f"{job_id}.json"
        if not path.exists():
            raise KeyError(job_id)
        return TrainingJob.model_validate_json(path.read_text(encoding="utf-8"))

    def jobs(self) -> tuple[TrainingJob, ...]:
        return tuple(
            TrainingJob.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted((self.root / "jobs").glob("*.json"))
        )

    def replace_job(self, job: TrainingJob) -> TrainingJob:
        _atomic_json(self.root / "jobs" / f"{job.job_id}.json", job)
        return job

    def evaluation(self, value: ModelOnlyEvaluation) -> ModelOnlyEvaluation:
        self._immutable(
            self.root / "evaluations" / f"{value.evaluation_id}.json", value
        )
        return value

    def evaluations(self) -> tuple[ModelOnlyEvaluation, ...]:
        return tuple(
            ModelOnlyEvaluation.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted((self.root / "evaluations").glob("*.json"))
        )

    def analysis(self, value: FailureAnalysis) -> FailureAnalysis:
        self._immutable(self.root / "analyses" / f"{value.analysis_id}.json", value)
        return value

    def proposal(self, value: InterventionProposal) -> InterventionProposal:
        self._immutable(self.root / "proposals" / f"{value.proposal_id}.json", value)
        return value


class PhaseOEngine:
    """Executes bounded, real local training only through FoundationTrainer."""

    _TRANSITIONS: ClassVar[dict[TrainingJobStatus, set[TrainingJobStatus]]] = {
        TrainingJobStatus.QUEUED: {
            TrainingJobStatus.VALIDATING,
            TrainingJobStatus.CANCELLED,
        },
        TrainingJobStatus.VALIDATING: {
            TrainingJobStatus.RUNNING,
            TrainingJobStatus.FAILED,
        },
        TrainingJobStatus.RUNNING: {
            TrainingJobStatus.CHECKPOINTING,
            TrainingJobStatus.EVALUATING,
            TrainingJobStatus.RESUMABLE,
            TrainingJobStatus.FAILED,
        },
        TrainingJobStatus.CHECKPOINTING: {
            TrainingJobStatus.RUNNING,
            TrainingJobStatus.RESUMABLE,
            TrainingJobStatus.FAILED,
        },
        TrainingJobStatus.EVALUATING: {
            TrainingJobStatus.COMPLETED,
            TrainingJobStatus.FAILED,
        },
        TrainingJobStatus.RESUMABLE: {
            TrainingJobStatus.VALIDATING,
            TrainingJobStatus.CANCELLED,
        },
        TrainingJobStatus.PAUSED: {
            TrainingJobStatus.QUEUED,
            TrainingJobStatus.CANCELLED,
        },
        TrainingJobStatus.COMPLETED: set(),
        TrainingJobStatus.FAILED: set(),
        TrainingJobStatus.CANCELLED: set(),
    }

    def __init__(
        self, root: Path, *, autonomy: AutonomyLevel = AutonomyLevel.MANUAL
    ) -> None:
        self.root, self.autonomy = root, autonomy
        self.store = PhaseOStore(root)
        self.development = DevelopmentOrchestrator(root)

    def register_dataset(
        self,
        *,
        dataset_id: str,
        release_dir: Path,
        tokenizer: UltronTokenizer,
        curriculum_roles: tuple[str, ...],
    ) -> DatasetRegistryRecord:
        release = DatasetRelease.model_validate_json(
            (release_dir / "manifest.json").read_text(encoding="utf-8")
        )
        documents = [
            ProcessedDocument.model_validate_json(line)
            for line in (release_dir / "documents.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if hashlib_sha := release.manifest_sha256:
            release_hash = hashlib_sha
        else:
            raise ValueError(
                "Corpus release is missing its immutable release-manifest hash."
            )
        reasons: list[str] = []
        if (
            not documents
            or not (release_dir / "train.jsonl").exists()
            or not (release_dir / "validation.jsonl").exists()
        ):
            reasons.append("MISSING_REQUIRED_SPLITS")
        if release.provenance_coverage < 1:
            reasons.append("INCOMPLETE_PROVENANCE")
        if release.leakage_checks.get("status") != "PASS":
            reasons.append("SPLIT_LEAKAGE")
        if any(item.provenance.license in {"", "UNSPECIFIED"} for item in documents):
            reasons.append("LICENSING_UNCERTAINTY")
        token_count = sum(len(tokenizer.encode(item.content)) + 1 for item in documents)
        source_types = Counter(item.provenance.source_type.value for item in documents)
        quality = (
            sum(item.quality_score for item in documents) / len(documents)
            if documents
            else None
        )
        total = max(1, len(documents))
        record = DatasetRegistryRecord(
            dataset_id=dataset_id,
            release_id=release.release_id,
            release_path=str(release_dir.resolve()),
            manifest_file_sha256=sha256_file(release_dir / "manifest.json"),
            release_manifest_sha256=release_hash,
            registered_at=datetime.now(UTC),
            document_count=len(documents),
            token_count=token_count,
            processing_version=release.processing_version,
            source_types=dict(source_types),
            license_distribution=dict(
                Counter(item.provenance.license for item in documents)
            ),
            languages=release.language_breakdown,
            domains=dict(release.composition.get("domain_documents", {})),
            quality_score=round(quality, 6) if quality is not None else None,
            duplicate_rate=float(
                release.duplicate_statistics.get("duplicate_rate", 0.0)
            ),
            near_duplicate_rate=float(
                release.duplicate_statistics.get("near_duplicates_removed", 0)
            )
            / total,
            contamination_status=str(release.leakage_checks.get("status", "UNKNOWN")),
            synthetic_percentage=round(
                source_types.get("CURATED_EXPERIENCE", 0) / total, 6
            ),
            teacher_generated_percentage=0.0,
            provenance_coverage=release.provenance_coverage,
            validation_status="PASS" if not reasons else "FAIL",
            curriculum_roles=curriculum_roles,
            eligibility=DatasetEligibility.ELIGIBLE
            if not reasons
            else DatasetEligibility.BLOCKED,
            eligibility_reasons=tuple(reasons),
        )
        return self.store.dataset(record)

    def register_checkpoint(
        self,
        checkpoint: Path,
        model_config: UltronModelConfig,
        *,
        lifecycle_status: str = "CANDIDATE",
    ) -> ModelArtifact:
        manifest = CheckpointStore.inspect(checkpoint)
        if not isinstance(manifest, CheckpointManifest):
            raise TypeError(
                "Phase O registers only complete resumable ULTRON checkpoints."
            )
        if manifest.architecture_config_sha256 != canonical_hash(
            model_config.model_dump(mode="json")
        ):
            raise ValueError(
                "Checkpoint does not match the supplied ULTRON model architecture."
            )
        if (
            manifest.lineage.get("origin") != "ULTRON_FROM_SCRATCH"
            or manifest.lineage.get("pretrained_weights") != "NONE"
        ):
            raise ValueError(
                "Phase O rejects non-proprietary or pretrained model lineage."
            )
        artifact = ModelArtifact(
            artifact_id=f"{manifest.training_run_id}-{manifest.checkpoint_id}",
            checkpoint_path=str(checkpoint.resolve()),
            checkpoint_sha256=sha256_file(checkpoint / "model.safetensors"),
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            parameter_count=model_config.parameter_estimate(),
            architecture_config_sha256=manifest.architecture_config_sha256,
            tokenizer_id=manifest.tokenizer_id,
            tokenizer_version=manifest.tokenizer_version,
            tokenizer_sha256=manifest.tokenizer_sha256,
            initialization=manifest.lineage.get("initialization", "UNKNOWN"),
            pretrained_weights=manifest.lineage.get("pretrained_weights", "UNKNOWN"),
            parent_checkpoint=manifest.parent_checkpoint,
            training_run_id=manifest.training_run_id,
            dataset_release_id=manifest.dataset_release_id,
            dataset_manifest_sha256=manifest.dataset_manifest_sha256,
            tokens_consumed=manifest.tokens_consumed,
            global_step=manifest.global_step,
            optimizer=manifest.optimizer_config,
            scheduler=manifest.scheduler_config,
            training_config=manifest.training_config,
            hardware=manifest.hardware,
            runtime=manifest.runtime,
            created_at=manifest.created_at,
            lifecycle_status=lifecycle_status,
        )
        return self.store.model(artifact)

    def queue_job(
        self,
        *,
        plan: CandidatePlan,
        dataset_id: str,
        tokenizer_path: Path,
        model_config: UltronModelConfig,
        training_config: TrainingConfig,
        resume_checkpoint: Path | None = None,
        initial_weights: Path | None = None,
    ) -> TrainingJob:
        if (
            self.development.candidate_states().get(plan.candidate_id)
            is not CandidateState.CANDIDATE
        ):
            raise ValueError(
                "Phase O jobs require a Phase N candidate in CANDIDATE state."
            )
        dataset = next(
            (item for item in self.store.datasets() if item.dataset_id == dataset_id),
            None,
        )
        if dataset is None or dataset.eligibility is not DatasetEligibility.ELIGIBLE:
            raise ValueError(
                "Training jobs require an eligible registered immutable dataset."
            )
        if (
            training_config.model_id,
            training_config.dataset_release_id,
            training_config.dataset_manifest_sha256,
        ) != (plan.model_id, plan.dataset_release_id, plan.dataset_manifest_sha256):
            raise ValueError(
                "Training config must exactly match the Phase N candidate model and dataset identity."
            )
        if (
            dataset.release_id != plan.dataset_release_id
            or dataset.release_manifest_sha256 != plan.dataset_manifest_sha256
        ):
            raise ValueError(
                "Registered dataset does not match the Phase N candidate plan."
            )
        if (
            tokenizer_path.resolve()
            .joinpath("tokenizer.sha256")
            .read_text(encoding="utf-8")
            .strip()
            != plan.tokenizer_sha256
        ):
            raise ValueError("Tokenizer does not match the Phase N candidate plan.")
        if resume_checkpoint and initial_weights:
            raise ValueError(
                "A job may resume optimizer state or initialize parent weights, not both."
            )
        requested_tokens = (
            training_config.steps
            * training_config.batch_size
            * model_config.max_sequence_length
            * training_config.gradient_accumulation_steps
        )
        resumed_tokens = 0
        if resume_checkpoint:
            manifest = CheckpointStore.inspect(resume_checkpoint)
            if not isinstance(manifest, CheckpointManifest):
                raise TypeError(
                    "A Phase O resume path must contain a complete checkpoint manifest."
                )
            resumed_tokens = manifest.tokens_consumed
        if initial_weights:
            parent = CheckpointStore.inspect(initial_weights.parent)
            if not isinstance(parent, CheckpointManifest):
                raise TypeError(
                    "Initial weights must come from a complete proprietary checkpoint."
                )
            if parent.architecture_config_sha256 != canonical_hash(
                model_config.model_dump(mode="json")
            ):
                raise ValueError(
                    "Initial weights do not match the requested model architecture."
                )
            if parent.tokenizer_sha256 != plan.tokenizer_sha256:
                raise ValueError(
                    "Initial weights do not match the Phase N candidate tokenizer."
                )
        if requested_tokens - resumed_tokens > plan.training_token_budget:
            raise ValueError(
                "Training configuration exceeds the candidate token budget."
            )
        job = TrainingJob(
            candidate_id=plan.candidate_id,
            experiment_id=plan.experiment_id,
            parent_model_id=plan.parent_experiment,
            dataset_id=dataset_id,
            dataset_release_id=dataset.release_id,
            release_path=dataset.release_path,
            tokenizer_path=str(tokenizer_path.resolve()),
            architecture_config=model_config.model_dump(mode="json"),
            training_config=training_config.model_dump(mode="json"),
            curriculum_stage=plan.curriculum_stage,
            hypothesis=plan.reason,
            expected_token_budget=plan.training_token_budget,
            checkpoint_interval=training_config.checkpoint_every,
            resume_checkpoint_path=str(resume_checkpoint.resolve())
            if resume_checkpoint
            else None,
            initial_weights_path=str(initial_weights.resolve())
            if initial_weights
            else None,
        )
        return self.store.job(job)

    def _transition(
        self, job: TrainingJob, status: TrainingJobStatus, detail: str, **changes: Any
    ) -> TrainingJob:
        if status not in self._TRANSITIONS[job.status]:
            raise ValueError(
                f"Illegal training-job transition: {job.status.value} -> {status.value}"
            )
        event = TrainingJobEvent(status=status, detail=detail)
        return self.store.replace_job(
            job.model_copy(
                update={"status": status, "events": (*job.events, event), **changes}
            )
        )

    def _verify_registered_dataset(self, job: TrainingJob) -> DatasetRegistryRecord:
        dataset = next(
            (
                item
                for item in self.store.datasets()
                if item.dataset_id == job.dataset_id
            ),
            None,
        )
        if dataset is None or dataset.eligibility is not DatasetEligibility.ELIGIBLE:
            raise ValueError("Job dataset is absent or no longer eligible.")
        manifest = Path(dataset.release_path) / "manifest.json"
        if sha256_file(manifest) != dataset.manifest_file_sha256:
            raise ValueError("Immutable dataset manifest changed after registration.")
        return dataset

    @staticmethod
    def _split_text(release: Path, split: str) -> str:
        return "\n".join(
            ProcessedDocument.model_validate_json(line).content
            for line in (release / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )

    def run_job(
        self,
        job_id: str,
        *,
        operator_id: str | None = None,
        train_sequences: Sequence[list[int] | tuple[list[int], list[int]]]
        | None = None,
        validation_sequences: Sequence[list[int] | tuple[list[int], list[int]]]
        | None = None,
    ) -> TrainingJob:
        """Run actual FoundationTrainer work and persist its real result/checkpoint."""
        job = self.store.get_job(job_id)
        if job.status not in {TrainingJobStatus.QUEUED, TrainingJobStatus.RESUMABLE}:
            raise ValueError("Only queued or resumable jobs can run.")
        job = self._transition(
            job,
            TrainingJobStatus.VALIDATING,
            "Checking immutable dataset and Phase N state.",
        )
        try:
            self._verify_registered_dataset(job)
            states = self.development.candidate_states()
            if states.get(job.candidate_id) is CandidateState.CANDIDATE:
                if (
                    self.development.ledger.policy()
                    and self.development.ledger.policy().allow_automatic_training_start
                ):
                    self.development.start_training(
                        job.candidate_id, lambda _: job.job_id
                    )
                elif operator_id:
                    self.development.record_operator_training_start(
                        job.candidate_id,
                        operator_id=operator_id,
                        local_job_id=job.job_id,
                    )
                else:
                    raise PermissionError(
                        "This candidate requires an identified operator to start training."
                    )
            elif states.get(job.candidate_id) is CandidateState.RESUMABLE:
                self.development.resume_training(
                    job.candidate_id, local_job_id=job.job_id
                )
            elif states.get(job.candidate_id) is not CandidateState.TRAINING:
                raise ValueError(
                    "Phase N candidate is not available for training or recovery."
                )
            job = self._transition(
                job,
                TrainingJobStatus.RUNNING,
                "Starting real local FoundationTrainer execution.",
                started_at=job.started_at or datetime.now(UTC),
                failure_reason=None,
            )
            tokenizer = UltronTokenizer(Path(job.tokenizer_path))
            config = TrainingConfig.model_validate(job.training_config).model_copy(
                update={"maximum_wall_seconds": job.maximum_wall_seconds}
            )
            model = UltronModelConfig.model_validate(job.architecture_config)
            resumed_tokens = 0
            if job.resume_checkpoint_path:
                prior = CheckpointStore.inspect(Path(job.resume_checkpoint_path))
                if not isinstance(prior, CheckpointManifest):
                    raise TypeError(
                        "A Phase O resume path must contain a complete checkpoint manifest."
                    )
                resumed_tokens = prior.tokens_consumed
            result: TrainingResult = FoundationTrainer(self.root).train(
                model_config=model,
                tokenizer=tokenizer,
                train_text=""
                if train_sequences is not None
                else self._split_text(Path(job.release_path), "train"),
                validation_text=""
                if validation_sequences is not None
                else self._split_text(Path(job.release_path), "validation"),
                train_sequences=train_sequences,
                validation_sequences=validation_sequences,
                config=config,
                resume_checkpoint=Path(job.resume_checkpoint_path)
                if job.resume_checkpoint_path
                else None,
                resume_weights=Path(job.initial_weights_path)
                if job.initial_weights_path
                else None,
                parent_checkpoint=Path(job.initial_weights_path).parent.name
                if job.initial_weights_path
                else None,
            )
            checkpoint = Path(result.checkpoint_path)
            self.register_checkpoint(checkpoint, model)
            manifest = CheckpointStore.inspect(checkpoint)
            assert isinstance(manifest, CheckpointManifest)
            candidate_tokens = result.tokens_processed - resumed_tokens
            if candidate_tokens < 0:
                raise ValueError(
                    "Resumed training reported fewer tokens than its valid parent checkpoint."
                )
            self.development.record_training_completion(
                TrainingCompletion(
                    candidate_id=job.candidate_id,
                    training_run_id=result.run_id,
                    checkpoint_id=manifest.checkpoint_id,
                    checkpoint_sha256=sha256_file(checkpoint / "model.safetensors"),
                    tokens_processed=candidate_tokens,
                    detail="Actual FoundationTrainer completion.",
                )
            )
            detail = "Checkpoint complete; model-only evaluation is required."
            if result.early_stop_reason and result.early_stop_reason.startswith(
                "WALL_CLOCK_LIMIT:"
            ):
                detail = f"{result.early_stop_reason}; training stopped at a complete checkpoint and requires model-only evaluation."
            job = self._transition(
                job,
                TrainingJobStatus.EVALUATING,
                detail,
                current_step=result.steps_completed,
                current_tokens=candidate_tokens,
                checkpoint_path=result.checkpoint_path,
                ended_at=result.completed_at,
            )
            return job
        except Exception as exc:
            latest = self._latest_valid_checkpoint(job)
            status = TrainingJobStatus.RESUMABLE if latest else TrainingJobStatus.FAILED
            if (
                self.development.candidate_states().get(job.candidate_id)
                is CandidateState.TRAINING
            ):
                self.development.record_training_interruption(
                    job.candidate_id, reason=f"{type(exc).__name__}: {exc}"
                )
            if (
                job.status in self._TRANSITIONS
                and status in self._TRANSITIONS[job.status]
            ):
                return self._transition(
                    job,
                    status,
                    f"{type(exc).__name__}: {exc}",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    resume_checkpoint_path=str(latest) if latest else None,
                    ended_at=datetime.now(UTC),
                )
            raise

    def _latest_valid_checkpoint(self, job: TrainingJob) -> Path | None:
        run_id = str(job.training_config.get("run_id", ""))
        directory = self.root / "models" / "base" / run_id
        candidates = (
            sorted(directory.glob("checkpoint-*"), reverse=True)
            if directory.exists()
            else []
        )
        for candidate in candidates:
            try:
                CheckpointStore.inspect(candidate)
                return candidate
            except CheckpointError:
                continue
        return None

    def recover_interrupted_jobs(self) -> tuple[TrainingJob, ...]:
        """Mark persisted jobs from a dead process resumable only with a valid checkpoint."""
        recovered: list[TrainingJob] = []
        for job in self.store.jobs():
            if job.status not in {
                TrainingJobStatus.RUNNING,
                TrainingJobStatus.CHECKPOINTING,
            }:
                continue
            checkpoint = self._latest_valid_checkpoint(job)
            status = (
                TrainingJobStatus.RESUMABLE if checkpoint else TrainingJobStatus.FAILED
            )
            if (
                self.development.candidate_states().get(job.candidate_id)
                is CandidateState.TRAINING
            ):
                self.development.record_training_interruption(
                    job.candidate_id, reason="Recovered after process interruption."
                )
            recovered.append(
                self._transition(
                    job,
                    status,
                    "Recovered after process interruption; checkpoint inspected.",
                    resume_checkpoint_path=str(checkpoint) if checkpoint else None,
                    failure_reason=None
                    if checkpoint
                    else "No valid checkpoint after interrupted process.",
                )
            )
        return tuple(recovered)

    def evaluate_model_only(
        self, job_id: str, suite: EvaluationSuite
    ) -> ModelOnlyEvaluation:
        job = self.store.get_job(job_id)
        if job.status is not TrainingJobStatus.EVALUATING or not job.checkpoint_path:
            raise ValueError(
                "Model-only evaluation requires a completed checkpoint awaiting evaluation."
            )
        tokenizer, model = (
            UltronTokenizer(Path(job.tokenizer_path)),
            UltronModelConfig.model_validate(job.architecture_config),
        )
        report = FoundationEvaluator().evaluate(
            Path(job.checkpoint_path), tokenizer, model, suite
        )
        record = ModelOnlyEvaluation(
            evaluation_id=f"{job.job_id}-{suite.suite_id}-{suite.version}",
            job_id=job.job_id,
            candidate_id=job.candidate_id,
            checkpoint_path=job.checkpoint_path,
            report=report,
            report_sha256=canonical_hash(report.model_dump(mode="json")),
        )
        self.store.evaluation(record)
        # The Phase N ledger receives the same base-model evidence, but only a
        # clearly named loss transformation.  It is not passed off as task
        # accuracy and cannot satisfy the default capability promotion policy.
        self.development.record_evaluation(
            EvaluationEvidence(
                evaluation_id=f"{record.evaluation_id}-phase-n",
                candidate_id=job.candidate_id,
                mode=EvaluationMode.BASE_MODEL,
                suite_id=suite.suite_id,
                suite_version=suite.version,
                suite_sha256=canonical_hash(suite.model_dump(mode="json")),
                report_sha256=record.report_sha256,
                metrics={
                    "teacher_forced_loss_proxy": 1.0 / (1.0 + report.validation_loss)
                },
                evaluator_version="FoundationEvaluator.MODEL_ONLY.v1",
            )
        )
        self._transition(
            job,
            TrainingJobStatus.COMPLETED,
            "Model-only evaluation completed; capability/regression suites remain required for promotion.",
        )
        return record

    def evaluate_capability_model_only(
        self,
        job_id: str,
        suite: ModelCapabilitySuite,
        *,
        baseline_run: CapabilityRun | None = None,
        max_new_tokens: int = 32,
    ) -> tuple[CapabilityRun, CapabilityComparison]:
        """Record primary task capability evidence from the local checkpoint only.

        ``baseline_run`` must come from the exact same immutable suite.  When
        omitted, the run is compared to itself solely to establish a first,
        operator-reviewed champion baseline; it cannot demonstrate improvement.
        """
        job = self.store.get_job(job_id)
        if job.status is not TrainingJobStatus.EVALUATING or not job.checkpoint_path:
            raise ValueError(
                "Capability evaluation requires a completed checkpoint awaiting evaluation."
            )
        evaluator = PhasePEngine(self.root)
        run = evaluator.evaluate_model_only(
            checkpoint=Path(job.checkpoint_path),
            tokenizer=UltronTokenizer(Path(job.tokenizer_path)),
            model_config=UltronModelConfig.model_validate(job.architecture_config),
            suite=suite,
            candidate_id=job.candidate_id,
            max_new_tokens=max_new_tokens,
        )
        comparison = compare_capability_runs(baseline_run or run, run)
        evaluator.store.write_immutable(
            "comparisons", f"comparison-{run.run_id}", comparison
        )
        self.development.record_evaluation(phase_n_evidence(run, comparison))
        self._transition(
            job,
            TrainingJobStatus.COMPLETED,
            "Immutable model-only capability and regression evidence completed.",
        )
        return run, comparison

    @staticmethod
    def compare_model_only(
        baseline: ModelOnlyEvaluation, candidate: ModelOnlyEvaluation
    ) -> EvaluationComparison:
        if (
            baseline.mode is not EvaluationMode.BASE_MODEL
            or candidate.mode is not EvaluationMode.BASE_MODEL
        ):
            raise ValueError("Only proprietary-model-only results are comparable here.")
        return FoundationEvaluator.compare(baseline.report, candidate.report)

    def analyze_capability_failures(self, report: CapabilityReport) -> FailureAnalysis:
        mapping = {
            Capability.MATH: FailureCategory.REASONING_FAILURE,
            Capability.REASONING: FailureCategory.REASONING_FAILURE,
            Capability.CODING: FailureCategory.REASONING_FAILURE,
            Capability.INSTRUCTION: FailureCategory.INSTRUCTION_FAILURE,
            Capability.TOOL_USE: FailureCategory.TOOL_FAILURE,
            Capability.MEMORY: FailureCategory.RETRIEVAL_FAILURE,
            Capability.LANGUAGE: FailureCategory.LANGUAGE_FAILURE,
            Capability.TRUTHFULNESS: FailureCategory.WRONG_KNOWLEDGE,
        }
        grouped: dict[FailureCategory, list[str]] = {}
        for result in report.results:
            if result.outcome != "PASS":
                grouped.setdefault(
                    mapping.get(result.capability, FailureCategory.EVALUATION_FAILURE),
                    [],
                ).append(result.task_id)
        analysis = FailureAnalysis(
            analysis_id=f"failure-{report.candidate_id}-{canonical_hash(report.model_dump(mode='json'))[:12]}",
            candidate_id=report.candidate_id,
            source_report_sha256=canonical_hash(report.model_dump(mode="json")),
            categories={key: tuple(value) for key, value in grouped.items()},
            summary="Task-owned capability failures grouped by declared task capability. This is intervention evidence, not ground truth about cause.",
        )
        return self.store.analysis(analysis)

    def propose_next_intervention(
        self, analysis: FailureAnalysis
    ) -> InterventionProposal | None:
        eligible = self.store.datasets()
        if not analysis.categories or not eligible:
            return None
        category, tasks = max(
            analysis.categories.items(), key=lambda item: len(item[1])
        )
        choices = tuple(
            record.dataset_id
            for record in eligible
            if record.eligibility is DatasetEligibility.ELIGIBLE
            and record.curriculum_roles
        )
        if not choices:
            return None
        stage = (
            "REASONING"
            if category is FailureCategory.REASONING_FAILURE
            else "GENERAL_KNOWLEDGE"
        )
        proposal = InterventionProposal(
            proposal_id=f"proposal-{analysis.analysis_id}",
            source_analysis_id=analysis.analysis_id,
            category=category,
            curriculum_stage=stage,
            dataset_ids=choices,
            rationale=f"{len(tasks)} task-owned {category.value} failures; eligible datasets are ranked before unvalidated acquisition.",
            value_score=round(
                len(tasks)
                / max(
                    1,
                    sum(
                        record.token_count
                        for record in eligible
                        if record.dataset_id in choices
                    ),
                ),
                12,
            ),
        )
        return self.store.proposal(proposal)

    @staticmethod
    def study_queue_requirements(
        root: Path,
    ) -> tuple[StudyQueueTrainingRequirement, ...]:
        requirements: list[StudyQueueTrainingRequirement] = []
        for event in StudyQueueStore(root).events(limit=100):
            snapshot = event.source_provenance.get("snapshot_path")
            verified = (
                event.decision is LearningDecision.LEARNING_DEMONSTRATED
                and event.validation_status == "VERIFIED"
            )
            requirements.append(
                StudyQueueTrainingRequirement(
                    event_id=str(event.event_id),
                    subject=event.subject,
                    source_url=event.source_id,
                    snapshot_path=snapshot,
                    source_sha256=event.source_provenance.get("sha256"),
                    training_eligible=False,
                    reason="VERIFIED_STUDY_TARGET_REQUIRES_LICENSED_PROVENANCE_AWARE_CORPUS_RELEASE"
                    if verified
                    else "STUDY_EVENT_NOT_A_VERIFIED_LEARNING_TARGET",
                )
            )
        return tuple(requirements)

    def monitor(self) -> DevelopmentMonitor:
        jobs, evaluations = self.store.jobs(), self.store.evaluations()
        capability_store = PhasePEngine(self.root).store
        capability_runs = capability_store.reads("runs", CapabilityRun)
        capability_comparisons = capability_store.reads(
            "comparisons", CapabilityComparison
        )
        opportunities = capability_store.reads("opportunities", LearningOpportunity)
        hypotheses = capability_store.reads("hypotheses", ExperimentHypothesis)
        traces = capability_store.reads("traces", DevelopmentTrace)
        phase_r_store = PhaseRStore(self.root)
        learning_attributions = phase_r_store.reads("learning", LearningAttribution)
        recommendations = phase_r_store.reads(
            "recommendations", NextExperimentRecommendation
        )
        phase_s_store = PhaseSStore(self.root)
        interventions = phase_s_store.reads("interventions", TrainingIntervention)
        training_recommendations = phase_s_store.reads(
            "recommendations", NextTrainingRecommendation
        )
        active = next(
            (
                job
                for job in jobs
                if job.status
                in {
                    TrainingJobStatus.RUNNING,
                    TrainingJobStatus.VALIDATING,
                    TrainingJobStatus.EVALUATING,
                    TrainingJobStatus.RESUMABLE,
                }
            ),
            None,
        )
        state = self.development.status()
        latest_opportunity = opportunities[-1] if opportunities else None
        return DevelopmentMonitor(
            champion_id=state.champion_id,
            current_candidate_id=active.candidate_id if active else None,
            current_job=active,
            current_dataset_id=active.dataset_id if active else None,
            current_curriculum=active.curriculum_stage if active else None,
            latest_evaluation=evaluations[-1] if evaluations else None,
            latest_capability_run=capability_runs[-1] if capability_runs else None,
            latest_capability_comparison=capability_comparisons[-1]
            if capability_comparisons
            else None,
            target_weakness=f"{latest_opportunity.domain.value}: {len(latest_opportunity.task_ids)} measured failed task(s)"
            if latest_opportunity
            else None,
            next_experiment=hypotheses[-1] if hypotheses else None,
            latest_development_trace=traces[-1] if traces else None,
            latest_learning_attribution=learning_attributions[-1]
            if learning_attributions
            else None,
            next_learning_recommendation=recommendations[-1]
            if recommendations
            else None,
            latest_training_intervention=interventions[-1] if interventions else None,
            next_training_recommendation=training_recommendations[-1]
            if training_recommendations
            else None,
            resource_state={
                "training_tokens_recorded": state.training_tokens_recorded,
                "active_job_wall_seconds": active.maximum_wall_seconds
                if active
                else None,
                "active_job_token_budget": active.expected_token_budget
                if active
                else None,
                "autonomy_level": self.autonomy.name,
            },
            training_tokens=sum(job.current_tokens for job in jobs),
            next_action=state.next_action.value,
            reason=state.reason,
            jobs_by_status={
                status.value: sum(job.status is status for job in jobs)
                for status in TrainingJobStatus
            },
        )

    def run_once(self, *, suite: EvaluationSuite | None = None) -> DevelopmentMonitor:
        """Execute at most one persisted development action under autonomy policy.

        This is a restart-safe state-machine tick, not a hidden background
        thread. A host can schedule it or use :meth:`run_continuously`.
        """
        self.recover_interrupted_jobs()
        jobs = self.store.jobs()
        waiting_evaluation = next(
            (job for job in jobs if job.status is TrainingJobStatus.EVALUATING), None
        )
        if (
            waiting_evaluation
            and suite
            and self.autonomy >= AutonomyLevel.EVALUATION_AND_REJECTION
        ):
            self.evaluate_model_only(waiting_evaluation.job_id, suite)
            return self.monitor()
        runnable = next(
            (
                job
                for job in jobs
                if job.status in {TrainingJobStatus.QUEUED, TrainingJobStatus.RESUMABLE}
            ),
            None,
        )
        policy = self.development.ledger.policy()
        if (
            runnable
            and self.autonomy >= AutonomyLevel.BOUNDED_TRAINING
            and policy
            and policy.allow_automatic_training_start
        ):
            self.run_job(runnable.job_id)
        return self.monitor()

    def run_continuously(
        self,
        *,
        max_cycles: int,
        poll_seconds: float = 1.0,
        suite: EvaluationSuite | None = None,
    ) -> tuple[DevelopmentMonitor, ...]:
        """A bounded host-side daemon loop; it stops after ``max_cycles``."""
        if max_cycles < 1 or poll_seconds < 0:
            raise ValueError(
                "max_cycles must be positive and poll_seconds cannot be negative."
            )
        snapshots: list[DevelopmentMonitor] = []
        for index in range(max_cycles):
            snapshots.append(self.run_once(suite=suite))
            if index + 1 < max_cycles and poll_seconds:
                time.sleep(poll_seconds)
        return tuple(snapshots)
