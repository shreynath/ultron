"""Immutable, evidence-oriented capability evaluation primitives.

This module deliberately does not generate answers or inject benchmark answers.
Callers supply a model/system output and a task-owned verifier records the result.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Capability(str, Enum):
    LANGUAGE = "language"
    MATH = "math"
    CODING = "coding"
    COMPUTER_SCIENCE = "computer_science"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    INSTRUCTION = "instruction"
    REASONING = "reasoning"
    RESEARCH = "research"
    TOOL_USE = "tool_use"
    PLANNING = "planning"
    MEMORY = "memory"
    TRUTHFULNESS = "truthfulness"
    WEB_FRESHNESS = "web_freshness"
    TASK_COMPLETION = "task_completion"


class CapabilityStatus(str, Enum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    UNCHANGED = "UNCHANGED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class TaskKind(str, Enum):
    EXACT = "EXACT"
    PYTHON = "PYTHON"
    TOOL_TRACE = "TOOL_TRACE"


class EvaluationTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    capability: Capability
    kind: TaskKind
    prompt: str
    expected: str | None = None
    timeout_seconds: float = Field(default=3, gt=0, le=30)


class CapabilitySuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    suite_id: str
    version: str
    created_at: datetime
    tasks: tuple[EvaluationTask, ...]
    dataset_hash: str
    tokenizer_version: str
    source_manifest_hash: str
    preprocessing_version: str


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    capability: Capability
    outcome: str
    score: float = Field(ge=0, le=1)
    detail: str


class CapabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    suite_id: str
    suite_version: str
    candidate_id: str
    created_at: datetime
    results: tuple[TaskResult, ...]
    scores: dict[str, float]
    generated_outputs: dict[str, str] = Field(default_factory=dict)


def create_suite(
    path: Path, *, suite_id: str, version: str, tasks: tuple[EvaluationTask, ...], tokenizer_version: str,
    source_manifest_hash: str, preprocessing_version: str,
) -> CapabilitySuite:
    if path.exists():
        raise ValueError("Capability suites are immutable; use a new path/version.")
    serial = [task.model_dump(mode="json") for task in tasks]
    dataset_hash = hashlib.sha256(json.dumps(serial, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    suite = CapabilitySuite(suite_id=suite_id, version=version, created_at=datetime.now(UTC), tasks=tasks, dataset_hash=dataset_hash, tokenizer_version=tokenizer_version, source_manifest_hash=source_manifest_hash, preprocessing_version=preprocessing_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(suite.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return suite


def execute_task(task: EvaluationTask, output: str) -> TaskResult:
    if task.kind is TaskKind.EXACT:
        passed = output.strip() == (task.expected or "").strip()
        return TaskResult(task_id=task.task_id, capability=task.capability, outcome="PASS" if passed else "FAIL", score=float(passed), detail="Exact task-owned comparison.")
    if task.kind is TaskKind.TOOL_TRACE:
        passed = output.strip() == (task.expected or "").strip()
        return TaskResult(task_id=task.task_id, capability=task.capability, outcome="PASS" if passed else "FAIL", score=float(passed), detail="Controlled tool trace comparison.")
    with tempfile.TemporaryDirectory(prefix="ultron-capability-") as temporary:
        script = Path(temporary) / "candidate.py"
        script.write_text(output, encoding="utf-8")
        if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").exists():
            return TaskResult(task_id=task.task_id, capability=task.capability, outcome="SANDBOX_UNAVAILABLE", score=0.0, detail="Refused to execute generated code without an OS sandbox.")
        policy = "\n".join((
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow file-read*)",
            f'(allow file-write* (subpath "{temporary}"))',
            '(allow file-read* (literal "/dev/null"))',
            '(allow file-write* (literal "/dev/null"))',
            "(deny network*)",
            f'(deny file-read* (subpath "{Path.home()}"))',
        ))

        try:
            limit_seconds = str(max(1, int(task.timeout_seconds)))
            interpreter = str(Path(sys.executable).resolve())
            completed = subprocess.run(["/usr/bin/sandbox-exec", "-p", policy, "/bin/sh", "-c", f"ulimit -t {limit_seconds}; ulimit -f 1024; exec \"$1\" -I \"$2\"", "sandbox", interpreter, str(script)], text=True, capture_output=True, timeout=task.timeout_seconds, check=False, cwd=temporary, env={"PATH": "/usr/bin:/bin", "HOME": temporary, "TMPDIR": temporary})
        except subprocess.TimeoutExpired:
            return TaskResult(task_id=task.task_id, capability=task.capability, outcome="TIMEOUT", score=0.0, detail="Controlled Python task timed out.")
    if completed.returncode:
        return TaskResult(task_id=task.task_id, capability=task.capability, outcome="RUNTIME_ERROR", score=0.0, detail=completed.stderr[-500:])
    passed = completed.stdout.strip() == (task.expected or "").strip()
    return TaskResult(task_id=task.task_id, capability=task.capability, outcome="PASS" if passed else "FAIL", score=float(passed), detail="Controlled Python stdout comparison.")


def evaluate_outputs(suite: CapabilitySuite, *, candidate_id: str, outputs: dict[str, str]) -> CapabilityReport:
    results = tuple(execute_task(task, outputs.get(task.task_id, "")) for task in suite.tasks)
    scores: dict[str, float] = {}
    for capability in Capability:
        matching = [result.score for result in results if result.capability is capability]
        if matching:
            scores[capability.value] = round(sum(matching) / len(matching), 6)
    return CapabilityReport(suite_id=suite.suite_id, suite_version=suite.version, candidate_id=candidate_id, created_at=datetime.now(UTC), results=results, scores=scores, generated_outputs=dict(outputs))


def evaluate_generator(suite: CapabilitySuite, *, candidate_id: str, generator, generation_config) -> CapabilityReport:
    """Run each immutable task through an ULTRON generator; no answer injection."""
    outputs = {task.task_id: generator.generate(task.prompt, generation_config).text for task in suite.tasks}
    return evaluate_outputs(suite, candidate_id=candidate_id, outputs=outputs)


def capability_matrix(baseline: CapabilityReport, candidate: CapabilityReport, *, minimum_tasks: int = 2, margin: float = 0.05) -> dict[str, CapabilityStatus]:
    if (baseline.suite_id, baseline.suite_version) != (candidate.suite_id, candidate.suite_version):
        raise ValueError("Capability reports must use the same immutable suite.")
    result: dict[str, CapabilityStatus] = {}
    for capability in Capability:
        base = [item for item in baseline.results if item.capability is capability]
        current = [item for item in candidate.results if item.capability is capability]
        if len(base) < minimum_tasks or len(current) < minimum_tasks:
            result[capability.value] = CapabilityStatus.INSUFFICIENT_EVIDENCE
            continue
        delta = candidate.scores[capability.value] - baseline.scores[capability.value]
        result[capability.value] = CapabilityStatus.IMPROVED if delta > margin else (CapabilityStatus.REGRESSED if delta < -margin else CapabilityStatus.UNCHANGED)
    return result
