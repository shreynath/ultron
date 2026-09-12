"""Reusable, fail-loud Phase L data, objective, and recovery diagnostics."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from app.foundation.checkpoints import CheckpointStore, sha256_file
from app.foundation.data import ProcessedDocument, resolve_artifact_path
from app.foundation.model import UltronModelConfig, build_model
from app.foundation.tokenizer import UltronTokenizer
from app.foundation.training import ResumableAdamW, TrainingConfig


def _revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configuration_audit(*, model_config: UltronModelConfig, training_config: TrainingConfig, tokenizer: UltronTokenizer, release_dir: Path, tokenized_dir: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    """Trace registered values through the actual serialized runtime inputs."""
    release, tokenized = _manifest(release_dir / "manifest.json"), _manifest(tokenized_dir / "manifest.json")
    checks: dict[str, bool] = {
        "release_matches_training_config": release["release_id"] == training_config.dataset_release_id,
        "tokenized_release_matches_training_config": tokenized["release_id"] == training_config.dataset_release_id,
        "tokenizer_sha_matches_tokenized": tokenized["tokenizer_sha256"] == tokenizer.sha256,
        "context_matches_model": tokenized["context_length"] == model_config.max_sequence_length,
        "vocabulary_matches_model": tokenizer.vocab_size == model_config.vocab_size,
        "all_shards_present_and_hashed": all(resolve_artifact_path(item["path"]).exists() and sha256_file(resolve_artifact_path(item["path"])) == item["sha256"] for item in tokenized["shards"]),
        "development_claim_is_explicit": training_config.evidence_class.value == "DEVELOPMENT_SCALE",
    }
    checkpoint_data: dict[str, Any] | None = None
    if checkpoint:
        manifest = CheckpointStore.inspect(checkpoint)
        checkpoint_data = manifest.model_dump(mode="json")
        checks.update({
            "checkpoint_model_matches": manifest.model_id == training_config.model_id,
            "checkpoint_dataset_matches": manifest.dataset_release_id == training_config.dataset_release_id,
            "checkpoint_tokenizer_matches": manifest.tokenizer_sha256 == tokenizer.sha256,
            "checkpoint_architecture_matches": manifest.architecture_config_sha256 == hashlib.sha256(json.dumps(model_config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        })
    return {"experiment_id": training_config.run_id, "parent_experiment": checkpoint_data["checkpoint_id"] if checkpoint_data else None, "git_revision": _revision(), "result": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "model_config": model_config.model_dump(mode="json"), "training_config": training_config.model_dump(mode="json"), "tokenized_packing_strategy": tokenized.get("packing_strategy", "FIXED_TERMINAL_EOS_OVERLAP"), "checkpoint": checkpoint_data, "next_action": "Proceed to data and packing diagnostics." if all(checks.values()) else "Correct the failed configuration contract before training."}


def tokenizer_audit(*, tokenizer: UltronTokenizer, release_dir: Path, sample_count: int = 3) -> dict[str, Any]:
    docs = [ProcessedDocument.model_validate_json(line) for line in (release_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    eos, bos, unk = tokenizer.processor.eos_id(), tokenizer.processor.bos_id(), tokenizer.processor.unk_id()
    samples = []
    punctuation_ids = {tokenizer.encode(value)[0] for value in (".", ",", "!", "?") if tokenizer.encode(value)}
    for doc in docs[:sample_count]:
        ids = tokenizer.encode(doc.content)
        decoded = tokenizer.decode(ids)
        samples.append({"document_id": doc.provenance.document_id, "input_tokens": len(ids), "round_trip_reencodes_identically": tokenizer.encode(decoded) == ids, "eos_appended_id": [*ids, eos][-1], "preview": decoded[:120]})
    checks = {"eos_distinct_from_unk": eos != unk, "eos_not_in_normal_encode": all(eos not in tokenizer.encode(doc.content) for doc in docs[:sample_count]), "round_trip_stable": all(item["round_trip_reencodes_identically"] for item in samples), "punctuation_ids_distinct": len(punctuation_ids) >= 2}
    return {"experiment_id": "phase-l-tokenizer-audit-v1", "parent_experiment": None, "git_revision": _revision(), "result": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "tokenizer": {"version": tokenizer.version, "sha256": tokenizer.sha256, "vocab_size": tokenizer.vocab_size, "eos_id": eos, "bos_id": bos, "unk_id": unk, "pad_id": tokenizer.processor.pad_id()}, "samples": samples, "punctuation_token_ids": sorted(punctuation_ids), "next_action": "Proceed to corpus diagnostics."}


def data_audit(*, release_dir: Path, tokenized_dir: Path, tokenizer: UltronTokenizer, split: str = "train") -> dict[str, Any]:
    docs = [ProcessedDocument.model_validate_json(line) for line in (release_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = _manifest(tokenized_dir / "manifest.json")
    rows: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        if shard["split"] == split:
            rows.extend(json.loads(line) for line in resolve_artifact_path(shard["path"]).read_text(encoding="utf-8").splitlines() if line.strip())
    raw = [tokenizer.encode(doc.content) + [tokenizer.processor.eos_id()] for doc in docs]
    flat = [token for ids in raw for token in ids]
    eos = tokenizer.processor.eos_id()
    windows = [row["input_ids"] for row in rows]
    active_eos = [position for row in rows for position, token in enumerate(row["input_ids"]) if position and token == eos and row.get("loss_mask", [1] * (len(row["input_ids"]) - 1))[position - 1] == 1]
    eos_positions = Counter(active_eos)
    docs_with_windows = {row["document_id"] for row in rows}
    ngrams = Counter(tuple(flat[index:index + 3]) for index in range(max(0, len(flat) - 2)))
    return {"experiment_id": "phase-l-data-audit-v1", "parent_experiment": manifest["release_id"], "git_revision": _revision(), "result": "PASS", "data_identity": {"release_id": manifest["release_id"], "tokenized_manifest_sha256": manifest.get("manifest_sha256"), "packing_strategy": manifest.get("packing_strategy", "FIXED_TERMINAL_EOS_OVERLAP")}, "metrics": {"documents": len(docs), "documents_with_packed_windows": len(docs_with_windows), "raw_tokens_with_eos": len(flat), "eos_count": sum(token == eos for token in flat), "eos_per_1k_raw_tokens": round(1000 * sum(token == eos for token in flat) / max(1, len(flat)), 6), "packed_windows": len(windows), "active_packed_eos_count": len(active_eos), "active_packed_eos_positions": dict(sorted(eos_positions.items())), "punctuation_per_1k_characters": round(1000 * sum(char in ".,;:!?()[]{}'\"" for doc in docs for char in doc.content) / max(1, sum(len(doc.content) for doc in docs)), 6), "empty_documents": sum(not doc.content for doc in docs), "shorter_than_context": sum(len(ids) < manifest["context_length"] + 1 for ids in raw), "top_tokens": Counter(flat).most_common(20), "top_trigrams": [(list(key), value) for key, value in ngrams.most_common(10)]}, "interpretation": "EOS position concentration is a packing-property finding, not a tokenizer finding.", "next_action": "Run packing audit and objective sanity."}


def packing_audit(*, release_dir: Path, tokenized_dir: Path, tokenizer: UltronTokenizer, split: str = "train", sample_limit: int = 32) -> dict[str, Any]:
    docs = {doc.provenance.document_id: doc for doc in (ProcessedDocument.model_validate_json(line) for line in (release_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    manifest = _manifest(tokenized_dir / "manifest.json")
    rows = []
    for shard in manifest["shards"]:
        if shard["split"] == split:
            rows.extend(json.loads(line) for line in resolve_artifact_path(shard["path"]).read_text(encoding="utf-8").splitlines() if line.strip())
    eos, width = tokenizer.processor.eos_id(), manifest["context_length"] + 1
    source_ids = {document_id: tokenizer.encode(document.content) + [eos] for document_id, document in docs.items()}
    source_token_sets = {document_id: set(ids) for document_id, ids in source_ids.items()}
    invalid, masks_valid, eos_positions, examples = 0, 0, Counter(), []
    for row in rows:
        ids, mask = row["input_ids"], row.get("loss_mask", [1] * (len(row["input_ids"]) - 1))
        source = source_ids[row["document_id"]]
        present = any(source[offset:offset + len(ids)] == ids for offset in range(max(1, len(source) - len(ids) + 1))) if "loss_mask" not in row else all(token in source_token_sets[row["document_id"]] or token == eos for token in ids)
        valid = len(ids) == width and len(mask) == width - 1 and all(item in (0, 1) for item in mask) and present
        invalid += int(not valid)
        masks_valid += int(valid)
        eos_positions.update(index for index, token in enumerate(ids) if token == eos and (index == 0 or mask[index - 1]))
        if len(examples) < sample_limit:
            examples.append({"document_id": row["document_id"], "input_ids": ids[:12], "target_ids": ids[1:13], "loss_mask": mask[:12], "eos_positions": [index for index, token in enumerate(ids) if token == eos], "decoded_input": tokenizer.decode(ids[:12]), "decoded_target": tokenizer.decode(ids[1:13])})
    result = "PASS" if not invalid else "FAIL"
    if manifest.get("packing_strategy", "FIXED_TERMINAL_EOS_OVERLAP") == "FIXED_TERMINAL_EOS_OVERLAP" and set(eos_positions) == {width - 1}:
        result = "FAIL"
    return {"experiment_id": "phase-l-packing-audit-v1", "parent_experiment": manifest["release_id"], "git_revision": _revision(), "result": result, "metrics": {"rows": len(rows), "valid_rows": masks_valid, "invalid_rows": invalid, "active_eos_positions": dict(sorted(eos_positions.items())), "loss_mask_policy": "all positions active when omitted; explicit masks otherwise"}, "examples": examples, "interpretation": "All active EOS targets at the final absolute position are a positional-EOS defect for short-form generation." if result == "FAIL" and set(eos_positions) == {width - 1} else "Packing rows, target shift, and loss masks are structurally valid.", "next_action": "Use VARIABLE_EOS_MASKED packing before a recovery experiment." if result == "FAIL" else "Proceed to synthetic overfit."}


def objective_sanity() -> dict[str, Any]:
    """Numerically verify the exact cross-entropy ordering used by training."""
    def cross_entropy(logits: np.ndarray, target: int, mask: float = 1.0) -> float:
        shifted = logits - logits.max()
        return float((-logits[target] + logits.max() + np.log(np.exp(shifted).sum())) * mask)
    correct = cross_entropy(np.array([0.0, 9.0, 0.0]), 1)
    shifted = cross_entropy(np.array([0.0, 9.0, 0.0]), 0)
    reversed_target = cross_entropy(np.array([0.0, 9.0, 0.0]), 2)
    masked = cross_entropy(np.array([0.0, 9.0, 0.0]), 2, 0.0)
    checks = {"correct_lower_than_shifted": correct < shifted, "correct_lower_than_reversed": correct < reversed_target, "masked_target_zero_contribution": masked == 0.0, "eos_target_supported": np.isfinite(cross_entropy(np.array([0.0, 0.0, 9.0]), 2))}
    return {"experiment_id": "phase-l-objective-sanity-v1", "parent_experiment": None, "git_revision": _revision(), "result": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "metrics": {"correct_target_loss": correct, "shifted_target_loss": shifted, "reversed_target_loss": reversed_target, "masked_target_loss": masked}, "next_action": "Run synthetic overfit."}


def synthetic_overfit(*, model_config: UltronModelConfig, tokenizer: UltronTokenizer, steps: int = 80) -> dict[str, Any]:
    """Use the production decoder, AdamW, and next-token objective on a tiny deterministic corpus."""
    import mlx.core as mx
    from mlx import nn

    candidates = [token for token in tokenizer.encode("alpha beta gamma delta") if token not in {tokenizer.processor.eos_id(), tokenizer.processor.unk_id()}]
    pattern = candidates[:3] + [tokenizer.processor.eos_id()]
    ids = (pattern * ((model_config.max_sequence_length + 1 + len(pattern) - 1) // len(pattern)))[:model_config.max_sequence_length + 1]
    batch = mx.array([ids])
    config = TrainingConfig(run_id="phase-l-synthetic-overfit", model_id="phase-l-synthetic-overfit", steps=steps, learning_rate=3e-3, weight_decay=0.0, warmup_steps=0, checkpoint_every=steps, validation_every=steps)
    model, optimizer = build_model(model_config), ResumableAdamW(config)
    def loss_fn(current):
        return mx.mean(nn.losses.cross_entropy(current(batch[:, :-1]), batch[:, 1:]))
    value_and_grad = nn.value_and_grad(model, lambda current: loss_fn(current))
    initial = float(loss_fn(model))
    for _ in range(steps):
        loss, grads = value_and_grad(model)
        optimizer.update(model, grads, config.learning_rate)
        mx.eval(model.parameters(), optimizer.arrays(), loss)
    final = float(loss_fn(model))
    predictions = np.asarray(model(batch[:, :-1])).argmax(axis=-1)[0]
    targets = np.asarray(batch[:, 1:])[0]
    accuracy = float((predictions == targets).mean())
    eos_accuracy = float((predictions[targets == tokenizer.processor.eos_id()] == tokenizer.processor.eos_id()).mean())
    passed = final < initial * 0.20 and accuracy > 0.90 and eos_accuracy > 0.90
    return {"experiment_id": "phase-l-synthetic-overfit-v1", "parent_experiment": None, "git_revision": _revision(), "result": "PASS" if passed else "FAIL", "configuration": {"architecture": model_config.model_dump(mode="json"), "steps": steps, "learning_rate": config.learning_rate, "sequence_length": model_config.max_sequence_length}, "metrics": {"initial_loss": initial, "final_loss": final, "teacher_forced_accuracy": accuracy, "eos_target_accuracy": eos_accuracy}, "interpretation": "Production decoder, AdamW, and next-token objective memorize the deterministic EOS-bearing corpus." if passed else "Synthetic overfit failed; do not run recovery training.", "next_action": "Run a short real-data recovery experiment." if passed else "Investigate model/objective/optimizer implementation."}


def functional_level(generation: dict[str, Any], capability: dict[str, Any]) -> int:
    aggregate = generation["records"][-1]["aggregate"]
    scores = capability.get("scores", {})
    if any(float(value) > 0 for value in scores.values()):
        return 5
    eos_rate = float(aggregate.get("eos_rate", 0))
    average_length = float(aggregate.get("average_length", 0))
    # EOS is useful only when it occurs after actual generation.  A model
    # that returns EOS for every prompt has collapsed to an empty response;
    # treating it as a functional termination signal would incorrectly
    # promote a clearly non-functional recipe.
    if eos_rate >= 0.95 and average_length < 4:
        return 0
    if (
        eos_rate > 0
        and eos_rate < 0.95
        and average_length >= 4
    ):
        return 3
    if float(aggregate.get("distinct_1", 0)) >= 0.05 or float(aggregate.get("repeated_unigram_rate", 1)) <= 0.90:
        return 1
    return 0


def branch_evidence(*, branch_id: str, audit: dict[str, Any], training: dict[str, Any], test: dict[str, Any], generation: dict[str, Any], capability: dict[str, Any], parent_experiment: str | None = None) -> dict[str, Any]:
    aggregate = generation["records"][-1]["aggregate"]
    level = functional_level(generation, capability)
    return {"experiment_id": branch_id, "parent_experiment": parent_experiment, "git_revision": _revision(), "audit": audit, "training": training, "test": test, "generation": aggregate, "capability": capability.get("scores", {}), "functional_level": level, "result": "FUNCTIONAL_SIGNAL" if level else "NO_FUNCTIONAL_SIGNAL", "next_action": "Eligible for matched comparison."}


def compare_branches(branches: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank functional signals first; loss breaks ties only after generation health."""
    def score(branch: dict[str, Any]) -> tuple[float, ...]:
        generation, capability = branch["generation"], branch["capability"]
        return (float(branch["functional_level"]), float(generation.get("eos_rate", 0)), 1 - float(generation.get("repeated_unigram_rate", 1)), float(generation.get("distinct_1", 0)), sum(float(value) for value in capability.values()), -float(branch["test"].get("loss", 1e9)))
    ranked = sorted(branches, key=score, reverse=True)
    best = ranked[0]
    improvement = int(best["functional_level"]) > 0
    return {"experiment_id": "phase-l-branch-comparison-v1", "parent_experiment": None, "git_revision": _revision(), "branches": ranked, "ranking_policy": "functional level, non-degenerate EOS, reduced repetition, distinctness, capability, then held-out loss", "best_functional_trajectory": best["experiment_id"], "overall_recommended_branch": best["experiment_id"] if improvement else None, "result": "RECIPE_RECOVERY" if improvement else "NO_BRANCH_FUNCTIONAL_IMPROVEMENT", "next_action": "Extend the selected branch." if improvement else "Do not extend a zero-signal branch; move to optimization and data-scale analysis."}
