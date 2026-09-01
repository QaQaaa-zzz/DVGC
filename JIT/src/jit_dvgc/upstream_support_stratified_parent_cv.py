"""TRAIN-only support-stratified unseen-parent diagnostic for C_up^0.

This diagnostic is the method-correction gate after the matched tiny-MLP
leave-one-entire-checkpoint stress test failed.  The checkpoint-domain holdout
is retained as an extrapolation stress result, but it is not the original JIT
requirement: the original validation design holds out parent groups while the
declared support spans the checkpoint families.

Each fold therefore holds out the same seed family across all three checkpoint
domains while keeping the other four seed families from all three domains in
TRAIN.  The model remains the single predeclared 76->8 tanh->1 tiny MLP.  No
consumed validation rows or predictions are read.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .iteration_train_evidence import canonical_sha256
from .policy_conditioned_continuation_field import (
    _cell_balanced_weights,
    _dataset,
    _metrics,
    _normalization,
    _transform,
)
from .upstream_checkpoint_domain_cv import _validate_rows
from .upstream_checkpoint_train_evidence import load_frozen_upstream_checkpoint_train_evidence
from .upstream_matched_checkpoint_domain_cv import _sigmoid, _tiny_mlp_parameter_count

CONFIG_SCHEMA = "jit_upstream_support_stratified_parent_cv_config_v1"
PROTOCOL_SCHEMA = "jit_upstream_support_stratified_parent_cv_protocol_v1"
SUMMARY_SCHEMA = "jit_upstream_support_stratified_parent_cv_summary_v1"
STATUS = "predeclared_after_tiny_mlp_checkpoint_extrapolation_stress_fail_before_support_parent_cv"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(value: Any, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be SHA-256")
    return text


def _seed_family(parent_group_id: str) -> str:
    text = str(parent_group_id)
    if "__" not in text:
        raise ValueError("parent group lacks seed-family separator")
    domain, seed = text.rsplit("__", 1)
    if not domain.startswith("transition_") or not seed.isdigit():
        raise ValueError("parent group seed-family format drift")
    return seed


def load_support_stratified_parent_cv_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported support-stratified upstream CV config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("support-stratified upstream CV protocol drift")
    if protocol.get("status") != STATUS:
        raise ValueError("support-stratified upstream CV status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("support-stratified upstream CV policy drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "frozen_upstream_train_manifest_sha256",
        "frozen_upstream_train_freeze_protocol_sha256",
        "prior_tiny_mlp_checkpoint_stress_summary_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if protocol.get("expected_combined") != {
        "candidate_count": 720,
        "positive_count": 639,
        "negative_count": 81,
        "parent_group_count": 15,
        "checkpoint_domain_count": 3,
        "seed_family_count": 5,
        "rows_per_parent_group": 48,
    }:
        raise ValueError("support-stratified expected counts drift")
    if protocol.get("required_domains") != [
        "transition_4988928",
        "transition_7987200",
        "transition_9977856",
    ]:
        raise ValueError("support-stratified checkpoint support drift")
    if protocol.get("required_seed_families") != [
        "1000001", "1000002", "1000003", "1000004", "1000005"
    ]:
        raise ValueError("support-stratified seed-family contract drift")
    if protocol.get("fold_design") != {
        "kind": "leave_one_seed_family_out_across_all_checkpoint_domains",
        "fold_count": 5,
        "heldout_parent_groups_per_fold": 3,
        "heldout_rows_per_fold": 144,
        "train_parent_groups_per_fold": 12,
        "train_rows_per_fold": 576,
        "checkpoint_domains_present_in_train_every_fold": True,
        "checkpoint_domains_present_in_heldout_every_fold": True,
        "physical_parent_groups_disjoint": True,
    }:
        raise ValueError("support-stratified fold design drift")
    if protocol.get("model") != {
        "family": "tiny_mlp_tanh",
        "input": "unified_actor_observation",
        "observation_size": 76,
        "hidden_units": 8,
        "activation": "tanh",
        "parameter_count": 625,
        "normalization": "fold_train_only_zscore_clip10",
        "sample_weighting": "equal_parent_label_cell_mass",
        "l2_weight": 0.01,
        "optimizer": "adam_full_batch_fixed_schedule",
        "steps": 4000,
        "learning_rate": 0.01,
        "seed_base": 848000,
    }:
        raise ValueError("support-stratified tiny-MLP contract drift")
    if _tiny_mlp_parameter_count(76, 8) != 625:
        raise ValueError("support-stratified tiny-MLP parameter-count drift")
    if protocol.get("diagnostic_gate") != {
        "minimum_pooled_oof_roc_auc": 0.70,
        "minimum_worst_domain_oof_roc_auc": 0.60,
        "require_positive_score_gap_in_every_domain": True,
        "minimum_mean_fold_roc_auc": 0.70,
        "minimum_worst_fold_roc_auc": 0.60,
        "require_positive_score_gap_in_every_fold": True,
    }:
        raise ValueError("support-stratified gate drift")
    if protocol.get("method_decision") != {
        "checkpoint_domain_holdout_reclassified_as_out_of_support_extrapolation_stress_test": True,
        "original_jit_requirement_is_group_disjoint_generalization_within_declared_support": True,
        "no_additional_model_family_search": True,
        "no_consumed_validation_reuse": True,
        "if_pass_next_gate_is_same_architecture_downstream_train_check": True,
        "if_fail_stop_tiny_mlp_and_make_explicit_continuation_method_decision": True,
    }:
        raise ValueError("support-stratified method-decision drift")
    if protocol.get("data_policy") != {
        "train_rows_only": True,
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "prior_checkpoint_stress_summary_read_for_provenance_only": True,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }:
        raise ValueError("support-stratified data-policy drift")
    if protocol.get("claim_boundary") != {
        "diagnostic_only": True,
        "continuation_field_reselected": False,
        "unified_continuation_architecture_frozen": False,
        "fresh_validation_bank_predeclared": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("support-stratified claim boundary drift")
    for field in ("frozen_upstream_train_evidence", "prior_tiny_mlp_checkpoint_stress_summary"):
        if not str(protocol.get(field, "")):
            raise ValueError(f"support-stratified {field} missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("support-stratified output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("support-stratified protocol SHA drift")
    return config


def _validate_prior_checkpoint_stress(protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(protocol["prior_tiny_mlp_checkpoint_stress_summary"]))
    summary = _read_object(path)
    declared = _sha(summary.get("summary_sha256"), field="prior tiny-MLP stress summary")
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if canonical_sha256(payload) != declared:
        raise ValueError("prior tiny-MLP stress summary self-hash drift")
    if declared != protocol["prior_tiny_mlp_checkpoint_stress_summary_sha256"]:
        raise ValueError("prior tiny-MLP stress summary identity drift")
    if summary.get("status") != "completed" or summary.get("model_family") != "tiny_mlp_tanh":
        raise ValueError("prior tiny-MLP stress result drift")
    if summary.get("checkpoint_domain_generalization_supported") is not False:
        raise ValueError("support-stratified correction requires failed checkpoint stress test")
    if summary.get("consumed_validation_rows_reused") is not False:
        raise ValueError("prior tiny-MLP stress reused consumed validation rows")
    if summary.get("consumed_validation_predictions_reused") is not False:
        raise ValueError("prior tiny-MLP stress reused consumed validation predictions")
    if summary.get("tube_1_authorized") is not False:
        raise ValueError("prior tiny-MLP stress unexpectedly authorized Tube_1")
    return {
        "summary_sha256": declared,
        "mean_checkpoint_stress_auc": float(summary["mean_logo_roc_auc"]),
        "worst_checkpoint_stress_auc": float(summary["worst_logo_roc_auc"]),
    }


def _fit_fold(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    heldout_seed_family: str,
    domains: Sequence[str],
    model_cfg: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_x, train_y, train_groups, _ = _dataset(train_rows)
    heldout_x, heldout_y, heldout_groups, heldout_states = _dataset(heldout_rows)
    if set(np.unique(train_y).tolist()) != {0.0, 1.0}:
        raise ValueError("support-stratified fold TRAIN lacks both labels")
    if set(np.unique(heldout_y).tolist()) != {0.0, 1.0}:
        raise ValueError("support-stratified held-out fold lacks both labels")
    if {_seed_family(group) for group in heldout_groups} != {heldout_seed_family}:
        raise ValueError("support-stratified held-out fold mixes seed families")
    if heldout_seed_family in {_seed_family(group) for group in train_groups}:
        raise ValueError("support-stratified fold leaks held-out seed family")
    heldout_domains = sorted({str(group).split("__", 1)[0] for group in heldout_groups})
    train_domains = sorted({str(group).split("__", 1)[0] for group in train_groups})
    if heldout_domains != list(domains) or train_domains != list(domains):
        raise ValueError("support-stratified fold does not preserve checkpoint support")
    if len(set(heldout_groups)) != 3 or len(heldout_rows) != 144:
        raise ValueError("support-stratified held-out fold size drift")
    if len(set(train_groups)) != 12 or len(train_rows) != 576:
        raise ValueError("support-stratified TRAIN fold size drift")

    sample_weights = _cell_balanced_weights(train_groups, train_y)
    mean, std = _normalization(train_x)
    x_train = _transform(train_x, mean, std)
    x_heldout = _transform(heldout_x, mean, std)
    jx = jnp.asarray(x_train)
    jy = jnp.asarray(train_y)
    jw = jnp.asarray(sample_weights)

    input_size = int(model_cfg["observation_size"])
    hidden_units = int(model_cfg["hidden_units"])
    if _tiny_mlp_parameter_count(input_size, hidden_units) != int(model_cfg["parameter_count"]):
        raise ValueError("support-stratified tiny-MLP parameter-count drift")
    key1, key2 = jax.random.split(jax.random.PRNGKey(int(seed)))
    scale1 = math.sqrt(2.0 / float(input_size + hidden_units))
    scale2 = math.sqrt(2.0 / float(hidden_units + 1))
    params = {
        "w1": jax.random.normal(key1, (input_size, hidden_units), dtype=jnp.float32) * scale1,
        "b1": jnp.zeros((hidden_units,), dtype=jnp.float32),
        "w2": jax.random.normal(key2, (hidden_units,), dtype=jnp.float32) * scale2,
        "b2": jnp.asarray(0.0, dtype=jnp.float32),
    }
    l2 = float(model_cfg["l2_weight"])
    optimizer = optax.adam(float(model_cfg["learning_rate"]))
    opt_state = optimizer.init(params)

    def logits(current, x):
        hidden = jnp.tanh(x @ current["w1"] + current["b1"])
        return hidden @ current["w2"] + current["b2"]

    def loss_fn(current):
        raw = logits(current, jx)
        bce = optax.sigmoid_binary_cross_entropy(raw, jy)
        data_loss = jnp.sum(jw * bce) / jnp.sum(jw)
        penalty = 0.5 * l2 * (
            jnp.sum(jnp.square(current["w1"])) + jnp.sum(jnp.square(current["w2"]))
        )
        return data_loss + penalty

    @jax.jit
    def step(current, state):
        loss, grads = jax.value_and_grad(loss_fn)(current)
        updates, next_state = optimizer.update(grads, state, current)
        return optax.apply_updates(current, updates), next_state, loss

    initial_loss = float(jax.device_get(loss_fn(params)))
    final_loss = initial_loss
    for _ in range(int(model_cfg["steps"])):
        params, opt_state, loss = step(params, opt_state)
        final_loss = float(jax.device_get(loss))
    if not math.isfinite(final_loss):
        raise ValueError("support-stratified optimizer became nonfinite")

    train_logits = np.asarray(jax.device_get(logits(params, jnp.asarray(x_train))), dtype=np.float64)
    heldout_logits = np.asarray(jax.device_get(logits(params, jnp.asarray(x_heldout))), dtype=np.float64)
    train_score = _sigmoid(train_logits)
    heldout_score = _sigmoid(heldout_logits)

    predictions: list[dict[str, Any]] = []
    for row, state, group, label, score in zip(
        heldout_rows, heldout_states, heldout_groups, heldout_y, heldout_score
    ):
        predictions.append({
            "state_sha256": str(state),
            "parent_group_id": str(group),
            "parent_domain_id": str(row["parent_domain_id"]),
            "seed_family": heldout_seed_family,
            "label": int(label),
            "score": float(score),
        })
    fold = {
        "heldout_seed_family": heldout_seed_family,
        "train_seed_families": sorted({_seed_family(group) for group in train_groups}),
        "train_checkpoint_domains": train_domains,
        "heldout_checkpoint_domains": heldout_domains,
        "train_parent_group_count": len(set(train_groups)),
        "heldout_parent_group_count": len(set(heldout_groups)),
        "train_metrics": _metrics(train_y, train_score),
        "heldout_metrics": _metrics(heldout_y, heldout_score),
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "seed": int(seed),
        "model_family": "tiny_mlp_tanh",
        "parameter_count": int(model_cfg["parameter_count"]),
    }
    return fold, predictions


def _classify(
    folds: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    domains: Sequence[str],
    gate_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if len(folds) != 5 or len(predictions) != 720:
        raise ValueError("support-stratified OOF coverage drift")
    states = [str(row["state_sha256"]) for row in predictions]
    if len(states) != len(set(states)):
        raise ValueError("support-stratified OOF repeats physical state")
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.float64)
    scores = np.asarray([float(row["score"]) for row in predictions], dtype=np.float64)
    pooled = _metrics(labels, scores)
    domain_metrics: dict[str, Any] = {}
    for domain in domains:
        selected = [row for row in predictions if row["parent_domain_id"] == domain]
        if len(selected) != 240:
            raise ValueError("support-stratified domain OOF coverage drift")
        domain_metrics[domain] = _metrics(
            np.asarray([int(row["label"]) for row in selected], dtype=np.float64),
            np.asarray([float(row["score"]) for row in selected], dtype=np.float64),
        )
    fold_aucs = [float(fold["heldout_metrics"]["roc_auc"]) for fold in folds]
    fold_gaps = [float(fold["heldout_metrics"]["score_gap"]) for fold in folds]
    domain_aucs = [float(domain_metrics[domain]["roc_auc"]) for domain in domains]
    domain_gaps = [float(domain_metrics[domain]["score_gap"]) for domain in domains]
    gate = {
        "pooled_oof_roc_auc_at_least_minimum": bool(
            float(pooled["roc_auc"]) >= float(gate_cfg["minimum_pooled_oof_roc_auc"])
        ),
        "worst_domain_oof_roc_auc_at_least_minimum": bool(
            min(domain_aucs) >= float(gate_cfg["minimum_worst_domain_oof_roc_auc"])
        ),
        "positive_score_gap_in_every_domain": bool(all(gap > 0.0 for gap in domain_gaps)),
        "mean_fold_roc_auc_at_least_minimum": bool(
            float(np.mean(fold_aucs)) >= float(gate_cfg["minimum_mean_fold_roc_auc"])
        ),
        "worst_fold_roc_auc_at_least_minimum": bool(
            min(fold_aucs) >= float(gate_cfg["minimum_worst_fold_roc_auc"])
        ),
        "positive_score_gap_in_every_fold": bool(all(gap > 0.0 for gap in fold_gaps)),
    }
    return {
        "pooled_oof_metrics": pooled,
        "domain_oof_metrics": domain_metrics,
        "mean_fold_roc_auc": float(np.mean(fold_aucs)),
        "worst_fold_roc_auc": float(np.min(fold_aucs)),
        "mean_fold_score_gap": float(np.mean(fold_gaps)),
        "fold_roc_auc": fold_aucs,
        "fold_score_gap": fold_gaps,
        "worst_domain_oof_roc_auc": float(np.min(domain_aucs)),
        "gate": gate,
        "support_stratified_parent_generalization_supported": bool(all(gate.values())),
    }


def run_support_stratified_parent_cv(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_support_stratified_parent_cv_config(config_path)
    protocol = dict(config["protocol"])
    prior = _validate_prior_checkpoint_stress(protocol)
    manifest, raw_rows = load_frozen_upstream_checkpoint_train_evidence(
        Path(str(protocol["frozen_upstream_train_evidence"]))
    )
    if manifest.get("manifest_sha256") != protocol["frozen_upstream_train_manifest_sha256"]:
        raise ValueError("support-stratified frozen TRAIN manifest drift")
    if manifest.get("freeze_protocol_sha256") != protocol["frozen_upstream_train_freeze_protocol_sha256"]:
        raise ValueError("support-stratified frozen TRAIN protocol drift")
    if manifest.get("matched_panel") is not True:
        raise ValueError("support-stratified CV requires matched-panel evidence")
    rows = _validate_rows(raw_rows, protocol)
    domains = list(protocol["required_domains"])
    seed_families = list(protocol["required_seed_families"])
    group_counts: dict[str, int] = {}
    for row in rows:
        group = str(row["parent_group_id"])
        group_counts[group] = group_counts.get(group, 0) + 1
    if set(group_counts.values()) != {48}:
        raise ValueError("support-stratified parent rows-per-group drift")
    observed_seed_families = sorted({_seed_family(row["parent_group_id"]) for row in rows})
    if observed_seed_families != seed_families:
        raise ValueError("support-stratified observed seed-family drift")

    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for index, heldout_seed in enumerate(seed_families):
        heldout = [row for row in rows if _seed_family(row["parent_group_id"]) == heldout_seed]
        train = [row for row in rows if _seed_family(row["parent_group_id"]) != heldout_seed]
        fold, fold_predictions = _fit_fold(
            train,
            heldout,
            heldout_seed_family=heldout_seed,
            domains=domains,
            model_cfg=protocol["model"],
            seed=int(protocol["model"]["seed_base"]) + index,
        )
        folds.append(fold)
        predictions.extend(fold_predictions)

    classification = _classify(folds, predictions, domains, protocol["diagnostic_gate"])
    supported = bool(classification["support_stratified_parent_generalization_supported"])
    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "oof_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    diagnosis = (
        "tiny_mlp_generalizes_to_unseen_parent_groups_within_declared_three_checkpoint_support"
        if supported
        else "tiny_mlp_does_not_generalize_to_unseen_parent_groups_within_declared_three_checkpoint_support"
    )
    next_gate = (
        "freeze 76->8 tanh->1 as the candidate shared continuation-field architecture and run the same architecture "
        "on downstream TRAIN-only group-disjoint diagnostics before any fresh validation is predeclared"
        if supported
        else "stop tiny-MLP repair; do not escalate model size or reuse consumed validation; make an explicit method "
        "decision about continuation representation or Tube construction"
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "train_only_upstream_support_stratified_unseen_parent_generalization_diagnostic",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "protocol_sha256": str(config["expected_protocol_sha256"]),
        "frozen_upstream_train_manifest_sha256": str(manifest["manifest_sha256"]),
        "candidate_count": len(rows),
        "positive_count": sum(int(row["label"]) for row in rows),
        "negative_count": sum(1 - int(row["label"]) for row in rows),
        "parent_group_count": len({str(row["parent_group_id"]) for row in rows}),
        "checkpoint_domains": domains,
        "seed_families": seed_families,
        "fold_count": len(folds),
        "folds": folds,
        **classification,
        "prior_checkpoint_extrapolation_stress": prior,
        "method_decision": dict(protocol["method_decision"]),
        "diagnosis": diagnosis,
        "environment_interactions": 0,
        "training_transitions": 0,
        "supervised_optimizer_steps": len(folds) * int(protocol["model"]["steps"]),
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "shared_up_down_architecture_candidate_authorized": supported,
        "fresh_validation_predeclaration_authorized": False,
        "tube_1_authorized": False,
        "next_scientific_gate": next_gate,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
