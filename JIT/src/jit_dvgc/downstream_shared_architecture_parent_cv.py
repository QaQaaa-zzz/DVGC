"""TRAIN-only downstream parent-group CV for the shared continuation architecture.

The upstream within-support gate authorized 76->8 tanh->1 as the candidate
shared C_up/C_down architecture.  This diagnostic applies that exact network
family and optimization contract to the frozen downstream TRAIN evidence.  It
uses leave-one-parent-group-out folds and never reads consumed validation rows
or predictions.
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

from .iteration_train_evidence import (
    canonical_sha256,
    load_frozen_iteration_train_evidence,
)
from .policy_conditioned_continuation_field import (
    _cell_balanced_weights,
    _dataset,
    _metrics,
    _normalization,
    _transform,
)
from .upstream_matched_checkpoint_domain_cv import _sigmoid, _tiny_mlp_parameter_count

CONFIG_SCHEMA = "jit_downstream_shared_architecture_parent_cv_config_v1"
PROTOCOL_SCHEMA = "jit_downstream_shared_architecture_parent_cv_protocol_v1"
SUMMARY_SCHEMA = "jit_downstream_shared_architecture_parent_cv_summary_v1"
STATUS = "predeclared_after_upstream_within_support_tiny_mlp_pass_before_downstream_parent_cv"

_EXPECTED_MODEL = {
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
    "seed_base": 849000,
}


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


def load_downstream_shared_architecture_parent_cv_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported downstream shared-architecture CV config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("downstream shared-architecture CV protocol drift")
    if protocol.get("status") != STATUS:
        raise ValueError("downstream shared-architecture CV status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("downstream shared-architecture CV policy drift")
    if protocol.get("phase") != "downstream":
        raise ValueError("downstream shared-architecture CV phase drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "frozen_train_manifest_sha256",
        "prior_upstream_support_summary_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if protocol.get("expected_downstream_train") != {
        "candidate_count": 2619,
        "positive_count": 2589,
        "negative_count": 30,
        "parent_group_count": 5,
    }:
        raise ValueError("downstream TRAIN count contract drift")
    if protocol.get("fold_design") != {
        "kind": "leave_one_parent_group_out",
        "fold_count": 5,
        "heldout_parent_groups_per_fold": 1,
        "train_parent_groups_per_fold": 4,
        "physical_parent_groups_disjoint": True,
        "every_row_heldout_exactly_once": True,
    }:
        raise ValueError("downstream fold design drift")
    if protocol.get("model") != _EXPECTED_MODEL:
        raise ValueError("downstream shared tiny-MLP contract drift")
    if _tiny_mlp_parameter_count(76, 8) != 625:
        raise ValueError("downstream tiny-MLP parameter-count drift")
    if protocol.get("diagnostic_gate") != {
        "minimum_pooled_oof_roc_auc": 0.70,
        "minimum_mean_fold_roc_auc": 0.70,
        "minimum_worst_fold_roc_auc": 0.60,
        "require_positive_score_gap_in_every_fold": True,
    }:
        raise ValueError("downstream diagnostic gate drift")
    if protocol.get("method_decision") != {
        "upstream_within_support_gate_passed": True,
        "shared_up_down_architecture_required": True,
        "architecture": "76->8_tanh->1",
        "phase_specific_weights_required": True,
        "phase_specific_thresholds_allowed": True,
        "no_downstream_architecture_search": True,
        "no_consumed_validation_reuse": True,
        "if_pass_freeze_shared_architecture_before_fresh_validation": True,
        "if_fail_do_not_freeze_shared_architecture": True,
    }:
        raise ValueError("downstream shared-architecture method decision drift")
    if protocol.get("data_policy") != {
        "train_rows_only": True,
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "prior_upstream_summary_read_for_provenance_only": True,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }:
        raise ValueError("downstream data policy drift")
    if protocol.get("claim_boundary") != {
        "diagnostic_only": True,
        "shared_continuation_architecture_frozen": False,
        "continuation_fields_refit": False,
        "fresh_validation_bank_predeclared": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("downstream claim boundary drift")
    for field in ("frozen_train_evidence", "prior_upstream_support_summary"):
        if not str(protocol.get(field, "")):
            raise ValueError(f"downstream shared-architecture {field} missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("downstream shared-architecture output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("downstream shared-architecture protocol SHA drift")
    return config


def _validate_prior_upstream_pass(protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(protocol["prior_upstream_support_summary"]))
    summary = _read_object(path)
    declared = _sha(summary.get("summary_sha256"), field="prior upstream support summary")
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if canonical_sha256(payload) != declared:
        raise ValueError("prior upstream support summary self-hash drift")
    if declared != protocol["prior_upstream_support_summary_sha256"]:
        raise ValueError("prior upstream support summary identity drift")
    if summary.get("status") != "completed":
        raise ValueError("prior upstream support diagnostic not completed")
    if summary.get("support_stratified_parent_generalization_supported") is not True:
        raise ValueError("downstream migration requires upstream within-support PASS")
    if summary.get("shared_up_down_architecture_candidate_authorized") is not True:
        raise ValueError("upstream did not authorize shared architecture candidate")
    if summary.get("consumed_validation_rows_reused") is not False:
        raise ValueError("prior upstream diagnostic reused consumed validation rows")
    if summary.get("consumed_validation_predictions_reused") is not False:
        raise ValueError("prior upstream diagnostic reused consumed validation predictions")
    if summary.get("tube_1_authorized") is not False:
        raise ValueError("prior upstream diagnostic unexpectedly authorized Tube_1")
    folds = summary.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("prior upstream fold evidence drift")
    for fold in folds:
        if fold.get("model_family") != "tiny_mlp_tanh" or int(fold.get("parameter_count", -1)) != 625:
            raise ValueError("prior upstream shared architecture drift")
    return {
        "summary_sha256": declared,
        "pooled_oof_roc_auc": float(summary["pooled_oof_metrics"]["roc_auc"]),
        "worst_fold_roc_auc": float(summary["worst_fold_roc_auc"]),
    }


def _validate_downstream_rows(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    actor = str(protocol["policy_actor_sha256"])
    payload = str(protocol["policy_payload_sha256"])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        if row.get("phase") != "downstream":
            continue
        if row.get("split") != "train":
            raise ValueError("downstream diagnostic accepts TRAIN rows only")
        if row.get("policy_actor_sha256") != actor or row.get("policy_payload_sha256") != payload:
            raise ValueError("downstream TRAIN policy identity drift")
        state = _sha(row.get("state_sha256"), field="downstream TRAIN state_sha256")
        if state in seen:
            raise ValueError("downstream TRAIN repeats physical state")
        seen.add(state)
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError("downstream TRAIN label must be binary")
        obs = np.asarray(row.get("actor_observation"), dtype=np.float32).reshape(-1)
        if obs.shape != (76,) or not np.isfinite(obs).all():
            raise ValueError("downstream TRAIN observation must be finite 76-D")
        if not str(row.get("parent_group_id", "")):
            raise ValueError("downstream TRAIN parent group missing")
        selected.append(row)

    expected = protocol["expected_downstream_train"]
    groups = sorted({str(row["parent_group_id"]) for row in selected})
    positive = sum(int(row["label"]) for row in selected)
    if (
        len(selected) != int(expected["candidate_count"])
        or positive != int(expected["positive_count"])
        or len(selected) - positive != int(expected["negative_count"])
        or len(groups) != int(expected["parent_group_count"])
    ):
        raise ValueError("downstream TRAIN evidence count drift")
    for group in groups:
        labels = {int(row["label"]) for row in selected if str(row["parent_group_id"]) == group}
        if labels != {0, 1}:
            raise ValueError("every downstream TRAIN parent group must contain both labels")
    return tuple(selected)


def _fit_fold(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    heldout_group: str,
    model_cfg: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_x, train_y, train_groups, _ = _dataset(train_rows)
    heldout_x, heldout_y, heldout_groups, heldout_states = _dataset(heldout_rows)
    if set(np.unique(train_y).tolist()) != {0.0, 1.0}:
        raise ValueError("downstream fold TRAIN lacks both labels")
    if set(np.unique(heldout_y).tolist()) != {0.0, 1.0}:
        raise ValueError("downstream held-out fold lacks both labels")
    if set(heldout_groups) != {heldout_group}:
        raise ValueError("downstream held-out fold mixes parent groups")
    if heldout_group in set(train_groups):
        raise ValueError("downstream fold leaks held-out parent group")
    if len(set(train_groups)) != 4:
        raise ValueError("downstream fold TRAIN group count drift")

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
        raise ValueError("downstream tiny-MLP parameter-count drift")
    key1, key2 = jax.random.split(jax.random.PRNGKey(int(seed)))
    scale1 = math.sqrt(2.0 / float(input_size + hidden_units))
    scale2 = math.sqrt(2.0 / float(hidden_units + 1))
    params = {
        "w1": jax.random.normal(key1, (input_size, hidden_units), dtype=jnp.float32) * scale1,
        "b1": jnp.zeros((hidden_units,), dtype=jnp.float32),
        "w2": jax.random.normal(key2, (hidden_units,), dtype=jnp.float32) * scale2,
        "b2": jnp.asarray(0.0, dtype=jnp.float32),
    }
    optimizer = optax.adam(float(model_cfg["learning_rate"]))
    opt_state = optimizer.init(params)
    l2 = float(model_cfg["l2_weight"])

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
        raise ValueError("downstream tiny-MLP optimizer became nonfinite")

    train_logits = np.asarray(jax.device_get(logits(params, jnp.asarray(x_train))), dtype=np.float64)
    heldout_logits = np.asarray(jax.device_get(logits(params, jnp.asarray(x_heldout))), dtype=np.float64)
    train_score = _sigmoid(train_logits)
    heldout_score = _sigmoid(heldout_logits)

    predictions: list[dict[str, Any]] = []
    for state, group, label, score in zip(heldout_states, heldout_groups, heldout_y, heldout_score):
        predictions.append({
            "state_sha256": str(state),
            "parent_group_id": str(group),
            "label": int(label),
            "score": float(score),
        })
    fold = {
        "heldout_parent_group_id": heldout_group,
        "train_parent_group_count": len(set(train_groups)),
        "heldout_parent_group_count": 1,
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
    gate_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if len(folds) != 5 or len(predictions) != 2619:
        raise ValueError("downstream OOF coverage drift")
    states = [str(row["state_sha256"]) for row in predictions]
    if len(states) != len(set(states)):
        raise ValueError("downstream OOF repeats physical state")
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.float64)
    scores = np.asarray([float(row["score"]) for row in predictions], dtype=np.float64)
    pooled = _metrics(labels, scores)
    aucs = [float(fold["heldout_metrics"]["roc_auc"]) for fold in folds]
    gaps = [float(fold["heldout_metrics"]["score_gap"]) for fold in folds]
    gate = {
        "pooled_oof_roc_auc_at_least_minimum": bool(
            float(pooled["roc_auc"]) >= float(gate_cfg["minimum_pooled_oof_roc_auc"])
        ),
        "mean_fold_roc_auc_at_least_minimum": bool(
            float(np.mean(aucs)) >= float(gate_cfg["minimum_mean_fold_roc_auc"])
        ),
        "worst_fold_roc_auc_at_least_minimum": bool(
            float(np.min(aucs)) >= float(gate_cfg["minimum_worst_fold_roc_auc"])
        ),
        "positive_score_gap_in_every_fold": bool(all(gap > 0.0 for gap in gaps)),
    }
    return {
        "pooled_oof_metrics": pooled,
        "mean_fold_roc_auc": float(np.mean(aucs)),
        "worst_fold_roc_auc": float(np.min(aucs)),
        "mean_fold_score_gap": float(np.mean(gaps)),
        "fold_roc_auc": aucs,
        "fold_score_gap": gaps,
        "gate": gate,
        "downstream_parent_generalization_supported": bool(all(gate.values())),
    }


def run_downstream_shared_architecture_parent_cv(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_downstream_shared_architecture_parent_cv_config(config_path)
    protocol = dict(config["protocol"])
    prior = _validate_prior_upstream_pass(protocol)
    manifest, raw_rows = load_frozen_iteration_train_evidence(
        Path(str(protocol["frozen_train_evidence"]))
    )
    if manifest.get("manifest_sha256") != protocol["frozen_train_manifest_sha256"]:
        raise ValueError("downstream frozen TRAIN manifest drift")
    if manifest.get("policy_actor_sha256") != protocol["policy_actor_sha256"]:
        raise ValueError("downstream frozen TRAIN actor drift")
    if manifest.get("policy_payload_sha256") != protocol["policy_payload_sha256"]:
        raise ValueError("downstream frozen TRAIN payload drift")

    rows = _validate_downstream_rows(raw_rows, protocol)
    groups = sorted({str(row["parent_group_id"]) for row in rows})
    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for index, heldout_group in enumerate(groups):
        heldout = [row for row in rows if str(row["parent_group_id"]) == heldout_group]
        train = [row for row in rows if str(row["parent_group_id"]) != heldout_group]
        fold, fold_predictions = _fit_fold(
            train,
            heldout,
            heldout_group=heldout_group,
            model_cfg=protocol["model"],
            seed=int(protocol["model"]["seed_base"]) + index,
        )
        folds.append(fold)
        predictions.extend(fold_predictions)

    classification = _classify(folds, predictions, protocol["diagnostic_gate"])
    supported = bool(classification["downstream_parent_generalization_supported"])
    diagnosis = (
        "shared_tiny_mlp_generalizes_to_unseen_downstream_parent_groups"
        if supported
        else "shared_tiny_mlp_does_not_generalize_to_unseen_downstream_parent_groups"
    )
    next_gate = (
        "freeze 76->8 tanh->1 as the shared C_up/C_down architecture, refit phase-specific fields on their full TRAIN evidence, and only then predeclare fresh independent validation"
        if supported
        else
        "do not freeze a shared continuation architecture and do not open fresh validation; make an explicit continuation-method decision without architecture search"
    )

    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "train_only_downstream_shared_architecture_unseen_parent_generalization_diagnostic",
        "iteration": 0,
        "policy_name": "pi_0",
        "phase": "downstream",
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "protocol_sha256": str(config["expected_protocol_sha256"]),
        "frozen_train_manifest_sha256": str(manifest["manifest_sha256"]),
        "prior_upstream_support": prior,
        "candidate_count": len(rows),
        "positive_count": sum(int(row["label"]) for row in rows),
        "negative_count": sum(1 - int(row["label"]) for row in rows),
        "parent_group_count": len(groups),
        "parent_groups": groups,
        "fold_count": len(folds),
        "folds": folds,
        **classification,
        "model_family": "tiny_mlp_tanh",
        "parameter_count": 625,
        "architecture": "76->8_tanh->1",
        "method_decision": dict(protocol["method_decision"]),
        "diagnosis": diagnosis,
        "environment_interactions": 0,
        "training_transitions": 0,
        "supervised_optimizer_steps": len(folds) * int(protocol["model"]["steps"]),
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "shared_continuation_architecture_authorized": supported,
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
