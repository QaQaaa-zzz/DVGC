"""TRAIN-only leave-one-parent-group-out diagnostic for failed C_up^0.

This diagnostic intentionally reuses the exact upstream v1 model family and
optimization contract. It never reads row-level validation evidence or
validation predictions. The already-consumed C0 summary is read only to bind
this diagnostic to the exact failed gate that triggered it.
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
    _score,
    _transform,
)

CONFIG_SCHEMA = "jit_upstream_train_logo_diagnostic_config_v1"
PROTOCOL_SCHEMA = "jit_upstream_train_logo_diagnostic_protocol_v1"
SUMMARY_SCHEMA = "jit_upstream_train_logo_diagnostic_summary_v1"


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


def load_upstream_train_logo_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported upstream TRAIN LOGO diagnostic config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("upstream TRAIN LOGO protocol is required")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("upstream TRAIN LOGO protocol schema drift")
    if protocol.get("status") != "predeclared_after_c0_gate_fail_before_train_diagnostic":
        raise ValueError("upstream TRAIN LOGO status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("upstream TRAIN LOGO policy identity drift")

    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "frozen_train_manifest_sha256",
        "prior_c0_summary_sha256",
        "prior_upstream_manifest_sha256",
        "prior_downstream_manifest_sha256",
    ):
        _sha(protocol.get(field), field=field)

    expected_train = protocol.get("expected_upstream_train")
    if expected_train != {
        "candidate_count": 571,
        "positive_count": 545,
        "negative_count": 26,
        "parent_group_count": 5,
    }:
        raise ValueError("upstream TRAIN LOGO expected count contract drift")

    expected_model = {
        "family": "linear_logistic",
        "input": "unified_actor_observation",
        "observation_size": 76,
        "normalization": "fold_train_only_zscore_clip10",
        "sample_weighting": "equal_parent_label_cell_mass",
        "l2_weight": 0.01,
        "optimizer": "adam_full_batch_fixed_schedule",
        "steps": 4000,
        "learning_rate": 0.01,
        "seed_base": 845000,
    }
    if protocol.get("model") != expected_model:
        raise ValueError("upstream TRAIN LOGO model contract drift")

    expected_gate = {
        "minimum_mean_logo_roc_auc": 0.70,
        "minimum_worst_logo_roc_auc": 0.55,
        "require_positive_score_gap_in_every_fold": True,
    }
    if protocol.get("diagnostic_gate") != expected_gate:
        raise ValueError("upstream TRAIN LOGO gate drift")

    expected_data_policy = {
        "train_rows_only": True,
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "prior_c0_gate_summary_read_for_provenance_only": True,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    if protocol.get("data_policy") != expected_data_policy:
        raise ValueError("upstream TRAIN LOGO data policy drift")

    expected_claims = {
        "diagnostic_only": True,
        "continuation_field_reselected": False,
        "fresh_validation_authorized": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }
    if protocol.get("claim_boundary") != expected_claims:
        raise ValueError("upstream TRAIN LOGO claim boundary drift")

    for field in ("frozen_train_evidence", "prior_c0_root"):
        if not str(protocol.get(field, "")):
            raise ValueError(f"upstream TRAIN LOGO {field} missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("upstream TRAIN LOGO output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("upstream TRAIN LOGO protocol SHA drift")
    return config


def _validate_prior_failed_c0(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root)
    summary = _read_object(root / "summary.json")
    declared = str(summary.get("summary_sha256", ""))
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if canonical_sha256(payload) != declared:
        raise ValueError("prior C0 summary self-hash drift")
    if declared != protocol["prior_c0_summary_sha256"]:
        raise ValueError("prior C0 summary identity drift")
    if summary.get("status") != "completed":
        raise ValueError("prior C0 run is not completed")
    if summary.get("calibration_passed") is not False:
        raise ValueError("prior C0 run did not fail overall calibration")
    if summary.get("tube_1_authorized") is not False:
        raise ValueError("prior C0 run unexpectedly authorized Tube_1")
    if summary.get("validation_used_for_parameter_fit") is not False:
        raise ValueError("prior C0 run used validation for parameter fit")
    if summary.get("test_data_used") is not False or summary.get("final_evaluation_data_used") is not False:
        raise ValueError("prior C0 run touched TEST/final data")

    upstream = _read_object(root / "upstream" / "manifest.json")
    downstream = _read_object(root / "downstream" / "manifest.json")
    if upstream.get("manifest_sha256") != protocol["prior_upstream_manifest_sha256"]:
        raise ValueError("prior C_up^0 manifest drift")
    if downstream.get("manifest_sha256") != protocol["prior_downstream_manifest_sha256"]:
        raise ValueError("prior C_down^0 manifest drift")
    if upstream.get("calibration_passed") is not False:
        raise ValueError("prior C_up^0 did not fail as declared")
    if downstream.get("calibration_passed") is not True:
        raise ValueError("prior C_down^0 did not pass as declared")
    return {
        "summary_sha256": declared,
        "upstream_manifest_sha256": upstream["manifest_sha256"],
        "downstream_manifest_sha256": downstream["manifest_sha256"],
        "upstream_validation_roc_auc": float(upstream["validation_metrics"]["roc_auc"]),
        "downstream_validation_roc_auc": float(downstream["validation_metrics"]["roc_auc"]),
    }


def _validate_train_rows(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    selected: list[dict[str, Any]] = []
    actor = str(protocol["policy_actor_sha256"])
    payload = str(protocol["policy_payload_sha256"])
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        if row.get("phase") != "upstream":
            continue
        if row.get("split") != "train":
            raise ValueError("upstream diagnostic accepts TRAIN rows only")
        if row.get("policy_actor_sha256") != actor:
            raise ValueError("upstream TRAIN actor identity drift")
        if row.get("policy_payload_sha256") != payload:
            raise ValueError("upstream TRAIN payload identity drift")
        state = _sha(row.get("state_sha256"), field="upstream TRAIN state_sha256")
        if state in seen:
            raise ValueError("upstream TRAIN repeats physical state")
        seen.add(state)
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError("upstream TRAIN label must be binary")
        obs = np.asarray(row.get("actor_observation"), dtype=np.float32).reshape(-1)
        if obs.shape != (76,) or not np.isfinite(obs).all():
            raise ValueError("upstream TRAIN observation must be finite 76-D")
        if not str(row.get("parent_group_id", "")):
            raise ValueError("upstream TRAIN parent group missing")
        selected.append(row)

    expected = protocol["expected_upstream_train"]
    positive = sum(int(row["label"]) for row in selected)
    groups = sorted({str(row["parent_group_id"]) for row in selected})
    if len(selected) != int(expected["candidate_count"]):
        raise ValueError("upstream TRAIN candidate count drift")
    if positive != int(expected["positive_count"]):
        raise ValueError("upstream TRAIN positive count drift")
    if len(selected) - positive != int(expected["negative_count"]):
        raise ValueError("upstream TRAIN negative count drift")
    if len(groups) != int(expected["parent_group_count"]):
        raise ValueError("upstream TRAIN parent group count drift")
    for group in groups:
        labels = {int(row["label"]) for row in selected if str(row["parent_group_id"]) == group}
        if labels != {0, 1}:
            raise ValueError("every upstream TRAIN parent group must contain both labels")
    return tuple(selected)


def _fit_fold(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    model_cfg: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    train_x, train_y, train_groups, _ = _dataset(train_rows)
    heldout_x, heldout_y, heldout_groups, _ = _dataset(heldout_rows)
    if {float(x) for x in np.unique(train_y)} != {0.0, 1.0}:
        raise ValueError("LOGO fold TRAIN lacks both labels")
    if {float(x) for x in np.unique(heldout_y)} != {0.0, 1.0}:
        raise ValueError("LOGO held-out group lacks both labels")
    if len(set(heldout_groups)) != 1:
        raise ValueError("LOGO held-out fold must contain one parent group")

    sample_weights = _cell_balanced_weights(train_groups, train_y)
    mean, std = _normalization(train_x)
    x_train = _transform(train_x, mean, std)
    x_heldout = _transform(heldout_x, mean, std)

    jx = jnp.asarray(x_train)
    jy = jnp.asarray(train_y)
    jw = jnp.asarray(sample_weights)
    l2 = float(model_cfg["l2_weight"])
    params = {
        "weight": jax.random.normal(
            jax.random.PRNGKey(int(seed)), (76,), dtype=jnp.float32
        ) * 0.001,
        "bias": jnp.asarray(0.0, dtype=jnp.float32),
    }
    optimizer = optax.adam(float(model_cfg["learning_rate"]))
    opt_state = optimizer.init(params)

    def loss_fn(current):
        logits = jx @ current["weight"] + current["bias"]
        bce = optax.sigmoid_binary_cross_entropy(logits, jy)
        data_loss = jnp.sum(jw * bce) / jnp.sum(jw)
        penalty = 0.5 * l2 * jnp.sum(jnp.square(current["weight"]))
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
        raise ValueError("upstream TRAIN LOGO optimizer became nonfinite")

    weight = np.asarray(jax.device_get(params["weight"]), dtype=np.float32)
    bias = float(jax.device_get(params["bias"]))
    train_score = _score(weight, bias, x_train)
    heldout_score = _score(weight, bias, x_heldout)
    return {
        "heldout_parent_group_id": str(next(iter(set(heldout_groups)))),
        "train_group_count": len(set(train_groups)),
        "train_metrics": _metrics(train_y, train_score),
        "heldout_metrics": _metrics(heldout_y, heldout_score),
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "seed": int(seed),
    }


def classify_logo_folds(
    folds: Sequence[Mapping[str, Any]],
    gate_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if not folds:
        raise ValueError("upstream TRAIN LOGO requires folds")
    aucs = [float(fold["heldout_metrics"]["roc_auc"]) for fold in folds]
    gaps = [float(fold["heldout_metrics"]["score_gap"]) for fold in folds]
    mean_auc = float(np.mean(aucs))
    worst_auc = float(np.min(aucs))
    gate = {
        "mean_logo_roc_auc_at_least_minimum": bool(
            mean_auc >= float(gate_cfg["minimum_mean_logo_roc_auc"])
        ),
        "worst_logo_roc_auc_at_least_minimum": bool(
            worst_auc >= float(gate_cfg["minimum_worst_logo_roc_auc"])
        ),
        "positive_score_gap_in_every_fold": bool(all(gap > 0.0 for gap in gaps)),
    }
    return {
        "mean_logo_roc_auc": mean_auc,
        "worst_logo_roc_auc": worst_auc,
        "mean_logo_score_gap": float(np.mean(gaps)),
        "fold_roc_auc": aucs,
        "fold_score_gap": gaps,
        "gate": gate,
        "train_group_generalization_supported": bool(all(gate.values())),
    }


def run_upstream_train_logo_diagnostic(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_upstream_train_logo_config(config_path)
    protocol = dict(config["protocol"])

    prior = _validate_prior_failed_c0(Path(str(protocol["prior_c0_root"])), protocol)
    train_manifest, train_rows = load_frozen_iteration_train_evidence(
        Path(str(protocol["frozen_train_evidence"]))
    )
    if train_manifest["manifest_sha256"] != protocol["frozen_train_manifest_sha256"]:
        raise ValueError("upstream TRAIN frozen manifest drift")
    if train_manifest["policy_actor_sha256"] != protocol["policy_actor_sha256"]:
        raise ValueError("upstream TRAIN frozen actor drift")
    if train_manifest["policy_payload_sha256"] != protocol["policy_payload_sha256"]:
        raise ValueError("upstream TRAIN frozen payload drift")

    upstream_rows = _validate_train_rows(train_rows, protocol)
    groups = sorted({str(row["parent_group_id"]) for row in upstream_rows})
    folds: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        fold_train = [row for row in upstream_rows if str(row["parent_group_id"]) != group]
        fold_heldout = [row for row in upstream_rows if str(row["parent_group_id"]) == group]
        folds.append(
            _fit_fold(
                fold_train,
                fold_heldout,
                model_cfg=protocol["model"],
                seed=int(protocol["model"]["seed_base"]) + index,
            )
        )

    classification = classify_logo_folds(folds, protocol["diagnostic_gate"])
    supported = bool(classification["train_group_generalization_supported"])
    if supported:
        diagnosis = (
            "current_linear_model_generalizes_across_existing_train_groups_but_failed_"
            "independent_validation_indicating_train_parent_domain_shift"
        )
        next_gate = (
            "do not change model using consumed validation; expand upstream TRAIN parent "
            "diversity with new TRAIN-only real-dynamics parents, then rerun TRAIN-only "
            "group generalization before predeclaring a fresh independent validation bank"
        )
    else:
        diagnosis = "current_linear_model_not_group_generalizable_within_existing_train_evidence"
        next_gate = (
            "revise upstream field representation or increase TRAIN parent diversity using "
            "TRAIN-only evidence; consumed validation remains unavailable for model selection; "
            "fresh validation is prohibited until a predeclared TRAIN-only group gate passes"
        )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "train_only_upstream_group_generalization_diagnostic",
        "iteration": 0,
        "policy_name": str(protocol["policy_name"]),
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "protocol_sha256": canonical_sha256(protocol),
        "frozen_train_manifest_sha256": str(protocol["frozen_train_manifest_sha256"]),
        "prior_c0": prior,
        "upstream_train_count": len(upstream_rows),
        "upstream_train_parent_group_ids": groups,
        "fold_count": len(folds),
        "folds": folds,
        **classification,
        "diagnosis": diagnosis,
        "fresh_validation_authorized": False,
        "tube_1_authorized": False,
        "environment_interactions": 0,
        "training_transitions": 0,
        "supervised_optimizer_steps": len(folds) * int(protocol["model"]["steps"]),
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "prior_c0_gate_summary_read_for_provenance_only": True,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
        "next_scientific_gate": next_gate,
    }
    summary["summary_sha256"] = canonical_sha256(summary)

    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
