"""TRAIN-only fair leave-one-checkpoint-domain-out diagnostics for C_up^0.

The original matched-panel diagnostic uses the fixed 76-D linear logistic
field.  After that model failed with acquisition-family confounding removed,
this module also supports exactly one predeclared low-capacity nonlinear repair:
a one-hidden-layer 8-unit tanh MLP.  Both variants use the same matched 720-row
TRAIN artifact, fold split, normalization, parent-label weighting, optimizer
schedule, and scientific gate.  Consumed validation is never read or reused.
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
from .upstream_checkpoint_domain_cv import (
    _fit_fold,
    _validate_rows,
    classify_checkpoint_domain_folds,
)
from .upstream_checkpoint_train_evidence import load_frozen_upstream_checkpoint_train_evidence

CONFIG_SCHEMA = "jit_upstream_checkpoint_domain_logo_config_v1"
PROTOCOL_SCHEMA = "jit_upstream_checkpoint_domain_logo_protocol_v1"
SUMMARY_SCHEMA = "jit_upstream_checkpoint_domain_logo_summary_v1"
STATUS = "predeclared_after_matched_panel_freeze_before_checkpoint_cv"
NONLINEAR_STATUS = "predeclared_after_matched_linear_cv_fail_before_single_nonlinear_train_cv"
MATCHED_STATUSES = (STATUS, NONLINEAR_STATUS)

_LINEAR_MODEL = {
    "family": "linear_logistic",
    "input": "unified_actor_observation",
    "observation_size": 76,
    "normalization": "fold_train_only_zscore_clip10",
    "sample_weighting": "equal_parent_label_cell_mass",
    "l2_weight": 0.01,
    "optimizer": "adam_full_batch_fixed_schedule",
    "steps": 4000,
    "learning_rate": 0.01,
    "seed_base": 846000,
}

_TINY_MLP_MODEL = {
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
    "seed_base": 847000,
}

_GATE = {
    "minimum_mean_logo_roc_auc": 0.70,
    "minimum_worst_logo_roc_auc": 0.60,
    "require_positive_score_gap_in_every_fold": True,
}

_DATA_POLICY = {
    "train_rows_only": True,
    "consumed_validation_rows_reused": False,
    "consumed_validation_predictions_reused": False,
    "test_data_used": False,
    "final_evaluation_data_used": False,
}

_CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "continuation_field_reselected": False,
    "fresh_validation_bank_predeclared": False,
    "tube_1_constructed": False,
    "pi_1_trained": False,
    "jce_jel_claim": False,
    "certified_safe_set_claim": False,
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


def _tiny_mlp_parameter_count(input_size: int, hidden_units: int) -> int:
    return int(input_size) * int(hidden_units) + int(hidden_units) + int(hidden_units) + 1


def _validate_prior_matched_linear_failure(path: Path, expected_sha: str) -> None:
    summary = _read_object(path)
    declared = _sha(summary.get("summary_sha256"), field="prior matched linear CV summary")
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if canonical_sha256(payload) != declared or declared != expected_sha:
        raise ValueError("prior matched linear CV summary identity drift")
    if summary.get("status") != "completed":
        raise ValueError("prior matched linear CV is not completed")
    if summary.get("protocol_sha256") != (
        "908e5b3c0e235b666dfe6368504e75343eae5a9d95dd3cab85daa1803981573d"
    ):
        raise ValueError("prior matched linear CV protocol drift")
    if summary.get("checkpoint_domain_generalization_supported") is not False:
        raise ValueError("nonlinear repair requires failed matched linear CV")
    if summary.get("fresh_validation_predeclaration_authorized") is not False:
        raise ValueError("prior matched linear CV unexpectedly authorized fresh validation")
    if summary.get("tube_1_authorized") is not False:
        raise ValueError("prior matched linear CV unexpectedly authorized Tube_1")
    if summary.get("consumed_validation_rows_reused") is not False:
        raise ValueError("prior matched linear CV reused consumed validation rows")
    if summary.get("consumed_validation_predictions_reused") is not False:
        raise ValueError("prior matched linear CV reused consumed validation predictions")
    interpretation = summary.get("interpretation", {})
    if interpretation.get("acquisition_family_confound_removed") is not True:
        raise ValueError("prior matched linear CV did not remove acquisition-family confounding")
    if interpretation.get("all_three_domains_use_same_locked_matched_panel") is not True:
        raise ValueError("prior matched linear CV did not use the locked matched panel")


def load_matched_checkpoint_domain_cv_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported matched checkpoint-domain LOGO config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("matched checkpoint-domain LOGO protocol drift")
    status = str(protocol.get("status", ""))
    if status not in MATCHED_STATUSES:
        raise ValueError("matched checkpoint-domain LOGO status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("matched checkpoint-domain LOGO policy drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "frozen_upstream_train_freeze_protocol_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if protocol.get("expected_combined") != {
        "candidate_count": 720,
        "positive_count": 639,
        "negative_count": 81,
        "parent_group_count": 15,
        "checkpoint_domain_count": 3,
    }:
        raise ValueError("matched checkpoint-domain expected counts drift")
    if protocol.get("required_domains") != [
        "transition_4988928",
        "transition_7987200",
        "transition_9977856",
    ]:
        raise ValueError("matched checkpoint-domain required domain order drift")
    expected_model = _LINEAR_MODEL if status == STATUS else _TINY_MLP_MODEL
    if protocol.get("model") != expected_model:
        raise ValueError("matched checkpoint-domain model contract drift")
    if status == NONLINEAR_STATUS:
        if _tiny_mlp_parameter_count(76, 8) != int(protocol["model"]["parameter_count"]):
            raise ValueError("tiny MLP parameter-count contract drift")
        for field in (
            "frozen_upstream_train_manifest_sha256",
            "prior_matched_linear_cv_summary_sha256",
        ):
            _sha(protocol.get(field), field=field)
        if not str(protocol.get("prior_matched_linear_cv_summary", "")):
            raise ValueError("tiny MLP repair requires prior matched linear CV summary")
        if protocol.get("method_decision") != {
            "reason": "matched_linear_checkpoint_cv_failed_after_acquisition_family_confound_was_removed",
            "single_repair_model_only": True,
            "hyperparameter_grid_search": False,
            "optimization_schedule_changed_from_linear": False,
            "fresh_validation_may_be_predeclared_only_if_train_checkpoint_gate_passes": True,
            "automatic_model_escalation_if_fail": False,
        }:
            raise ValueError("tiny MLP repair method-decision contract drift")
        expected_interpretation = {
            "checkpoint_domain_is_parent_group_transition_prefix": True,
            "all_three_domains_use_same_locked_matched_panel": True,
            "acquisition_family_confound_removed": True,
            "actor_observation_is_three_real_fifo_frames_plus_jump_signal": True,
            "tiny_mlp_adds_one_low_capacity_nonlinear_interaction_layer": True,
        }
    else:
        expected_interpretation = {
            "checkpoint_domain_is_parent_group_transition_prefix": True,
            "all_three_domains_use_same_locked_matched_panel": True,
            "acquisition_family_confound_removed": True,
            "failure_can_be_interpreted_as_checkpoint_domain_or_representation_generalization_limit": True,
        }
    if protocol.get("diagnostic_gate") != _GATE:
        raise ValueError("matched checkpoint-domain gate drift")
    if protocol.get("data_policy") != _DATA_POLICY:
        raise ValueError("matched checkpoint-domain data policy drift")
    if protocol.get("interpretation") != expected_interpretation:
        raise ValueError("matched checkpoint-domain interpretation drift")
    if protocol.get("claim_boundary") != _CLAIM_BOUNDARY:
        raise ValueError("matched checkpoint-domain claim boundary drift")
    if not str(protocol.get("frozen_upstream_train_evidence", "")):
        raise ValueError("matched checkpoint-domain frozen evidence missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("matched checkpoint-domain output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("matched checkpoint-domain protocol SHA drift")
    return config


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _fit_tiny_mlp_fold(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    heldout_domain: str,
    model_cfg: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    train_x, train_y, train_groups, _ = _dataset(train_rows)
    heldout_x, heldout_y, heldout_groups, _ = _dataset(heldout_rows)
    if set(np.unique(train_y).tolist()) != {0.0, 1.0}:
        raise ValueError("tiny MLP checkpoint fold TRAIN lacks both labels")
    if set(np.unique(heldout_y).tolist()) != {0.0, 1.0}:
        raise ValueError("tiny MLP checkpoint held-out fold lacks both labels")
    if {str(group).split("__", 1)[0] for group in heldout_groups} != {heldout_domain}:
        raise ValueError("tiny MLP held-out rows mix checkpoint domains")
    if heldout_domain in {str(group).split("__", 1)[0] for group in train_groups}:
        raise ValueError("tiny MLP fold leaks held-out checkpoint domain")

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
        raise ValueError("tiny MLP parameter count drift")
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
        raise ValueError("tiny MLP checkpoint CV optimizer became nonfinite")

    train_logits = np.asarray(jax.device_get(logits(params, jnp.asarray(x_train))), dtype=np.float64)
    heldout_logits = np.asarray(jax.device_get(logits(params, jnp.asarray(x_heldout))), dtype=np.float64)
    train_score = _sigmoid(train_logits)
    heldout_score = _sigmoid(heldout_logits)
    return {
        "heldout_checkpoint_domain": heldout_domain,
        "train_checkpoint_domains": sorted({str(group).split("__", 1)[0] for group in train_groups}),
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


def run_matched_checkpoint_domain_cv(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_matched_checkpoint_domain_cv_config(config_path)
    protocol = dict(config["protocol"])
    status = str(protocol["status"])
    manifest, raw_rows = load_frozen_upstream_checkpoint_train_evidence(
        Path(str(protocol["frozen_upstream_train_evidence"]))
    )
    if manifest.get("freeze_protocol_sha256") != protocol["frozen_upstream_train_freeze_protocol_sha256"]:
        raise ValueError("matched checkpoint-domain frozen TRAIN protocol drift")
    if manifest.get("policy_actor_sha256") != protocol["policy_actor_sha256"]:
        raise ValueError("matched checkpoint-domain frozen actor drift")
    if manifest.get("policy_payload_sha256") != protocol["policy_payload_sha256"]:
        raise ValueError("matched checkpoint-domain frozen payload drift")
    if manifest.get("matched_panel") is not True:
        raise ValueError("matched checkpoint-domain requires a matched-panel frozen artifact")
    if status == NONLINEAR_STATUS:
        if manifest.get("manifest_sha256") != protocol["frozen_upstream_train_manifest_sha256"]:
            raise ValueError("tiny MLP matched TRAIN manifest drift")
        _validate_prior_matched_linear_failure(
            Path(str(protocol["prior_matched_linear_cv_summary"])),
            str(protocol["prior_matched_linear_cv_summary_sha256"]),
        )
    rows = _validate_rows(raw_rows, protocol)
    domains = list(protocol["required_domains"])
    folds: list[dict[str, Any]] = []
    fitter = _fit_tiny_mlp_fold if status == NONLINEAR_STATUS else _fit_fold
    for index, heldout_domain in enumerate(domains):
        heldout = [row for row in rows if row["parent_domain_id"] == heldout_domain]
        train = [row for row in rows if row["parent_domain_id"] != heldout_domain]
        folds.append(
            fitter(
                train,
                heldout,
                heldout_domain=heldout_domain,
                model_cfg=protocol["model"],
                seed=int(protocol["model"]["seed_base"]) + index,
            )
        )
    classification = classify_checkpoint_domain_folds(folds, protocol["diagnostic_gate"])
    supported = bool(classification["checkpoint_domain_generalization_supported"])

    if status == NONLINEAR_STATUS:
        diagnosis = (
            "tiny_mlp_train_checkpoint_generalization_supported"
            if supported
            else "tiny_mlp_train_checkpoint_generalization_not_supported"
        )
        next_gate = (
            "freeze this single-repair TRAIN checkpoint result and predeclare one fresh independent upstream validation bank; "
            "do not reuse consumed validation and keep Tube_1 locked until fresh validation passes"
            if supported
            else
            "stop automatic upstream model escalation; do not reuse consumed validation or collect more same-panel data; "
            "make an explicit scientific method decision before any further C_up model change"
        )
        artifact_role = "train_only_upstream_single_repair_tiny_mlp_checkpoint_generalization_diagnostic"
    else:
        diagnosis = (
            "current_linear_model_generalizes_across_three_matched_train_checkpoint_domains"
            if supported
            else "current_linear_model_does_not_generalize_across_three_matched_train_checkpoint_domains"
        )
        next_gate = (
            "freeze this matched checkpoint-domain CV result and predeclare one fresh independent upstream validation bank; "
            "do not reuse consumed validation and do not construct Tube_1 yet"
            if supported
            else
            "acquisition-family confounding is removed; do not reuse consumed validation; revise the TRAIN-side upstream "
            "continuation representation/model before any fresh validation bank is predeclared"
        )
        artifact_role = "train_only_upstream_matched_checkpoint_domain_generalization_diagnostic"

    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": artifact_role,
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
        "fold_count": len(folds),
        "folds": folds,
        **classification,
        "model_family": str(protocol["model"]["family"]),
        "parameter_count": int(protocol["model"].get("parameter_count", 77)),
        "diagnosis": diagnosis,
        "interpretation": dict(protocol["interpretation"]),
        "environment_interactions": 0,
        "training_transitions": 0,
        "supervised_optimizer_steps": len(folds) * int(protocol["model"]["steps"]),
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "fresh_validation_predeclaration_authorized": supported,
        "tube_1_authorized": False,
        "next_scientific_gate": next_gate,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    if status == NONLINEAR_STATUS:
        summary["method_decision"] = dict(protocol["method_decision"])
        summary["prior_matched_linear_cv_summary_sha256"] = str(
            protocol["prior_matched_linear_cv_summary_sha256"]
        )
    summary["summary_sha256"] = canonical_sha256(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
