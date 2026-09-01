"""TRAIN-only leave-one-checkpoint-domain-out diagnostic for C_up^0.

The diagnostic uses the exact same 76-D linear logistic model family that
already passed leave-one-seed/group-out inside transition_4988928.  It never
reads the consumed validation rows or predictions.  The held-out unit is now an
entire checkpoint transition domain rather than one seed trajectory.
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
    _score,
    _transform,
)
from .upstream_checkpoint_train_evidence import (
    checkpoint_domain,
    load_frozen_upstream_checkpoint_train_evidence,
)

CONFIG_SCHEMA = "jit_upstream_checkpoint_domain_logo_config_v1"
PROTOCOL_SCHEMA = "jit_upstream_checkpoint_domain_logo_protocol_v1"
SUMMARY_SCHEMA = "jit_upstream_checkpoint_domain_logo_summary_v1"


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


def load_upstream_checkpoint_domain_cv_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported upstream checkpoint-domain LOGO config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("upstream checkpoint-domain LOGO protocol required")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("upstream checkpoint-domain LOGO schema drift")
    if protocol.get("status") != "predeclared_after_three_domain_train_freeze_before_checkpoint_cv":
        raise ValueError("upstream checkpoint-domain LOGO status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("upstream checkpoint-domain LOGO policy drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "frozen_upstream_train_freeze_protocol_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if protocol.get("expected_combined") != {
        "candidate_count": 1051,
        "positive_count": 963,
        "negative_count": 88,
        "parent_group_count": 15,
        "checkpoint_domain_count": 3,
    }:
        raise ValueError("upstream checkpoint-domain expected counts drift")
    if protocol.get("required_domains") != [
        "transition_4988928",
        "transition_7987200",
        "transition_9977856",
    ]:
        raise ValueError("upstream checkpoint-domain required domain order drift")
    if protocol.get("model") != {
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
    }:
        raise ValueError("upstream checkpoint-domain model contract drift")
    if protocol.get("diagnostic_gate") != {
        "minimum_mean_logo_roc_auc": 0.70,
        "minimum_worst_logo_roc_auc": 0.60,
        "require_positive_score_gap_in_every_fold": True,
    }:
        raise ValueError("upstream checkpoint-domain gate drift")
    if protocol.get("data_policy") != {
        "train_rows_only": True,
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }:
        raise ValueError("upstream checkpoint-domain data policy drift")
    if protocol.get("interpretation") != {
        "checkpoint_domain_is_parent_group_transition_prefix": True,
        "legacy_4988928_acquisition_family_is_heterogeneous": True,
        "new_7987200_and_9977856_domains_use_locked_matched_panel": True,
        "failure_cannot_be_attributed_to_checkpoint_shift_alone": True,
    }:
        raise ValueError("upstream checkpoint-domain interpretation drift")
    if protocol.get("claim_boundary") != {
        "diagnostic_only": True,
        "continuation_field_reselected": False,
        "fresh_validation_bank_predeclared": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("upstream checkpoint-domain claim boundary drift")
    if not str(protocol.get("frozen_upstream_train_evidence", "")):
        raise ValueError("upstream checkpoint-domain frozen evidence path missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("upstream checkpoint-domain output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("upstream checkpoint-domain protocol SHA drift")
    return config


def _validate_rows(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    actor = str(protocol["policy_actor_sha256"])
    payload = str(protocol["policy_payload_sha256"])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        if row.get("split") != "train" or row.get("phase") != "upstream":
            raise ValueError("checkpoint-domain CV accepts upstream TRAIN rows only")
        if row.get("policy_actor_sha256") != actor or row.get("policy_payload_sha256") != payload:
            raise ValueError("checkpoint-domain CV policy identity drift")
        state = _sha(row.get("state_sha256"), field="checkpoint-domain CV state_sha256")
        if state in seen:
            raise ValueError("checkpoint-domain CV repeats physical state")
        seen.add(state)
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError("checkpoint-domain CV label must be binary")
        obs = np.asarray(row.get("actor_observation"), dtype=np.float32).reshape(-1)
        if obs.shape != (76,) or not np.isfinite(obs).all():
            raise ValueError("checkpoint-domain CV observation must be finite 76-D")
        group = str(row.get("parent_group_id", ""))
        domain = checkpoint_domain(group)
        if str(row.get("parent_domain_id", domain)) != domain:
            raise ValueError("checkpoint-domain CV parent_domain_id drift")
        row["parent_domain_id"] = domain
        selected.append(row)
    expected = protocol["expected_combined"]
    positive = sum(int(row["label"]) for row in selected)
    groups = {str(row["parent_group_id"]) for row in selected}
    domains = sorted({str(row["parent_domain_id"]) for row in selected})
    if (
        len(selected) != int(expected["candidate_count"])
        or positive != int(expected["positive_count"])
        or len(selected) - positive != int(expected["negative_count"])
        or len(groups) != int(expected["parent_group_count"])
        or len(domains) != int(expected["checkpoint_domain_count"])
        or domains != list(protocol["required_domains"])
    ):
        raise ValueError("checkpoint-domain CV combined evidence counts drift")
    for group in sorted(groups):
        labels = {int(row["label"]) for row in selected if row["parent_group_id"] == group}
        if labels != {0, 1}:
            raise ValueError("checkpoint-domain CV requires both labels in every parent group")
    return tuple(selected)


def _fit_fold(
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
        raise ValueError("checkpoint-domain fold TRAIN lacks both labels")
    if set(np.unique(heldout_y).tolist()) != {0.0, 1.0}:
        raise ValueError("checkpoint-domain held-out fold lacks both labels")
    if {checkpoint_domain(group) for group in heldout_groups} != {heldout_domain}:
        raise ValueError("checkpoint-domain held-out rows mix domains")
    if heldout_domain in {checkpoint_domain(group) for group in train_groups}:
        raise ValueError("checkpoint-domain fold leaks held-out domain into TRAIN")

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
        raise ValueError("checkpoint-domain CV optimizer became nonfinite")
    weight = np.asarray(jax.device_get(params["weight"]), dtype=np.float32)
    bias = float(jax.device_get(params["bias"]))
    train_score = _score(weight, bias, x_train)
    heldout_score = _score(weight, bias, x_heldout)
    return {
        "heldout_checkpoint_domain": heldout_domain,
        "train_checkpoint_domains": sorted({checkpoint_domain(group) for group in train_groups}),
        "train_parent_group_count": len(set(train_groups)),
        "heldout_parent_group_count": len(set(heldout_groups)),
        "train_metrics": _metrics(train_y, train_score),
        "heldout_metrics": _metrics(heldout_y, heldout_score),
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "seed": int(seed),
    }


def classify_checkpoint_domain_folds(
    folds: Sequence[Mapping[str, Any]], gate_cfg: Mapping[str, Any]
) -> dict[str, Any]:
    if len(folds) != 3:
        raise ValueError("checkpoint-domain LOGO requires exactly three folds")
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
        "checkpoint_domain_generalization_supported": bool(all(gate.values())),
    }


def run_upstream_checkpoint_domain_cv(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_upstream_checkpoint_domain_cv_config(config_path)
    protocol = dict(config["protocol"])
    manifest, raw_rows = load_frozen_upstream_checkpoint_train_evidence(
        Path(str(protocol["frozen_upstream_train_evidence"]))
    )
    if manifest.get("freeze_protocol_sha256") != protocol["frozen_upstream_train_freeze_protocol_sha256"]:
        raise ValueError("checkpoint-domain CV frozen TRAIN protocol drift")
    if manifest.get("policy_actor_sha256") != protocol["policy_actor_sha256"]:
        raise ValueError("checkpoint-domain CV frozen actor drift")
    if manifest.get("policy_payload_sha256") != protocol["policy_payload_sha256"]:
        raise ValueError("checkpoint-domain CV frozen payload drift")
    rows = _validate_rows(raw_rows, protocol)
    domains = list(protocol["required_domains"])
    folds: list[dict[str, Any]] = []
    for index, heldout_domain in enumerate(domains):
        heldout = [row for row in rows if row["parent_domain_id"] == heldout_domain]
        train = [row for row in rows if row["parent_domain_id"] != heldout_domain]
        folds.append(
            _fit_fold(
                train,
                heldout,
                heldout_domain=heldout_domain,
                model_cfg=protocol["model"],
                seed=int(protocol["model"]["seed_base"]) + index,
            )
        )
    classification = classify_checkpoint_domain_folds(folds, protocol["diagnostic_gate"])
    supported = bool(classification["checkpoint_domain_generalization_supported"])
    diagnosis = (
        "current_linear_model_generalizes_across_three_train_checkpoint_domains"
        if supported
        else "current_linear_model_does_not_generalize_across_three_train_checkpoint_domains"
    )
    next_gate = (
        "freeze the checkpoint-domain CV result and predeclare a fresh independent upstream validation bank; "
        "do not reuse the consumed seed-1000006 validation and do not construct Tube_1 yet"
        if supported
        else
        "do not use consumed validation for model selection; revise or expand TRAIN-side upstream evidence/representation "
        "before any fresh validation bank is predeclared"
    )
    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "train_only_upstream_checkpoint_domain_generalization_diagnostic",
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
        "diagnosis": diagnosis,
        "interpretation": dict(protocol["interpretation"]),
        "environment_interactions": 0,
        "training_transitions": 0,
        "supervised_optimizer_steps": int(protocol["model"]["steps"]) * len(folds),
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "fresh_validation_predeclaration_authorized": supported,
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
