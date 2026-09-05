"""Freeze the shared C_up/C_down architecture and refit phase-specific fields on TRAIN only.

This is the production handoff after both TRAIN-side unseen-parent diagnostics pass.
The architecture is shared; weights and random seeds are phase-specific.  No validation,
TEST, final-evaluation data, environment interaction, or PPO transition is consumed here.
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

from .config import file_sha256
from .iteration_train_evidence import canonical_sha256, load_frozen_iteration_train_evidence
from .policy_conditioned_continuation_field import (
    _cell_balanced_weights,
    _dataset,
    _metrics,
    _normalization,
    _transform,
)
from .upstream_checkpoint_train_evidence import load_frozen_upstream_checkpoint_train_evidence
from .upstream_matched_checkpoint_domain_cv import _sigmoid, _tiny_mlp_parameter_count

CONFIG_SCHEMA = "jit_shared_continuation_field_refit_config_v1"
PROTOCOL_SCHEMA = "jit_shared_continuation_field_refit_protocol_v1"
ARCHITECTURE_SCHEMA = "jit_shared_continuation_architecture_v1"
FIELD_SCHEMA = "jit_shared_continuation_field_v1"
SUMMARY_SCHEMA = "jit_shared_continuation_field_refit_summary_v1"
STATUS = "predeclared_after_upstream_and_downstream_train_gates_pass_before_full_train_refit"

_MODEL_CONTRACT = {
    "family": "tiny_mlp_tanh",
    "input": "unified_actor_observation",
    "observation_size": 76,
    "hidden_units": 8,
    "activation": "tanh",
    "parameter_count": 625,
    "normalization": "train_only_zscore_clip10",
    "sample_weighting": "equal_parent_label_cell_mass",
    "l2_weight": 0.01,
    "optimizer": "adam_full_batch_fixed_schedule",
    "steps": 4000,
    "learning_rate": 0.01,
    "phase_specific_seeds": {"upstream": 850001, "downstream": 850002},
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


def load_shared_continuation_field_refit_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported shared continuation refit config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("shared continuation refit protocol drift")
    if protocol.get("status") != STATUS:
        raise ValueError("shared continuation refit status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("shared continuation refit policy drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "upstream_train_manifest_sha256",
        "downstream_train_manifest_sha256",
        "upstream_train_gate_summary_sha256",
        "downstream_train_gate_summary_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if protocol.get("architecture") != _MODEL_CONTRACT:
        raise ValueError("shared continuation architecture contract drift")
    if _tiny_mlp_parameter_count(76, 8) != 625:
        raise ValueError("shared continuation architecture parameter-count drift")
    if protocol.get("expected_train") != {
        "upstream": {
            "candidate_count": 720,
            "positive_count": 639,
            "negative_count": 81,
            "parent_group_count": 15,
        },
        "downstream": {
            "candidate_count": 2619,
            "positive_count": 2589,
            "negative_count": 30,
            "parent_group_count": 5,
        },
    }:
        raise ValueError("shared continuation expected TRAIN counts drift")
    if protocol.get("method_decision") != {
        "shared_up_down_architecture_required": True,
        "phase_specific_weights_required": True,
        "phase_specific_calibration_required": True,
        "architecture_selection_complete": True,
        "no_additional_architecture_search": True,
        "full_train_refit_uses_no_validation": True,
        "fresh_validation_may_be_predeclared_after_successful_refit": True,
    }:
        raise ValueError("shared continuation method decision drift")
    if protocol.get("data_policy") != {
        "train_only": True,
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }:
        raise ValueError("shared continuation data policy drift")
    if protocol.get("claim_boundary") != {
        "policy_conditioned_empirical_continuation_fields": True,
        "architecture_frozen": True,
        "fields_calibrated": False,
        "fresh_validation_bank_predeclared": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("shared continuation claim boundary drift")
    for field in (
        "upstream_train_evidence",
        "downstream_train_evidence",
        "upstream_train_gate_summary",
        "downstream_train_gate_summary",
    ):
        if not str(protocol.get(field, "")):
            raise ValueError(f"shared continuation {field} missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("shared continuation output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("shared continuation refit protocol SHA drift")
    return config


def _validate_gate_summary(
    path: Path,
    *,
    expected_sha: str,
    kind: str,
) -> dict[str, Any]:
    summary = _read_object(Path(path))
    declared = _sha(summary.get("summary_sha256"), field=f"{kind} TRAIN gate summary")
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if canonical_sha256(payload) != declared or declared != expected_sha:
        raise ValueError(f"{kind} TRAIN gate summary identity drift")
    if summary.get("status") != "completed":
        raise ValueError(f"{kind} TRAIN gate not completed")
    if kind == "upstream":
        if summary.get("support_stratified_parent_generalization_supported") is not True:
            raise ValueError("upstream TRAIN support gate did not pass")
        if summary.get("shared_up_down_architecture_candidate_authorized") is not True:
            raise ValueError("upstream TRAIN gate did not authorize shared architecture candidate")
        folds = summary.get("folds")
        if not isinstance(folds, list) or len(folds) != 5:
            raise ValueError("upstream TRAIN gate fold provenance drift")
        if any(
            not isinstance(fold, Mapping)
            or fold.get("model_family") != "tiny_mlp_tanh"
            or int(fold.get("parameter_count", -1)) != 625
            for fold in folds
        ):
            raise ValueError("upstream TRAIN gate architecture drift")
    elif kind == "downstream":
        if summary.get("downstream_parent_generalization_supported") is not True:
            raise ValueError("downstream TRAIN parent gate did not pass")
        if summary.get("shared_continuation_architecture_authorized") is not True:
            raise ValueError("downstream TRAIN gate did not authorize shared architecture")
        if summary.get("architecture") != "76->8_tanh->1":
            raise ValueError("downstream TRAIN gate architecture drift")
        if summary.get("model_family") != "tiny_mlp_tanh" or int(summary.get("parameter_count", -1)) != 625:
            raise ValueError("downstream TRAIN gate model drift")
    else:
        raise ValueError("unknown TRAIN gate kind")
    if summary.get("consumed_validation_rows_reused") is not False:
        raise ValueError(f"{kind} TRAIN gate reused validation rows")
    if summary.get("consumed_validation_predictions_reused") is not False:
        raise ValueError(f"{kind} TRAIN gate reused validation predictions")
    if summary.get("test_data_used") is not False or summary.get("final_evaluation_data_used") is not False:
        raise ValueError(f"{kind} TRAIN gate touched TEST/final data")
    return summary


def _validate_train_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    actor: str,
    payload: str,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        if row.get("phase") != phase:
            continue
        if row.get("split") != "train":
            raise ValueError(f"{phase} refit accepts TRAIN rows only")
        if row.get("policy_actor_sha256") != actor or row.get("policy_payload_sha256") != payload:
            raise ValueError(f"{phase} refit policy identity drift")
        state = _sha(row.get("state_sha256"), field=f"{phase} TRAIN state")
        if state in seen:
            raise ValueError(f"{phase} refit repeats physical state")
        seen.add(state)
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError(f"{phase} refit label must be binary")
        obs = np.asarray(row.get("actor_observation"), dtype=np.float32).reshape(-1)
        if obs.shape != (76,) or not np.isfinite(obs).all():
            raise ValueError(f"{phase} refit observation must be finite 76-D")
        if not str(row.get("parent_group_id", "")):
            raise ValueError(f"{phase} refit parent group missing")
        selected.append(row)
    positive = sum(int(row["label"]) for row in selected)
    groups = sorted({str(row["parent_group_id"]) for row in selected})
    actual = {
        "candidate_count": len(selected),
        "positive_count": positive,
        "negative_count": len(selected) - positive,
        "parent_group_count": len(groups),
    }
    if actual != dict(expected):
        raise ValueError(f"{phase} refit TRAIN count drift: {actual}")
    for group in groups:
        labels = {int(row["label"]) for row in selected if str(row["parent_group_id"]) == group}
        if labels != {0, 1}:
            raise ValueError(f"{phase} refit requires both labels in every parent group")
    return tuple(selected)


def _fit_phase(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    model_cfg: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    x, y, groups, _states = _dataset(rows)
    sample_weights = _cell_balanced_weights(groups, y)
    mean, std = _normalization(x)
    x_train = _transform(x, mean, std)
    jx = jnp.asarray(x_train)
    jy = jnp.asarray(y)
    jw = jnp.asarray(sample_weights)

    input_size = int(model_cfg["observation_size"])
    hidden_units = int(model_cfg["hidden_units"])
    if _tiny_mlp_parameter_count(input_size, hidden_units) != int(model_cfg["parameter_count"]):
        raise ValueError("shared continuation tiny-MLP parameter count drift")
    seed = int(model_cfg["phase_specific_seeds"][phase])
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
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

    def logits(current, values):
        hidden = jnp.tanh(values @ current["w1"] + current["b1"])
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
        raise ValueError(f"{phase} shared continuation refit became nonfinite")

    train_logits = np.asarray(jax.device_get(logits(params, jnp.asarray(x_train))), dtype=np.float64)
    train_score = _sigmoid(train_logits)
    train_metrics = _metrics(y, train_score)

    phase_output = output / phase
    phase_output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "w1": np.asarray(jax.device_get(params["w1"]), dtype=np.float32),
        "b1": np.asarray(jax.device_get(params["b1"]), dtype=np.float32),
        "w2": np.asarray(jax.device_get(params["w2"]), dtype=np.float32),
        "b2": np.asarray(jax.device_get(params["b2"]), dtype=np.float32),
        "mean": mean,
        "std": std,
    }
    np.savez(phase_output / "field.npz", **arrays)
    manifest = {
        "schema": FIELD_SCHEMA,
        "status": "completed_uncalibrated",
        "phase": phase,
        "field_name": "C_up^0" if phase == "upstream" else "C_down^0",
        "architecture": "76->8_tanh->1",
        "model_family": "tiny_mlp_tanh",
        "parameter_count": 625,
        "input": "unified_actor_observation",
        "observation_size": 76,
        "phase_specific_weights": True,
        "phase_specific_calibration_required": True,
        "score_semantics": "regularized policy-conditioned empirical continuation score; not a certified probability or safe-set certificate",
        "normalization": "TRAIN-only z-score with std floor 1e-6 and clip +/-10",
        "sample_weighting": "equal total mass for every TRAIN (parent_group_id,label) cell",
        "l2_weight": l2,
        "optimizer": str(model_cfg["optimizer"]),
        "optimizer_steps": int(model_cfg["steps"]),
        "learning_rate": float(model_cfg["learning_rate"]),
        "seed": seed,
        "initial_objective": initial_loss,
        "final_objective": final_loss,
        "train_metrics": train_metrics,
        "acceptance_threshold": None,
        "calibrated": False,
        "field_file_sha256": file_sha256(phase_output / "field.npz"),
        "environment_interactions": 0,
        "training_transitions": 0,
        "validation_rows_used": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (phase_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def fit_shared_continuation_fields(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_shared_continuation_field_refit_config(config_path)
    protocol = dict(config["protocol"])
    upstream_gate = _validate_gate_summary(
        Path(str(protocol["upstream_train_gate_summary"])),
        expected_sha=str(protocol["upstream_train_gate_summary_sha256"]),
        kind="upstream",
    )
    downstream_gate = _validate_gate_summary(
        Path(str(protocol["downstream_train_gate_summary"])),
        expected_sha=str(protocol["downstream_train_gate_summary_sha256"]),
        kind="downstream",
    )

    upstream_manifest, upstream_raw = load_frozen_upstream_checkpoint_train_evidence(
        Path(str(protocol["upstream_train_evidence"]))
    )
    if upstream_manifest.get("manifest_sha256") != protocol["upstream_train_manifest_sha256"]:
        raise ValueError("shared continuation upstream TRAIN manifest drift")
    downstream_manifest, downstream_raw = load_frozen_iteration_train_evidence(
        Path(str(protocol["downstream_train_evidence"]))
    )
    if downstream_manifest.get("manifest_sha256") != protocol["downstream_train_manifest_sha256"]:
        raise ValueError("shared continuation downstream TRAIN manifest drift")

    actor = str(protocol["policy_actor_sha256"])
    payload = str(protocol["policy_payload_sha256"])
    for name, manifest in (("upstream", upstream_manifest), ("downstream", downstream_manifest)):
        if manifest.get("policy_actor_sha256") != actor or manifest.get("policy_payload_sha256") != payload:
            raise ValueError(f"shared continuation {name} TRAIN policy drift")

    upstream_rows = _validate_train_rows(
        upstream_raw,
        phase="upstream",
        actor=actor,
        payload=payload,
        expected=protocol["expected_train"]["upstream"],
    )
    downstream_rows = _validate_train_rows(
        downstream_raw,
        phase="downstream",
        actor=actor,
        payload=payload,
        expected=protocol["expected_train"]["downstream"],
    )

    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)

    architecture = {
        "schema": ARCHITECTURE_SCHEMA,
        "status": "frozen",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": actor,
        "policy_payload_sha256": payload,
        "architecture": dict(protocol["architecture"]),
        "upstream_train_gate_summary_sha256": upstream_gate["summary_sha256"],
        "downstream_train_gate_summary_sha256": downstream_gate["summary_sha256"],
        "shared_up_down_architecture": True,
        "phase_specific_weights_required": True,
        "phase_specific_calibration_required": True,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    architecture["architecture_manifest_sha256"] = canonical_sha256(architecture)
    (output / "architecture_manifest.json").write_text(
        json.dumps(architecture, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    phase_manifests = {
        "upstream": _fit_phase(upstream_rows, phase="upstream", model_cfg=protocol["architecture"], output=output),
        "downstream": _fit_phase(downstream_rows, phase="downstream", model_cfg=protocol["architecture"], output=output),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "shared_architecture_frozen_phase_specific_full_train_refit",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": actor,
        "policy_payload_sha256": payload,
        "fit_protocol_sha256": str(config["expected_protocol_sha256"]),
        "fit_config_file_sha256": file_sha256(config_path),
        "architecture": "76->8_tanh->1",
        "model_family": "tiny_mlp_tanh",
        "parameter_count_per_phase": 625,
        "architecture_manifest_sha256": architecture["architecture_manifest_sha256"],
        "upstream_train_manifest_sha256": upstream_manifest["manifest_sha256"],
        "downstream_train_manifest_sha256": downstream_manifest["manifest_sha256"],
        "upstream_train_gate_summary_sha256": upstream_gate["summary_sha256"],
        "downstream_train_gate_summary_sha256": downstream_gate["summary_sha256"],
        "phase_manifests": {
            phase: {
                "manifest_sha256": phase_manifests[phase]["manifest_sha256"],
                "field_file_sha256": phase_manifests[phase]["field_file_sha256"],
                "train_metrics": phase_manifests[phase]["train_metrics"],
                "calibrated": False,
            }
            for phase in ("upstream", "downstream")
        },
        "architecture_frozen": True,
        "phase_specific_weights_fitted": True,
        "fields_calibrated": False,
        "environment_interactions": 0,
        "training_transitions": 0,
        "validation_rows_used": 0,
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "fresh_validation_predeclaration_authorized": True,
        "tube_1_authorized": False,
        "next_scientific_gate": "predeclare and execute one fresh group-disjoint independent validation bank for both frozen shared-architecture phase fields; calibrate phase-specific thresholds only on that fresh bank before Tube_1",
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary