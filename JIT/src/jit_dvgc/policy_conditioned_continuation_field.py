"""Policy-conditioned continuation fields C_up^k / C_down^k.

Iteration-0 uses a deliberately low-complexity linear logistic score over the
76-D unified actor observation. TRAIN descendants are weighted so every
(parent_group_id, label) cell contributes equal total loss, limiting domination
by dense descendants from one parent or one class. Validation is never used to
fit model parameters or tune hyperparameters; it is used once to calibrate a
conservative acceptance threshold and evaluate the predeclared gate.
"""
from __future__ import annotations

from collections import Counter
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
from .iteration_validation_evidence import load_frozen_iteration_validation_evidence


CONFIG_SCHEMA = "jit_policy_conditioned_continuation_field_config_v1"
PROTOCOL_SCHEMA = "jit_policy_conditioned_continuation_field_protocol_v1"
FIELD_SCHEMA = "jit_policy_conditioned_continuation_field_v1"
SUMMARY_SCHEMA = "jit_policy_conditioned_continuation_field_summary_v1"


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


def load_continuation_field_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported continuation field config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("continuation field protocol is required")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("continuation field protocol schema drift")
    if protocol.get("status") != "predeclared_before_model_fit":
        raise ValueError("continuation field protocol must be predeclared before fit")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("continuation field policy identity drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "frozen_train_manifest_sha256",
        "validation_scientific_protocol_sha256",
        "validation_runtime_protocol_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if not str(protocol.get("frozen_train_evidence", "")):
        raise ValueError("continuation field frozen TRAIN path missing")
    if not str(protocol.get("frozen_validation_evidence", "")):
        raise ValueError("continuation field frozen validation path missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("continuation field output path missing")

    expected_model = {
        "family": "linear_logistic",
        "input": "unified_actor_observation",
        "observation_size": 76,
        "normalization": "train_only_zscore_clip10",
        "sample_weighting": "equal_parent_label_cell_mass",
        "l2_weight": 0.01,
        "optimizer": "adam_full_batch_fixed_schedule",
        "steps": 4000,
        "learning_rate": 0.01,
        "seeds": {"upstream": 840001, "downstream": 840002},
    }
    if protocol.get("model") != expected_model:
        raise ValueError("continuation field fixed model contract drift")

    expected_calibration = {
        "decision_rule": "accept_if_score_strictly_greater_than_max_validation_negative_score",
        "minimum_validation_roc_auc": 0.70,
        "minimum_validation_positive_recall": 0.20,
        "require_accepted_positive_in_every_validation_parent": True,
        "validation_hyperparameter_search": False,
        "threshold_is_safety_certificate": False,
    }
    if protocol.get("calibration") != expected_calibration:
        raise ValueError("continuation field calibration contract drift")

    counts = protocol.get("expected_counts")
    if not isinstance(counts, Mapping) or set(counts) != {"train", "validation"}:
        raise ValueError("continuation field expected counts drift")
    for split in ("train", "validation"):
        if set(counts[split]) != {"upstream", "downstream"}:
            raise ValueError("continuation field phase count contract drift")
        for phase in ("upstream", "downstream"):
            row = counts[split][phase]
            if set(row) != {"candidate_count", "positive_count", "negative_count"}:
                raise ValueError("continuation field count record drift")
            if int(row["candidate_count"]) != int(row["positive_count"]) + int(row["negative_count"]):
                raise ValueError("continuation field count arithmetic drift")
            if int(row["positive_count"]) <= 0 or int(row["negative_count"]) <= 0:
                raise ValueError("continuation field requires both classes in each split")

    expected_claims = {
        "policy_conditioned_empirical_continuation_field": True,
        "certified_probability_claim": False,
        "certified_safe_set_claim": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
    }
    if protocol.get("claim_boundary") != expected_claims:
        raise ValueError("continuation field claim boundary drift")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("continuation field protocol SHA drift")
    return config


def _split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    split: str,
    actor_sha256: str,
    payload_sha256: str,
) -> tuple[dict[str, Any], ...]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        if row.get("phase") != phase:
            continue
        if row.get("split") != split:
            raise ValueError(f"{phase} evidence contains wrong split")
        if row.get("policy_actor_sha256") != actor_sha256:
            raise ValueError(f"{phase} evidence actor identity drift")
        if row.get("policy_payload_sha256") != payload_sha256:
            raise ValueError(f"{phase} evidence payload identity drift")
        state_sha = _sha(row.get("state_sha256"), field=f"{phase} state_sha256")
        if state_sha in seen:
            raise ValueError(f"{phase} evidence contains duplicate physical state")
        seen.add(state_sha)
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError(f"{phase} evidence label must be binary")
        obs = np.asarray(row.get("actor_observation"), dtype=np.float32).reshape(-1)
        if obs.shape != (76,) or not np.isfinite(obs).all():
            raise ValueError(f"{phase} evidence actor observation must be finite 76-D")
        if not str(row.get("parent_group_id", "")):
            raise ValueError(f"{phase} evidence parent group missing")
        selected.append(row)
    if not selected:
        raise ValueError(f"{phase} {split} evidence is empty")
    if {int(row["label"]) for row in selected} != {0, 1}:
        raise ValueError(f"{phase} {split} evidence must contain both classes")
    return tuple(selected)


def _dataset(rows: Sequence[Mapping[str, Any]]):
    x = np.stack(
        [np.asarray(row["actor_observation"], dtype=np.float32) for row in rows]
    ).astype(np.float32)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.float32)
    groups = tuple(str(row["parent_group_id"]) for row in rows)
    states = tuple(str(row["state_sha256"]) for row in rows)
    return x, y, groups, states


def _cell_balanced_weights(groups: Sequence[str], targets: np.ndarray) -> np.ndarray:
    cells = Counter((str(group), int(label)) for group, label in zip(groups, targets))
    group_names = sorted(set(str(group) for group in groups))
    for group in group_names:
        if cells[(group, 0)] <= 0 or cells[(group, 1)] <= 0:
            raise ValueError("TRAIN weighting requires both labels in every parent group")
    weights = np.asarray(
        [1.0 / float(cells[(str(group), int(label))]) for group, label in zip(groups, targets)],
        dtype=np.float32,
    )
    weights *= np.float32(len(weights) / np.sum(weights))
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("continuation field sample weights are invalid")
    return weights


def _normalization(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_x, axis=0).astype(np.float32)
    std = np.std(train_x, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    positives = score[y == 1.0]
    negatives = score[y == 0.0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("ROC AUC requires both classes")
    return float(
        np.mean(
            (positives[:, None] > negatives[None, :])
            + 0.5 * (positives[:, None] == negatives[None, :])
        )
    )


def _metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return {
        "count": int(len(y)),
        "positive_count": int(np.sum(y == 1.0)),
        "negative_count": int(np.sum(y == 0.0)),
        "roc_auc": _auc(y, score),
        "positive_mean_score": float(np.mean(score[y == 1.0])),
        "negative_mean_score": float(np.mean(score[y == 0.0])),
        "score_gap": float(np.mean(score[y == 1.0]) - np.mean(score[y == 0.0])),
    }


def _score(weights: np.ndarray, bias: float, x: np.ndarray) -> np.ndarray:
    logits = np.asarray(x, dtype=np.float64) @ np.asarray(weights, dtype=np.float64) + float(bias)
    logits = np.clip(logits, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


def _fit_phase(
    *,
    phase: str,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    model_cfg = protocol["model"]
    cal_cfg = protocol["calibration"]
    train_x, train_y, train_groups, train_states = _dataset(train_rows)
    validation_x, validation_y, validation_groups, validation_states = _dataset(validation_rows)
    if set(train_states).intersection(validation_states):
        raise ValueError(f"{phase} TRAIN/validation exact physical state overlap")

    sample_weights = _cell_balanced_weights(train_groups, train_y)
    mean, std = _normalization(train_x)
    x_train = _transform(train_x, mean, std)
    x_validation = _transform(validation_x, mean, std)

    jx = jnp.asarray(x_train)
    jy = jnp.asarray(train_y)
    jw = jnp.asarray(sample_weights)
    l2 = float(model_cfg["l2_weight"])
    seed = int(model_cfg["seeds"][phase])
    params = {
        "weight": jax.random.normal(jax.random.PRNGKey(seed), (76,), dtype=jnp.float32) * 0.001,
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
        raise ValueError(f"{phase} continuation field fit became nonfinite")

    weight = np.asarray(jax.device_get(params["weight"]), dtype=np.float32)
    bias = float(jax.device_get(params["bias"]))
    train_score = _score(weight, bias, x_train)
    validation_score = _score(weight, bias, x_validation)

    negative_scores = validation_score[validation_y == 0.0]
    if len(negative_scores) == 0:
        raise ValueError(f"{phase} validation contains no negatives for calibration")
    threshold = float(np.max(negative_scores))
    accepted = validation_score > threshold
    accepted_negative = int(np.sum(accepted & (validation_y == 0.0)))
    if accepted_negative != 0:
        raise ValueError(f"{phase} conservative threshold accepted a validation negative")
    positive_mask = validation_y == 1.0
    positive_recall = float(np.mean(accepted[positive_mask]))
    accepted_positive_groups = sorted(
        {
            validation_groups[index]
            for index in range(len(validation_groups))
            if bool(accepted[index]) and validation_y[index] == 1.0
        }
    )
    validation_positive_groups = sorted(
        {
            validation_groups[index]
            for index in range(len(validation_groups))
            if validation_y[index] == 1.0
        }
    )
    if set(validation_positive_groups) != set(validation_groups):
        missing = sorted(set(validation_groups) - set(validation_positive_groups))
        raise ValueError(f"{phase} validation parent(s) lack positive support: {missing}")

    train_metrics = _metrics(train_y, train_score)
    validation_metrics = _metrics(validation_y, validation_score)
    gate = {
        "validation_roc_auc_at_least_minimum": bool(
            validation_metrics["roc_auc"] >= float(cal_cfg["minimum_validation_roc_auc"])
        ),
        "validation_positive_recall_at_least_minimum": bool(
            positive_recall >= float(cal_cfg["minimum_validation_positive_recall"])
        ),
        "accepted_positive_in_every_validation_parent": bool(
            set(accepted_positive_groups) == set(validation_positive_groups)
        ),
        "accepted_validation_negative_count_zero": accepted_negative == 0,
    }
    calibration_passed = all(gate.values())

    phase_output = output / phase
    phase_output.mkdir(parents=True, exist_ok=False)
    np.savez(
        phase_output / "field.npz",
        weight=weight,
        bias=np.asarray(bias, dtype=np.float32),
        mean=mean,
        std=std,
    )
    predictions = []
    for index, row in enumerate(validation_rows):
        predictions.append(
            {
                "state_sha256": str(row["state_sha256"]),
                "parent_group_id": str(row["parent_group_id"]),
                "label": int(row["label"]),
                "score": float(validation_score[index]),
                "accepted": bool(accepted[index]),
            }
        )
    (phase_output / "validation_predictions.json").write_text(
        json.dumps(predictions, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": FIELD_SCHEMA,
        "status": "completed",
        "phase": phase,
        "iteration": int(protocol["iteration"]),
        "policy_name": str(protocol["policy_name"]),
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "field_name": "C_up^0" if phase == "upstream" else "C_down^0",
        "input": "unified_actor_observation",
        "observation_size": 76,
        "model_family": "linear_logistic",
        "parameter_count": 77,
        "score_semantics": (
            "regularized policy-conditioned empirical continuation score; "
            "not a certified probability or safe-set certificate"
        ),
        "normalization": "TRAIN-only z-score with std floor 1e-6 and clip +/-10",
        "sample_weighting": "equal total mass for every TRAIN (parent_group_id,label) cell",
        "l2_weight": float(model_cfg["l2_weight"]),
        "optimizer": str(model_cfg["optimizer"]),
        "optimizer_steps": int(model_cfg["steps"]),
        "learning_rate": float(model_cfg["learning_rate"]),
        "seed": seed,
        "initial_objective": initial_loss,
        "final_objective": final_loss,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "calibration_rule": str(cal_cfg["decision_rule"]),
        "acceptance_threshold_exclusive": threshold,
        "validation_positive_recall_at_threshold": positive_recall,
        "accepted_validation_positive_groups": accepted_positive_groups,
        "validation_positive_groups": validation_positive_groups,
        "calibration_gate": gate,
        "calibration_passed": calibration_passed,
        "field_file_sha256": file_sha256(phase_output / "field.npz"),
        "validation_predictions_sha256": file_sha256(
            phase_output / "validation_predictions.json"
        ),
        "environment_interactions": 0,
        "training_transitions": 0,
        "validation_rows_used_for_parameter_fit": 0,
        "validation_rows_used_for_threshold_calibration": int(len(validation_rows)),
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (phase_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def fit_policy_conditioned_continuation_fields(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_continuation_field_config(config_path)
    protocol = dict(config["protocol"])

    train_manifest, train_all = load_frozen_iteration_train_evidence(
        Path(str(protocol["frozen_train_evidence"]))
    )
    if train_manifest["manifest_sha256"] != protocol["frozen_train_manifest_sha256"]:
        raise ValueError("continuation field frozen TRAIN manifest drift")
    if train_manifest["policy_actor_sha256"] != protocol["policy_actor_sha256"]:
        raise ValueError("continuation field TRAIN actor drift")
    if train_manifest["policy_payload_sha256"] != protocol["policy_payload_sha256"]:
        raise ValueError("continuation field TRAIN payload drift")

    validation_manifest, _validation_candidates, validation_all = (
        load_frozen_iteration_validation_evidence(
            Path(str(protocol["frozen_validation_evidence"]))
        )
    )
    if validation_manifest["policy_actor_sha256"] != protocol["policy_actor_sha256"]:
        raise ValueError("continuation field validation actor drift")
    if validation_manifest["policy_payload_sha256"] != protocol["policy_payload_sha256"]:
        raise ValueError("continuation field validation payload drift")
    if (
        validation_manifest["scientific_protocol_sha256"]
        != protocol["validation_scientific_protocol_sha256"]
    ):
        raise ValueError("continuation field validation scientific protocol drift")
    if (
        validation_manifest["runtime_protocol_sha256"]
        != protocol["validation_runtime_protocol_sha256"]
    ):
        raise ValueError("continuation field validation runtime protocol drift")
    if validation_manifest.get("validation_rows_may_enter_train_or_tube") is not False:
        raise ValueError("continuation field validation artifact leakage policy drift")

    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    phase_manifests: dict[str, Any] = {}
    for phase in ("upstream", "downstream"):
        train_rows = _split_rows(
            train_all,
            phase=phase,
            split="train",
            actor_sha256=protocol["policy_actor_sha256"],
            payload_sha256=protocol["policy_payload_sha256"],
        )
        validation_rows = _split_rows(
            validation_all,
            phase=phase,
            split="validation",
            actor_sha256=protocol["policy_actor_sha256"],
            payload_sha256=protocol["policy_payload_sha256"],
        )
        expected_train = protocol["expected_counts"]["train"][phase]
        expected_validation = protocol["expected_counts"]["validation"][phase]
        for rows, expected, name in (
            (train_rows, expected_train, "TRAIN"),
            (validation_rows, expected_validation, "validation"),
        ):
            positive = sum(int(row["label"]) for row in rows)
            actual = {
                "candidate_count": len(rows),
                "positive_count": positive,
                "negative_count": len(rows) - positive,
            }
            if actual != expected:
                raise ValueError(f"{phase} {name} count drift")
        phase_manifests[phase] = _fit_phase(
            phase=phase,
            train_rows=train_rows,
            validation_rows=validation_rows,
            protocol=protocol,
            output=output,
        )

    calibration_passed = all(
        bool(phase_manifests[phase]["calibration_passed"])
        for phase in ("upstream", "downstream")
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "iteration": int(protocol["iteration"]),
        "policy_name": str(protocol["policy_name"]),
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "fit_protocol_sha256": canonical_sha256(protocol),
        "fit_config_file_sha256": file_sha256(config_path),
        "frozen_train_manifest_sha256": str(train_manifest["manifest_sha256"]),
        "frozen_validation_manifest_sha256": str(validation_manifest["manifest_sha256"]),
        "phase_manifests": {
            phase: {
                "manifest_sha256": phase_manifests[phase]["manifest_sha256"],
                "calibration_passed": bool(phase_manifests[phase]["calibration_passed"]),
                "validation_roc_auc": float(
                    phase_manifests[phase]["validation_metrics"]["roc_auc"]
                ),
                "validation_positive_recall_at_threshold": float(
                    phase_manifests[phase]["validation_positive_recall_at_threshold"]
                ),
                "acceptance_threshold_exclusive": float(
                    phase_manifests[phase]["acceptance_threshold_exclusive"]
                ),
            }
            for phase in ("upstream", "downstream")
        },
        "calibration_passed": calibration_passed,
        "tube_1_authorized": calibration_passed,
        "environment_interactions": 0,
        "training_transitions": 0,
        "validation_used_for_parameter_fit": False,
        "validation_used_for_threshold_calibration": True,
        "validation_rows_may_enter_train_or_tube": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
        "next_scientific_gate": (
            "if and only if both phase calibration gates pass, freeze C_up^0/C_down^0 "
            "and construct Tube_1 from TRAIN states only with core retention; otherwise stop "
            "and revise the field model using TRAIN-side evidence without reusing validation "
            "for hyperparameter search"
        ),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def score_continuation_field(field_dir: Path, observations: np.ndarray) -> np.ndarray:
    field_dir = Path(field_dir)
    manifest = _read_object(field_dir / "manifest.json")
    if manifest.get("schema") != FIELD_SCHEMA or manifest.get("status") != "completed":
        raise ValueError("invalid continuation field manifest")
    payload = np.load(field_dir / "field.npz")
    x = np.asarray(observations, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 76 or not np.isfinite(x).all():
        raise ValueError("continuation field observations must be finite Nx76")
    transformed = _transform(
        x,
        np.asarray(payload["mean"], dtype=np.float32),
        np.asarray(payload["std"], dtype=np.float32),
    )
    return _score(
        np.asarray(payload["weight"], dtype=np.float32),
        float(np.asarray(payload["bias"])),
        transformed,
    )
