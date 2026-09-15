"""Small supervised V_up model trained from frozen-expert continuation labels."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from flax import linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .upstream_boundary import canonical_sha256, file_sha256
from .upstream_boundary_lock import load_boundary_lock

VALUE_MODEL_SCHEMA = "jit_upstream_value_model_v1"


class ContinuationValueMLP(nn.Module):
    hidden_sizes: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, observation):
        x = observation
        for width in self.hidden_sizes:
            x = nn.Dense(width)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x)[..., 0]


@dataclass(frozen=True)
class ValueExamples:
    observations: np.ndarray
    targets: np.ndarray
    metadata: tuple[dict[str, Any], ...]

    @property
    def count(self) -> int:
        return int(self.targets.shape[0])


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("entries", payload.get("labels", []))
    if not isinstance(payload, list):
        raise ValueError(f"labels must be a list: {path}")
    return [dict(row) for row in payload]


def _rows_for_split(path: Path, split: str, *, require_all: bool) -> list[dict[str, Any]]:
    rows = _load_rows(path)
    if require_all and any(str(row.get("split", "")) != split for row in rows):
        raise ValueError(f"{path} must contain {split} rows only")
    selected = [row for row in rows if str(row.get("split", "")) == split]
    if not selected:
        raise ValueError(f"{path} contains no {split} rows")
    return selected


def _make_examples(rows: Sequence[Mapping[str, Any]], *, actor_sha256: str) -> ValueExamples:
    observations: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []
    seen: dict[str, tuple[float, np.ndarray]] = {}
    observation_size: int | None = None

    for row in rows:
        if str(row.get("expert_actor_sha256", "")) != actor_sha256:
            raise ValueError("V_up labels use a different frozen expert actor")
        branch_count = int(row.get("branch_count", 0))
        success_count = int(row.get("success_count", -1))
        if branch_count <= 0 or success_count not in (0, branch_count):
            raise ValueError("first-pass V_up expects closed deterministic binary labels")
        target = float(success_count / branch_count)
        obs = np.asarray(row.get("actor_observation", ()), dtype=np.float32).reshape(-1)
        if obs.size == 0 or not np.all(np.isfinite(obs)):
            raise ValueError("V_up actor_observation must be finite and non-empty")
        if observation_size is None:
            observation_size = int(obs.size)
        elif int(obs.size) != observation_size:
            raise ValueError("V_up actor_observation sizes differ")
        state_hash = str(row.get("state_sha256", ""))
        if not state_hash:
            raise ValueError("V_up label is missing state_sha256")
        if state_hash in seen:
            previous_target, previous_obs = seen[state_hash]
            if previous_target != target or not np.array_equal(previous_obs, obs):
                raise ValueError("same physical state has conflicting V_up supervision")
            continue
        seen[state_hash] = (target, obs.copy())
        observations.append(obs)
        targets.append(target)
        metadata.append(
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "state_sha256": state_hash,
                "parent_group_id": str(row.get("parent_group_id", "")),
                "seed": int(row.get("seed", -1)),
                "role": str(row.get("role", "")),
                "split": str(row.get("split", "")),
            }
        )

    if not observations:
        raise ValueError("V_up dataset is empty after physical-state deduplication")
    y = np.asarray(targets, dtype=np.float32)
    if not ({0.0, 1.0} <= set(float(x) for x in np.unique(y))):
        raise ValueError("V_up training data must contain both success and failure")
    return ValueExamples(np.stack(observations).astype(np.float32), y, tuple(metadata))


def build_upstream_value_datasets(
    nominal_labels: Path,
    boundary_train_labels: Path,
    boundary_validation_labels: Path,
    lock_path: Path,
) -> tuple[ValueExamples, ValueExamples, dict[str, Any]]:
    """Build TRAIN/validation arrays while never selecting TEST rows."""
    lock = load_boundary_lock(lock_path)
    actor_sha = str(lock["frozen_pi_up_actor_sha256"])

    nominal_train = _rows_for_split(nominal_labels, "train", require_all=False)
    nominal_validation = _rows_for_split(nominal_labels, "validation", require_all=False)
    boundary_train = _rows_for_split(boundary_train_labels, "train", require_all=True)
    boundary_validation = _rows_for_split(boundary_validation_labels, "validation", require_all=True)

    if any(str(row.get("boundary_protocol_sha256", "")) != str(lock["train_protocol_sha256"]) for row in boundary_train):
        raise ValueError("boundary TRAIN labels do not match the locked TRAIN protocol")
    if any(str(row.get("lock_sha256", "")) != str(lock["lock_sha256"]) for row in boundary_validation):
        raise ValueError("boundary validation labels do not match the locked protocol")

    train = _make_examples([*nominal_train, *boundary_train], actor_sha256=actor_sha)
    validation = _make_examples([*nominal_validation, *boundary_validation], actor_sha256=actor_sha)
    if train.observations.shape[1] != validation.observations.shape[1]:
        raise ValueError("TRAIN and validation observation widths differ")

    provenance = {
        "lock_sha256": lock["lock_sha256"],
        "expert_actor_sha256": actor_sha,
        "nominal_labels_sha256": file_sha256(nominal_labels),
        "boundary_train_labels_sha256": file_sha256(boundary_train_labels),
        "boundary_validation_labels_sha256": file_sha256(boundary_validation_labels),
        "train_count": train.count,
        "validation_count": validation.count,
        "observation_size": int(train.observations.shape[1]),
        "test_data_used": False,
    }
    return train, validation, provenance


def binary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("binary metric arrays must be same-shape and non-empty")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite in [0,1]")
    eps = 1e-7
    clipped = np.clip(p, eps, 1.0 - eps)
    prediction = (p >= 0.5).astype(np.float64)
    positives = y == 1.0
    negatives = y == 0.0
    tp_rate = float(np.mean(prediction[positives] == 1.0)) if np.any(positives) else None
    tn_rate = float(np.mean(prediction[negatives] == 0.0)) if np.any(negatives) else None
    balanced = ((tp_rate + tn_rate) / 2.0) if tp_rate is not None and tn_rate is not None else None

    auc = None
    if np.any(positives) and np.any(negatives):
        pos_scores = p[positives][:, None]
        neg_scores = p[negatives][None, :]
        auc = float(np.mean((pos_scores > neg_scores) + 0.5 * (pos_scores == neg_scores)))

    ece = 0.0
    for lower in np.linspace(0.0, 0.8, 5):
        upper = lower + 0.2
        mask = (p >= lower) & ((p < upper) if upper < 1.0 else (p <= upper))
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))

    positive_mean = float(np.mean(p[positives])) if np.any(positives) else None
    negative_mean = float(np.mean(p[negatives])) if np.any(negatives) else None
    return {
        "count": int(y.size),
        "positive_count": int(np.sum(positives)),
        "negative_count": int(np.sum(negatives)),
        "prevalence": float(np.mean(y)),
        "bce": float(np.mean(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy_at_0_5": float(np.mean(prediction == y)),
        "balanced_accuracy_at_0_5": balanced,
        "roc_auc": auc,
        "ece_5bin": float(ece),
        "positive_mean_score": positive_mean,
        "negative_mean_score": negative_mean,
        "positive_negative_score_gap": (positive_mean - negative_mean) if positive_mean is not None and negative_mean is not None else None,
    }


def _normalize(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0).astype(np.float32)
    std = np.std(train, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (
        np.clip((train - mean) / std, -10.0, 10.0).astype(np.float32),
        np.clip((validation - mean) / std, -10.0, 10.0).astype(np.float32),
        mean,
        std,
    )


def _probabilities(model: ContinuationValueMLP, params, observations: np.ndarray) -> np.ndarray:
    logits = model.apply({"params": params}, jnp.asarray(observations, dtype=jnp.float32))
    return np.asarray(jax.device_get(jax.nn.sigmoid(logits)), dtype=np.float64)


def train_upstream_value_model(
    nominal_labels: Path,
    boundary_train_labels: Path,
    boundary_validation_labels: Path,
    lock_path: Path,
    output_dir: Path,
    *,
    hidden_sizes: Sequence[int] = (64, 64),
    steps: int = 2000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 830001,
) -> dict[str, Any]:
    """Train a fixed-schedule first-pass V_up; validation never changes the boundary protocol."""
    hidden = tuple(int(width) for width in hidden_sizes)
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden_sizes must be positive")
    if steps <= 0 or not (learning_rate > 0.0) or weight_decay < 0.0:
        raise ValueError("invalid V_up optimization hyperparameters")

    train, validation, provenance = build_upstream_value_datasets(
        nominal_labels, boundary_train_labels, boundary_validation_labels, lock_path
    )
    x_train, x_validation, mean, std = _normalize(train.observations, validation.observations)
    y_train = jnp.asarray(train.targets, dtype=jnp.float32)
    jax_train = jnp.asarray(x_train, dtype=jnp.float32)

    model = ContinuationValueMLP(hidden_sizes=hidden)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, x_train.shape[1]), dtype=jnp.float32))["params"]
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    def loss_fn(current_params):
        logits = model.apply({"params": current_params}, jax_train)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y_train))

    @jax.jit
    def update(current_params, current_opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        return new_params, new_opt_state, loss

    initial_loss = float(jax.device_get(loss_fn(params)))
    final_loss = initial_loss
    for _ in range(int(steps)):
        params, opt_state, loss = update(params, opt_state)
        final_loss = float(jax.device_get(loss))
    if not math.isfinite(final_loss):
        raise ValueError("V_up optimizer produced nonfinite loss")

    train_prob = _probabilities(model, params, x_train)
    validation_prob = _probabilities(model, params, x_validation)
    metrics = {
        "train": binary_metrics(train.targets, train_prob),
        "validation": binary_metrics(validation.targets, validation_prob),
        "initial_train_bce": initial_loss,
        "final_optimizer_train_bce": final_loss,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    params_path = output_dir / "params.msgpack"
    params_path.write_bytes(serialization.to_bytes(params))
    normalization_path = output_dir / "normalization.npz"
    np.savez(normalization_path, mean=mean, std=std)

    predictions = []
    for meta, target, probability in zip(validation.metadata, validation.targets, validation_prob):
        predictions.append({**meta, "target": float(target), "probability": float(probability)})
    (output_dir / "validation_predictions.json").write_text(
        json.dumps(predictions, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    manifest = {
        "schema": VALUE_MODEL_SCHEMA,
        "status": "completed",
        "target": "V_up",
        "input": "actor_observation",
        "target_semantics": "probability that frozen pi_up_star reaches Apex before retained failure",
        "hidden_sizes": list(hidden),
        "activation": "tanh",
        "optimizer": "adamw_full_batch_fixed_schedule",
        "optimizer_steps": int(steps),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        **provenance,
        "params_sha256": file_sha256(params_path),
        "normalization_sha256": file_sha256(normalization_path),
        "environment_interactions": 0,
        "training_transitions": 0,
        "test_data_used": False,
        "model_selection_note": "first-pass fixed schedule; validation is reported, not used to retune boundary collection",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics}


def score_upstream_value(model_dir: Path, observations: np.ndarray) -> np.ndarray:
    """Load a saved first-pass V_up model and return continuation scores in [0,1]."""
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != VALUE_MODEL_SCHEMA:
        raise ValueError("unsupported V_up model manifest")
    hidden = tuple(int(x) for x in manifest["hidden_sizes"])
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.ndim != 2 or obs.shape[1] != int(manifest["observation_size"]):
        raise ValueError("V_up inference observation shape mismatch")
    norm = np.load(model_dir / "normalization.npz")
    mean, std = np.asarray(norm["mean"], np.float32), np.asarray(norm["std"], np.float32)
    normalized = np.clip((obs - mean) / std, -10.0, 10.0).astype(np.float32)
    model = ContinuationValueMLP(hidden_sizes=hidden)
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((1, obs.shape[1]), dtype=jnp.float32))["params"]
    params = serialization.from_bytes(template, (model_dir / "params.msgpack").read_bytes())
    return _probabilities(model, params, normalized)
