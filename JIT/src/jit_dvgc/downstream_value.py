"""Small supervised V_down model from frozen-expert continuation labels."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .expert_freeze import load_frozen_manifest
from .upstream_boundary import canonical_sha256, file_sha256
from .upstream_value import ContinuationValueMLP, ValueExamples, binary_metrics

VALUE_MODEL_SCHEMA = "jit_downstream_value_model_v1"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("entries", payload.get("labels", []))
    if not isinstance(payload, list):
        raise ValueError(f"labels must be a list: {path}")
    rows = [dict(row) for row in payload]
    if not rows:
        raise ValueError("V_down labels are empty")
    return rows


def _make_examples(rows: Sequence[Mapping[str, Any]], *, actor_sha256: str) -> ValueExamples:
    observations: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []
    seen: dict[str, tuple[float, np.ndarray]] = {}
    observation_size: int | None = None

    for row in rows:
        if str(row.get("expert_actor_sha256", "")) != actor_sha256:
            raise ValueError("V_down labels use a different frozen expert actor")
        branch_count = int(row.get("branch_count", 0))
        success_count = int(row.get("success_count", -1))
        if branch_count <= 0 or success_count not in (0, branch_count):
            raise ValueError("first-pass V_down expects closed deterministic binary labels")
        target = float(success_count / branch_count)
        obs = np.asarray(row.get("actor_observation", ()), dtype=np.float32).reshape(-1)
        if obs.size == 0 or not np.all(np.isfinite(obs)):
            raise ValueError("V_down actor_observation must be finite and non-empty")
        if observation_size is None:
            observation_size = int(obs.size)
        elif int(obs.size) != observation_size:
            raise ValueError("V_down actor_observation sizes differ")
        state_hash = str(row.get("state_sha256", ""))
        if not state_hash:
            raise ValueError("V_down label is missing state_sha256")
        if state_hash in seen:
            previous_target, previous_obs = seen[state_hash]
            if previous_target != target or not np.array_equal(previous_obs, obs):
                raise ValueError("same physical state has conflicting V_down supervision")
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
                "source_bank": str(row.get("source_bank", "")),
            }
        )

    if not observations:
        raise ValueError("V_down dataset is empty after physical-state deduplication")
    targets_array = np.asarray(targets, dtype=np.float32)
    if set(float(x) for x in np.unique(targets_array)) != {0.0, 1.0}:
        raise ValueError("V_down split must contain both success and failure")
    return ValueExamples(
        np.stack(observations).astype(np.float32),
        targets_array,
        tuple(metadata),
    )


def build_downstream_value_datasets(
    labels_path: Path,
    frozen_manifest: Path,
) -> tuple[ValueExamples, ValueExamples, dict[str, Any]]:
    """Build TRAIN/validation V_down arrays while never selecting TEST rows."""
    frozen = load_frozen_manifest(frozen_manifest)
    down_record = frozen["experts"]["pi_down_star"]
    actor_sha = str(down_record["actor_sha256"])
    rows = _load_rows(labels_path)

    protocol_hashes = {str(row.get("protocol_sha256", "")) for row in rows}
    if "" in protocol_hashes or len(protocol_hashes) != 1:
        raise ValueError("V_down labels must share one continuation protocol")
    train_rows = [row for row in rows if str(row.get("split", "")) == "train"]
    validation_rows = [row for row in rows if str(row.get("split", "")) == "validation"]
    test_rows = [row for row in rows if str(row.get("split", "")) == "test"]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("V_down labels must contain train/validation/test splits")

    train = _make_examples(train_rows, actor_sha256=actor_sha)
    validation = _make_examples(validation_rows, actor_sha256=actor_sha)
    if train.observations.shape[1] != validation.observations.shape[1]:
        raise ValueError("V_down TRAIN and validation observation widths differ")
    train_hashes = {row["state_sha256"] for row in train.metadata}
    validation_hashes = {row["state_sha256"] for row in validation.metadata}
    if train_hashes.intersection(validation_hashes):
        raise ValueError("V_down physical states overlap TRAIN and validation")

    provenance = {
        "expert_actor_sha256": actor_sha,
        "expert_payload_sha256": str(down_record["payload_sha256"]),
        "expert_config_sha256": str(down_record["config_sha256"]),
        "xml_sha256": str(down_record["xml_sha256"]),
        "labels_sha256": file_sha256(labels_path),
        "continuation_protocol_sha256": next(iter(protocol_hashes)),
        "train_count": train.count,
        "validation_count": validation.count,
        "declared_test_count": len(test_rows),
        "observation_size": int(train.observations.shape[1]),
        "test_data_used": False,
    }
    return train, validation, provenance


def _normalize(train: np.ndarray, validation: np.ndarray):
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


def train_downstream_value_model(
    labels_path: Path,
    frozen_manifest: Path,
    output_dir: Path,
    *,
    hidden_sizes: Sequence[int] = (64, 64),
    steps: int = 2000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 830002,
) -> dict[str, Any]:
    """Train fixed-schedule first-pass V_down without using TEST data."""
    hidden = tuple(int(width) for width in hidden_sizes)
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden_sizes must be positive")
    if steps <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid V_down optimization hyperparameters")

    train, validation, provenance = build_downstream_value_datasets(
        labels_path, frozen_manifest
    )
    x_train, x_validation, mean, std = _normalize(
        train.observations, validation.observations
    )
    y_train = jnp.asarray(train.targets, dtype=jnp.float32)
    jax_train = jnp.asarray(x_train, dtype=jnp.float32)

    model = ContinuationValueMLP(hidden_sizes=hidden)
    params = model.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, x_train.shape[1]), dtype=jnp.float32),
    )["params"]
    optimizer = optax.adamw(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    opt_state = optimizer.init(params)

    def loss_fn(current_params):
        logits = model.apply({"params": current_params}, jax_train)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y_train))

    @jax.jit
    def update(current_params, current_opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, new_opt_state = optimizer.update(
            grads, current_opt_state, current_params
        )
        new_params = optax.apply_updates(current_params, updates)
        return new_params, new_opt_state, loss

    initial_loss = float(jax.device_get(loss_fn(params)))
    final_loss = initial_loss
    for _ in range(int(steps)):
        params, opt_state, loss = update(params, opt_state)
        final_loss = float(jax.device_get(loss))
    if not math.isfinite(final_loss):
        raise ValueError("V_down optimizer produced nonfinite loss")

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

    predictions = [
        {**meta, "target": float(target), "probability": float(probability)}
        for meta, target, probability in zip(
            validation.metadata, validation.targets, validation_prob
        )
    ]
    (output_dir / "validation_predictions.json").write_text(
        json.dumps(predictions, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": VALUE_MODEL_SCHEMA,
        "status": "completed",
        "target": "V_down",
        "input": "actor_observation",
        "target_semantics": "probability that frozen pi_down_star completes valid landing and short recovery",
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
        "model_selection_note": "first-pass fixed schedule; validation is reported and TEST remains untouched",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest, "metrics": metrics}
