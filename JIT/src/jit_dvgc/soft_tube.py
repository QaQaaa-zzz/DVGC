"""TRAIN-only learned Soft Tube construction from frozen phase value models."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from .handoff_snapshot import load_snapshot
from .upstream_boundary import canonical_sha256, file_sha256, physical_state_sha256
from .upstream_value import ContinuationValueMLP


SOFT_TUBE_SCHEMA = "jit_soft_tube_v1"
WEIGHT_FLOOR = 0.05
WEIGHT_SCALE = 0.95
PHASE_MIXTURE = {"upstream": 0.5, "downstream": 0.5}


@dataclass(frozen=True)
class SoftTubeInputs:
    frozen_experts: Path
    up_model_dir: Path
    down_model_dir: Path
    up_nominal_labels: Path
    up_nominal_catalog: Path
    up_nominal_protocol: Path
    up_boundary_labels: Path
    up_boundary_catalog: Path
    up_boundary_protocol: Path
    down_labels: Path
    down_catalog: Path
    down_protocol: Path


@dataclass(frozen=True)
class SoftTubeArtifact:
    root: Path
    manifest: Mapping[str, Any]
    entries: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class _Source:
    name: str
    phase: str
    labels: Path
    catalog: Path
    protocol: Path
    protocol_field: str


ScoreFunction = Callable[[Path, np.ndarray], np.ndarray]


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_manifest_sha(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return canonical_sha256(payload)


def _require_hash(actual: str, expected: Any, message: str) -> None:
    if actual != str(expected):
        raise ValueError(message)


def _load_frozen(path: Path) -> tuple[dict[str, Any], str]:
    frozen = _read_json(path)
    if frozen.get("schema") != "jit_frozen_phase_experts_v1":
        raise ValueError("unsupported frozen expert manifest")
    if frozen.get("status") != "frozen":
        raise ValueError("phase experts are not frozen")
    experts = frozen.get("experts", {})
    if set(experts) != {"pi_up_star", "pi_down_star"}:
        raise ValueError("frozen manifest must contain exactly pi_up_star and pi_down_star")
    up, down = experts["pi_up_star"], experts["pi_down_star"]
    if up.get("xml_sha256") != down.get("xml_sha256"):
        raise ValueError("frozen expert XML identity mismatch")
    return frozen, file_sha256(path)


def _load_model(
    model_dir: Path, *, target: str, actor_sha256: str
) -> dict[str, Any]:
    root = Path(model_dir)
    manifest = _read_json(root / "manifest.json")
    expected_schema = {
        "V_up": "jit_upstream_value_model_v1",
        "V_down": "jit_downstream_value_model_v1",
    }[target]
    if manifest.get("schema") != expected_schema or manifest.get("target") != target:
        raise ValueError(f"unsupported {target} model manifest")
    if manifest.get("status") != "completed":
        raise ValueError(f"{target} model is not completed")
    if manifest.get("test_data_used") is not False:
        raise ValueError(f"{target} manifest does not prove TEST exclusion")
    if manifest.get("expert_actor_sha256") != actor_sha256:
        raise ValueError(f"{target} frozen expert actor mismatch")
    _require_hash(
        _canonical_manifest_sha(manifest),
        manifest.get("manifest_sha256"),
        f"{target} manifest SHA-256 mismatch",
    )
    _require_hash(
        file_sha256(root / "params.msgpack"),
        manifest.get("params_sha256"),
        f"{target} params SHA-256 mismatch",
    )
    _require_hash(
        file_sha256(root / "normalization.npz"),
        manifest.get("normalization_sha256"),
        f"{target} normalization SHA-256 mismatch",
    )
    return manifest


def _default_score(model_dir: Path, observations: np.ndarray) -> np.ndarray:
    root = Path(model_dir)
    manifest = _read_json(root / "manifest.json")
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != int(manifest["observation_size"]):
        raise ValueError(f"{manifest['target']} inference observation shape mismatch")
    with np.load(root / "normalization.npz") as normalization:
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        std = np.asarray(normalization["std"], dtype=np.float32)
    normalized = np.clip((obs - mean) / std, -10.0, 10.0).astype(np.float32)
    hidden = tuple(int(width) for width in manifest["hidden_sizes"])
    model = ContinuationValueMLP(hidden_sizes=hidden)
    template = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, obs.shape[1]), dtype=jnp.float32),
    )["params"]
    params = serialization.from_bytes(
        template, (root / "params.msgpack").read_bytes()
    )
    logits = model.apply({"params": params}, jnp.asarray(normalized))
    return np.asarray(jax.device_get(jax.nn.sigmoid(logits)), dtype=np.float64)


def _rows(payload: Any, *, context: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("entries", payload.get("labels", []))
    if not isinstance(payload, list):
        raise ValueError(f"{context} rows must be a list")
    return [dict(row) for row in payload]


def _train_rows(path: Path) -> list[dict[str, Any]]:
    # Split is the only field consulted before non-TRAIN rows are discarded.
    selected = [
        row
        for row in _rows(_read_json(path), context=str(path))
        if str(row.get("split", "")) == "train"
    ]
    if not selected:
        raise ValueError(f"{path} has no TRAIN rows")
    return selected


def _catalog_index(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    payload = _read_json(path)
    entries = _rows(payload, context=str(path))
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in entries:
        key = (
            str(row.get("state_sha256", "")),
            str(row.get("source_bank", "")),
            str(row.get("snapshot", "")),
        )
        if not all(key) or key in index:
            raise ValueError(f"{path} has an invalid or duplicate catalog identity")
        index[key] = row
    return payload, index


def _validate_protocol(
    source: _Source,
    *,
    frozen_sha256: str,
    actor_sha256: str,
    xml_sha256: str,
) -> tuple[dict[str, Any], str]:
    protocol = _read_json(source.protocol)
    if protocol.get("frozen_manifest_sha256") != frozen_sha256:
        raise ValueError(f"{source.name} frozen manifest identity mismatch")
    actor_field = (
        "frozen_pi_up_actor_sha256"
        if source.name == "up_boundary"
        else "expert_actor_sha256"
    )
    if protocol.get(actor_field) != actor_sha256:
        raise ValueError(f"{source.name} expert actor identity mismatch")
    if protocol.get("xml_sha256") != xml_sha256:
        raise ValueError(f"{source.name} XML identity mismatch")
    if source.name != "up_boundary":
        _require_hash(
            file_sha256(source.catalog),
            protocol.get("catalog_sha256"),
            f"{source.name} catalog SHA-256 mismatch",
        )
    elif protocol.get("split") != "train":
        raise ValueError("up_boundary protocol is not TRAIN-only")
    train_seeds = tuple(int(seed) for seed in protocol.get("train_seeds", ()))
    if not train_seeds:
        raise ValueError(f"{source.name} protocol has no TRAIN seed allowlist")
    return protocol, file_sha256(source.protocol)


def _selected_entries(
    source: _Source,
    *,
    protocol: Mapping[str, Any],
    actor_sha256: str,
    xml_sha256: str,
    observation_size: int,
) -> list[dict[str, Any]]:
    catalog, index = _catalog_index(source.catalog)
    if source.name == "up_nominal":
        if catalog.get("frozen_pi_up_actor_sha256") != actor_sha256:
            raise ValueError("up_nominal catalog actor identity mismatch")
        if catalog.get("xml_sha256") != xml_sha256:
            raise ValueError("up_nominal catalog XML identity mismatch")
    if source.name == "up_boundary":
        if catalog.get("split") != "train":
            raise ValueError("up_boundary catalog is not TRAIN-only")
        if catalog.get("protocol_sha256") != protocol.get("protocol_sha256"):
            raise ValueError("up_boundary catalog protocol mismatch")

    allowed_seeds = {int(seed) for seed in protocol["train_seeds"]}
    selected: list[dict[str, Any]] = []
    for label in _train_rows(source.labels):
        if int(label.get("seed", -1)) not in allowed_seeds:
            raise ValueError(f"{source.name} TRAIN row uses a non-TRAIN seed")
        if label.get("expert_actor_sha256") != actor_sha256:
            raise ValueError(f"{source.name} label expert actor mismatch")
        if source.name == "up_boundary":
            if label.get("boundary_protocol_sha256") != protocol.get("protocol_sha256"):
                raise ValueError("up_boundary label protocol mismatch")
        elif label.get("protocol_sha256") != protocol.get("protocol_sha256"):
            raise ValueError(f"{source.name} label protocol mismatch")

        key = (
            str(label.get("state_sha256", "")),
            str(label.get("source_bank", "")),
            str(label.get("snapshot", "")),
        )
        catalog_row = index.get(key)
        if catalog_row is None:
            raise ValueError(f"{source.name} TRAIN label has no exact catalog row")
        for field in ("seed", "parent_group_id", "role", "tick"):
            if catalog_row.get(field) != label.get(field):
                raise ValueError(f"{source.name} catalog/label {field} mismatch")

        snapshot_path = source.catalog.parent / key[1] / key[2]
        if not snapshot_path.is_dir():
            raise FileNotFoundError(f"selected TRAIN snapshot is missing: {snapshot_path}")
        snapshot = load_snapshot(snapshot_path)
        if snapshot.xml_sha256 != xml_sha256:
            raise ValueError(f"{source.name} snapshot XML identity mismatch")
        if physical_state_sha256(snapshot) != key[0]:
            raise ValueError(f"{source.name} snapshot physical-state SHA-256 mismatch")

        observation = np.asarray(label.get("actor_observation", ()), dtype=np.float32).reshape(-1)
        if observation.shape != (observation_size,) or not np.all(np.isfinite(observation)):
            raise ValueError(f"{source.name} TRAIN actor observation is invalid")
        branch_count = int(label.get("branch_count", 0))
        success_count = int(label.get("success_count", -1))
        if branch_count <= 0 or success_count not in (0, branch_count):
            raise ValueError(f"{source.name} TRAIN label is not closed binary supervision")

        selected.append(
            {
                "phase": source.phase,
                "split": "train",
                "candidate_id": str(label.get("candidate_id", "")),
                "snapshot": str(snapshot_path.resolve()),
                "source_bank": key[1],
                "state_sha256": key[0],
                "parent_group_id": str(label["parent_group_id"]),
                "seed": int(label["seed"]),
                "role": str(label["role"]),
                "tick": int(label["tick"]),
                "source_checkpoint": catalog_row.get(
                    "source_checkpoint", catalog_row.get("anchor_source_checkpoint")
                ),
                "source_training_transitions": catalog_row.get(
                    "source_training_transitions",
                    catalog_row.get("anchor_source_training_transitions"),
                ),
                "label_target": float(success_count / branch_count),
                "actor_observation": observation,
                "source_name": source.name,
                "source_catalog": str(source.catalog.resolve()),
                "source_catalog_sha256": file_sha256(source.catalog),
                "source_labels": str(source.labels.resolve()),
                "source_labels_sha256": file_sha256(source.labels),
                "source_protocol": str(source.protocol.resolve()),
                "source_protocol_sha256": file_sha256(source.protocol),
                "continuation_protocol_sha256": str(label.get("protocol_sha256", "")),
            }
        )
    return selected


def _deduplicate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_hash = row["state_sha256"]
        previous = seen.get(state_hash)
        if previous is None:
            seen[state_hash] = row
            result.append(row)
            continue
        if previous["phase"] != row["phase"]:
            raise ValueError(f"physical state {state_hash} has conflicting phase assignments")
        comparable = (
            "snapshot",
            "source_bank",
            "parent_group_id",
            "seed",
            "role",
            "tick",
            "label_target",
        )
        if any(previous[field] != row[field] for field in comparable) or not np.array_equal(
            previous["actor_observation"], row["actor_observation"]
        ):
            raise ValueError(f"physical state {state_hash} has conflicting provenance")
    return result


def _score_phase(
    rows: list[dict[str, Any]], model_dir: Path, scorer: ScoreFunction
) -> None:
    if not rows:
        raise ValueError("Soft Tube phase has no TRAIN support")
    observations = np.stack([row["actor_observation"] for row in rows])
    scores = np.asarray(scorer(Path(model_dir), observations), dtype=np.float64).reshape(-1)
    if scores.shape != (len(rows),) or not np.all(np.isfinite(scores)):
        raise ValueError("value scorer returned an invalid shape or nonfinite value")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("value scorer returned a score outside [0,1]")
    for row, score in zip(rows, scores):
        row["value_score"] = float(score)
        row["sampling_weight"] = float(WEIGHT_FLOOR + WEIGHT_SCALE * score)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(value)
        for name, value in zip(
            ("min", "q25", "median", "q75", "max"),
            np.quantile(array, (0.0, 0.25, 0.5, 0.75, 1.0)),
        )
    }


def _diagnostics(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, Any] = {}
    for phase in ("upstream", "downstream"):
        rows = [row for row in entries if row["phase"] == phase]
        by_phase[phase] = {
            "count": len(rows),
            "score_quantiles": _quantiles([row["value_score"] for row in rows]),
            "weight_quantiles": _quantiles([row["sampling_weight"] for row in rows]),
            "label_distribution": dict(
                sorted(Counter(str(int(row["label_target"])) for row in rows).items())
            ),
            "source_bank_distribution": dict(
                sorted(Counter(row["source_bank"] for row in rows).items())
            ),
            "role_distribution": dict(sorted(Counter(row["role"] for row in rows).items())),
            "near_boundary": [
                {
                    "state_sha256": row["state_sha256"],
                    "role": row["role"],
                    "value_score": row["value_score"],
                    "sampling_weight": row["sampling_weight"],
                }
                for row in sorted(rows, key=lambda item: abs(item["value_score"] - 0.5))[:10]
            ],
        }
    return {"total_count": len(entries), "by_phase": by_phase, "test_data_used": False}


def _serializable_entry(
    row: Mapping[str, Any], *, model_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            **row,
            "actor_observation": None,
            "value_model_manifest_sha256": model_manifest["manifest_sha256"],
            "value_model_params_sha256": model_manifest["params_sha256"],
            "value_model_normalization_sha256": model_manifest["normalization_sha256"],
            "value_model_target": model_manifest["target"],
        }.items()
        if value is not None and key != "actor_observation"
    }


def build_soft_tube(
    inputs: SoftTubeInputs,
    output_dir: Path,
    *,
    score_up: ScoreFunction | None = None,
    score_down: ScoreFunction | None = None,
) -> SoftTubeArtifact:
    """Build one provenance-complete learned Soft Tube from TRAIN snapshots."""
    inputs = SoftTubeInputs(**{name: Path(getattr(inputs, name)) for name in inputs.__dataclass_fields__})
    frozen, frozen_sha = _load_frozen(inputs.frozen_experts)
    up_record = frozen["experts"]["pi_up_star"]
    down_record = frozen["experts"]["pi_down_star"]
    xml_sha = str(up_record["xml_sha256"])
    up_model = _load_model(
        inputs.up_model_dir, target="V_up", actor_sha256=up_record["actor_sha256"]
    )
    down_model = _load_model(
        inputs.down_model_dir,
        target="V_down",
        actor_sha256=down_record["actor_sha256"],
    )
    _require_hash(
        file_sha256(inputs.up_nominal_labels),
        up_model.get("nominal_labels_sha256"),
        "V_up nominal labels SHA-256 mismatch",
    )
    _require_hash(
        file_sha256(inputs.up_boundary_labels),
        up_model.get("boundary_train_labels_sha256"),
        "V_up boundary TRAIN labels SHA-256 mismatch",
    )
    _require_hash(
        file_sha256(inputs.down_labels),
        down_model.get("labels_sha256"),
        "V_down labels SHA-256 mismatch",
    )
    if down_model.get("xml_sha256") not in (None, xml_sha):
        raise ValueError("V_down model XML identity mismatch")

    sources = (
        _Source(
            "up_nominal",
            "upstream",
            inputs.up_nominal_labels,
            inputs.up_nominal_catalog,
            inputs.up_nominal_protocol,
            "protocol_sha256",
        ),
        _Source(
            "up_boundary",
            "upstream",
            inputs.up_boundary_labels,
            inputs.up_boundary_catalog,
            inputs.up_boundary_protocol,
            "boundary_protocol_sha256",
        ),
        _Source(
            "down_nominal",
            "downstream",
            inputs.down_labels,
            inputs.down_catalog,
            inputs.down_protocol,
            "protocol_sha256",
        ),
    )
    protocols: dict[str, tuple[dict[str, Any], str]] = {}
    for source in sources:
        actor_sha = (
            up_record["actor_sha256"]
            if source.phase == "upstream"
            else down_record["actor_sha256"]
        )
        protocols[source.name] = _validate_protocol(
            source,
            frozen_sha256=frozen_sha,
            actor_sha256=actor_sha,
            xml_sha256=xml_sha,
        )
    if protocols["up_boundary"][0].get("nominal_catalog_sha256") != file_sha256(
        inputs.up_nominal_catalog
    ):
        raise ValueError("up_boundary protocol nominal catalog mismatch")
    if protocols["up_boundary"][0].get("nominal_labels_sha256") != file_sha256(
        inputs.up_nominal_labels
    ):
        raise ValueError("up_boundary protocol nominal labels mismatch")
    if down_model.get("continuation_protocol_sha256") != protocols["down_nominal"][0].get(
        "protocol_sha256"
    ):
        raise ValueError("V_down continuation protocol mismatch")

    raw_rows: list[dict[str, Any]] = []
    for source in sources:
        raw_rows.extend(
            _selected_entries(
                source,
                protocol=protocols[source.name][0],
                actor_sha256=(
                    up_record["actor_sha256"]
                    if source.phase == "upstream"
                    else down_record["actor_sha256"]
                ),
                xml_sha256=xml_sha,
                observation_size=int(
                    up_model["observation_size"]
                    if source.phase == "upstream"
                    else down_model["observation_size"]
                ),
            )
        )
    rows = _deduplicate(raw_rows)
    up_rows = [row for row in rows if row["phase"] == "upstream"]
    down_rows = [row for row in rows if row["phase"] == "downstream"]
    _score_phase(up_rows, inputs.up_model_dir, score_up or _default_score)
    _score_phase(down_rows, inputs.down_model_dir, score_down or _default_score)
    serializable = tuple(
        _serializable_entry(
            row, model_manifest=up_model if row["phase"] == "upstream" else down_model
        )
        for row in [*up_rows, *down_rows]
    )
    diagnostics = _diagnostics(serializable)
    manifest = {
        "schema": SOFT_TUBE_SCHEMA,
        "status": "completed",
        "artifact_role": "learned_soft_feasibility_tube",
        "certified_safe": False,
        "training_guidance_only": True,
        "test_data_used": False,
        "validation_data_used": False,
        "training_transitions": 0,
        "environment_interactions": 0,
        "xml_sha256": xml_sha,
        "frozen_experts_sha256": frozen_sha,
        "pi_up_actor_sha256": up_record["actor_sha256"],
        "pi_down_actor_sha256": down_record["actor_sha256"],
        "value_models": {
            "V_up": {
                key: up_model[key]
                for key in ("manifest_sha256", "params_sha256", "normalization_sha256")
            },
            "V_down": {
                key: down_model[key]
                for key in ("manifest_sha256", "params_sha256", "normalization_sha256")
            },
        },
        "source_hashes": {
            source.name: {
                "catalog_sha256": file_sha256(source.catalog),
                "labels_sha256": file_sha256(source.labels),
                "protocol_sha256": file_sha256(source.protocol),
                "declared_protocol_sha256": protocols[source.name][0].get(
                    "protocol_sha256"
                ),
            }
            for source in sources
        },
        "weighting": {
            "mapping": "sampling_weight = 0.05 + 0.95 * value_score",
            "floor": WEIGHT_FLOOR,
            "scale": WEIGHT_SCALE,
            "monotonic": True,
            "nonzero_support": True,
            "validation_tuned": False,
        },
        "phase_mixture": PHASE_MIXTURE,
        "entry_count": len(serializable),
        "upstream_count": len(up_rows),
        "downstream_count": len(down_rows),
    }
    manifest["entries_sha256"] = canonical_sha256({"entries": list(serializable)})
    manifest["diagnostics_sha256"] = canonical_sha256(diagnostics)
    manifest["manifest_sha256"] = canonical_sha256(manifest)

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Soft Tube output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, payload in (
            ("entries.json", list(serializable)),
            ("diagnostics.json", diagnostics),
            ("manifest.json", manifest),
        ):
            (temporary / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return SoftTubeArtifact(output, manifest, serializable, diagnostics)


def load_soft_tube(path: Path) -> SoftTubeArtifact:
    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema") != SOFT_TUBE_SCHEMA or manifest.get("status") != "completed":
        raise ValueError("unsupported or incomplete Soft Tube artifact")
    _require_hash(
        _canonical_manifest_sha(manifest),
        manifest.get("manifest_sha256"),
        "Soft Tube manifest SHA-256 mismatch",
    )
    entries = tuple(_rows(_read_json(root / "entries.json"), context="Soft Tube entries"))
    diagnostics = _read_json(root / "diagnostics.json")
    _require_hash(
        canonical_sha256({"entries": list(entries)}),
        manifest.get("entries_sha256"),
        "Soft Tube entries SHA-256 mismatch",
    )
    _require_hash(
        canonical_sha256(diagnostics),
        manifest.get("diagnostics_sha256"),
        "Soft Tube diagnostics SHA-256 mismatch",
    )
    if manifest.get("test_data_used") is not False or manifest.get("validation_data_used") is not False:
        raise ValueError("Soft Tube does not prove split isolation")
    if any(entry.get("split") != "train" for entry in entries):
        raise ValueError("Soft Tube contains a non-TRAIN entry")
    return SoftTubeArtifact(root, manifest, entries, diagnostics)
