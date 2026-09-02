"""Deterministic phase-balanced sampling from a validated learned Soft Tube."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .handoff_snapshot import load_snapshot
from .snapshot_pool import SnapshotPool
from .soft_tube import (
    PHASE_MIXTURE,
    SOFT_TUBE_SCHEMA,
    WEIGHT_FLOOR,
    WEIGHT_SCALE,
    SoftTubeArtifact,
    load_soft_tube,
)

PHASE_UPSTREAM = 0
PHASE_DOWNSTREAM = 1
CORE_REPLAY_SCHEMA = "jit_tube_rsi_core_replay_v1"


def _legacy_physical_state_sha256(snapshot: Any) -> str:
    """Hash legacy qpos/qvel without depending on old upstream acquisition code."""
    digest = hashlib.sha256()
    digest.update(np.asarray(snapshot.qpos).tobytes())
    digest.update(np.asarray(snapshot.qvel).tobytes())
    return digest.hexdigest()


def _snapshot_state_sha(path: Path) -> tuple[str, str, int | None]:
    identity = json.loads((Path(path) / "identity.json").read_text(encoding="utf-8"))
    schema = str(identity.get("schema", ""))
    if schema == "handoff_snapshot_v1":
        snapshot = load_snapshot(path)
        return _legacy_physical_state_sha256(snapshot), schema, None
    if schema == "jit_unified_envelope_snapshot_v1":
        from .unified_envelope_snapshot import (
            load_unified_envelope_snapshot,
            physical_state_sha256 as unified_state_sha256,
        )

        snapshot = load_unified_envelope_snapshot(path)
        return unified_state_sha256(snapshot), schema, int(snapshot.active_phase)
    raise ValueError(f"unsupported Soft Tube snapshot schema: {schema}")


def _validate_score_semantics(entry: Mapping[str, Any], phase: str) -> None:
    target = str(entry.get("value_model_target", ""))
    score_source = entry.get("score_source")
    if score_source is None:
        expected = "V_up" if phase == "upstream" else "V_down"
        if target != expected:
            raise ValueError("Soft Tube legacy core entry uses a cross-phase value model")
        return
    if not isinstance(score_source, Mapping):
        raise ValueError("Soft Tube score_source must be an object")
    if score_source.get("kind") != "policy_conditioned_continuation_field":
        raise ValueError("Soft Tube expansion has unsupported score source")

    field_name = str(score_source.get("field_name", ""))
    expected_prefix = "C_up^" if phase == "upstream" else "C_down^"
    iteration_text = field_name.removeprefix(expected_prefix)
    if (
        target != field_name
        or not field_name.startswith(expected_prefix)
        or not iteration_text.isdigit()
    ):
        raise ValueError("Soft Tube expansion uses a cross-phase or invalid continuation field")

    threshold = float(score_source.get("acceptance_threshold_exclusive", np.nan))
    score = float(entry.get("value_score", np.nan))
    if not np.isfinite(threshold) or not score > threshold:
        raise ValueError("Soft Tube expansion does not satisfy its strict C threshold")
    if (
        score_source.get("selection_rule")
        != "TRAIN_label_positive_and_score_strictly_greater_than_threshold"
    ):
        raise ValueError("Soft Tube expansion selection rule drift")
    if int(entry.get("continuation_label", -1)) != 1:
        raise ValueError("Soft Tube expansion must retain only positive TRAIN labels")


def normalize_core_replay_contract(
    contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the optional source-core replay contract used during Tube-RSI."""
    if contract is None:
        return None
    raw = dict(contract)
    expected_keys = {
        "schema",
        "selection",
        "core_probability",
        "expansion_probability",
        "core_within_source",
        "expansion_within_source",
        "source_core_definition",
    }
    if set(raw) != expected_keys:
        raise ValueError("Tube-RSI core replay fields drift")
    if raw.get("schema") != CORE_REPLAY_SCHEMA:
        raise ValueError("unsupported Tube-RSI core replay schema")
    expected_fixed = {
        "selection": "phase_then_source_then_entry",
        "core_within_source": "uniform",
        "expansion_within_source": "value_weighted",
        "source_core_definition": "first_core_retained_count_entries",
    }
    if any(raw.get(key) != value for key, value in expected_fixed.items()):
        raise ValueError("Tube-RSI core replay semantics drift")
    core = float(raw["core_probability"])
    expansion = float(raw["expansion_probability"])
    if not math.isfinite(core) or not math.isfinite(expansion):
        raise ValueError("Tube-RSI source probabilities must be finite")
    if not (0.0 < core < 1.0) or not (0.0 < expansion < 1.0):
        raise ValueError("Tube-RSI source probabilities must lie strictly inside (0, 1)")
    if not math.isclose(core + expansion, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Tube-RSI source probabilities must sum to one")
    return {
        **raw,
        "core_probability": core,
        "expansion_probability": expansion,
    }


def _partition_entries(
    loaded: SoftTubeArtifact,
    contract: Mapping[str, Any] | None,
) -> dict[str, list[tuple[dict[str, Any], bool]]]:
    normalized = normalize_core_replay_contract(contract)
    core_retained_count = int(loaded.manifest.get("core_retained_count", 0))
    if normalized is not None:
        if core_retained_count <= 0 or core_retained_count >= len(loaded.entries):
            raise ValueError("Tube-RSI core replay requires a nonempty retained core and expansion")
        if int(loaded.manifest.get("source_tube_entry_count", -1)) != core_retained_count:
            raise ValueError("Tube-RSI retained-core partition identity drift")

    by_phase: dict[str, list[tuple[dict[str, Any], bool]]] = {
        "upstream": [],
        "downstream": [],
    }
    for global_index, original in enumerate(loaded.entries):
        entry = dict(original)
        phase = str(entry.get("phase", ""))
        if phase not in by_phase:
            raise ValueError(f"Soft Tube contains unsupported phase: {phase}")
        by_phase[phase].append((entry, global_index < core_retained_count))

    if normalized is not None:
        for phase, rows in by_phase.items():
            core_count = sum(is_core for _, is_core in rows)
            expansion_count = len(rows) - core_count
            if core_count <= 0 or expansion_count <= 0:
                raise ValueError(
                    f"Tube-RSI core replay requires core and expansion support in {phase}"
                )
    return by_phase


def _sampling_logits(
    rows: list[tuple[dict[str, Any], bool]],
    contract: Mapping[str, Any] | None,
) -> jax.Array:
    weights = np.asarray(
        [float(entry["sampling_weight"]) for entry, _ in rows], dtype=np.float64
    )
    normalized = normalize_core_replay_contract(contract)
    if normalized is None:
        return jnp.log(jnp.asarray(weights, dtype=jnp.float32))

    core_mask = np.asarray([is_core for _, is_core in rows], dtype=bool)
    expansion_mask = ~core_mask
    probabilities = np.zeros(len(rows), dtype=np.float64)
    probabilities[core_mask] = float(normalized["core_probability"]) / int(
        np.sum(core_mask)
    )
    expansion_weights = weights[expansion_mask]
    expansion_total = float(np.sum(expansion_weights))
    if not math.isfinite(expansion_total) or expansion_total <= 0.0:
        raise ValueError("Tube-RSI expansion sampling weights are invalid")
    probabilities[expansion_mask] = (
        float(normalized["expansion_probability"])
        * expansion_weights
        / expansion_total
    )
    if not np.isclose(float(np.sum(probabilities)), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("Tube-RSI source-balanced sampling probabilities do not sum to one")
    return jnp.log(jnp.asarray(probabilities, dtype=jnp.float32))


def describe_tube_sampling(
    artifact: SoftTubeArtifact | Path,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the declared Tube sampling distribution without env interactions."""
    loaded = load_soft_tube(artifact) if isinstance(artifact, (str, Path)) else artifact
    normalized = normalize_core_replay_contract(contract)
    by_phase = _partition_entries(loaded, normalized)
    phases: dict[str, Any] = {}
    for phase, rows in by_phase.items():
        logits = np.asarray(jax.device_get(_sampling_logits(rows, normalized)), dtype=np.float64)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= np.sum(probabilities)
        core_mask = np.asarray([is_core for _, is_core in rows], dtype=bool)
        phases[phase] = {
            "entry_count": len(rows),
            "core_count": int(np.sum(core_mask)),
            "expansion_count": int(np.sum(~core_mask)),
            "core_probability": float(np.sum(probabilities[core_mask])),
            "expansion_probability": float(np.sum(probabilities[~core_mask])),
        }
    return {
        "phase_mixture": dict(PHASE_MIXTURE),
        "core_replay_contract": normalized,
        "phases": phases,
    }


def _match_prng_key_representation(
    sampled_key: jax.Array, reference_key: jax.Array
) -> jax.Array:
    """Preserve sampled key data while matching the caller's JAX key representation."""
    key_data = jax.random.key_data(sampled_key)
    if jax.dtypes.issubdtype(reference_key.dtype, jax.dtypes.prng_key):
        return jax.random.wrap_key_data(key_data)
    return key_data


@dataclass(frozen=True)
class TubeRSIPool:
    artifact: SoftTubeArtifact
    snapshot_pool: SnapshotPool
    upstream_weights: jax.Array
    downstream_weights: jax.Array
    upstream_sampling_logits: jax.Array
    downstream_sampling_logits: jax.Array
    upstream_count: int
    downstream_count: int
    upstream_core_count: int
    downstream_core_count: int
    core_replay_contract: Mapping[str, Any] | None

    @classmethod
    def from_artifact(
        cls,
        artifact: SoftTubeArtifact | Path,
        *,
        compatibility: Mapping[str, Any],
        core_replay_contract: Mapping[str, Any] | None = None,
    ) -> "TubeRSIPool":
        loaded = (
            load_soft_tube(artifact)
            if isinstance(artifact, (str, Path))
            else artifact
        )
        manifest = loaded.manifest
        if manifest.get("schema") != SOFT_TUBE_SCHEMA or manifest.get("status") != "completed":
            raise ValueError("unsupported or incomplete Soft Tube artifact")
        if manifest.get("test_data_used") is not False or manifest.get("validation_data_used") is not False:
            raise ValueError("Soft Tube does not prove validation/TEST exclusion")
        if manifest.get("phase_mixture") != PHASE_MIXTURE:
            raise ValueError("Soft Tube phase mixture is not the fixed 50/50 contract")

        normalized_contract = normalize_core_replay_contract(core_replay_contract)
        by_phase = _partition_entries(loaded, normalized_contract)
        for phase_rows in by_phase.values():
            for entry, _ in phase_rows:
                if entry.get("split") != "train":
                    raise ValueError("Soft Tube contains a non-TRAIN entry")
                phase = str(entry.get("phase", ""))
                score = float(entry.get("value_score", np.nan))
                weight = float(entry.get("sampling_weight", np.nan))
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("Soft Tube entry has an invalid score")
                if not np.isfinite(weight) or weight <= 0.0:
                    raise ValueError("Soft Tube entry must have a positive sampling weight")
                expected_weight = WEIGHT_FLOOR + WEIGHT_SCALE * score
                if not np.isclose(weight, expected_weight, rtol=0.0, atol=1.0e-12):
                    raise ValueError("Soft Tube entry violates the fixed weighting mapping")
                _validate_score_semantics(entry, phase)
                snapshot_path = Path(str(entry.get("snapshot", "")))
                if not snapshot_path.is_dir():
                    raise FileNotFoundError(f"Soft Tube snapshot is missing: {snapshot_path}")
                state_sha, schema, active_phase = _snapshot_state_sha(snapshot_path)
                if state_sha != entry.get("state_sha256"):
                    raise ValueError("Soft Tube snapshot physical-state identity mismatch")
                declared_schema = entry.get("snapshot_schema")
                if declared_schema is not None and declared_schema != schema:
                    raise ValueError("Soft Tube snapshot schema declaration mismatch")
                if active_phase is not None:
                    expected_phase = (
                        PHASE_UPSTREAM if phase == "upstream" else PHASE_DOWNSTREAM
                    )
                    if active_phase != expected_phase:
                        raise ValueError("Soft Tube unified snapshot phase mismatch")

        if not by_phase["upstream"] or not by_phase["downstream"]:
            raise ValueError("Tube-RSI requires nonempty upstream and downstream support")
        ordered = [
            *(entry for entry, _ in by_phase["upstream"]),
            *(entry for entry, _ in by_phase["downstream"]),
        ]
        pool = SnapshotPool.from_paths(
            (Path(entry["snapshot"]) for entry in ordered),
            compatibility=compatibility,
        )
        up_weights = jnp.asarray(
            [entry["sampling_weight"] for entry, _ in by_phase["upstream"]],
            dtype=jnp.float32,
        )
        down_weights = jnp.asarray(
            [entry["sampling_weight"] for entry, _ in by_phase["downstream"]],
            dtype=jnp.float32,
        )
        return cls(
            loaded,
            pool,
            up_weights,
            down_weights,
            _sampling_logits(by_phase["upstream"], normalized_contract),
            _sampling_logits(by_phase["downstream"], normalized_contract),
            len(by_phase["upstream"]),
            len(by_phase["downstream"]),
            sum(is_core for _, is_core in by_phase["upstream"]),
            sum(is_core for _, is_core in by_phase["downstream"]),
            normalized_contract,
        )

    def sample_at(
        self,
        phase_index: jax.Array | int,
        entry_index: jax.Array | int,
    ):
        phase = jnp.asarray(phase_index, dtype=jnp.int32)
        local = jnp.asarray(entry_index, dtype=jnp.int32)
        global_index = jnp.where(
            phase == PHASE_UPSTREAM,
            local,
            jnp.asarray(self.upstream_count, dtype=jnp.int32) + local,
        )
        sample = self.snapshot_pool.sample_at_index(global_index)
        sample["tube_phase"] = phase
        sample["tube_entry_index"] = local
        sample["tube_global_index"] = global_index
        return sample

    def sample_phase(self, rng: jax.Array, phase_index: jax.Array | int):
        phase = jnp.asarray(phase_index, dtype=jnp.int32)
        up_index = jax.random.categorical(rng, self.upstream_sampling_logits)
        down_index = jax.random.categorical(rng, self.downstream_sampling_logits)
        entry_index = jnp.where(phase == PHASE_UPSTREAM, up_index, down_index)
        return self.sample_at(phase, entry_index)

    def sample(self, rng: jax.Array):
        phase_key, entry_key = jax.random.split(rng)
        phase = jax.random.bernoulli(
            phase_key, PHASE_MIXTURE["downstream"]
        ).astype(jnp.int32)
        sample = self.sample_phase(entry_key, phase)
        sample["rng"] = _match_prng_key_representation(sample["rng"], rng)
        return sample