"""Deterministic phase-balanced sampling from a validated learned Soft Tube."""

from __future__ import annotations

from dataclasses import dataclass
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
from .upstream_boundary import physical_state_sha256


PHASE_UPSTREAM = 0
PHASE_DOWNSTREAM = 1


@dataclass(frozen=True)
class TubeRSIPool:
    artifact: SoftTubeArtifact
    snapshot_pool: SnapshotPool
    upstream_weights: jax.Array
    downstream_weights: jax.Array
    upstream_count: int
    downstream_count: int

    @classmethod
    def from_artifact(
        cls,
        artifact: SoftTubeArtifact | Path,
        *,
        compatibility: Mapping[str, Any],
    ) -> "TubeRSIPool":
        loaded = load_soft_tube(artifact) if isinstance(artifact, (str, Path)) else artifact
        manifest = loaded.manifest
        if manifest.get("schema") != SOFT_TUBE_SCHEMA or manifest.get("status") != "completed":
            raise ValueError("unsupported or incomplete Soft Tube artifact")
        if manifest.get("test_data_used") is not False or manifest.get("validation_data_used") is not False:
            raise ValueError("Soft Tube does not prove validation/TEST exclusion")
        if manifest.get("phase_mixture") != PHASE_MIXTURE:
            raise ValueError("Soft Tube phase mixture is not the fixed 50/50 contract")

        by_phase: dict[str, list[dict[str, Any]]] = {
            "upstream": [],
            "downstream": [],
        }
        for original in loaded.entries:
            entry = dict(original)
            if entry.get("split") != "train":
                raise ValueError("Soft Tube contains a non-TRAIN entry")
            phase = str(entry.get("phase", ""))
            if phase not in by_phase:
                raise ValueError(f"Soft Tube contains unsupported phase: {phase}")
            score = float(entry.get("value_score", np.nan))
            weight = float(entry.get("sampling_weight", np.nan))
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("Soft Tube entry has an invalid value score")
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError("Soft Tube entry must have a positive sampling weight")
            expected_weight = WEIGHT_FLOOR + WEIGHT_SCALE * score
            if not np.isclose(weight, expected_weight, rtol=0.0, atol=1.0e-12):
                raise ValueError("Soft Tube entry violates the fixed weighting mapping")
            expected_target = "V_up" if phase == "upstream" else "V_down"
            if entry.get("value_model_target") != expected_target:
                raise ValueError("Soft Tube entry uses a cross-phase value model")
            snapshot_path = Path(str(entry.get("snapshot", "")))
            if not snapshot_path.is_dir():
                raise FileNotFoundError(f"Soft Tube snapshot is missing: {snapshot_path}")
            snapshot = load_snapshot(snapshot_path)
            if physical_state_sha256(snapshot) != entry.get("state_sha256"):
                raise ValueError("Soft Tube snapshot physical-state identity mismatch")
            by_phase[phase].append(entry)

        if not by_phase["upstream"] or not by_phase["downstream"]:
            raise ValueError("Tube-RSI requires nonempty upstream and downstream support")
        ordered = [*by_phase["upstream"], *by_phase["downstream"]]
        pool = SnapshotPool.from_paths(
            (Path(entry["snapshot"]) for entry in ordered),
            compatibility=compatibility,
        )
        up_weights = jnp.asarray(
            [entry["sampling_weight"] for entry in by_phase["upstream"]],
            dtype=jnp.float32,
        )
        down_weights = jnp.asarray(
            [entry["sampling_weight"] for entry in by_phase["downstream"]],
            dtype=jnp.float32,
        )
        return cls(
            loaded,
            pool,
            up_weights,
            down_weights,
            len(by_phase["upstream"]),
            len(by_phase["downstream"]),
        )

    def sample_at(self, phase_index: jax.Array | int, entry_index: jax.Array | int):
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
        up_index = jax.random.categorical(rng, jnp.log(self.upstream_weights))
        down_index = jax.random.categorical(rng, jnp.log(self.downstream_weights))
        entry_index = jnp.where(phase == PHASE_UPSTREAM, up_index, down_index)
        return self.sample_at(phase, entry_index)

    def sample(self, rng: jax.Array):
        phase_key, entry_key = jax.random.split(rng)
        phase = jax.random.bernoulli(
            phase_key, PHASE_MIXTURE["downstream"]
        ).astype(jnp.int32)
        return self.sample_phase(entry_key, phase)
