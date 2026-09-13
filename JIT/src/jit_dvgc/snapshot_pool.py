"""JAX-sampleable snapshot pool supporting legacy and unified Tube entries."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .handoff_snapshot import HandoffSnapshot, load_snapshot


DOWN_EVENT_FIELDS = (
    "airborne_seen",
    "valid_contact_seen",
    "contact_x",
    "post_contact_ticks",
    "recovery_success",
)


def _load_any_snapshot(path: Path) -> tuple[Any, str]:
    directory = Path(path)
    identity = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
    schema = str(identity.get("schema", ""))
    if schema == "handoff_snapshot_v1":
        return load_snapshot(directory), schema
    if schema == "jit_unified_envelope_snapshot_v1":
        # Local import avoids: unified_envelope_snapshot -> tube_rsi -> snapshot_pool.
        from .unified_envelope_snapshot import load_unified_envelope_snapshot

        return load_unified_envelope_snapshot(directory), schema
    raise ValueError(f"unsupported Tube snapshot schema: {schema}")


@dataclass
class SnapshotPool:
    snapshots: tuple[Any, ...]
    snapshot_schemas: tuple[str, ...]
    parent_group_ids: tuple[str, ...]
    compatibility: Mapping[str, Any]

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[Path],
        *,
        compatibility: Mapping[str, Any],
    ) -> "SnapshotPool":
        loaded = tuple(_load_any_snapshot(Path(path)) for path in paths)
        if not loaded:
            raise ValueError("snapshot pool must not be empty")
        items = tuple(item for item, _schema in loaded)
        schemas = tuple(schema for _item, schema in loaded)
        for item in items:
            if item.compatibility_identity != compatibility:
                raise ValueError("snapshot pool compatibility identity mismatch")
        groups = tuple(
            f"transition_{item.config_sha256}__{item.parent_trajectory}"
            for item in items
        )
        return cls(items, schemas, groups, compatibility)

    @classmethod
    def from_closed_bank(
        cls,
        bank: Path,
        *,
        compatibility: Mapping[str, Any],
    ) -> "SnapshotPool":
        root = Path(bank)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("status") != "closed":
            raise ValueError("snapshot pool source bank must be closed")
        index = json.loads((root / "index.json").read_text())
        return cls.from_paths(
            (root / row["snapshot"] for row in index),
            compatibility=compatibility,
        )

    def _common_stack(self, name: str) -> jax.Array:
        return jax.device_put(
            np.stack([np.asarray(getattr(snapshot, name)) for snapshot in self.snapshots])
        )

    @staticmethod
    def _up_events(snapshot: Any) -> Mapping[str, Any]:
        return snapshot.events if isinstance(snapshot, HandoffSnapshot) else snapshot.up_events

    @staticmethod
    def _down_events(snapshot: Any) -> Mapping[str, Any]:
        if isinstance(snapshot, HandoffSnapshot):
            return {
                "airborne_seen": np.asarray(False),
                "valid_contact_seen": np.asarray(False),
                "contact_x": np.asarray(0.0, dtype=np.float32),
                "post_contact_ticks": np.asarray(0, dtype=np.int32),
                "recovery_success": np.asarray(False),
            }
        return snapshot.down_events

    @staticmethod
    def _unified_metadata(snapshot: Any) -> Mapping[str, Any]:
        if isinstance(snapshot, HandoffSnapshot):
            return {
                "start_phase": np.asarray(0, dtype=np.int32),
                "phase_transitioned": np.asarray(False),
                "episode_step": np.asarray(0, dtype=np.int32),
                "phase_episode_step": np.asarray(0, dtype=np.int32),
                "episode_return": np.asarray(0.0, dtype=np.float32),
            }
        return {
            "start_phase": np.asarray(snapshot.start_phase, dtype=np.int32),
            "phase_transitioned": np.asarray(snapshot.phase_transitioned),
            "episode_step": np.asarray(snapshot.episode_step, dtype=np.int32),
            "phase_episode_step": np.asarray(
                snapshot.phase_episode_step, dtype=np.int32
            ),
            "episode_return": np.asarray(snapshot.episode_return, dtype=np.float32),
        }

    def sample(self, rng: jax.Array) -> dict[str, Any]:
        index = jax.random.randint(rng, (), 0, len(self.snapshots))
        return self.sample_at_index(index)

    def sample_at_index(self, index: jax.Array) -> dict[str, Any]:
        """Select a fixed item with a JAX scalar index (jit/vmap compatible)."""
        result = {
            name: self._common_stack(name)[index]
            for name in (
                "qpos",
                "qvel",
                "observation_fifo",
                "observation",
                "last_action",
                "ctrl",
            )
        }
        # Snapshot payloads persist PRNG keys as raw uint32[2] key data for stable
        # serialization. Runtime JAX state uses typed keys, matching natural resets
        # and canonical snapshot restore, so mixed reset selection has one PyTree
        # dtype contract under jit/vmap.
        result["rng"] = jax.random.wrap_key_data(
            jnp.asarray(self._common_stack("rng")[index], dtype=jnp.uint32)
        )
        result["history_valid_count"] = jnp.asarray(
            [snapshot.history_valid_count for snapshot in self.snapshots], dtype=jnp.int32
        )[index]
        result["tick"] = jnp.asarray(
            [
                snapshot.tick
                if isinstance(snapshot, HandoffSnapshot)
                else snapshot.source_tick
                for snapshot in self.snapshots
            ],
            dtype=jnp.int32,
        )[index]
        up_keys = tuple(self._up_events(self.snapshots[0]))
        result["events"] = {
            key: jnp.asarray(
                [self._up_events(snapshot)[key] for snapshot in self.snapshots]
            )[index]
            for key in up_keys
        }
        result["down_events"] = {
            key: jnp.asarray(
                [self._down_events(snapshot)[key] for snapshot in self.snapshots]
            )[index]
            for key in DOWN_EVENT_FIELDS
        }
        for key in (
            "start_phase",
            "phase_transitioned",
            "episode_step",
            "phase_episode_step",
            "episode_return",
        ):
            result[key] = jnp.asarray(
                [self._unified_metadata(snapshot)[key] for snapshot in self.snapshots]
            )[index]
        result["preserve_unified_context"] = jnp.asarray(
            [
                schema == "jit_unified_envelope_snapshot_v1"
                for schema in self.snapshot_schemas
            ],
            dtype=bool,
        )[index]
        result["parent_group_index"] = index
        return result

    def sample_index(self, index: int) -> dict[str, Any]:
        """Materialize a deterministic indexed item for host-side restore tests."""
        snapshot = self.snapshots[int(index)]
        return {
            "index": int(index),
            **{
                name: np.asarray(getattr(snapshot, name)).copy()
                for name in (
                    "qpos",
                    "qvel",
                    "observation_fifo",
                    "observation",
                    "last_action",
                    "ctrl",
                    "rng",
                )
            },
        }

    def materialize(self, item: Mapping[str, Any]) -> Any:
        """Reattach immutable provenance to a sampled host-side item."""
        source = self.snapshots[int(item["index"])]
        if not isinstance(source, HandoffSnapshot):
            return source
        return HandoffSnapshot(
            qpos=item["qpos"],
            qvel=item["qvel"],
            observation_fifo=item["observation_fifo"],
            history_valid_count=source.history_valid_count,
            observation=item["observation"],
            last_action=item["last_action"],
            ctrl=item["ctrl"],
            rng=item["rng"],
            events=source.events,
            tick=source.tick,
            parent_trajectory=source.parent_trajectory,
            parent_tick=source.parent_tick,
            config_sha256=source.config_sha256,
            xml_sha256=source.xml_sha256,
            policy_sha256=source.policy_sha256,
            policy_identity=source.policy_identity,
            compatibility_identity=source.compatibility_identity,
        )

    def snapshot(self, index: int) -> Any:
        return self.snapshots[int(index)]
