"""JAX-sampleable, provenance-preserving handoff snapshot pool."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import jax
import jax.numpy as jnp
import numpy as np
from .handoff_snapshot import HandoffSnapshot, load_snapshot

@dataclass
class SnapshotPool:
    snapshots: tuple[HandoffSnapshot, ...]
    parent_group_ids: tuple[str, ...]
    compatibility: Mapping[str, Any]

    @classmethod
    def from_paths(cls, paths: Iterable[Path], *, compatibility: Mapping[str, Any]) -> "SnapshotPool":
        items = tuple(load_snapshot(Path(p)) for p in paths)
        if not items:
            raise ValueError("snapshot pool must not be empty")
        for item in items:
            if item.compatibility_identity != compatibility:
                raise ValueError("snapshot pool compatibility identity mismatch")
        groups = tuple(f"transition_{item.config_sha256}__{item.parent_trajectory}" for item in items)
        return cls(items, groups, compatibility)

    @classmethod
    def from_closed_bank(cls, bank: Path, *, compatibility: Mapping[str, Any]) -> "SnapshotPool":
        root = Path(bank)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("status") != "closed":
            raise ValueError("snapshot pool source bank must be closed")
        index = json.loads((root / "index.json").read_text())
        return cls.from_paths((root / row["snapshot"] for row in index), compatibility=compatibility)

    def _stack(self, name: str) -> jax.Array:
        return jax.device_put(np.stack([getattr(s, name) for s in self.snapshots]))

    def sample(self, rng: jax.Array) -> dict[str, Any]:
        index = jax.random.randint(rng, (), 0, len(self.snapshots))
        result = {name: self._stack(name)[index] for name in ("qpos", "qvel", "observation_fifo", "observation", "last_action", "ctrl", "rng")}
        result["history_valid_count"] = jnp.asarray([s.history_valid_count for s in self.snapshots], dtype=jnp.int32)[index]
        result["tick"] = jnp.asarray([s.tick for s in self.snapshots], dtype=jnp.int32)[index]
        result["events"] = {key: jnp.asarray([s.events[key] for s in self.snapshots])[index] for key in self.snapshots[0].events}
        result["parent_group_index"] = index
        return result

    def sample_index(self, index: int) -> dict[str, Any]:
        """Materialize a deterministic indexed item for host-side restore tests."""
        return {"index": int(index), **{name: getattr(self.snapshots[int(index)], name).copy() for name in ("qpos", "qvel", "observation_fifo", "observation", "last_action", "ctrl", "rng")}}

    def materialize(self, item: Mapping[str, Any]) -> HandoffSnapshot:
        """Reattach immutable provenance to a sampled pytree item."""
        source = self.snapshots[int(item["index"])]
        return HandoffSnapshot(qpos=item["qpos"], qvel=item["qvel"], observation_fifo=item["observation_fifo"], history_valid_count=source.history_valid_count, observation=item["observation"], last_action=item["last_action"], ctrl=item["ctrl"], rng=item["rng"], events=source.events, tick=source.tick, parent_trajectory=source.parent_trajectory, parent_tick=source.parent_tick, config_sha256=source.config_sha256, xml_sha256=source.xml_sha256, policy_sha256=source.policy_sha256, policy_identity=source.policy_identity, compatibility_identity=source.compatibility_identity)

    def snapshot(self, index: int) -> HandoffSnapshot:
        return self.snapshots[int(index)]
