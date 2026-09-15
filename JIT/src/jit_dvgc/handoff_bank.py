"""Bounded, provenance-preserving handoff snapshot bank collection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable
from collections import deque
import numpy as np
import jax

from .handoff_snapshot import HandoffSnapshot, save_snapshot

OFFSETS = (-10, -5, -1, 0, 1, 5, 10)

def pytree_sha256(tree: Any) -> str:
    """Hash full JAX pytree structure and leaf bytes (without repr truncation)."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    digest = hashlib.sha256(str(treedef).encode("utf-8"))
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()

@dataclass
class BankCollector:
    output_dir: Path
    purpose: str
    checkpoint: str
    config_sha256: str
    xml_sha256: str
    policy_sha256: str
    interaction_budget: int
    seen: set[str] | None = None
    entries: list[dict[str, Any]] | None = None
    transitions: int = 0

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.seen = set() if self.seen is None else self.seen
        self.entries = [] if self.entries is None else self.entries

    @staticmethod
    def _state_hash(snapshot: HandoffSnapshot) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(snapshot.qpos).tobytes())
        digest.update(np.asarray(snapshot.qvel).tobytes())
        return digest.hexdigest()

    def add(self, snapshot: HandoffSnapshot, *, seed: int, role: str, parent_trajectory: str) -> bool:
        key = self._state_hash(snapshot)
        if key in self.seen:
            return False
        self.seen.add(key)
        ordinal = len(self.entries)
        path = self.output_dir / "snapshots" / f"snapshot_{ordinal:06d}"
        save_snapshot(path, snapshot)
        qpos, qvel = np.asarray(snapshot.qpos), np.asarray(snapshot.qvel)
        entry = {"snapshot": str(path.relative_to(self.output_dir)), "state_sha256": key,
                 "role": role, "seed": int(seed), "parent_trajectory": parent_trajectory,
                 "parent_tick": snapshot.parent_tick, "tick": snapshot.tick,
                 "z": float(qpos[2]), "vx": float(qvel[0]), "vz": float(qvel[2]),
                 "pitch": float(qpos[4]) if qpos.size > 4 else float("nan"),
                 "pitch_rate": float(qvel[4]) if qvel.size > 4 else float("nan")}
        self.entries.append(entry)
        return True

    def close(self, *, status: str = "closed", failure: str | None = None) -> dict[str, Any]:
        if status not in ("closed", "failed"):
            raise ValueError("bank status must be closed or failed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": "handoff_bank_v1", "status": status, "purpose": self.purpose,
                    "checkpoint": self.checkpoint, "config_sha256": self.config_sha256,
                    "xml_sha256": self.xml_sha256, "policy_sha256": self.policy_sha256,
                    "interaction_accounting": {"budget": self.interaction_budget,
                    "snapshots": len(self.entries), "actual_transitions": self.transitions,
                    "max_transitions": self.interaction_budget, "max_ticks": getattr(self, "max_ticks", None)},
                    "snapshot_count": len(self.entries), "failure": failure}
        (self.output_dir / "index.json").write_text(json.dumps(self.entries, indent=2, sort_keys=True) + "\n")
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


def select_offsets(apex_tick: int, *, max_tick: int, offsets: Iterable[int] = OFFSETS) -> dict[int, str]:
    """Map fixed offsets to semantic roles, omitting unavailable ticks."""
    roles = {-10: "pre_apex", -5: "pre_apex", -1: "nearest_pre_apex", 0: "nearest_apex",
             1: "post_apex", 5: "post_apex", 10: "early_descent"}
    return {apex_tick + offset: roles[offset] for offset in offsets if 0 <= apex_tick + offset <= max_tick}


def collect_rollout(collector: BankCollector, states: list[Any], *, seed: int,
                    capture: Callable[[Any, str, int], HandoffSnapshot],
                    parent_trajectory: str, apex_tick: int | None = None,
                    contact_tick: int | None = None) -> int:
    """Collect one bounded rollout; states are ordered state-at-tick values."""
    if not states:
        return 0
    apex_tick = apex_tick if apex_tick is not None else next((int(s.info["episode_step"]) for s in states if bool(np.asarray(s.info.get("events").apex_seen))), -1)
    selected = select_offsets(apex_tick, max_tick=len(states)-1) if apex_tick >= 0 else {}
    if contact_tick is not None and 0 <= contact_tick < len(states):
        selected[contact_tick] = "pre_contact"
    added = 0
    for tick, role in sorted(selected.items()):
        snapshot = capture(states[tick], parent_trajectory, tick)
        added += collector.add(snapshot, seed=seed, role=role, parent_trajectory=parent_trajectory)
    return added


def collect_streaming_rollout(collector: BankCollector, initial_state: Any, *, seed: int,
                              step: Callable[[Any], Any], capture: Callable[[Any, str, int], HandoffSnapshot],
                              parent_trajectory: str, max_ticks: int) -> int:
    """Bounded-memory collector; never retains the complete MJX state trace."""
    history = deque(maxlen=11)
    state = initial_state
    apex_tick = None
    captured: set[int] = set()
    added = 0
    roles = {-10: "pre_apex", -5: "pre_apex", -1: "nearest_pre_apex", 0: "nearest_apex", 1: "post_apex", 5: "post_apex", 10: "early_descent"}
    for tick in range(max_ticks + 1):
        history.append((tick, state))
        if apex_tick is None and bool(np.asarray(state.info["events"].apex_seen)):
            apex_tick = tick
        if apex_tick is not None:
            for offset, role in roles.items():
                target = apex_tick + offset
                if target in captured or target < 0 or target != tick and not any(t == target for t, _ in history):
                    continue
                found = next((s for t, s in history if t == target), None)
                if found is not None:
                    added += collector.add(capture(found, parent_trajectory, target), seed=seed, role=role, parent_trajectory=parent_trajectory)
                    captured.add(target)
        if bool(state.info.get("terminated", False)) or bool(state.info.get("truncated", False)) or tick == max_ticks:
            break
        state = step(state)
        collector.transitions += 1
    return added
