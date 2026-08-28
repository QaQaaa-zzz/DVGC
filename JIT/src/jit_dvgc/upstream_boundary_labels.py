"""TRAIN-only continuation labeling for reachable V_up boundary candidates."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from .continuation_labels import DEFAULT_TRAIN_SEEDS, derive_branch_key
from .env import TwoPhaseBikeEnv
from .expert_freeze import load_frozen_manifest, verify_frozen_record
from .handoff_snapshot import load_snapshot
from .ppo import make_checkpoint_policy
from .upstream_boundary import BOUNDARY_CATALOG_SCHEMA, canonical_sha256, file_sha256
from .upstream_labels import _candidate_id, _run_upstream_branch

BOUNDARY_LABEL_SCHEMA = "jit_upstream_boundary_labels_v1"


def _load_train_catalog(path: Path, train_seeds: Sequence[int]) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != BOUNDARY_CATALOG_SCHEMA:
        raise ValueError("boundary labeler requires a reachable-boundary catalog")
    rows = [dict(row) for row in payload.get("entries", [])]
    if not rows:
        raise ValueError("boundary catalog is empty")
    allowed = {int(seed) for seed in train_seeds}
    if not allowed:
        raise ValueError("train_seeds must be non-empty")
    parents: dict[str, int] = {}
    for row in rows:
        required = {"source_bank", "parent_group_id", "seed", "role", "tick", "snapshot", "state_sha256"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"boundary row missing fields: {sorted(missing)}")
        seed = int(row["seed"])
        if str(row.get("split", row.get("anchor_split", ""))) != "train" or seed not in allowed:
            raise ValueError("boundary labeler accepts TRAIN rows only")
        parent = str(row["parent_group_id"])
        previous = parents.setdefault(parent, seed)
        if previous != seed:
            raise ValueError("parent lineage maps to multiple numeric seeds")
    return rows


def label_train_boundary_continuations(
    frozen_manifest: Path,
    catalog: Path,
    output_dir: Path,
    *,
    max_ticks: int = 400,
    protocol_seed: int = 820402,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
) -> dict[str, Any]:
    """Label one locked TRAIN boundary bank with frozen deterministic pi_up_star."""
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    rows = _load_train_catalog(catalog, train_seeds)
    frozen = load_frozen_manifest(frozen_manifest)
    record = frozen["experts"]["pi_up_star"]
    config, payload = verify_frozen_record(record)
    if config.phase != "propulsion_ascent":
        raise ValueError("pi_up_star must use propulsion_ascent config")

    env = TwoPhaseBikeEnv(config)
    policy = make_checkpoint_policy(env, payload, deterministic=True)
    step_fn = jax.jit(env.step)
    protocol = {
        "schema": BOUNDARY_LABEL_SCHEMA,
        "target": "V_up",
        "split": "train",
        "question": "frozen pi_up_star reaches Apex before a retained terminal",
        "policy_mode": "deterministic",
        "branches": 1,
        "max_ticks": int(max_ticks),
        "protocol_seed": int(protocol_seed),
        "train_seeds": sorted({int(seed) for seed in train_seeds}),
        "frozen_manifest_sha256": file_sha256(frozen_manifest),
        "catalog_sha256": file_sha256(catalog),
        "expert_actor_sha256": record["actor_sha256"],
        "expert_payload_sha256": record["payload_sha256"],
        "expert_config_sha256": record["config_sha256"],
        "xml_sha256": record["xml_sha256"],
        "training_transitions": 0,
    }
    protocol_hash = canonical_sha256(protocol)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(json.dumps({**protocol, "protocol_sha256": protocol_hash}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    base = Path(catalog).parent
    labeled = []
    transitions = 0
    outcomes: Counter[str] = Counter()
    for index, row in enumerate(rows):
        candidate_id = _candidate_id(row)
        snapshot = load_snapshot(base / row["source_bank"] / row["snapshot"])
        branch_seed, base_key = derive_branch_key(protocol_seed=protocol_seed, candidate_id=candidate_id, branch_index=0)
        branch = _run_upstream_branch(env, policy, snapshot, branch_seed=branch_seed, base_key=base_key, branch_index=0, max_ticks=max_ticks, step_fn=step_fn)
        transitions += int(branch["transitions"])
        outcomes[str(branch["reason"])] += 1
        success = int(bool(branch["success"]))
        labeled.append({
            "candidate_id": candidate_id,
            "source_bank": row["source_bank"],
            "seed": int(row["seed"]),
            "parent_group_id": row["parent_group_id"],
            "role": row["role"],
            "tick": int(row["tick"]),
            "state_sha256": row.get("state_sha256"),
            "snapshot": row["snapshot"],
            "split": "train",
            "actor_observation": np.asarray(snapshot.observation, dtype=np.float32).tolist(),
            "branch_count": 1,
            "success_count": success,
            "physical_failure_count": int(bool(branch["physical_failure"])),
            "task_failure_count": int(bool(branch["task_failure"])),
            "timeout_count": int(bool(branch["timeout"])),
            "empirical_success_rate": float(success),
            "branches": [branch],
            "expert_actor_sha256": record["actor_sha256"],
            "protocol_sha256": protocol_hash,
            "boundary_protocol_sha256": row.get("protocol_sha256"),
            "boundary_candidate_index": index,
        })

    report = {
        "schema": BOUNDARY_LABEL_SCHEMA,
        "status": "completed",
        "target": "V_up",
        "split": "train",
        "protocol_sha256": protocol_hash,
        "candidate_count": len(labeled),
        "branch_rollout_count": len(labeled),
        "success_rollouts": sum(row["success_count"] for row in labeled),
        "physical_failure_rollouts": sum(row["physical_failure_count"] for row in labeled),
        "task_failure_rollouts": sum(row["task_failure_count"] for row in labeled),
        "timeout_rollouts": sum(row["timeout_count"] for row in labeled),
        "closed_outcome_counts": dict(sorted(outcomes.items())),
        "labeling_transitions": transitions,
        "training_transitions": 0,
    }
    if report["success_rollouts"] + report["physical_failure_rollouts"] + report["task_failure_rollouts"] + report["timeout_rollouts"] != report["branch_rollout_count"]:
        raise ValueError("boundary continuation outcome accounting did not close")
    (output_dir / "labels.json").write_text(json.dumps(labeled, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
