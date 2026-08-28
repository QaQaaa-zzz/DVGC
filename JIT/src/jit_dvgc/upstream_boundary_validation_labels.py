"""Continuation labels for the locked validation V_up boundary bank."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .continuation_labels import derive_branch_key
from .env import TwoPhaseBikeEnv
from .expert_freeze import load_frozen_manifest, verify_frozen_record
from .handoff_snapshot import load_snapshot
from .ppo import make_checkpoint_policy
from .upstream_boundary import BOUNDARY_CATALOG_SCHEMA, canonical_sha256, file_sha256
from .upstream_boundary_lock import load_boundary_lock
from .upstream_labels import _candidate_id, _run_upstream_branch

VALIDATION_LABEL_SCHEMA = "jit_upstream_boundary_validation_labels_v1"


def _load_validation_catalog(path: Path, lock: dict[str, Any]) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != BOUNDARY_CATALOG_SCHEMA or payload.get("split") != "validation":
        raise ValueError("validation labeler requires a validation reachable-boundary catalog")
    if str(payload.get("lock_sha256", "")) != str(lock["lock_sha256"]):
        raise ValueError("validation catalog does not match boundary lock")
    rows = [dict(row) for row in payload.get("entries", [])]
    if not rows:
        raise ValueError("validation boundary catalog is empty")
    allowed = {int(seed) for seed in lock.get("validation_seeds", ())}
    test = {int(seed) for seed in lock.get("test_seeds", ())}
    parents: dict[str, int] = {}
    for row in rows:
        required = {"source_bank", "parent_group_id", "seed", "role", "tick", "snapshot", "state_sha256"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"validation boundary row missing fields: {sorted(missing)}")
        seed = int(row["seed"])
        if str(row.get("anchor_split", "")) != "validation" or seed not in allowed or seed in test:
            raise ValueError("validation boundary labels accept validation seeds only")
        if str(row.get("lock_sha256", "")) != str(lock["lock_sha256"]):
            raise ValueError("validation row lock hash mismatch")
        parent = str(row["parent_group_id"])
        previous = parents.setdefault(parent, seed)
        if previous != seed:
            raise ValueError("validation parent lineage maps to multiple numeric seeds")
    return rows


def label_validation_boundary_continuations(
    frozen_manifest: Path,
    catalog: Path,
    lock_path: Path,
    output_dir: Path,
    *,
    max_ticks: int = 400,
    protocol_seed: int = 820405,
) -> dict[str, Any]:
    """Label validation states with the same frozen deterministic pi_up_star."""
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    lock = load_boundary_lock(lock_path)
    rows = _load_validation_catalog(catalog, lock)
    frozen = load_frozen_manifest(frozen_manifest)
    record = frozen["experts"]["pi_up_star"]
    config, payload = verify_frozen_record(record)
    if config.phase != "propulsion_ascent":
        raise ValueError("pi_up_star must use propulsion_ascent config")
    for key, actual in (
        ("frozen_pi_up_actor_sha256", record["actor_sha256"]),
        ("frozen_pi_up_payload_sha256", record["payload_sha256"]),
        ("frozen_pi_up_config_sha256", record["config_sha256"]),
        ("xml_sha256", record["xml_sha256"]),
    ):
        if str(lock[key]) != str(actual):
            raise ValueError(f"frozen pi_up_star identity mismatch for {key}")

    env = TwoPhaseBikeEnv(config)
    policy = make_checkpoint_policy(env, payload, deterministic=True)
    step_fn = jax.jit(env.step)
    protocol = {
        "schema": VALIDATION_LABEL_SCHEMA,
        "target": "V_up",
        "split": "validation",
        "question": "frozen pi_up_star reaches Apex before a retained terminal",
        "policy_mode": "deterministic",
        "branches": 1,
        "max_ticks": int(max_ticks),
        "protocol_seed": int(protocol_seed),
        "validation_seeds": list(lock["validation_seeds"]),
        "lock_sha256": lock["lock_sha256"],
        "train_protocol_sha256": lock["train_protocol_sha256"],
        "frozen_manifest_sha256": file_sha256(frozen_manifest),
        "catalog_sha256": file_sha256(catalog),
        "expert_actor_sha256": record["actor_sha256"],
        "expert_payload_sha256": record["payload_sha256"],
        "expert_config_sha256": record["config_sha256"],
        "xml_sha256": record["xml_sha256"],
        "test_interaction_count": 0,
        "training_transitions": 0,
    }
    protocol_hash = canonical_sha256(protocol)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(
        json.dumps({**protocol, "protocol_sha256": protocol_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    base = Path(catalog).parent
    labeled: list[dict[str, Any]] = []
    transitions = 0
    outcomes: Counter[str] = Counter()
    for index, row in enumerate(rows):
        candidate_id = _candidate_id(row)
        snapshot = load_snapshot(base / row["source_bank"] / row["snapshot"])
        branch_seed, base_key = derive_branch_key(protocol_seed=protocol_seed, candidate_id=candidate_id, branch_index=0)
        branch = _run_upstream_branch(
            env,
            policy,
            snapshot,
            branch_seed=branch_seed,
            base_key=base_key,
            branch_index=0,
            max_ticks=max_ticks,
            step_fn=step_fn,
        )
        transitions += int(branch["transitions"])
        outcomes[str(branch["reason"])] += 1
        success = int(bool(branch["success"]))
        labeled.append(
            {
                "candidate_id": candidate_id,
                "source_bank": row["source_bank"],
                "seed": int(row["seed"]),
                "parent_group_id": row["parent_group_id"],
                "role": row["role"],
                "tick": int(row["tick"]),
                "state_sha256": row.get("state_sha256"),
                "snapshot": row["snapshot"],
                "split": "validation",
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
                "lock_sha256": lock["lock_sha256"],
                "boundary_candidate_index": index,
            }
        )

    report = {
        "schema": VALIDATION_LABEL_SCHEMA,
        "status": "completed",
        "target": "V_up",
        "split": "validation",
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": protocol_hash,
        "candidate_count": len(labeled),
        "branch_rollout_count": len(labeled),
        "success_rollouts": sum(row["success_count"] for row in labeled),
        "physical_failure_rollouts": sum(row["physical_failure_count"] for row in labeled),
        "task_failure_rollouts": sum(row["task_failure_count"] for row in labeled),
        "timeout_rollouts": sum(row["timeout_count"] for row in labeled),
        "closed_outcome_counts": dict(sorted(outcomes.items())),
        "labeling_transitions": transitions,
        "test_interaction_count": 0,
        "training_transitions": 0,
    }
    if report["success_rollouts"] + report["physical_failure_rollouts"] + report["task_failure_rollouts"] + report["timeout_rollouts"] != report["branch_rollout_count"]:
        raise ValueError("validation continuation outcome accounting did not close")
    (output_dir / "labels.json").write_text(json.dumps(labeled, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
