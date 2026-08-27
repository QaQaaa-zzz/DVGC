"""Policy-bound continuation labels for the frozen downstream expert."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import jax
import numpy as np

from .constants import END_REASONS
from .env import TwoPhaseBikeEnv
from .evaluation import capture_episode
from .expert_freeze import load_frozen_manifest, verify_frozen_record
from .handoff_snapshot import compatibility_identity
from .phase_d_smoke import catalog_sha256
from .ppo import make_checkpoint_policy
from .snapshot_pool import SnapshotPool


CONTINUATION_LABEL_SCHEMA = "jit_continuation_labels_v1"


def assign_parent_splits(
    parent_group_ids: Iterable[str], *, split_seed: int = 820301
) -> dict[str, str]:
    """Create guaranteed parent-disjoint 2/3, 1/6, 1/6 splits."""
    groups = sorted(set(str(value) for value in parent_group_ids))
    if len(groups) < 3:
        raise ValueError("continuation labels require at least three parent groups")
    ranked = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{split_seed}|{value}".encode()).hexdigest(),
    )
    n = len(ranked)
    n_train = max(1, (2 * n) // 3)
    n_validation = max(1, n // 6)
    if n_train + n_validation >= n:
        n_train = n - 2
        n_validation = 1
    result: dict[str, str] = {}
    for value in ranked[:n_train]:
        result[value] = "train"
    for value in ranked[n_train : n_train + n_validation]:
        result[value] = "validation"
    for value in ranked[n_train + n_validation :]:
        result[value] = "test"
    if set(result) != set(groups) or set(result.values()) != {"train", "validation", "test"}:
        raise ValueError("invalid parent split")
    return result


def derive_branch_key(
    *, protocol_seed: int, candidate_id: str, branch_index: int
) -> tuple[int, jax.Array]:
    digest = hashlib.sha256(
        f"{int(protocol_seed)}|{candidate_id}|{int(branch_index)}".encode()
    ).digest()
    seed = int.from_bytes(digest[:4], "little", signed=False)
    return seed, jax.random.PRNGKey(seed)


def _candidate_id(row: Mapping[str, Any]) -> str:
    stable = "|".join(
        str(row.get(key, ""))
        for key in ("source_bank", "parent_group_id", "seed", "role", "tick", "state_sha256")
    )
    return hashlib.sha256(stable.encode()).hexdigest()


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = list(payload.get("entries", []))
    if not rows:
        raise ValueError("snapshot catalog is empty")
    required = {"source_bank", "parent_group_id", "seed", "role", "tick", "snapshot"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"catalog row missing fields: {sorted(missing)}")
    return rows


def _protocol_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def label_downstream_continuations(
    frozen_manifest: Path,
    catalog: Path,
    output_dir: Path,
    *,
    branches: int = 1,
    max_ticks: int = 100,
    protocol_seed: int = 820301,
    stochastic_policy: bool = False,
    split_seed: int = 820301,
) -> dict[str, Any]:
    """Relabel every catalog candidate under one frozen pi_down_star."""
    if branches <= 0:
        raise ValueError("branches must be positive")
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    if not stochastic_policy and branches != 1:
        raise ValueError("deterministic continuation labeling must use exactly one branch")

    frozen = load_frozen_manifest(frozen_manifest)
    up_record = frozen["experts"]["pi_up_star"]
    down_record = frozen["experts"]["pi_down_star"]
    up_config, _ = verify_frozen_record(up_record)
    down_config, down_payload = verify_frozen_record(down_record)

    rows = _load_catalog(catalog)
    parent_splits = assign_parent_splits(
        (row["parent_group_id"] for row in rows), split_seed=split_seed
    )
    base = Path(catalog).parent
    paths = [base / row["source_bank"] / row["snapshot"] for row in rows]
    source_env = TwoPhaseBikeEnv(up_config)
    pool = SnapshotPool.from_paths(paths, compatibility=compatibility_identity(source_env))
    env = TwoPhaseBikeEnv(down_config, snapshot_pool=pool)
    policy = make_checkpoint_policy(
        env, down_payload, deterministic=not stochastic_policy
    )
    reset_index_fn = jax.jit(env.reset_descent_index)
    step_fn = jax.jit(env.step)

    protocol = {
        "schema": CONTINUATION_LABEL_SCHEMA,
        "target": "V_down",
        "question": "frozen pi_down_star completes valid landing and short recovery",
        "frozen_manifest_sha256": hashlib.sha256(Path(frozen_manifest).read_bytes()).hexdigest(),
        "expert_payload_sha256": down_record["payload_sha256"],
        "expert_actor_sha256": down_record["actor_sha256"],
        "expert_config_sha256": down_record["config_sha256"],
        "xml_sha256": down_record["xml_sha256"],
        "catalog_sha256": catalog_sha256(catalog),
        "branches": int(branches),
        "max_ticks": int(max_ticks),
        "protocol_seed": int(protocol_seed),
        "policy_mode": "stochastic" if stochastic_policy else "deterministic",
        "split_seed": int(split_seed),
        "training_transitions": 0,
    }
    protocol_hash = _protocol_sha256(protocol)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(
        json.dumps({**protocol, "protocol_sha256": protocol_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    labeled: list[dict[str, Any]] = []
    total_transitions = 0
    closed_outcomes: Counter[str] = Counter()
    split_candidate_counts: Counter[str] = Counter()
    split_success_counts: Counter[str] = Counter()

    try:
        for index, row in enumerate(rows):
            candidate_id = _candidate_id(row)
            snapshot = pool.snapshot(index)
            branch_rows = []
            for branch_index in range(branches):
                branch_seed, base_key = derive_branch_key(
                    protocol_seed=protocol_seed,
                    candidate_id=candidate_id,
                    branch_index=branch_index,
                )
                step_counter = 0

                def branch_policy(obs):
                    nonlocal step_counter
                    key = jax.random.fold_in(base_key, step_counter)
                    step_counter += 1
                    return policy(obs, key)

                trace = capture_episode(
                    env,
                    branch_policy,
                    seed=branch_seed,
                    horizon=max_ticks,
                    reset_fn=lambda _key, i=index: reset_index_fn(
                        jax.numpy.asarray(i, dtype=jax.numpy.int32)
                    ),
                    step_fn=step_fn,
                )
                terminal = trace.frames[-1]
                reason = END_REASONS.get(terminal.end_code, f"unknown_{terminal.end_code}")
                total_transitions += trace.environment_transitions
                closed_outcomes[reason] += 1
                branch_rows.append(
                    {
                        "branch_index": branch_index,
                        "branch_seed": branch_seed,
                        "transitions": trace.environment_transitions,
                        "success": bool(terminal.success),
                        "physical_failure": bool(terminal.physical_failure),
                        "timeout": bool(terminal.timeout),
                        "end_code": int(terminal.end_code),
                        "reason": reason,
                    }
                )

            success_count = sum(int(branch["success"]) for branch in branch_rows)
            physical_failure_count = sum(
                int(branch["physical_failure"]) for branch in branch_rows
            )
            timeout_count = sum(int(branch["timeout"]) for branch in branch_rows)
            if success_count + physical_failure_count + timeout_count != branches:
                raise ValueError("continuation outcomes did not close")
            split = parent_splits[str(row["parent_group_id"])]
            split_candidate_counts[split] += 1
            split_success_counts[split] += int(success_count > 0)
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
                    "split": split,
                    "actor_observation": np.asarray(snapshot.observation, dtype=np.float32).tolist(),
                    "branch_count": branches,
                    "success_count": success_count,
                    "physical_failure_count": physical_failure_count,
                    "timeout_count": timeout_count,
                    "empirical_success_rate": success_count / branches,
                    "branches": branch_rows,
                    "expert_actor_sha256": down_record["actor_sha256"],
                    "protocol_sha256": protocol_hash,
                }
            )

        report = {
            "schema": CONTINUATION_LABEL_SCHEMA,
            "status": "completed",
            "target": "V_down",
            "protocol_sha256": protocol_hash,
            "training_transitions": 0,
            "labeling_transitions": total_transitions,
            "candidate_count": len(labeled),
            "branch_rollout_count": len(labeled) * branches,
            "success_rollouts": sum(row["success_count"] for row in labeled),
            "physical_failure_rollouts": sum(
                row["physical_failure_count"] for row in labeled
            ),
            "timeout_rollouts": sum(row["timeout_count"] for row in labeled),
            "closed_outcome_counts": dict(sorted(closed_outcomes.items())),
            "split_parent_counts": {
                split: len({group for group, value in parent_splits.items() if value == split})
                for split in ("train", "validation", "test")
            },
            "split_candidate_counts": dict(sorted(split_candidate_counts.items())),
            "split_positive_candidate_counts": dict(sorted(split_success_counts.items())),
            "parent_split": parent_splits,
        }
        if (
            report["success_rollouts"]
            + report["physical_failure_rollouts"]
            + report["timeout_rollouts"]
            != report["branch_rollout_count"]
        ):
            raise ValueError("global continuation outcome accounting did not close")
        (output_dir / "labels.json").write_text(
            json.dumps(labeled, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    except Exception as exc:
        failure = {
            "schema": CONTINUATION_LABEL_SCHEMA,
            "status": "engineering_error",
            "target": "V_down",
            "protocol_sha256": protocol_hash,
            "training_transitions": 0,
            "labeling_transitions": total_transitions,
            "completed_candidates": len(labeled),
            "reason": f"{type(exc).__name__}: {exc}",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
