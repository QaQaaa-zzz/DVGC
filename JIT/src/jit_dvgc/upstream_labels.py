"""Policy-bound V_up continuation labels for the frozen pi_up_star."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from .constants import END_REASONS
from .continuation_labels import (
    CONTINUATION_LABEL_SCHEMA,
    DEFAULT_TEST_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALIDATION_SEEDS,
    assign_global_seed_splits,
    derive_branch_key,
)
from .env import TwoPhaseBikeEnv
from .expert_freeze import load_frozen_manifest, verify_frozen_record
from .handoff_snapshot import load_snapshot
from .phase_d_smoke import catalog_sha256
from .ppo import make_checkpoint_policy


def _candidate_id(row: Mapping[str, Any]) -> str:
    stable = "|".join(
        str(row.get(key, ""))
        for key in (
            "source_bank",
            "parent_group_id",
            "seed",
            "role",
            "tick",
            "state_sha256",
        )
    )
    return hashlib.sha256(stable.encode()).hexdigest()


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = list(payload.get("entries", []))
    if not rows:
        raise ValueError("snapshot catalog is empty")
    required = {
        "source_bank",
        "parent_group_id",
        "seed",
        "role",
        "tick",
        "snapshot",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"catalog row missing fields: {sorted(missing)}")
    return rows


def _protocol_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _seed_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    test_seeds: Sequence[int],
) -> tuple[dict[int, str], dict[str, str]]:
    seed_splits = assign_global_seed_splits(
        (int(row["seed"]) for row in rows),
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )
    parent_splits: dict[str, str] = {}
    for row in rows:
        parent_group_id = str(row["parent_group_id"])
        split = seed_splits[int(row["seed"])]
        previous = parent_splits.setdefault(parent_group_id, split)
        if previous != split:
            raise ValueError("parent group maps to multiple global-seed splits")
    return seed_splits, parent_splits


def _host_bool(value: Any) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _host_int(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)))


def classify_upstream_outcome(
    *,
    apex_seen: bool,
    terminated: bool,
    truncated: bool,
    physical_failure: bool,
    timeout: bool,
    end_code: int,
    label_horizon: bool = False,
) -> dict[str, Any]:
    """Close one V_up continuation into one mutually exclusive outcome."""
    if apex_seen and not terminated and not truncated:
        return {
            "success": True,
            "physical_failure": False,
            "task_failure": False,
            "timeout": False,
            "end_code": int(end_code),
            "reason": "apex_success",
        }
    if physical_failure:
        return {
            "success": False,
            "physical_failure": True,
            "task_failure": False,
            "timeout": False,
            "end_code": int(end_code),
            "reason": END_REASONS.get(
                int(end_code), f"unknown_{int(end_code)}"
            ),
        }
    if terminated:
        return {
            "success": False,
            "physical_failure": False,
            "task_failure": True,
            "timeout": False,
            "end_code": int(end_code),
            "reason": END_REASONS.get(
                int(end_code), f"unknown_{int(end_code)}"
            ),
        }
    if truncated or timeout or label_horizon:
        return {
            "success": False,
            "physical_failure": False,
            "task_failure": False,
            "timeout": True,
            "end_code": int(end_code),
            "reason": (
                "label_horizon"
                if label_horizon and not (truncated or timeout)
                else END_REASONS.get(
                    int(end_code), f"unknown_{int(end_code)}"
                )
            ),
        }
    raise ValueError("upstream continuation outcome is not closed")


def _run_upstream_branch(
    env: TwoPhaseBikeEnv,
    policy: Any,
    snapshot: Any,
    *,
    branch_seed: int,
    base_key: jax.Array,
    branch_index: int,
    max_ticks: int,
    step_fn: Any,
) -> dict[str, Any]:
    state = env.restore_handoff_snapshot(snapshot)
    if _host_bool(state.info["events"].apex_seen):
        raise ValueError("V_up candidate must be strictly pre-Apex")

    transitions = 0
    for step_index in range(max_ticks):
        result = policy(state.obs, jax.random.fold_in(base_key, step_index))
        action = result[0] if isinstance(result, tuple) else result
        state = step_fn(state, action)
        transitions += 1

        apex_seen = _host_bool(state.info["events"].apex_seen)
        terminated = _host_bool(state.info.get("terminated", False))
        truncated = _host_bool(state.info.get("truncated", False))
        physical_failure = _host_bool(
            state.info.get("physical_failure", False)
        )
        timeout = _host_bool(state.info.get("timeout", False))
        end_code = _host_int(state.info.get("end_code", 0))
        if apex_seen or terminated or truncated or timeout:
            outcome = classify_upstream_outcome(
                apex_seen=apex_seen,
                terminated=terminated,
                truncated=truncated,
                physical_failure=physical_failure,
                timeout=timeout,
                end_code=end_code,
            )
            return {
                "branch_index": int(branch_index),
                "branch_seed": int(branch_seed),
                "transitions": int(transitions),
                **outcome,
            }

    outcome = classify_upstream_outcome(
        apex_seen=False,
        terminated=False,
        truncated=False,
        physical_failure=False,
        timeout=False,
        end_code=0,
        label_horizon=True,
    )
    return {
        "branch_index": int(branch_index),
        "branch_seed": int(branch_seed),
        "transitions": int(transitions),
        **outcome,
    }


def label_upstream_continuations(
    frozen_manifest: Path,
    catalog: Path,
    output_dir: Path,
    *,
    branches: int = 1,
    max_ticks: int = 400,
    protocol_seed: int = 820302,
    stochastic_policy: bool = False,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    validation_seeds: Sequence[int] = DEFAULT_VALIDATION_SEEDS,
    test_seeds: Sequence[int] = DEFAULT_TEST_SEEDS,
) -> dict[str, Any]:
    """Relabel pre-Apex candidates under the frozen pi_up_star."""
    if branches <= 0:
        raise ValueError("branches must be positive")
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    if not stochastic_policy and branches != 1:
        raise ValueError(
            "deterministic continuation labeling must use exactly one branch"
        )

    frozen = load_frozen_manifest(frozen_manifest)
    up_record = frozen["experts"]["pi_up_star"]
    up_config, up_payload = verify_frozen_record(up_record)
    if up_config.phase != "propulsion_ascent":
        raise ValueError("pi_up_star must use propulsion_ascent config")

    rows = _load_catalog(catalog)
    seed_splits, parent_splits = _seed_splits(
        rows,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )
    env = TwoPhaseBikeEnv(up_config)
    policy = make_checkpoint_policy(
        env, up_payload, deterministic=not stochastic_policy
    )
    step_fn = jax.jit(env.step)

    protocol = {
        "schema": CONTINUATION_LABEL_SCHEMA,
        "target": "V_up",
        "question": (
            "frozen pi_up_star reaches Apex before a retained terminal"
        ),
        "frozen_manifest_sha256": hashlib.sha256(
            Path(frozen_manifest).read_bytes()
        ).hexdigest(),
        "expert_payload_sha256": up_record["payload_sha256"],
        "expert_actor_sha256": up_record["actor_sha256"],
        "expert_config_sha256": up_record["config_sha256"],
        "xml_sha256": up_record["xml_sha256"],
        "catalog_sha256": catalog_sha256(catalog),
        "branches": int(branches),
        "max_ticks": int(max_ticks),
        "protocol_seed": int(protocol_seed),
        "policy_mode": "stochastic" if stochastic_policy else "deterministic",
        "split_unit": "global_seed",
        "train_seeds": sorted(
            seed for seed, split in seed_splits.items() if split == "train"
        ),
        "validation_seeds": sorted(
            seed
            for seed, split in seed_splits.items()
            if split == "validation"
        ),
        "test_seeds": sorted(
            seed for seed, split in seed_splits.items() if split == "test"
        ),
        "training_transitions": 0,
    }
    protocol_hash = _protocol_sha256(protocol)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {**protocol, "protocol_sha256": protocol_hash},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    base = Path(catalog).parent
    labeled: list[dict[str, Any]] = []
    total_transitions = 0
    closed_outcomes: Counter[str] = Counter()
    split_candidate_counts: Counter[str] = Counter()
    split_success_counts: Counter[str] = Counter()

    try:
        for row in rows:
            candidate_id = _candidate_id(row)
            snapshot = load_snapshot(
                base / row["source_bank"] / row["snapshot"]
            )
            branch_rows = []
            for branch_index in range(branches):
                branch_seed, base_key = derive_branch_key(
                    protocol_seed=protocol_seed,
                    candidate_id=candidate_id,
                    branch_index=branch_index,
                )
                branch = _run_upstream_branch(
                    env,
                    policy,
                    snapshot,
                    branch_seed=branch_seed,
                    base_key=base_key,
                    branch_index=branch_index,
                    max_ticks=max_ticks,
                    step_fn=step_fn,
                )
                total_transitions += int(branch["transitions"])
                closed_outcomes[str(branch["reason"])] += 1
                branch_rows.append(branch)

            success_count = sum(
                int(branch["success"]) for branch in branch_rows
            )
            physical_failure_count = sum(
                int(branch["physical_failure"]) for branch in branch_rows
            )
            task_failure_count = sum(
                int(branch["task_failure"]) for branch in branch_rows
            )
            timeout_count = sum(
                int(branch["timeout"]) for branch in branch_rows
            )
            if (
                success_count
                + physical_failure_count
                + task_failure_count
                + timeout_count
                != branches
            ):
                raise ValueError("continuation outcomes did not close")

            split = seed_splits[int(row["seed"])]
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
                    "actor_observation": np.asarray(
                        snapshot.observation, dtype=np.float32
                    ).tolist(),
                    "branch_count": branches,
                    "success_count": success_count,
                    "physical_failure_count": physical_failure_count,
                    "task_failure_count": task_failure_count,
                    "timeout_count": timeout_count,
                    "empirical_success_rate": success_count / branches,
                    "branches": branch_rows,
                    "expert_actor_sha256": up_record["actor_sha256"],
                    "protocol_sha256": protocol_hash,
                }
            )

        report = {
            "schema": CONTINUATION_LABEL_SCHEMA,
            "status": "completed",
            "target": "V_up",
            "protocol_sha256": protocol_hash,
            "training_transitions": 0,
            "labeling_transitions": total_transitions,
            "candidate_count": len(labeled),
            "branch_rollout_count": len(labeled) * branches,
            "success_rollouts": sum(
                row["success_count"] for row in labeled
            ),
            "physical_failure_rollouts": sum(
                row["physical_failure_count"] for row in labeled
            ),
            "task_failure_rollouts": sum(
                row["task_failure_count"] for row in labeled
            ),
            "timeout_rollouts": sum(
                row["timeout_count"] for row in labeled
            ),
            "closed_outcome_counts": dict(sorted(closed_outcomes.items())),
            "split_unit": "global_seed",
            "seed_split": {
                str(seed): split
                for seed, split in sorted(seed_splits.items())
            },
            "split_parent_counts": {
                split: len(
                    {
                        group
                        for group, value in parent_splits.items()
                        if value == split
                    }
                )
                for split in ("train", "validation", "test")
            },
            "split_candidate_counts": dict(
                sorted(split_candidate_counts.items())
            ),
            "split_positive_candidate_counts": dict(
                sorted(split_success_counts.items())
            ),
            "parent_split": parent_splits,
        }
        if (
            report["success_rollouts"]
            + report["physical_failure_rollouts"]
            + report["task_failure_rollouts"]
            + report["timeout_rollouts"]
            != report["branch_rollout_count"]
        ):
            raise ValueError(
                "global continuation outcome accounting did not close"
            )
        (output_dir / "labels.json").write_text(
            json.dumps(labeled, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    except Exception as exc:
        failure = {
            "schema": CONTINUATION_LABEL_SCHEMA,
            "status": "engineering_error",
            "target": "V_up",
            "protocol_sha256": protocol_hash,
            "training_transitions": 0,
            "labeling_transitions": total_transitions,
            "completed_candidates": len(labeled),
            "reason": f"{type(exc).__name__}: {exc}",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
