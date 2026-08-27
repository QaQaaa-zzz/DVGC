#!/usr/bin/env python3
"""Collect nominal V_up continuation candidates from real Phase U trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import jax

from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from jit_dvgc.env import TwoPhaseBikeEnv
from jit_dvgc.expert_freeze import load_frozen_manifest, verify_frozen_record
from jit_dvgc.handoff_bank import BankCollector, pytree_sha256
from jit_dvgc.ppo import make_checkpoint_policy
from jit_dvgc.upstream_candidates import (
    UPSTREAM_SELECTION_ROLES,
    collect_upstream_streaming_rollout,
)


DEFAULT_SEEDS = tuple(range(1000001, 1000009))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-manifest", type=Path, required=True)
    p.add_argument("--source-checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--max-ticks", type=int, default=400)
    args = p.parse_args()

    if args.max_ticks <= 0:
        p.error("--max-ticks must be positive")
    seeds = tuple(int(seed) for seed in args.seeds)
    if len(set(seeds)) != len(seeds):
        p.error("--seeds must be unique")
    if not seeds:
        p.error("--seeds must be non-empty")

    frozen = load_frozen_manifest(args.frozen_manifest)
    pi_up_record = frozen["experts"]["pi_up_star"]
    config, _ = verify_frozen_record(pi_up_record)
    if config.phase != "propulsion_ascent":
        raise ValueError("pi_up_star must use propulsion_ascent config")

    env = TwoPhaseBikeEnv(config)
    identity = CheckpointIdentity(
        config_sha256=config.config_sha256,
        xml_sha256=env._bundle.xml_sha256,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )
    if pi_up_record["xml_sha256"] != env._bundle.xml_sha256:
        raise ValueError("frozen pi_up_star XML identity mismatch")

    source_records = []
    seen_transitions: set[int] = set()
    for checkpoint_path in args.source_checkpoints:
        payload = load_checkpoint(checkpoint_path, expected=identity)
        transitions = int(payload.training_transitions)
        if transitions in seen_transitions:
            raise ValueError(f"duplicate source training transition: {transitions}")
        seen_transitions.add(transitions)
        source_records.append(
            {
                "path": Path(checkpoint_path),
                "payload": payload,
                "training_transitions": transitions,
                "actor_sha256": pytree_sha256(payload.actor_params),
            }
        )
    source_records.sort(key=lambda row: row["training_transitions"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    catalog_entries: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    total_transitions = 0
    role_counts: Counter[str] = Counter()

    reset_fn = jax.jit(env.reset_natural)

    try:
        for source in source_records:
            transitions = source["training_transitions"]
            bank_name = f"source_{transitions}"
            bank_dir = output_dir / bank_name
            collector = BankCollector(
                bank_dir,
                "nominal V_up continuation candidate collection",
                str(source["path"]),
                config.config_sha256,
                env._bundle.xml_sha256,
                source["actor_sha256"],
                len(seeds) * args.max_ticks,
            )
            collector.max_ticks = args.max_ticks
            source_policy = make_checkpoint_policy(
                env, source["payload"], deterministic=True
            )
            policy_step = jax.jit(
                lambda state, key: env.step(state, source_policy(state.obs, key)[0])
            )

            for seed in seeds:
                base_key = jax.random.PRNGKey(seed)
                state = reset_fn(base_key)
                parent_group_id = f"transition_{transitions}__{seed}"
                collect_upstream_streaming_rollout(
                    collector,
                    state,
                    seed=seed,
                    step=lambda s, tick, key=base_key: policy_step(
                        s, jax.random.fold_in(key, tick)
                    ),
                    capture=lambda s, parent, tick, policy_sha=source[
                        "actor_sha256"
                    ]: env.capture_handoff_snapshot(
                        s,
                        policy_sha256=policy_sha,
                        parent_trajectory=parent,
                        parent_tick=tick,
                    ),
                    parent_trajectory=parent_group_id,
                    max_ticks=args.max_ticks,
                )

            manifest = collector.close()
            total_transitions += int(collector.transitions)
            for entry in collector.entries:
                role_counts[str(entry["role"])] += 1
                catalog_entries.append(
                    {
                        **entry,
                        "source_bank": bank_name,
                        "parent_group_id": entry["parent_trajectory"],
                        "source_checkpoint": str(source["path"]),
                        "source_training_transitions": transitions,
                        "source_actor_sha256": source["actor_sha256"],
                    }
                )
            source_summaries.append(
                {
                    "source_bank": bank_name,
                    "checkpoint": str(source["path"]),
                    "training_transitions": transitions,
                    "actor_sha256": source["actor_sha256"],
                    "snapshot_count": len(collector.entries),
                    "collection_transitions": int(collector.transitions),
                    "status": manifest["status"],
                }
            )

        report = {
            "schema": "jit_upstream_candidate_catalog_v1",
            "status": "completed",
            "target": "V_up",
            "selection": "semantic_pre_apex_nominal",
            "selection_roles": list(UPSTREAM_SELECTION_ROLES),
            "frozen_pi_up_actor_sha256": pi_up_record["actor_sha256"],
            "config_sha256": config.config_sha256,
            "xml_sha256": env._bundle.xml_sha256,
            "seeds": list(seeds),
            "source_count": len(source_records),
            "parent_count": len(source_records) * len(seeds),
            "candidate_count": len(catalog_entries),
            "collection_transitions": total_transitions,
            "role_counts": dict(sorted(role_counts.items())),
            "sources": source_summaries,
            "entries": catalog_entries,
            "training_transitions": 0,
        }
        _write_json(output_dir / "catalog.json", report)
        _write_json(
            output_dir / "summary.json",
            {key: value for key, value in report.items() if key != "entries"},
        )
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "entries"},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        failure = {
            "schema": "jit_upstream_candidate_catalog_v1",
            "status": "engineering_error",
            "target": "V_up",
            "candidate_count": len(catalog_entries),
            "collection_transitions": total_transitions,
            "training_transitions": 0,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        _write_json(output_dir / "summary.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
