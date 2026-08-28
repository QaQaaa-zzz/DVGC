#!/usr/bin/env python3
"""Audit and collect the TRAIN V_up reachable-boundary pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from jit_dvgc.constants import ACTION_ORDER
from jit_dvgc.continuation_labels import DEFAULT_TRAIN_SEEDS
from jit_dvgc.env import TwoPhaseBikeEnv
from jit_dvgc.expert_freeze import load_frozen_manifest, verify_frozen_record
from jit_dvgc.ppo import make_checkpoint_policy
from jit_dvgc.upstream_boundary import (
    DEFAULT_BOUNDARY_DURATIONS,
    DEFAULT_BOUNDARY_STRENGTHS,
    DEFAULT_NEAR_ATOL,
    audit_and_select_train_anchors,
    collect_reachable_boundary_candidates,
    file_sha256,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-training-transitions", type=int, default=4_988_928)
    parser.add_argument("--negative-role", default="ascending_entry")
    parser.add_argument("--failure-reason", default="pitch_limit")
    parser.add_argument(
        "--guard-roles",
        nargs="*",
        default=["jump_zone_entry", "height_entry"],
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_BOUNDARY_STRENGTHS),
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=list(DEFAULT_BOUNDARY_DURATIONS),
    )
    parser.add_argument(
        "--action-names",
        nargs="+",
        choices=list(ACTION_ORDER),
        default=list(ACTION_ORDER),
        help="action axes to perturb; defaults to all axes",
    )
    parser.add_argument(
        "--signs",
        type=int,
        nargs="+",
        choices=(-1, 1),
        default=[-1, 1],
        help="perturbation signs to use; defaults to both -1 and +1",
    )
    parser.add_argument("--protocol-seed", type=int, default=820401)
    parser.add_argument("--max-negative-anchors", type=int, default=1)
    parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_SEEDS),
    )
    parser.add_argument("--near-qpos-atol", type=float, default=DEFAULT_NEAR_ATOL)
    parser.add_argument("--near-qvel-atol", type=float, default=DEFAULT_NEAR_ATOL)
    parser.add_argument(
        "--near-observation-atol", type=float, default=DEFAULT_NEAR_ATOL
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="write audit.json without constructing MuJoCo/MJX or collecting candidates",
    )
    args = parser.parse_args()

    train_seeds = tuple(int(seed) for seed in args.train_seeds)
    if not train_seeds or len(set(train_seeds)) != len(train_seeds):
        parser.error("--train-seeds must be non-empty and unique")
    if len(set(args.action_names)) != len(args.action_names):
        parser.error("--action-names must be unique")
    if len(set(args.signs)) != len(args.signs):
        parser.error("--signs must be unique")

    selected, audit = audit_and_select_train_anchors(
        args.catalog,
        args.labels,
        train_seeds=train_seeds,
        source_training_transitions=args.source_training_transitions,
        negative_role=args.negative_role,
        failure_reason=args.failure_reason,
        guard_roles=tuple(args.guard_roles),
        max_negative_anchors=args.max_negative_anchors,
        qpos_atol=args.near_qpos_atol,
        qvel_atol=args.near_qvel_atol,
        observation_atol=args.near_observation_atol,
    )

    if args.audit_only:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(output_dir / "audit.json", audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0

    frozen = load_frozen_manifest(args.frozen_manifest)
    pi_up_record = frozen["experts"]["pi_up_star"]
    config, payload = verify_frozen_record(pi_up_record)
    if config.phase != "propulsion_ascent":
        raise ValueError("pi_up_star must use propulsion_ascent config")

    env = TwoPhaseBikeEnv(config)
    if pi_up_record["xml_sha256"] != env._bundle.xml_sha256:
        raise ValueError("frozen pi_up_star XML identity mismatch")

    policy = make_checkpoint_policy(env, payload, deterministic=True)
    step_fn = jax.jit(env.step)
    base_key = jax.random.PRNGKey(args.protocol_seed)

    def policy_action(state, variant_index: int, perturb_step: int):
        variant_key = jax.random.fold_in(base_key, int(variant_index))
        step_key = jax.random.fold_in(variant_key, int(perturb_step))
        result = policy(state.obs, step_key)
        action = result[0] if isinstance(result, tuple) else result
        return np.asarray(jax.device_get(action), dtype=np.float32)

    def capture(state, anchor):
        return env.capture_handoff_snapshot(
            state,
            policy_sha256=pi_up_record["actor_sha256"],
            parent_trajectory=str(anchor.row["parent_group_id"]),
            parent_tick=int(anchor.row["tick"]),
            policy_identity="pi_up_star_reachable_boundary_perturbation",
        )

    protocol = {
        "target": "V_up",
        "purpose": "unique reachable continuation boundary pilot",
        "protocol_seed": int(args.protocol_seed),
        "frozen_manifest_sha256": file_sha256(args.frozen_manifest),
        "frozen_pi_up_actor_sha256": pi_up_record["actor_sha256"],
        "frozen_pi_up_payload_sha256": pi_up_record["payload_sha256"],
        "frozen_pi_up_config_sha256": pi_up_record["config_sha256"],
        "xml_sha256": pi_up_record["xml_sha256"],
        "nominal_catalog_sha256": file_sha256(args.catalog),
        "nominal_labels_sha256": file_sha256(args.labels),
        "source_training_transitions": int(args.source_training_transitions),
        "negative_role": str(args.negative_role),
        "failure_reason": str(args.failure_reason),
        "guard_roles": list(args.guard_roles),
        "train_seeds": sorted(train_seeds),
        "max_negative_anchors": int(args.max_negative_anchors),
        "selected_action_names": list(args.action_names),
        "selected_signs": list(args.signs),
        "near_duplicate_tolerances": {
            "qpos_atol": float(args.near_qpos_atol),
            "qvel_atol": float(args.near_qvel_atol),
            "observation_atol": float(args.near_observation_atol),
        },
        "state_generation": (
            "restore real nominal snapshot; recompute deterministic pi_up_star action; "
            "add one bounded selected action-basis perturbation for duration ticks; "
            "env.step; exclude terminal/Apex; capture real resulting snapshot"
        ),
    }

    report = collect_reachable_boundary_candidates(
        selected,
        args.output_dir,
        restore=env.restore_handoff_snapshot,
        policy_action=policy_action,
        step=step_fn,
        capture=capture,
        protocol=protocol,
        strengths=tuple(args.strengths),
        durations=tuple(args.durations),
        action_names=tuple(args.action_names),
        signs=tuple(args.signs),
    )
    _write_json(Path(args.output_dir) / "audit.json", audit)
    print(
        json.dumps(
            {
                "audit": audit,
                "collection": {
                    key: value for key, value in report.items() if key != "entries"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
