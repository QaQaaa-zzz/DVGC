"""Evaluate policy and bounded-sequence Takeoff controllers on one fixed bank."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd

from cli.search_takeoff_actions import SEQUENCES, action_at, reference_action_sequence
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, load_params, save_json

SUCCESSFUL_BOUNDED_SEQUENCES = (
    "hold", "extend_half", "hip_full_knee_half", "hip_half_knee_full",
    "preload_then_extend", "hold_then_extend", "reference_time_aligned",
)


def _rollout(env, step, row, seed, horizon, action_fn, support, noise, hip_q, knee_q):
    key = jax.random.PRNGKey(seed)
    state = restore_snapshot(env, row, key)
    reward = defaultdict(float)
    trace = []
    reason = "horizon_exhaustion"
    for tick in range(horizon):
        key, action_key, noise_key = jax.random.split(key, 3)
        action = jp.asarray(action_fn(state, action_key, tick, row), jp.float32)
        if noise:
            action = jp.clip(action + noise * jax.random.normal(noise_key, action.shape), -1, 1)
        state = step(state, action)
        for name, value in state.metrics.items():
            if name == "reward" or name.startswith("reward/"):
                reward[name] += float(np.asarray(jax.device_get(value)))
        qpos = np.asarray(jax.device_get(state.data.qpos))
        qvel = np.asarray(jax.device_get(state.data.qvel))
        ctrl = np.asarray(jax.device_get(state.data.ctrl))
        feature = np.asarray(jax.device_get(env._physical_feature(state.data)))
        entry = bool(float(np.asarray(jax.device_get(state.metrics["event/stage_entry"]))) > .5)
        trace.append({
            "tick": tick + 1, "action": np.asarray(action).tolist(),
            "hip": float(qpos[hip_q]), "knee": float(qpos[knee_q]),
            "root_z": float(qpos[2]), "vertical_velocity": float(qvel[2]),
            "pitch": float(feature[4]), "roll": float(feature[3]),
            "next_stage_entry": entry,
            "_qpos": qpos, "_qvel": qvel, "_ctrl": ctrl,
        })
        if entry:
            for frame in trace:
                contact = support.measure(frame.pop("_qpos"), frame.pop("_qvel"), frame.pop("_ctrl"))
                frame["wheel_contacts"] = int(contact["wheel_contacts"])
                frame["wheel_clearance_min"] = float(contact["wheel_min"])
                frame["body_contacts"] = int(contact["body_contacts"])
            return True, tick + 1, "next_stage_entry", dict(reward), trace
        if float(np.asarray(jax.device_get(state.done))) > .5:
            code = int(np.asarray(jax.device_get(state.info["end_code"])))
            reason = END_REASON.get(code, f"unknown_{code}")
            break
    for frame in trace:
        frame.pop("_qpos", None); frame.pop("_qvel", None); frame.pop("_ctrl", None)
    return False, None, reason, dict(reward), trace


def _summary(rows):
    groups = {}
    for kind in ("canonical_compressed", "reference_aligned_compressed", "all"):
        selected = rows if kind == "all" else [row for row in rows if row["candidate_kind"] == kind]
        ids = {row["candidate_id"] for row in selected}
        successful = {row["candidate_id"] for row in selected if row["success"]}
        groups[kind] = {
            "states": len(ids), "branches": len(selected),
            "successful_unique_states": len(successful),
            "successful_branches": sum(row["success"] for row in selected),
            "termination_reasons": dict(Counter(row["reason"] for row in selected)),
            "time_to_ascent": {
                "mean": float(np.mean([row["entry_tick"] for row in selected if row["success"]]))
                if successful else None,
                "min": min((row["entry_tick"] for row in selected if row["success"]), default=None),
                "max": max((row["entry_tick"] for row in selected if row["success"]), default=None),
            },
        }
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--policy", action="append", default=[], help="name=policy_dir")
    parser.add_argument("--include-bounded-sequences", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--action-noise", type=float, default=.03)
    parser.add_argument("--seed", type=int, default=10_100_000)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", default="data/reference_jump.csv")
    args = parser.parse_args()

    bank = SnapshotBank.load(args.bank)
    contract = bank.metadata.get("evaluation_contract", {})
    if contract.get("version") != "takeoff_balanced_eval_v1":
        raise SystemExit("fixed balanced Takeoff evaluation bank required")
    cfg = load_config(args.config, {
        "training_stage": "takeoff", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "takeoff_to_ascent",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    support = GroundSupportSolver(cfg.xml_path)
    hip_q = int(env.mj_model.jnt_qposadr[env.mj_model.joint("hip_joint").id])
    knee_q = int(env.mj_model.jnt_qposadr[env.mj_model.joint("knee_joint").id])
    reference = pd.read_csv(args.reference)
    controllers = []
    for spec in args.policy:
        name, path = spec.split("=", 1)
        inference = build_inference(env, load_params(Path(path) / "params.pkl"), deterministic=True)
        controllers.append((name, path, lambda state, key, tick, row, f=inference: f(state.obs, key)[0]))
    if args.include_bounded_sequences:
        names = SUCCESSFUL_BOUNDED_SEQUENCES
        for name in names:
            def scripted(state, key, tick, row, sequence_name=name):
                sequence = (reference_action_sequence(reference, row, args.horizon)
                            if sequence_name == "reference_time_aligned" else SEQUENCES[sequence_name])
                return action_at(sequence, tick)
            controllers.append((f"bounded:{name}", None, scripted))
    outcomes = []
    by_controller = {}
    for ci, (name, path, action_fn) in enumerate(controllers):
        rows = []
        for i, row in enumerate(bank.records):
            for branch in range(args.branches):
                success, tick, reason, rewards, trace = _rollout(
                    env, step, row, args.seed + ci * 1_000_000 + i * 100 + branch,
                    args.horizon, action_fn, support, args.action_noise, hip_q, knee_q,
                )
                item = {
                    "controller": name, "controller_policy": path,
                    "candidate_id": row["id"], "candidate_kind": row["candidate_kind"],
                    "reference_index": row.get("reference_index"), "branch": branch,
                    "success": success, "entry_tick": tick, "reason": reason,
                    "reward_breakdown": rewards, "trace": trace if success else [],
                }
                rows.append(item); outcomes.append(item)
        by_controller[name] = {
            "policy": path,
            "policy_sha256": file_sha256(Path(path) / "params.pkl") if path else None,
            "strata": _summary(rows),
        }
    union_rows = []
    for row in bank.records:
        candidate = [item for item in outcomes if item["candidate_id"] == row["id"]]
        union_rows.extend(candidate)
    union = _summary(union_rows)
    payload = {
        "status": "PASS", "artifact_role": "takeoff_controller_proposal_bank_evaluation",
        "bank": str(Path(args.bank).resolve()), "bank_sha256": file_sha256(args.bank),
        "evaluation_contract_sha256": contract["sha256"],
        "controllers": by_controller, "union_of_controllers": union,
        "outcomes": outcomes,
        "interpretation": "fresh current-protocol controller-bank evidence; failures are not physical negatives",
    }
    save_json(args.output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
