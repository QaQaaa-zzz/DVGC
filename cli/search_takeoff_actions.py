"""Policy-free bounded action scan on authenticity-audited Takeoff resets."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd

from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import STAGE_ID, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json
from dvgc.stage_reachability import evaluate_entry


SEQUENCES = {
    "hold": [(50, [0.0, 0.0, 0.0, 0.0])],
    "extend_half": [(50, [0.0, 0.0, 0.5, 0.5])],
    "extend_full": [(50, [0.0, 0.0, 1.0, 1.0])],
    "hip_full_knee_half": [(50, [0.0, 0.0, 1.0, 0.5])],
    "hip_half_knee_full": [(50, [0.0, 0.0, 0.5, 1.0])],
    "drive_extend_full": [(50, [0.0, 0.5, 1.0, 1.0])],
    "preload_then_extend": [(5, [0.0, 0.0, -0.5, -0.5]), (45, [0.0, 0.0, 1.0, 1.0])],
    "hold_then_extend": [(5, [0.0, 0.0, 0.0, 0.0]), (45, [0.0, 0.0, 1.0, 1.0])],
}


def action_at(spec, tick):
    elapsed = 0
    for steps, action in spec:
        elapsed += int(steps)
        if tick < elapsed:
            return jp.asarray(action, jp.float32)
    return jp.asarray(spec[-1][1], jp.float32)


def reference_action_sequence(reference: pd.DataFrame, row: dict, horizon: int):
    """Return the time-aligned open-loop reference controls for one proposal.

    This is an action-direction/controllability probe, not a rollout label or
    policy.  Candidate provenance chooses the first row; no success outcome is
    consulted.
    """
    start = row.get("reference_index")
    if start is None:
        start = row.get("source_reference_index")
    if start is None:
        start = row.get("source_index", 0)
    start = int(start)
    start = max(0, min(start, len(reference) - 1))
    actions = []
    for tick in range(horizon):
        source = reference.iloc[min(start + tick, len(reference) - 1)]
        actions.append(
            (
                1,
                [
                    float(source.action_steering),
                    float(source.action_rearwheel),
                    float(source.action_hip),
                    float(source.action_knee),
                ],
            )
        )
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9_930_000)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", default="data/reference_jump.csv")
    args = parser.parse_args()
    bank = SnapshotBank.load(args.bank)
    if bank.metadata.get("reset_protocol", {}).get("version") != "takeoff_reset_authenticity_v3":
        raise SystemExit("Action scan requires takeoff_reset_authenticity_v3")
    rows = bank.records[:args.states]
    cfg = load_config(args.config, {
        "training_stage": "takeoff", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    reference = pd.read_csv(args.reference)
    step_fn = jax.jit(env.step)
    support = GroundSupportSolver(cfg.xml_path)
    hip_q = int(env.mj_model.jnt_qposadr[env.mj_model.joint("hip_joint").id])
    knee_q = int(env.mj_model.jnt_qposadr[env.mj_model.joint("knee_joint").id])
    root_q = int(env.mj_model.jnt_qposadr[env.mj_model.joint("floating_base_joint").id])
    outcomes = []
    reasons = Counter()
    successful_states = set()

    for state_index, row in enumerate(rows):
        sequences = dict(SEQUENCES)
        sequences["reference_time_aligned"] = reference_action_sequence(
            reference, row, args.horizon
        )
        for sequence_name, sequence in sequences.items():
            state = restore_snapshot(env, row, jax.random.PRNGKey(args.seed + state_index))
            previous_vz = float(np.asarray(jax.device_get(state.data.qvel[2])))
            trace = []
            success = False
            entry_tick = None
            reason = "horizon_exhaustion"
            for tick in range(args.horizon):
                action = action_at(sequence, tick)
                state = step_fn(state, action)
                qpos = np.asarray(jax.device_get(state.data.qpos))
                qvel = np.asarray(jax.device_get(state.data.qvel))
                ctrl = np.asarray(jax.device_get(state.data.ctrl))
                contact = support.measure(qpos, qvel, ctrl)
                sample = sample_from_state(env, state, previous_vz)
                sample["dual_wheel_airborne"] = bool(
                    sample["dual_wheel_airborne"]
                    and contact["wheel_contacts"] == 0
                    and contact["wheel_min"] >= 0.002
                )
                entry = evaluate_entry("takeoff", sample, cfg)
                actuator_force = np.asarray(
                    jax.device_get(getattr(state.data, "actuator_force", state.data.ctrl))
                )
                trace.append({
                    "tick": tick + 1,
                    "action": np.asarray(action).tolist(),
                    "hip": float(qpos[hip_q]), "knee": float(qpos[knee_q]),
                    "hip_velocity": float(qvel[int(env.mj_model.jnt_dofadr[env.mj_model.joint("hip_joint").id])]),
                    "knee_velocity": float(qvel[int(env.mj_model.jnt_dofadr[env.mj_model.joint("knee_joint").id])]),
                    "hip_torque": float(actuator_force[env.mj_model.actuator("cmd_hip_f").id]),
                    "knee_torque": float(actuator_force[env.mj_model.actuator("cmd_knee_f").id]),
                    "root_z": float(qpos[root_q + 2]), "vertical_velocity": float(qvel[2]),
                    "wheel_clearance_min": float(contact["wheel_min"]),
                    "wheel_contacts": int(contact["wheel_contacts"]),
                    "body_contacts": int(contact["body_contacts"]),
                    "next_stage_entry": bool(entry["valid"]),
                })
                if entry["valid"]:
                    success, entry_tick, reason = True, tick + 1, "next_stage_entry"
                    successful_states.add(row["id"])
                    break
                if float(np.asarray(jax.device_get(state.done))) > 0.5:
                    code = int(np.asarray(jax.device_get(state.info["end_code"])))
                    reason = END_REASON.get(code, f"unknown_{code}")
                    break
                previous_vz = float(sample["physical_feature"][8])
            reasons[reason] += 1
            outcomes.append({
                "state_index": state_index, "candidate_id": row["id"],
                "candidate_kind": row["candidate_kind"],
                "reference_index": row.get("reference_index"),
                "sequence": sequence_name,
                "action_sequence": [{"steps": n, "action": a} for n, a in sequence],
                "success": success, "entry_tick": entry_tick,
                "failure_reason": None if success else reason,
                "trace": trace,
            })
    payload = {
        "status": "PASS",
        "artifact_role": "takeoff_bounded_action_search",
        "policy_used": False,
        "bank": str(Path(args.bank).resolve()),
        "bank_sha256": file_sha256(args.bank),
        "reset_protocol_sha256": bank.metadata["reset_protocol_sha256"],
        "states": len(rows),
        "sequences": len(SEQUENCES) + 1,
        "horizon": int(args.horizon),
        "successful_unique_states": len(successful_states),
        "successful_branches": sum(item["success"] for item in outcomes),
        "termination_reasons": dict(reasons),
        "outcomes": outcomes,
    }
    save_json(args.output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
