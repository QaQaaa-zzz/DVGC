"""Bounded roll-feedback scan on one natural-start MJX lineage."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.runtime import save_json


def _natural_takeoff(env, step, seed):
    state = env.reset(jax.random.PRNGKey(seed))
    ticks = 0
    while int(state.info["phase"]) < 1 and not float(state.done):
        state = step(state, np.zeros(env.action_size, np.float32))
        ticks += 1
    if float(state.done):
        raise RuntimeError("natural reset terminated before Takeoff")
    return state, ticks


def _rollout(env, step, start, knots, width, horizon, kr, kw, drive_offset):
    state = start
    previous_vz = float(state.data.qvel[2])
    airborne = apex = landing = final = False
    apex_tick = None
    maximum_x = float(state.data.qpos[0])
    maximum_z = float(state.data.qpos[2])
    minimum_roll_margin = float("inf")
    for tick in range(horizon):
        if tick // width < len(knots):
            action = knots[tick // width].copy()
        else:
            action = np.asarray([0.0, 1.0, 0.0, 0.0], np.float32)
        feature = np.asarray(jax.device_get(env._physical_feature(state.data)))
        action[0] = np.clip(
            action[0] + kr * feature[3] + kw * feature[9], -1.0, 1.0
        )
        action[1] = np.clip(action[1] + drive_offset, -1.0, 1.0)
        state = step(state, action)
        feature = np.asarray(jax.device_get(env._physical_feature(state.data)))
        airborne |= bool(int(state.info["had_airborne"]))
        vz = float(feature[8])
        if airborne and previous_vz > 0.0 and vz <= 0.0 and not float(state.done):
            apex = True
            apex_tick = tick + 1 if apex_tick is None else apex_tick
        previous_vz = vz
        landing |= bool(int(state.info["had_valid_landing"]))
        final |= bool(int(state.info["recovery_success"]))
        maximum_x = max(maximum_x, float(feature[0]))
        maximum_z = max(maximum_z, float(feature[2]))
        minimum_roll_margin = min(
            minimum_roll_margin,
            np.deg2rad(float(env._config.max_roll_deg)) - abs(float(feature[3])),
        )
        if float(state.done):
            break
    return {
        "kr": float(kr),
        "kw": float(kw),
        "drive_offset": float(drive_offset),
        "steps": tick + 1,
        "airborne": airborne,
        "apex": apex,
        "apex_tick": apex_tick,
        "landing": landing,
        "final_recovery": final,
        "maximum_x": maximum_x,
        "maximum_z": maximum_z,
        "minimum_roll_margin": minimum_roll_margin,
        "termination_reason": END_REASON.get(
            int(state.info["end_code"]), "unknown"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-support", required=True)
    parser.add_argument("--initial-sequence", required=True)
    parser.add_argument("--seed", type=int, default=12_180_000)
    parser.add_argument("--knot-width", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.initial_sequence).read_text())
    knots = np.asarray(payload["action_knots"], np.float32)
    support = SnapshotBank.load(args.stage_support)
    config = load_config(args.config, {
        "training_stage": "full",
        "use_bank_resets": False,
        "stage_reachability_objective": "",
        "expert_chain_termination": False,
        "domain_randomization": False,
        "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(
        config, snapshot_bank=SnapshotBank(), stage_support_bank=support
    )
    step = jax.jit(env.step)
    start, approach_ticks = _natural_takeoff(env, step, args.seed)
    gains = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0)
    rate_gains = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    drive_offsets = (0.0, 0.3)
    rows = [
        _rollout(
            env, step, start, knots, args.knot_width, args.horizon,
            kr, kw, drive,
        )
        for kr, kw, drive in itertools.product(
            gains, rate_gains, drive_offsets
        )
    ]
    rows.sort(key=lambda row: (
        row["final_recovery"], row["landing"], row["apex"],
        row["maximum_x"], row["steps"],
    ), reverse=True)
    report = {
        "status": "PASS" if rows[0]["landing"] else "FAIL",
        "artifact_role": "mjx_natural_continuous_feedback_scan",
        "development_runtime": "MJX",
        "fixed_dynamics_variant": True,
        "snapshot_restore": False,
        "ppo_authorization": False,
        "natural_approach_ticks": approach_ticks,
        "evaluated": len(rows),
        "best": rows[0],
        "top10": rows[:10],
        "stage_support_sha256": file_sha256(args.stage_support),
        "xml_sha256": file_sha256(config.xml_path),
        "effective_solver": env._effective_mjx_solver,
    }
    save_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(40)


if __name__ == "__main__":
    main()
