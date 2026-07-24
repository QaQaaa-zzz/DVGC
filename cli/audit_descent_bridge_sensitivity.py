"""Event-local finite-difference audit for the natural MJX Descent bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.continuous import DescentSupportMatcher
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.runtime import save_json


CHANNELS = ("steer", "drive", "hip", "knee")


def _reference_action(env, cfg, state, reference, tick, pulse_ticks, offset, stride):
    if tick < pulse_ticks:
        return np.asarray([0.0, 1.0, 0.25, 1.0], np.float32)
    row = reference.iloc[min(offset + (tick - pulse_ticks) * stride,
                             len(reference) - 1)]
    qpos = np.asarray(state.data.qpos)
    hip = float(qpos[env._joint_qpos["hip_joint"]])
    knee = float(qpos[env._joint_qpos["knee_joint"]])
    hip_target = float(row.hip_position)
    knee_target = float(row.knee_position)
    hip_action = (
        (hip_target - cfg.hip_initial) / (cfg.hip_max - cfg.hip_initial)
        if hip_target >= cfg.hip_initial
        else (hip_target - cfg.hip_initial) / (cfg.hip_initial - cfg.hip_min)
    )
    knee_action = (knee - knee_target) / cfg.knee_action_target_delta
    return np.clip(
        np.asarray([0.0, 1.0, hip_action, knee_action], np.float32),
        -1.0, 1.0,
    )


def _distance(env, matcher, state, apex_crossed):
    entry, distance = matcher.evaluate(state, apex_crossed=apex_crossed)
    feature = np.asarray(env._physical_feature(state.data), np.float64)
    return entry, distance, feature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default="data/reference_jump.csv")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seed", type=int, default=12_320_000)
    parser.add_argument("--offset", type=int, default=105)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--pulse-ticks", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--horizon", type=int, default=80)
    args = parser.parse_args()
    support = SnapshotBank.load(args.support_bank)
    cfg = load_config(args.config, {
        "training_stage": "full",
        "use_bank_resets": False,
        "stage_reachability_objective": "",
        "expert_chain_termination": False,
        "domain_randomization": False,
        "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(
        cfg, snapshot_bank=SnapshotBank(), stage_support_bank=support
    )
    step = jax.jit(env.step)
    matcher = DescentSupportMatcher(env)
    reference = pd.read_csv(args.reference)
    state = env.reset(jax.random.PRNGKey(args.seed))
    approach_ticks = 0
    while int(state.info["phase"]) < 1 and not float(state.done):
        state = step(state, np.asarray([0.0, 1.0, 0.0, 0.0], np.float32))
        approach_ticks += 1
    states = [state]
    actions = []
    trace = []
    previous_vz = float(state.data.qvel[2])
    positive_seen = previous_vz > 0.0
    apex_tick = None
    handoff_tick = None
    closest = None
    for tick in range(args.horizon):
        action = _reference_action(
            env, cfg, state, reference, tick, args.pulse_ticks,
            args.offset, args.stride,
        )
        state = step(state, action)
        actions.append(action)
        states.append(state)
        vz = float(state.data.qvel[2])
        positive_seen |= vz > 0.0
        if (
            apex_tick is None and positive_seen
            and previous_vz > 0.0 and vz <= 0.0 and not float(state.done)
        ):
            apex_tick = tick + 1
            handoff_tick = apex_tick + 1
        previous_vz = vz
        entry, distance, feature = _distance(
            env, matcher, state, apex_crossed=apex_tick is not None
        )
        row = {
            "tick": tick + 1,
            "distance": distance,
            "entry": entry,
            "feature": feature.tolist(),
            "action": action.tolist(),
            "end_code": int(state.info["end_code"]),
        }
        trace.append(row)
        if (
            apex_tick is not None and not float(state.done)
            and (closest is None or distance < closest["distance"])
        ):
            closest = row
        if float(state.done):
            break
    failure_tick = len(trace) if float(state.done) else None
    if closest is None:
        raise RuntimeError("nominal trajectory never produced a descending state")

    sensitivities = []
    for window in (4, 8, 12):
        start_tick = max(0, closest["tick"] - window)
        actual_window = closest["tick"] - start_tick
        for channel in range(4):
            endpoints = {}
            for sign in (-1.0, 1.0):
                branch = states[start_tick]
                branch_apex = apex_tick is not None and start_tick >= apex_tick
                best_distance = float("inf")
                minimum_pitch_margin = float("inf")
                failed = False
                endpoint_feature = None
                for local in range(actual_window):
                    global_tick = start_tick + local
                    nominal = _reference_action(
                        env, cfg, branch, reference, global_tick,
                        args.pulse_ticks, args.offset, args.stride,
                    )
                    residual = np.zeros(4, np.float32)
                    residual[channel] = sign * args.epsilon
                    branch = step(branch, np.clip(nominal + residual, -1, 1))
                    vz = float(branch.data.qvel[2])
                    branch_apex |= (
                        global_tick + 1 >= (apex_tick or args.horizon + 1)
                    )
                    _, distance, endpoint_feature = _distance(
                        env, matcher, branch, apex_crossed=branch_apex
                    )
                    pitch_margin = (
                        np.deg2rad(float(cfg.max_pitch_deg))
                        - abs(float(endpoint_feature[4]))
                    )
                    best_distance = min(best_distance, distance)
                    minimum_pitch_margin = min(
                        minimum_pitch_margin, pitch_margin
                    )
                    if float(branch.done):
                        failed = True
                        break
                endpoints[sign] = {
                    "distance": best_distance,
                    "pitch_margin": minimum_pitch_margin,
                    "feature": endpoint_feature.tolist(),
                    "failed": failed,
                    "termination_reason": END_REASON.get(
                        int(branch.info["end_code"]), "unknown"
                    ),
                }
            minus, plus = endpoints[-1.0], endpoints[1.0]
            derivative = (
                np.asarray(plus["feature"]) - np.asarray(minus["feature"])
            ) / (2.0 * args.epsilon)
            sensitivities.append({
                "window_ticks": window,
                "actual_ticks": actual_window,
                "start_tick": start_tick,
                "channel": CHANNELS[channel],
                "feature_derivative": {
                    name: float(derivative[index])
                    for index, name in enumerate(
                        support.metadata["stage_entry_matcher"]["feature_names"]
                    )
                    if name in (
                        "x", "z", "vx", "vz", "pitch", "wy",
                        "rearwheel_velocity",
                    )
                },
                "distance_derivative": (
                    plus["distance"] - minus["distance"]
                ) / (2.0 * args.epsilon),
                "pitch_margin_derivative": (
                    plus["pitch_margin"] - minus["pitch_margin"]
                ) / (2.0 * args.epsilon),
                "minus": minus,
                "plus": plus,
            })
    useful = [
        row for row in sensitivities
        if not row["minus"]["failed"] and not row["plus"]["failed"]
        and (
            abs(row["distance_derivative"]) >= 0.1
            or abs(row["pitch_margin_derivative"]) >= 0.01
        )
    ]
    major_residual_response = [
        row for row in useful
        if abs(row["distance_derivative"]) >= 1.0
        and any(
            abs(row["feature_derivative"].get(name, 0.0)) >= threshold
            for name, threshold in (
                ("vx", 0.25),
                ("rearwheel_velocity", 1.0),
                ("z", 0.02),
            )
        )
    ]
    selected_window = min(
        (row["window_ticks"] for row in major_residual_response), default=12
    )
    report = {
        "status": "PASS",
        "artifact_role": "descent_support_entry_bridge_sensitivity",
        "development_runtime": "MJX",
        "effective_solver": env._effective_mjx_solver,
        "fixed_dynamics_variant": True,
        "snapshot_discovery_only": True,
        "ppo_authorization": False,
        "nominal": {
            "approach_ticks": approach_ticks,
            "apex_tick": apex_tick,
            "handoff_tick": handoff_tick,
            "closest_tick": closest["tick"],
            "closest_distance": closest["distance"],
            "closest_feature": closest["feature"],
            "failure_tick": failure_tick,
            "failure_reason": END_REASON.get(
                int(state.info["end_code"]), "unknown"
            ),
            "pitch_failure_after_handoff_ticks": (
                failure_tick - handoff_tick
                if failure_tick is not None and handoff_tick is not None
                else None
            ),
        },
        "selected_shortest_window_ticks": selected_window,
        "sensitivities": sensitivities,
        "support_bank_sha256": file_sha256(args.support_bank),
        "xml_sha256": file_sha256(cfg.xml_path),
        "matcher_sha256": support.metadata[
            "stage_entry_matcher"
        ]["matcher_sha256"],
    }
    save_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "nominal": report["nominal"],
        "selected_shortest_window_ticks": selected_window,
        "useful_sensitivities": useful,
    }, indent=2))


if __name__ == "__main__":
    main()
