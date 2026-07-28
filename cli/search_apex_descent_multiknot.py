"""Finite-horizon multi-knot Apex->Descent bridge search and failure typing."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import frozen_rollout, restore_snapshot
from dvgc.runtime import build_inference, save_json
from dvgc.stage_reachability import evaluate_entry


FEATURE_NAMES = (
    "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz",
    "wx", "wy", "wz", "steer", "hip", "knee", "rearwheel_velocity",
)


def is_dynamically_reached_apex(row: dict) -> bool:
    """Recognize legacy and event-aligned dynamic Apex proposals."""
    return bool(
        row.get("candidate_kind") == "apex_dynamically_reached"
        or (
            row.get("candidate_kind") == "stage_entry_snapshot"
            and row.get("entry_from_stage") == "ascent"
            and row.get("entry_to_stage") == "apex"
        )
    )


def _round_a():
    specs = []
    correction = (
        (0., 0.), (-.2, -.2), (-.35, -.15), (-.15, -.35),
        (.15, -.25), (-.25, .15),
    )
    for coast in (4, 8, 12):
        for duration in (8, 16):
            for hip, knee in correction:
                specs.append({
                    "round": "A", "coast": coast, "correction_duration": duration,
                    "hip": hip, "knee": knee, "post_duration": 16,
                    "post_hip": float(np.clip(-.35 * hip, -.2, .2)),
                    "post_knee": float(np.clip(-.35 * knee, -.2, .2)),
                })
    return specs


def _round_b(best, seed, count):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        rows.append({
            "round": "B",
            "coast": int(np.clip(best["coast"] + rng.integers(-4, 5), 0, 20)),
            "correction_duration": int(np.clip(
                best["correction_duration"] + rng.integers(-6, 7), 4, 24
            )),
            "hip": float(np.clip(best["hip"] + rng.normal(0, .15), -.6, .4)),
            "knee": float(np.clip(best["knee"] + rng.normal(0, .15), -.6, .4)),
            "post_duration": int(rng.integers(8, 25)),
            "post_hip": float(np.clip(best["post_hip"] + rng.normal(0, .08), -.3, .3)),
            "post_knee": float(np.clip(best["post_knee"] + rng.normal(0, .08), -.3, .3)),
        })
    return rows


def _action(spec, tick):
    if tick < spec["coast"]:
        return jp.zeros((4,), jp.float32)
    if tick < spec["coast"] + spec["correction_duration"]:
        return jp.asarray([0., 0., spec["hip"], spec["knee"]], jp.float32)
    if tick < spec["coast"] + spec["correction_duration"] + spec["post_duration"]:
        return jp.asarray(
            [0., 0., spec["post_hip"], spec["post_knee"]], jp.float32
        )
    return jp.zeros((4,), jp.float32)


def _nearest(feature, support, center, scale):
    squared = np.square((support - feature[None, :]) / scale)
    index = int(np.argmin(np.sum(squared, axis=1)))
    return float(np.sqrt(np.sum(squared[index]))), index, squared[index]


def _failure_class(result, radius):
    if result["valid_descent_entry"]:
        if not result["downstream_controller_success"]:
            return "downstream_controller_gap"
        if not result["final_landing_recovery"]:
            return "final_recovery_gap"
        return "full_downstream_success"
    if not result["apex_crossed"]:
        return "apex_not_crossed"
    if result["failure_reason"] in ("pitch_limit", "roll_limit"):
        return "pose_instability_before_descent"
    if result["minimum_joint_margin"] < .01:
        return "joint_margin_exhausted"
    if result["minimum_support_distance"] is not None and result["minimum_support_distance"] > radius:
        return "support_metric_mismatch"
    return "crossed_apex_detector_missed"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apex-bank", required=True)
    p.add_argument("--support-bank", required=True)
    p.add_argument("--descent-policy", required=True)
    p.add_argument("--landing-policy", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--downstream-horizon", type=int, default=200)
    p.add_argument("--round-b-proposals", type=int, default=48)
    p.add_argument("--seed", type=int, default=11_100_000)
    a = p.parse_args()
    bank = SnapshotBank.load(a.apex_bank)
    support_bank = SnapshotBank.load(a.support_bank)
    support_metadata = dict(support_bank.metadata)
    support_metadata["support_features"] = [
        r["physical_feature"] for r in support_bank.records
    ]
    matcher = support_bank.metadata["stage_entry_matcher"]
    center = np.asarray(matcher["center"], float)
    scale = np.asarray(matcher["scale"], float)
    support = np.asarray([r["physical_feature"] for r in support_bank.records], float)
    dp, dc, dm = load_bundle(a.descent_policy, verify_files=True)
    lp, lc, lm = load_bundle(a.landing_policy, verify_files=True)
    cfg = load_config(a.config, {
        **dc, "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    lcfg = load_config(a.config, {
        **lc, "training_stage": "landing", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(),
                         stage_support_bank=support_bank)
    lenv = OrangeBikeDVGC(lcfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step); lstep = jax.jit(lenv.step)
    dinfer = build_inference(env, dp, deterministic=True)
    linfer = build_inference(lenv, lp, deterministic=True)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    hip_id = model.joint("hip_joint").id; knee_id = model.joint("knee_joint").id
    hip_q = int(model.jnt_qposadr[hip_id]); knee_q = int(model.jnt_qposadr[knee_id])

    def rollout(row, spec, proposal_index):
        seed = a.seed + proposal_index
        key = jax.random.PRNGKey(seed)
        state = restore_snapshot(env, row, key)
        previous_vz = float(np.asarray(state.data.qvel[2]))
        seen_positive = previous_vz > .02
        zero_crossing = None; first_negative = None
        trace, min_distance, min_margin = [], float("inf"), float("inf")
        valid = False; reason = "horizon_exhaustion"
        entry_snapshot = None; energy = 0.
        for tick in range(a.horizon):
            action = _action(spec, tick)
            energy += float(np.sum(np.square(np.asarray(action))))
            state = step(state, action)
            sample = sample_from_state(env, state, previous_vz)
            feature = np.asarray(sample["physical_feature"], float)
            seen_positive = seen_positive or feature[8] > .02
            if zero_crossing is None and seen_positive and feature[8] <= 0:
                zero_crossing = tick + 1
            if first_negative is None and feature[8] < -.05:
                first_negative = tick + 1
            distance, nearest, contribution = _nearest(feature, support, center, scale)
            min_distance = min(min_distance, distance)
            qpos = np.asarray(state.data.qpos)
            hip_margin = min(qpos[hip_q] - model.jnt_range[hip_id, 0],
                             model.jnt_range[hip_id, 1] - qpos[hip_q])
            knee_margin = min(qpos[knee_q] - model.jnt_range[knee_id, 0],
                              model.jnt_range[knee_id, 1] - qpos[knee_q])
            min_margin = min(min_margin, hip_margin, knee_margin)
            entry = evaluate_entry("apex", sample, cfg, support_metadata)
            trace.append({
                "tick": tick + 1, "root_x": float(feature[0]),
                "root_z": float(feature[2]), "vertical_velocity": float(feature[8]),
                "roll": float(feature[3]), "pitch": float(feature[4]),
                "angular_velocity": feature[9:12].tolist(),
                "hip": float(feature[13]), "knee": float(feature[14]),
                "hip_margin": float(hip_margin), "knee_margin": float(knee_margin),
                "action": np.asarray(action).tolist(),
                "support_distance": distance,
                "nearest_support_id": support_bank.records[nearest]["id"],
                "squared_distance_contribution": dict(
                    zip(FEATURE_NAMES, contribution.tolist())
                ),
                "descent_conditions": entry["entry_quality"],
                "valid_descent_entry": bool(entry["valid"]),
            })
            if entry["valid"]:
                valid = True; reason = "valid_descent_entry"
                entry_snapshot = env.snapshot_record(state, "flight"); break
            if float(np.asarray(state.done)) > .5:
                code = int(np.asarray(state.info["end_code"]))
                reason = END_REASON.get(code, f"unknown_{code}"); break
            previous_vz = float(feature[8])
        downstream = False; final = False; downstream_reason = None
        if entry_snapshot is not None:
            dstate = state
            previous_vz = float(np.asarray(dstate.data.qvel[2]))
            for dtick in range(a.downstream_horizon):
                key, ak = jax.random.split(key)
                daction, _ = dinfer(dstate.obs, ak)
                dstate = step(dstate, daction)
                dsample = sample_from_state(env, dstate, previous_vz)
                landing = evaluate_entry("descent", dsample, cfg)
                if landing["valid"]:
                    downstream = True
                    lsnap = env.snapshot_record(dstate, "landing")
                    lkey = jax.random.PRNGKey(seed + 60_000_000)
                    _, outcome = frozen_rollout(
                        lenv, linfer, restore_snapshot(lenv, lsnap, lkey), lkey,
                        horizon=a.downstream_horizon, step_fn=lstep,
                    )
                    final = bool(outcome["final"])
                    downstream_reason = END_REASON.get(outcome["end_code"], "unknown")
                    break
                if float(np.asarray(dstate.done)) > .5:
                    code = int(np.asarray(dstate.info["end_code"]))
                    downstream_reason = END_REASON.get(code, f"unknown_{code}"); break
                previous_vz = float(dsample["physical_feature"][8])
        result = {
            "candidate_id": row["id"], "candidate_kind": row.get("candidate_kind"),
            "independent_parent": row.get(
                "independent_trajectory_parent_id",
                row.get("source_parent_id", row.get("trajectory_parent_id")),
            ),
            "dynamic_evidence": is_dynamically_reached_apex(row),
            "proposal_index": proposal_index, "seed": seed, "parameters": spec,
            "valid_descent_entry": valid, "time_to_descent_entry": tick + 1 if valid else None,
            "apex_crossed": zero_crossing is not None,
            "vertical_velocity_zero_crossing_tick": zero_crossing,
            "first_negative_vertical_velocity_tick": first_negative,
            "minimum_support_distance": None if np.isinf(min_distance) else min_distance,
            "minimum_joint_margin": float(min_margin),
            "failure_reason": None if valid else reason,
            "physical_failure_tick": tick + 1 if reason in ("pitch_limit", "roll_limit") else None,
            "action_saturation_fraction": float(np.mean([
                np.mean(np.abs(t["action"]) >= .999) for t in trace
            ])) if trace else 0.,
            "action_energy": energy,
            "downstream_controller_success": downstream,
            "final_landing_recovery": final,
            "downstream_termination_reason": downstream_reason,
            "trace": trace,
        }
        result["failure_mode"] = _failure_class(result, float(matcher["radius"]))
        return result

    round_a, counter = [], 0
    best_by_state = {}
    for row in bank.records:
        state_rows = []
        for spec in _round_a():
            result = rollout(row, spec, counter); counter += 1
            state_rows.append(result); round_a.append(result)
            if result["valid_descent_entry"]:
                break
        best_by_state[row["id"]] = max(state_rows, key=lambda r: (
            int(r["valid_descent_entry"]), int(r["downstream_controller_success"]),
            int(r["final_landing_recovery"]), int(r["failure_reason"] is None or
                                                 r["failure_reason"] == "horizon_exhaustion"),
            -(r["minimum_support_distance"] or 1e9), -r["action_energy"],
        ))
    dynamic_success_a = [r for r in round_a if r["valid_descent_entry"]
                         and r["dynamic_evidence"]]
    round_b = []
    if not dynamic_success_a:
        for ri, row in enumerate(bank.records):
            if not is_dynamically_reached_apex(row):
                continue
            base = best_by_state[row["id"]]["parameters"]
            for spec in _round_b(base, a.seed + 70_000_000 + ri,
                                 a.round_b_proposals):
                result = rollout(row, spec, counter); counter += 1
                round_b.append(result)
                if result["valid_descent_entry"]:
                    break
    outcomes = round_a + round_b
    dynamic_success = [r for r in outcomes if r["valid_descent_entry"]
                       and r["dynamic_evidence"]]
    success_parents = {r["independent_parent"] for r in dynamic_success}
    full_final = [r for r in dynamic_success if r["final_landing_recovery"]]
    payload = {
        "status": "PASS", "artifact_role": "apex_descent_multiknot_bounded_search",
        "apex_bank_sha256": file_sha256(a.apex_bank),
        "support_bank_sha256": file_sha256(a.support_bank),
        "descent_policy_hash": file_sha256(Path(a.descent_policy) / "params.pkl"),
        "landing_policy_hash": file_sha256(Path(a.landing_policy) / "params.pkl"),
        "matcher_sha256": matcher["matcher_sha256"],
        "matcher_radius_unchanged": matcher["radius"],
        "horizon": a.horizon,
        "horizon_rationale": "100 ticks covers coast, physical vz zero crossing, negative-vz formation, and post-Apex correction; downstream gets a separate 200-tick continuation",
        "states": len(bank.records),
        "dynamic_states": sum(is_dynamically_reached_apex(r) for r in bank.records),
        "round_a_branches": len(round_a), "round_b_executed": bool(round_b),
        "round_b_branches": len(round_b),
        "dynamic_descent_positive_unique": len({
            r["candidate_id"] for r in dynamic_success
        }),
        "dynamic_descent_positive_parents": len(success_parents),
        "dynamic_descent_positive_branches": len(dynamic_success),
        "downstream_controller_success_branches": sum(
            r["downstream_controller_success"] for r in dynamic_success
        ),
        "final_recovery_branches": len(full_final),
        "failure_modes": dict(Counter(r["failure_mode"] for r in outcomes
                                      if r["dynamic_evidence"])),
        "termination_reasons": dict(Counter(
            "valid_descent_entry" if r["valid_descent_entry"] else r["failure_reason"]
            for r in outcomes if r["dynamic_evidence"]
        )),
        "apex_training_authorized": False,
        "authorization_requires_bank_and_two_parent_gates": True,
        "outcomes": outcomes,
    }
    save_json(a.output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
