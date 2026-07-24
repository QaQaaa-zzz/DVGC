"""Minimal event-aligned pre-Apex horizon and momentum audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from cli.discover_apex_feedback_bridge import (
    TERMINAL_FEATURES,
    TERMINAL_INDEX,
    _actions,
    _downstream,
    _state_score,
)
from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.centroidal import replay_centroidal
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, save_json


def _canonical_class(value):
    return (
        "apex_local_response_detected"
        if value == "apex_local_correctable" else value
    )


def _select_starts(records, desired=(0, -4, -6, -8)):
    """Choose unique recorded starts within one tick of requested offsets."""
    selected = []
    used = set()
    for target in desired:
        row = min(records, key=lambda item: (
            abs(int(item["relative_to_apex"]) - target),
            -int(item["relative_to_apex"]),
        ))
        if abs(int(row["relative_to_apex"]) - target) > 1 or row["id"] in used:
            continue
        selected.append((target, row))
        used.add(row["id"])
    return selected


def _summary(model, state, diagnostic):
    momentum = replay_centroidal(
        model, np.asarray(state.data.qpos), np.asarray(state.data.qvel),
        np.asarray(state.data.ctrl),
    )
    feature = diagnostic["feature"]
    return {
        "roll": float(feature[3]), "pitch": float(feature[4]),
        "vx": float(feature[6]), "vz": float(feature[8]),
        "angular_velocity": feature[9:12].tolist(),
        "hip": float(feature[13]), "knee": float(feature[14]),
        "joint_margin": float(diagnostic["joint_margin"]),
        "formal_descent_support_entry": bool(diagnostic["entry"]["valid"]),
        "stable_physical_descent": bool(diagnostic["stable"]),
        "done": bool(diagnostic["done"]),
        "terminal_cluster_distance": float(diagnostic["target_distance"]),
        "system_com": momentum["system_com"],
        "centroidal_angular_momentum": momentum[
            "centroidal_angular_momentum"
        ],
        "momentum_crosscheck_linf": momentum[
            "angular_momentum_crosscheck_linf"
        ],
    }


def _plan(
    env, step, model, state, previous_vz, warm_plan, prediction_horizon,
    control_horizon, support_metadata, terminal_target, terminal_center,
    terminal_scale, *, capture_candidates,
):
    candidates = []
    for index, action in enumerate(_actions()):
        sequence = [action] * min(control_horizon, prediction_horizon)
        tail = list(warm_plan[control_horizon:prediction_horizon])
        sequence.extend(tail)
        sequence.extend(
            [jp.zeros((4,), jp.float32)]
            * (prediction_horizon - len(sequence))
        )
        probe = state
        probe_vz = previous_vz
        diagnostic = None
        for planned_action in sequence:
            probe = step(probe, planned_action)
            _, diagnostic = _state_score(
                env, probe, probe_vz, support_metadata, model,
                terminal_target, terminal_center, terminal_scale,
                float(np.sum(np.square(np.asarray(planned_action)))),
            )
            probe_vz = float(diagnostic["feature"][8])
            if diagnostic["done"] or diagnostic["entry"]["valid"]:
                break
        score, _ = _state_score(
            env, probe, probe_vz, support_metadata, model,
            terminal_target, terminal_center, terminal_scale,
            float(sum(np.sum(np.square(np.asarray(x))) for x in sequence)),
        )
        candidates.append((
            score, -float(np.sum(np.square(np.asarray(action)))), -index,
            sequence, action, probe, diagnostic,
        ))
    best = max(candidates, key=lambda row: row[:3])
    details = None
    if capture_candidates:
        details = [{
            "action_index": -int(row[2]),
            "first_action": np.asarray(row[4]).tolist(),
            "score": float(row[0]),
            "terminal": _summary(model, row[5], row[6]),
        } for row in candidates]
    return list(best[3]), {
        "first_action": np.asarray(best[4]).tolist(),
        "selected_score": float(best[0]),
        "planned_terminal": _summary(model, best[5], best[6]),
        "candidate_terminals": details,
    }


def _run(
    runtime, model, start, seed, prediction_horizon, support_metadata,
    terminal_target, terminal_center, terminal_scale, controller_horizon,
    control_horizon, downstream_horizon,
):
    env, step, *_ = runtime
    key = jax.random.PRNGKey(seed)
    state = restore_snapshot(env, start, key)
    neutral_state = restore_snapshot(env, start, key)
    previous_vz = float(np.asarray(state.data.qvel[2]))
    neutral_previous_vz = previous_vz
    initial_sample = sample_from_state(env, state, previous_vz)
    initial_feature = np.asarray(initial_sample["physical_feature"], float)
    warm_plan = [jp.zeros((4,), jp.float32)] * prediction_horizon
    active_plan = list(warm_plan)
    plans = []
    trace = []
    stable_count = 0
    max_stable_count = 0
    stable_snapshot = None
    formal_snapshot = None
    reason = "horizon_exhaustion"
    action_response_tick = None
    apex_tick = None
    for tick in range(controller_horizon):
        if tick % control_horizon == 0:
            shifted = active_plan[control_horizon:] + (
                [jp.zeros((4,), jp.float32)] * control_horizon
            )
            active_plan, plan = _plan(
                env, step, model, state, previous_vz, shifted,
                prediction_horizon, control_horizon, support_metadata,
                terminal_target, terminal_center, terminal_scale,
                capture_candidates=(tick == 0),
            )
            plan.update({
                "plan_tick": tick,
                "actual_target_tick": tick + prediction_horizon,
            })
            plans.append(plan)
            warm_plan = list(active_plan)
        action = active_plan[tick % control_horizon]
        state = step(state, action)
        if float(np.asarray(neutral_state.done)) <= .5:
            neutral_state = step(neutral_state, jp.zeros((4,), jp.float32))
            neutral_sample = sample_from_state(
                env, neutral_state, neutral_previous_vz
            )
            neutral_feature = np.asarray(
                neutral_sample["physical_feature"], float
            )
            neutral_previous_vz = float(neutral_feature[8])
        else:
            neutral_feature = None
        _, diagnostic = _state_score(
            env, state, previous_vz, support_metadata, model,
            terminal_target, terminal_center, terminal_scale,
            float(np.sum(np.square(np.asarray(action)))),
        )
        feature = diagnostic["feature"]
        stable_count = stable_count + 1 if diagnostic["stable"] else 0
        max_stable_count = max(max_stable_count, stable_count)
        if stable_count >= 16 and stable_snapshot is None:
            stable_snapshot = env.snapshot_record(state, "flight")
        if diagnostic["apex"] and apex_tick is None:
            apex_tick = tick + 1
        if neutral_feature is not None:
            delta_roll = abs(float(feature[3] - neutral_feature[3]))
            delta_wx = abs(float(feature[9] - neutral_feature[9]))
            if (action_response_tick is None
                    and (delta_roll >= .02 or delta_wx >= .2)):
                action_response_tick = tick + 1
        momentum = replay_centroidal(
            model, np.asarray(state.data.qpos), np.asarray(state.data.qvel),
            np.asarray(state.data.ctrl),
        )
        trace.append({
            "tick": tick + 1, "action": np.asarray(action).tolist(),
            "roll": float(feature[3]), "pitch": float(feature[4]),
            "vz": float(feature[8]), "angular_velocity": feature[9:12].tolist(),
            "centroidal_angular_momentum":
                momentum["centroidal_angular_momentum"],
            "system_com": momentum["system_com"],
            "robot_terrain_contact_count": len(
                momentum["robot_terrain_contacts"]
            ),
            "net_terrain_impulse": (
                np.asarray(momentum["net_terrain_force"]) * .02
            ).tolist(),
            "net_terrain_angular_impulse": (
                np.asarray(momentum["net_terrain_torque_about_com"]) * .02
            ).tolist(),
            "formal_descent_support_entry": bool(diagnostic["entry"]["valid"]),
            "stable_run_ticks": stable_count,
            "done": bool(diagnostic["done"]),
        })
        if diagnostic["entry"]["valid"]:
            formal_snapshot = env.snapshot_record(state, "flight")
            reason = "formal_descent_support_entry"
            break
        if diagnostic["done"]:
            code = int(np.asarray(state.info["end_code"]))
            reason = END_REASON.get(code, f"unknown_{code}")
            break
        previous_vz = float(feature[8])

    actual_by_tick = {row["tick"]: row for row in trace}
    for plan in plans:
        actual = actual_by_tick.get(plan["actual_target_tick"])
        plan["actual_terminal"] = actual
        if actual is not None:
            predicted = plan["planned_terminal"]
            plan["prediction_error"] = {
                "roll": actual["roll"] - predicted["roll"],
                "pitch": actual["pitch"] - predicted["pitch"],
                "vz": actual["vz"] - predicted["vz"],
                "angular_velocity": (
                    np.asarray(actual["angular_velocity"])
                    - np.asarray(predicted["angular_velocity"])
                ).tolist(),
                "centroidal_angular_momentum": (
                    np.asarray(actual["centroidal_angular_momentum"])
                    - np.asarray(predicted["centroidal_angular_momentum"])
                ).tolist(),
            }
    downstream = {
        "descent_controller_success": False,
        "final_landing_recovery": False,
        "descent_termination_reason": None,
    }
    source = formal_snapshot or stable_snapshot
    if source is not None:
        downstream = _downstream(
            runtime,
            restore_snapshot(env, source, jax.random.fold_in(key, 7_000_000)),
            jax.random.fold_in(key, 8_000_000), support_metadata,
            downstream_horizon, allow_without_formal=formal_snapshot is None,
        )
    first_actions = [plan["first_action"] for plan in plans]
    return {
        "action_response_latency_ticks": action_response_tick,
        "response_reference": "paired_zero_action_counterfactual",
        "physical_apex_tick": apex_tick,
        "max_stable_descent_ticks": max_stable_count,
        "stable_16_ticks": stable_snapshot is not None,
        "formal_descent_support_entry": formal_snapshot is not None,
        "termination_reason": reason,
        "failure_minus_apex_ticks": (
            len(trace) - apex_tick
            if apex_tick is not None and reason in ("roll_limit", "pitch_limit")
            else None
        ),
        "first_selected_action": first_actions[0],
        "selected_action_changes": int(sum(
            not np.allclose(first_actions[i], first_actions[i - 1])
            for i in range(1, len(first_actions))
        )),
        **downstream,
        "plans": plans,
        "trace": trace,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authority-bank", required=True)
    p.add_argument("--authority-report", required=True)
    p.add_argument("--support-bank", required=True)
    p.add_argument("--terminal-bank", required=True)
    p.add_argument("--descent-policy", required=True)
    p.add_argument("--landing-policy", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--parent", action="append", required=True)
    p.add_argument("--prediction-horizon", action="append", type=int,
                   default=[])
    p.add_argument("--controller-horizon", type=int, default=40)
    p.add_argument("--control-horizon", type=int, default=2)
    p.add_argument("--downstream-horizon", type=int, default=200)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--seed", type=int, default=11_700_000)
    a = p.parse_args()
    horizons = a.prediction_horizon or [3, 6, 9, 12]
    authority_bank = SnapshotBank.load(a.authority_bank)
    authority = json.loads(Path(a.authority_report).read_text())
    support = SnapshotBank.load(a.support_bank)
    support_metadata = dict(support.metadata)
    support_metadata["support_features"] = [
        row["physical_feature"] for row in support.records
    ]
    terminal = SnapshotBank.load(a.terminal_bank)
    center = np.asarray(terminal.metadata["normalization_center"], float)
    scale = np.asarray(terminal.metadata["normalization_scale"], float)
    target = np.asarray([
        [(row["physical_feature"][TERMINAL_INDEX[name]] - center[i]) / scale[i]
         for i, name in enumerate(TERMINAL_FEATURES)]
        for row in terminal.records
    ], float)
    dp, dc, _ = load_bundle(a.descent_policy, verify_files=True)
    lp, lc, _ = load_bundle(a.landing_policy, verify_files=True)
    cfg = load_config(a.config, {
        **dc, "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    lcfg = load_config(a.config, {
        **lc, "training_stage": "landing", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(
        cfg, snapshot_bank=SnapshotBank(), stage_support_bank=support
    )
    lenv = OrangeBikeDVGC(lcfg, snapshot_bank=SnapshotBank())
    runtime = (
        env, jax.jit(env.step), build_inference(env, dp, deterministic=True),
        lenv, jax.jit(lenv.step),
        build_inference(lenv, lp, deterministic=True),
    )
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    outcomes = []
    selected = {}
    for parent_index, display_parent in enumerate(a.parent):
        info = authority["parent_results"][display_parent]
        rows = [
            row for row in authority_bank.records
            if row["trajectory_parent_id"] == info["parent_id"]
        ]
        starts = _select_starts(rows)
        selected[display_parent] = [{
            "requested_offset": requested,
            "actual_offset": row["relative_to_apex"], "snapshot_id": row["id"],
        } for requested, row in starts]
        for start_index, (requested, start) in enumerate(starts):
            for horizon in horizons:
                result = _run(
                    runtime, model, start,
                    a.seed + parent_index * 100_000 + start_index * 1000 + horizon,
                    horizon, support_metadata, target, center, scale,
                    a.controller_horizon, a.control_horizon,
                    a.downstream_horizon,
                )
                outcomes.append({
                    "parent": display_parent, "parent_id": info["parent_id"],
                    "control_authority_class": _canonical_class(
                        info["classification"]
                    ),
                    "requested_start_relative_to_apex": requested,
                    "actual_start_relative_to_apex": start["relative_to_apex"],
                    "start_snapshot_id": start["id"],
                    "prediction_horizon": horizon,
                    **result,
                })
    payload = {
        "status": "PASS",
        "artifact_role": "minimal_event_aligned_apex_horizon_audit",
        "diagnostic_only": True, "apex_ppo_authorized": False,
        "prediction_horizons": horizons,
        "control_horizon_ticks": a.control_horizon,
        "replanning_interval_ticks": a.control_horizon,
        "controller_horizon_ticks": a.controller_horizon,
        "action_candidates": [np.asarray(x).tolist() for x in _actions()],
        "authority_bank_sha256": file_sha256(a.authority_bank),
        "support_bank_sha256": file_sha256(a.support_bank),
        "terminal_bank_sha256": file_sha256(a.terminal_bank),
        "xml_sha256": file_sha256(cfg.xml_path),
        "selected_starts": selected,
        "outcomes": outcomes,
        "summary": {
            "runs": len(outcomes),
            "stable_16_ticks": sum(x["stable_16_ticks"] for x in outcomes),
            "formal_descent_support_entry": sum(
                x["formal_descent_support_entry"] for x in outcomes
            ),
            "downstream_controller_success": sum(
                x["descent_controller_success"] for x in outcomes
            ),
            "final_landing_recovery": sum(
                x["final_landing_recovery"] for x in outcomes
            ),
            "termination_reasons": dict(Counter(
                x["termination_reason"] for x in outcomes
            )),
        },
    }
    save_json(a.output, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
