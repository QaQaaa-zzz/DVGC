"""Low-dimensional natural-lineage search into frozen Descent support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from cli.audit_descent_bridge_sensitivity import _reference_action
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.continuous import DescentSupportMatcher, save_trajectory
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import build_inference, save_json


def _run(env, step, matcher, reference, cfg, descent_infer, landing_infer,
         params, seed, horizon, window_start=7, window_end=15,
         launch_delay=0, segments=2, approach_tail_start=None, record=False):
    state = env.reset(jax.random.PRNGKey(seed))
    key = jax.random.PRNGKey(seed + 1)
    tick = 0
    while int(state.info["phase"]) < 1 and not float(state.done):
        action = np.asarray([0.0, 1.0, 0.0, 0.0], np.float32)
        if approach_tail_start is not None and tick >= approach_tail_start:
            action[2:] += params[segments * 2:segments * 2 + 2]
            action = np.clip(action, -1.0, 1.0)
        state = step(state, action)
        tick += 1
    approach_ticks = tick
    previous_vz = float(state.data.qvel[2])
    positive_seen = previous_vz > 0.0
    apex_tick = None
    support_tick = None
    landing_tick = None
    best_distance = float("inf")
    closest_detail = None
    minimum_pitch_margin = float("inf")
    stable_descent_ticks = 0
    arrays = {
        "qpos": [np.asarray(state.data.qpos)],
        "qvel": [np.asarray(state.data.qvel)],
        "action": [np.zeros(4, np.float32)],
        "phase": [int(state.info["phase"])],
        "end_code": [int(state.info["end_code"])],
    }
    events = []
    for local_tick in range(horizon):
        if support_tick is None:
            control_tick = local_tick - launch_delay
            if control_tick < 0:
                action = np.asarray([0.0, 1.0, 0.0, 0.0], np.float32)
            else:
                action = _reference_action(
                    env, cfg, state, reference, control_tick, 3, 105, 10
                )
            if window_start <= control_tick < window_end:
                segment = min(
                    (control_tick - window_start) * segments
                    // max(window_end - window_start, 1),
                    segments - 1,
                )
                action = action.copy()
                action[2] += params[segment * 2]
                action[3] += params[segment * 2 + 1]
                action = np.clip(action, -1.0, 1.0)
            controller = "nominal_plus_local_residual"
        elif int(state.info["had_valid_landing"]) == 0:
            key, action_key = jax.random.split(key)
            action, _ = descent_infer(state.obs, action_key)
            action = np.asarray(action)
            controller = "frozen_descent"
            stable_descent_ticks += 1
        else:
            key, action_key = jax.random.split(key)
            action, _ = landing_infer(state.obs, action_key)
            action = np.asarray(action)
            controller = "frozen_landing"
        state = step(state, np.clip(action, -1.0, 1.0))
        tick += 1
        feature = np.asarray(env._physical_feature(state.data), np.float64)
        vz = float(feature[8])
        positive_seen |= vz > 0.0
        if (
            apex_tick is None and positive_seen and previous_vz > 0.0
            and vz <= 0.0 and not float(state.done)
        ):
            apex_tick = local_tick + 1
            events.append({"event": "apex", "tick": tick})
        previous_vz = vz
        entry, distance = matcher.evaluate(
            state, apex_crossed=apex_tick is not None
        )
        if distance < best_distance:
            normalized = (feature - matcher.center) / matcher.scale
            delta = normalized[None, :] - matcher.features
            nearest = int(np.argmin(np.linalg.norm(delta, axis=1)))
            raw_target = (
                matcher.features[nearest] * matcher.scale + matcher.center
            )
            closest_detail = {
                "tick": tick,
                "local_tick": local_tick + 1,
                "feature": feature.tolist(),
                "nearest_support_index": nearest,
                "nearest_support_feature": raw_target.tolist(),
                "raw_error": (feature - raw_target).tolist(),
                "normalized_error": delta[nearest].tolist(),
                "squared_contribution": np.square(delta[nearest]).tolist(),
            }
            best_distance = distance
        minimum_pitch_margin = min(
            minimum_pitch_margin,
            np.deg2rad(float(cfg.max_pitch_deg)) - abs(float(feature[4])),
        )
        if entry and support_tick is None:
            support_tick = tick
            events.append({
                "event": "formal_descent_support_entry",
                "tick": tick,
                "distance": distance,
            })
        if int(state.info["had_valid_landing"]) and landing_tick is None:
            landing_tick = tick
            events.append({"event": "valid_landing", "tick": tick})
        if record:
            arrays["qpos"].append(np.asarray(state.data.qpos))
            arrays["qvel"].append(np.asarray(state.data.qvel))
            arrays["action"].append(np.asarray(action))
            arrays["phase"].append(int(state.info["phase"]))
            arrays["end_code"].append(int(state.info["end_code"]))
        if float(state.done):
            break
    final = bool(int(state.info["recovery_success"]))
    outcome = {
        "params": np.asarray(params).tolist(),
        "approach_ticks": approach_ticks,
        "steps": tick,
        "apex_tick": apex_tick,
        "support_entry": support_tick is not None,
        "support_tick": support_tick,
        "stable_descent_ticks": stable_descent_ticks,
        "landing": landing_tick is not None,
        "landing_tick": landing_tick,
        "final_recovery": final,
        "best_support_distance": best_distance,
        "closest_support_detail": closest_detail,
        "minimum_pitch_margin": minimum_pitch_margin,
        "termination_reason": END_REASON.get(
            int(state.info["end_code"]), "unknown"
        ),
        "events": events,
    }
    # A proposal that terminates before the bridge window is complete is not a
    # valid local improvement, even when it avoids the nominal trajectory's
    # later pitch-limit termination.
    bridge_window_complete = (
        tick >= approach_ticks + launch_delay + window_end
    )
    outcome["bridge_window_complete"] = bridge_window_complete
    outcome["rank"] = (
        int(final),
        int(outcome["landing"]),
        int(outcome["support_entry"]),
        min(stable_descent_ticks, 100),
        int(bridge_window_complete),
        -best_distance,
        minimum_pitch_margin,
    )
    return outcome, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-bank", required=True)
    parser.add_argument("--descent-policy", required=True)
    parser.add_argument("--landing-policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", default="data/reference_jump.csv")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seed", type=int, default=12_330_000)
    parser.add_argument("--horizon", type=int, default=220)
    parser.add_argument("--window-start", type=int, default=7)
    parser.add_argument("--window-end", type=int, default=15)
    parser.add_argument("--residual-bound", type=float, default=0.30)
    parser.add_argument("--launch-delay", type=int, default=0)
    parser.add_argument("--segments", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--approach-tail-start",
        type=int,
        help="natural-reset tick for an additional two-parameter hip/knee tail",
    )
    parser.add_argument(
        "--screen-delays",
        help="comma-separated launch delays; evaluate nominal only and exit",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    support = SnapshotBank.load(args.support_bank)
    dp, dc, _ = load_bundle(args.descent_policy, verify_files=True)
    lp, _, _ = load_bundle(args.landing_policy, verify_files=True)
    cfg = load_config(args.config, {
        **dc,
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
    descent_infer = build_inference(env, dp, deterministic=True)
    landing_infer = build_inference(env, lp, deterministic=True)
    reference = pd.read_csv(args.reference)

    params = np.zeros(
        args.segments * 2 + (2 if args.approach_tail_start is not None else 0),
        np.float32,
    )
    if args.screen_delays:
        delays = [int(value) for value in args.screen_delays.split(",")]
        rows = []
        for delay in delays:
            result, _ = _run(
                env, step, matcher, reference, cfg,
                descent_infer, landing_infer, params, args.seed, args.horizon,
                args.window_start, args.window_end, delay, args.segments,
                args.approach_tail_start,
            )
            rows.append({"launch_delay": delay, **result})
        save_json(output / "timing_screen.json", {
            "status": "PASS",
            "continuous_natural_lineage": True,
            "snapshot_restore": False,
            "rows": rows,
        })
        print(json.dumps(rows, indent=2))
        return
    evaluations = []
    baseline, _ = _run(
        env, step, matcher, reference, cfg, descent_infer, landing_infer,
        params, args.seed, args.horizon, args.window_start, args.window_end,
        args.launch_delay, args.segments, args.approach_tail_start,
    )
    evaluations.append(baseline)
    best = baseline
    for radius in (0.16, 0.08, 0.04, 0.02):
        improved = True
        while improved:
            improved = False
            for dimension in range(len(params)):
                for sign in (-1.0, 1.0):
                    proposal = params.copy()
                    proposal[dimension] = np.clip(
                        proposal[dimension] + sign * radius,
                        -args.residual_bound, args.residual_bound,
                    )
                    result, _ = _run(
                        env, step, matcher, reference, cfg,
                        descent_infer, landing_infer, proposal,
                        args.seed, args.horizon,
                        args.window_start, args.window_end,
                        args.launch_delay,
                        args.segments, args.approach_tail_start,
                    )
                    result["radius"] = radius
                    result["dimension"] = dimension
                    evaluations.append(result)
                    if tuple(result["rank"]) > tuple(best["rank"]):
                        params = proposal
                        best = result
                        improved = True
                        break
                if improved:
                    break
    strict_runs = []
    strict_arrays = None
    for repeat in range(20):
        result, arrays = _run(
            env, step, matcher, reference, cfg, descent_infer, landing_infer,
            params, args.seed, args.horizon,
            args.window_start, args.window_end, args.launch_delay,
            args.segments, args.approach_tail_start, record=True,
        )
        strict_runs.append(result)
        if strict_arrays is None:
            strict_arrays = arrays
        else:
            for name in strict_arrays:
                if not np.array_equal(
                    np.asarray(strict_arrays[name]), np.asarray(arrays[name])
                ):
                    raise RuntimeError(
                        f"natural exact replay diverged in {name}, repeat {repeat}"
                    )
    trajectory_path = output / "best_natural_trajectory.npz"
    trajectory_sha = save_trajectory(trajectory_path, {
        name: np.asarray(values) for name, values in strict_arrays.items()
    })
    report = {
        "status": "PASS" if best["final_recovery"] else "FAIL",
        "artifact_role": "descent_support_entry_bridge_local_search",
        "development_runtime": "MJX",
        "effective_solver": env._effective_mjx_solver,
        "fixed_dynamics_variant": True,
        "continuous_natural_lineage": True,
        "snapshot_restore": False,
        "search_dimensions": len(params),
        "parameterization": [
            f"segment_{segment}_{joint}"
            for segment in range(1, args.segments + 1)
            for joint in ("hip", "knee")
        ] + (
            ["approach_tail_hip", "approach_tail_knee"]
            if args.approach_tail_start is not None else []
        ),
        "window_ticks": [args.window_start, args.window_end],
        "launch_delay": args.launch_delay,
        "approach_tail_start": args.approach_tail_start,
        "residual_bound": args.residual_bound,
        "trust_region_schedule": [0.16, 0.08, 0.04, 0.02],
        "evaluations": len(evaluations),
        "baseline": baseline,
        "best": best,
        "formal_support_entries": sum(
            row["support_entry"] for row in evaluations
        ),
        "landing_count": sum(row["landing"] for row in evaluations),
        "final_recovery_count": sum(
            row["final_recovery"] for row in evaluations
        ),
        "natural_exact_replay": {
            "repeats": 20,
            "bit_exact": True,
            "outcome_identical": len({
                json.dumps(row, sort_keys=True) for row in strict_runs
            }) == 1,
        },
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": trajectory_sha,
        "support_bank_sha256": file_sha256(args.support_bank),
        "descent_policy_sha256": file_sha256(
            Path(args.descent_policy) / "params.pkl"
        ),
        "landing_policy_sha256": file_sha256(
            Path(args.landing_policy) / "params.pkl"
        ),
        "xml_sha256": file_sha256(cfg.xml_path),
        "matcher_sha256": support.metadata[
            "stage_entry_matcher"
        ]["matcher_sha256"],
        "ppo_authorization": False,
    }
    save_json(output / "report.json", report)
    save_json(output / "evaluations.json", {"rows": evaluations})
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in ("baseline",)
    }, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(40)


if __name__ == "__main__":
    main()
