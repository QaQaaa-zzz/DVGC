"""Resumable single-world MJX search from natural reset to Descent support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.runtime import save_json


def _natural_takeoff_state(env, step, key, approach_drive):
    state = env.reset(key)
    approach_action = np.asarray(
        [0.0, approach_drive, 0.0, 0.0], np.float32
    )
    ticks = 0
    while int(state.info["phase"]) < 1 and not float(state.done):
        state = step(state, approach_action)
        ticks += 1
    if float(state.done):
        raise RuntimeError("natural reset terminated before Takeoff")
    return state, ticks


def _evaluate(
    env, step, start, sequence, width, matcher, maximum_z_weight,
    support_distance_weight, post_apex_weight, forward_weight, failure_penalty,
):
    support, center, scale, metadata = matcher
    previous_vz = float(start.data.qvel[2])
    positive = airborne = apex = entry = False
    state = start
    best_distance = float("inf")
    best_feature = None
    maximum_z = float(state.data.qpos[2])
    post_apex_ticks = 0
    for tick in range(len(sequence) * width):
        state = step(state, sequence[tick // width])
        feature = np.asarray(env._physical_feature(state.data), float)
        airborne |= bool(int(state.info["had_airborne"]))
        vz = float(feature[8])
        positive |= airborne and vz > 0.05
        apex |= (
            airborne and positive and previous_vz > 0.0 and vz <= 0.0
            and not float(state.done)
        )
        previous_vz = vz
        maximum_z = max(maximum_z, float(feature[2]))
        distance = float(np.min(np.linalg.norm(
            support - ((feature - center) / scale)[None, :], axis=1
        )))
        pose_cost = (
            (feature[3] / 0.4) ** 2 + (feature[4] / 0.6) ** 2
            + (feature[9] / 3.0) ** 2 + (feature[10] / 3.0) ** 2
        )
        if apex and not float(state.done):
            post_apex_ticks += 1
            if distance < best_distance:
                best_distance = distance
                best_feature = feature.copy()
                best_pose_cost = float(pose_cost)
        envelope = metadata["reference_envelope"]
        entry = bool(
            apex and not float(state.done)
            and envelope["x"]["min"] - 0.2 <= feature[0]
            <= envelope["x"]["max"] + 0.2
            and envelope["z"]["min"] - 0.08 <= feature[2]
            <= envelope["z"]["max"] + 0.08
            and metadata["descent_vz_min"] <= feature[8]
            <= metadata["descent_vz_max"]
            and abs(feature[9]) <= metadata["max_abs_roll_rate"]
            and abs(feature[10]) <= metadata["max_abs_pitch_rate"]
            and distance <= metadata["radius"]
        )
        if entry or float(state.done):
            break
    done = bool(float(state.done))
    score = (
        1_000_000_000.0 * entry
        + 10_000_000.0 * apex
        + float(post_apex_weight) * post_apex_ticks
        + 100_000.0 * airborne
        + float(maximum_z_weight) * maximum_z
        + float(forward_weight) * (
            float(best_feature[0]) if best_feature is not None else 0.0
        )
        - float(support_distance_weight) * (
            best_distance if apex else 0.0
        )
        - 600.0 * (best_pose_cost if apex else 0.0)
        - float(failure_penalty) * done
    )
    return {
        "score": float(score),
        "entry": entry,
        "apex": apex,
        "airborne": airborne,
        "done": done,
        "best_support_distance": (
            None if not np.isfinite(best_distance) else best_distance
        ),
        "best_feature": (
            None if best_feature is None else best_feature.tolist()
        ),
        "maximum_z": maximum_z,
        "post_apex_ticks": post_apex_ticks,
        "steps": tick + 1,
        "termination_reason": END_REASON.get(
            int(state.info["end_code"]), "unknown"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-support", required=True)
    parser.add_argument("--initial-sequence")
    parser.add_argument("--knots", type=int, default=24)
    parser.add_argument("--pulse-knots", type=int)
    parser.add_argument("--pulse-action", default="0,1,1,0.5")
    parser.add_argument("--tail-action", default="0,1,0,0")
    parser.add_argument("--initial-prefix-knots", type=int, default=0)
    parser.add_argument("--initial-prefix-action", default="0,1,0,1")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite", type=int, default=6)
    parser.add_argument("--knot-width", type=int, default=2)
    parser.add_argument("--maximum-z-weight", type=float, default=2_000.0)
    parser.add_argument("--support-distance-weight", type=float, default=1_000.0)
    parser.add_argument("--post-apex-weight", type=float, default=200_000.0)
    parser.add_argument("--forward-weight", type=float, default=0.0)
    parser.add_argument("--failure-penalty", type=float, default=5_000.0)
    parser.add_argument("--seed", type=int, default=12_150_000)
    parser.add_argument("--approach-drive", type=float, default=0.0)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    support_bank = SnapshotBank.load(args.stage_support)
    metadata = support_bank.metadata["stage_entry_matcher"]
    raw_support = np.asarray(
        [row["physical_feature"] for row in support_bank.records], float
    )
    center = np.asarray(metadata["center"], float)
    scale = np.asarray(metadata["scale"], float)
    matcher = ((raw_support - center) / scale, center, scale, metadata)
    config = load_config(args.config, {
        "training_stage": "full",
        "use_bank_resets": False,
        "stage_reachability_objective": "",
        "expert_chain_termination": False,
        "domain_randomization": False,
        "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(
        config, snapshot_bank=SnapshotBank(), stage_support_bank=support_bank
    )
    step = jax.jit(env.step)
    start, approach_ticks = _natural_takeoff_state(
        env, step, jax.random.PRNGKey(args.seed), args.approach_drive
    )
    progress = output / "progress.json"
    start_iteration = 0
    if progress.exists():
        saved = json.loads(progress.read_text())
        mean = np.asarray(saved["mean"], np.float32)
        std = np.asarray(saved["std"], np.float32)
        best = saved.get("best")
        start_iteration = int(saved["completed_iterations"])
    else:
        if args.initial_sequence:
            payload = json.loads(Path(args.initial_sequence).read_text())
            payload = payload.get("iteration_best", payload)
            mean = np.asarray(payload["action_knots"], np.float32)
            initial_std = float(payload.get("initial_std", 0.20))
        elif args.pulse_knots is not None:
            pulse = np.fromstring(args.pulse_action, sep=",", dtype=np.float32)
            tail = np.fromstring(args.tail_action, sep=",", dtype=np.float32)
            if pulse.shape != (4,) or tail.shape != (4,):
                raise SystemExit("pulse/tail action must contain four values")
            if not 0 <= args.pulse_knots <= args.knots:
                raise SystemExit("pulse-knots must be within [0, knots]")
            mean = np.repeat(tail[None, :], args.knots, axis=0)
            mean[:args.pulse_knots] = pulse
            initial_std = 0.30
        else:
            raise SystemExit(
                "--initial-sequence or --pulse-knots is required for a new search"
            )
        if args.initial_prefix_knots:
            prefix = np.fromstring(
                args.initial_prefix_action, sep=",", dtype=np.float32
            )
            if prefix.shape != (4,):
                raise SystemExit("initial-prefix-action must contain four values")
            if not 0 <= args.initial_prefix_knots <= len(mean):
                raise SystemExit("initial-prefix-knots exceeds sequence length")
            mean[:args.initial_prefix_knots] = prefix
        std = np.full_like(mean, initial_std)
        best = None
    rng = np.random.default_rng(args.seed)
    # Resuming reproduces the exact proposal stream without storing RNG internals.
    if start_iteration:
        rng.normal(size=(start_iteration, args.population, *mean.shape))
    for iteration in range(start_iteration, args.iterations):
        sequences = np.clip(
            mean[None, ...]
            + rng.normal(size=(args.population, *mean.shape))
            * std[None, ...],
            -1.0, 1.0,
        ).astype(np.float32)
        sequences[0] = mean
        rows = [
            _evaluate(
                env, step, start, sequence, args.knot_width, matcher,
                args.maximum_z_weight, args.support_distance_weight,
                args.post_apex_weight, args.forward_weight,
                args.failure_penalty,
            )
            for sequence in sequences
        ]
        order = sorted(
            range(len(rows)), key=lambda index: rows[index]["score"],
            reverse=True,
        )
        apex_order = [index for index in order if rows[index]["apex"]]
        pool = apex_order or order
        elite = sequences[pool[:args.elite]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0) * 0.9, 0.03)
        candidate = {
            **rows[order[0]],
            "action_knots": sequences[order[0]].tolist(),
            "iteration": iteration,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
        save_json(output / f"iteration_{iteration:03d}.json", {
            "iteration": iteration,
            "population": args.population,
            "entry_count": sum(row["entry"] for row in rows),
            "apex_count": sum(row["apex"] for row in rows),
            "airborne_count": sum(row["airborne"] for row in rows),
            "iteration_best": candidate,
        })
        save_json(progress, {
            "status": "ACTIVE",
            "completed_iterations": iteration + 1,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "best": best,
        })
    strict = _evaluate(
        env, step, start, np.asarray(best["action_knots"], np.float32),
        args.knot_width, matcher, args.maximum_z_weight,
        args.support_distance_weight, args.post_apex_weight,
        args.forward_weight, args.failure_penalty,
    )
    report = {
        "status": "PASS" if strict["entry"] else "FAIL",
        "artifact_role": "mjx_natural_continuous_bridge_search",
        "ppo_authorization": False,
        "development_runtime": "MJX",
        "fixed_dynamics_variant": True,
        "snapshot_restore": False,
        "natural_approach_ticks": approach_ticks,
        "approach_drive": args.approach_drive,
        "iterations": args.iterations,
        "population": args.population,
        "knot_width": args.knot_width,
        "maximum_z_weight": args.maximum_z_weight,
        "support_distance_weight": args.support_distance_weight,
        "post_apex_weight": args.post_apex_weight,
        "forward_weight": args.forward_weight,
        "failure_penalty": args.failure_penalty,
        "strict_natural_replay": strict,
        "action_knots": best["action_knots"],
        "stage_support_sha256": file_sha256(args.stage_support),
        "xml_sha256": file_sha256(config.xml_path),
        "effective_solver": env._effective_mjx_solver,
    }
    save_json(output / "report.json", report)
    save_json(progress, {
        "status": report["status"],
        "completed_iterations": args.iterations,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "best": best,
    })
    print(json.dumps(report, indent=2))
    if not strict["entry"]:
        raise SystemExit(40)


if __name__ == "__main__":
    main()
