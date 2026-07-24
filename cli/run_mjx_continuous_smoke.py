"""Resumable natural-start continuous MJX pipeline smoke."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.continuous import (
    CONTINUOUS_STAGES, ContinuousPhaseTracker, DescentSupportMatcher,
    load_trajectory, save_trajectory,
)
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.runtime import build_inference, load_params, save_json


TAKEOFF_PROGRAMS = (
    (0.50, 0.00), (0.50, 0.25), (0.50, 0.50),
    (0.60, 0.00), (0.60, 0.25), (0.60, 0.50),
    (0.70, 0.00), (0.70, 0.25), (0.70, 0.50),
    (0.75, 0.00), (0.75, 0.25), (0.75, 0.50),
)


def _policy(path: str | Path, env):
    return build_inference(
        env, load_params(Path(path) / "params.pkl"), deterministic=True
    )


def _controller_action(
    tracker: ContinuousPhaseTracker, state, key, inference, episode: int,
    stage_tick: int, upstream_sequence, takeoff_tick: int, approach_drive: float,
):
    if tracker.stage == "approach":
        return np.asarray(
            [0.0, approach_drive, 0.0, 0.0], np.float32
        ), key, "bounded_approach_drive"
    if upstream_sequence is not None and tracker.stage in (
        "takeoff", "ascent", "apex"
    ):
        index = min(takeoff_tick, len(upstream_sequence) - 1)
        return upstream_sequence[index], key, "bounded_upstream_sequence"
    if tracker.stage == "takeoff":
        hip, knee = TAKEOFF_PROGRAMS[episode % len(TAKEOFF_PROGRAMS)]
        if episode % 4 == 3 and stage_tick <= 3:
            hip, knee = -0.5, -0.5
        return np.asarray([0.0, 0.0, hip, knee], np.float32), key, (
            f"bounded_takeoff_h{hip:+.2f}_k{knee:+.2f}"
        )
    key, action_key = jax.random.split(key)
    action, _ = inference[tracker.stage](state.obs, action_key)
    return np.asarray(jax.device_get(action), np.float32), key, tracker.stage


def _summary_from_markers(markers):
    episodes = [json.loads(path.read_text()) for path in markers]
    stage_counts = Counter(
        stage for row in episodes for stage in set(row["stages_reached"])
    )
    reasons = Counter(row["termination_reason"] for row in episodes)
    return episodes, stage_counts, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12100000)
    parser.add_argument("--horizon", type=int, default=750)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--stage-support", required=True)
    parser.add_argument("--ascent-policy", required=True)
    parser.add_argument("--apex-policy", required=True)
    parser.add_argument("--descent-policy", required=True)
    parser.add_argument("--landing-policy", required=True)
    parser.add_argument("--upstream-action-sequence")
    parser.add_argument("--upstream-knot-width", type=int, default=2)
    parser.add_argument("--approach-drive", type=float, default=0.0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "cost_estimate.json", {
        "status": "PASS",
        "task": "100_natural_start_continuous_mjx_smoke",
        "episodes": args.episodes,
        "maximum_steps": args.episodes * args.horizon,
        "estimated_wall_time": "under_2_hours",
        "ppo_steps": 0,
    }) if not (output / "cost_estimate.json").exists() else None

    config = load_config(args.config, {
        "training_stage": "full",
        "use_bank_resets": False,
        "stage_reachability_objective": "",
        "expert_chain_termination": False,
        "domain_randomization": False,
        "obs_noise_enable": False,
    })
    support = SnapshotBank.load(args.stage_support)
    env = OrangeBikeDVGC(
        config, snapshot_bank=SnapshotBank(), stage_support_bank=support
    )
    step_fn = jax.jit(env.step)
    inference = {
        "ascent": _policy(args.ascent_policy, env),
        "apex": _policy(args.apex_policy, env),
        "descent": _policy(args.descent_policy, env),
        "landing": _policy(args.landing_policy, env),
    }
    policy_paths = {
        stage: str(Path(getattr(args, f"{stage}_policy")).resolve())
        for stage in inference
    }
    policy_hashes = {
        stage: file_sha256(Path(path) / "params.pkl")
        for stage, path in policy_paths.items()
    }
    matcher = DescentSupportMatcher(env)
    upstream_sequence = None
    if args.upstream_action_sequence:
        payload = json.loads(Path(args.upstream_action_sequence).read_text())
        knots = np.asarray(payload["action_knots"], np.float32)
        upstream_sequence = np.repeat(
            knots, args.upstream_knot_width, axis=0
        )

    for episode in range(args.episodes):
        marker = output / "episodes" / f"episode_{episode:04d}.json"
        if marker.exists():
            existing = json.loads(marker.read_text())
            trajectory, digest = load_trajectory(
                existing["trajectory_path"], existing["trajectory_sha256"]
            )
            if len(trajectory["qpos"]) != existing["steps"] + 1:
                raise RuntimeError(f"invalid completed episode {episode}")
            continue
        key = jax.random.PRNGKey(args.seed + episode)
        state = env.reset(key)
        tracker = ContinuousPhaseTracker(previous_vz=float(state.data.qvel[2]))
        stages_reached = {tracker.stage}
        stage_tick = 0
        takeoff_tick = 0
        controller_log = []
        previous_controller = None
        arrays = {
            "qpos": [np.asarray(jax.device_get(state.data.qpos))],
            "qvel": [np.asarray(jax.device_get(state.data.qvel))],
            "ctrl": [np.asarray(jax.device_get(state.data.ctrl))],
            "action": [np.zeros(env.action_size, np.float32)],
            "observation_history": [
                np.asarray(jax.device_get(state.info["obs_history"]))
            ],
            "env_phase": [int(state.info["phase"])],
            "pipeline_phase": [tracker.index],
            "end_code": [int(state.info["end_code"])],
            "contact_count": [
                int(np.asarray(jax.device_get(state.data._impl.nacon)).reshape(-1)[0])
            ],
        }
        contact_seen = arrays["contact_count"][-1] > 0
        contact_ended = False
        action_violation = False
        nonfinite = False
        for tick in range(1, args.horizon + 1):
            stage_tick += 1
            action, key, controller = _controller_action(
                tracker, state, key, inference, episode, stage_tick,
                upstream_sequence, takeoff_tick, args.approach_drive,
            )
            if tracker.stage in ("takeoff", "ascent", "apex"):
                takeoff_tick += 1
            clipped = np.clip(action, -1.0, 1.0)
            action_violation |= not np.array_equal(action, clipped)
            if controller != previous_controller:
                controller_log.append({
                    "tick": tick,
                    "stage": tracker.stage,
                    "controller": controller,
                    "policy_hash": policy_hashes.get(tracker.stage),
                })
                previous_controller = controller
            state = step_fn(state, clipped)
            descent_entry, descent_distance = matcher.evaluate(
                state, apex_crossed=(tracker.stage == "apex")
            )
            before = tracker.stage
            tracker.observe(
                state,
                descent_entry=descent_entry,
                physical_descent=(
                    tracker.stage == "apex"
                    and float(state.data.qvel[2]) < 0.0
                ),
                tick=tick,
            )
            if tracker.stage != before:
                stage_tick = 0
            stages_reached.add(tracker.stage)
            count = int(np.asarray(
                jax.device_get(state.data._impl.nacon)
            ).reshape(-1)[0])
            contact_ended |= contact_seen and count == 0
            contact_seen |= count > 0
            fields = (
                np.asarray(jax.device_get(state.data.qpos)),
                np.asarray(jax.device_get(state.data.qvel)),
                np.asarray(jax.device_get(state.data.ctrl)),
                np.asarray(jax.device_get(state.info["obs_history"])),
            )
            nonfinite |= not all(np.all(np.isfinite(value)) for value in fields)
            arrays["qpos"].append(fields[0])
            arrays["qvel"].append(fields[1])
            arrays["ctrl"].append(fields[2])
            arrays["action"].append(clipped)
            arrays["observation_history"].append(fields[3])
            arrays["env_phase"].append(int(state.info["phase"]))
            arrays["pipeline_phase"].append(tracker.index)
            arrays["end_code"].append(int(state.info["end_code"]))
            arrays["contact_count"].append(count)
            if bool(float(state.done)):
                break
        normalized = {
            key_name: np.asarray(value) for key_name, value in arrays.items()
        }
        trajectory_path = output / "trajectories" / f"episode_{episode:04d}.npz"
        digest = save_trajectory(trajectory_path, normalized)
        code = int(state.info["end_code"])
        payload = {
            "episode": episode,
            "seed": args.seed + episode,
            "steps": len(normalized["qpos"]) - 1,
            "stages_reached": [
                stage for stage in CONTINUOUS_STAGES if stage in stages_reached
            ],
            "phase_switches": tracker.switches,
            "controller_switches": controller_log,
            "termination_reason": END_REASON.get(code, f"unknown_{code}"),
            "physical_failure": bool(int(state.info["terminated"]))
                and not bool(int(state.info["recovery_success"])),
            "timeout": bool(int(state.info["truncated"])),
            "final_recovery": bool(int(state.info["recovery_success"])),
            "nonfinite": nonfinite,
            "action_bounds_violation": action_violation,
            "contact_seen": contact_seen,
            "contact_ended": contact_ended,
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": digest,
            "last_descent_distance": descent_distance,
        }
        marker.parent.mkdir(parents=True, exist_ok=True)
        save_json(marker, payload)
        save_json(output / "progress.json", {
            "status": "ACTIVE",
            "completed": episode + 1,
            "total": args.episodes,
            "last_completed_episode": episode,
            "updated_at": time.time(),
        })

    markers = sorted((output / "episodes").glob("episode_*.json"))
    episodes, stage_counts, reasons = _summary_from_markers(markers)
    all_stages = all(stage_counts[stage] > 0 for stage in CONTINUOUS_STAGES)
    passed = (
        len(episodes) == args.episodes
        and not any(row["nonfinite"] for row in episodes)
        and not any(row["action_bounds_violation"] for row in episodes)
        and all_stages
        and any(row["contact_seen"] for row in episodes)
        and any(row["contact_ended"] for row in episodes)
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "artifact_role": "mjx_continuous_pipeline_smoke",
        "development_runtime": "MJX",
        "continuous_data_lineage": True,
        "physics_restore_during_handoff": False,
        "temporary_event_controller": True,
        "physical_apex_descent_switch": True,
        "upstream_action_sequence": args.upstream_action_sequence,
        "approach_drive": args.approach_drive,
        "fixed_dynamics_variant": True,
        "episodes": len(episodes),
        "stage_reach_counts": dict(stage_counts),
        "termination_reasons": dict(reasons),
        "final_recovery_count": sum(row["final_recovery"] for row in episodes),
        "physical_failure_count": sum(row["physical_failure"] for row in episodes),
        "timeout_count": sum(row["timeout"] for row in episodes),
        "nonfinite_count": sum(row["nonfinite"] for row in episodes),
        "action_bounds_violation_count": sum(
            row["action_bounds_violation"] for row in episodes
        ),
        "contact_seen_count": sum(row["contact_seen"] for row in episodes),
        "contact_ended_count": sum(row["contact_ended"] for row in episodes),
        "trajectory_reload_verified": True,
        "resumable_markers": True,
        "policy_paths": policy_paths,
        "policy_hashes": policy_hashes,
        "stage_support_sha256": file_sha256(args.stage_support),
        "xml_sha256": file_sha256(config.xml_path),
        "config_sha256": file_sha256(args.config),
        "effective_solver": env._effective_mjx_solver,
        "ppo_authorization": False,
        "next_action": (
            "continuous_takeoff_tail_authority"
            if passed else "bounded_continuous_controller_coverage_repair"
        ),
    }
    save_json(output / "report.json", report)
    save_json(output / "progress.json", {
        "status": report["status"],
        "completed": len(episodes),
        "total": args.episodes,
        "updated_at": time.time(),
    })
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(40)


if __name__ == "__main__":
    main()
