"""Two-parent receding-horizon Apex-to-Descent bridge feasibility pilot."""
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
from dvgc.certification import DYNAMICS_VARIANTS
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import frozen_rollout, restore_snapshot
from dvgc.runtime import build_inference, save_json
from dvgc.stage_reachability import evaluate_entry


TERMINAL_FEATURES = (
    "x", "z", "roll", "pitch", "vx", "vz", "wx", "wy", "wz", "hip", "knee",
)
TERMINAL_INDEX = {
    "x": 0, "z": 2, "roll": 3, "pitch": 4, "vx": 6, "vz": 8,
    "wx": 9, "wy": 10, "wz": 11, "hip": 13, "knee": 14,
}


def _actions():
    pairs = (
        (0., 0.), (.18, 0.), (-.18, 0.), (0., .18), (0., -.18),
        (.18, .18), (-.18, -.18), (.18, -.18), (-.18, .18),
        (.32, .12), (-.32, -.12), (.12, .32), (-.12, -.32),
    )
    return [jp.asarray([0., 0., hip, knee], jp.float32) for hip, knee in pairs]


def _bridge_has_no_physical_failure(row):
    return row["termination_reason"] in (
        "formal_descent_support_entry", "horizon_exhaustion", "stage_timeout",
    )


def _terminal_distance(feature, target, center, scale):
    query = np.asarray([
        feature[TERMINAL_INDEX[name]] for name in TERMINAL_FEATURES
    ], float)
    normalized = (query - center) / scale
    return float(np.min(np.linalg.norm(target - normalized[None, :], axis=1)))


def _joint_margin(model, qpos):
    values = []
    for name in ("hip_joint", "knee_joint"):
        joint = model.joint(name)
        address = int(model.jnt_qposadr[joint.id])
        values.append(min(
            qpos[address] - model.jnt_range[joint.id, 0],
            model.jnt_range[joint.id, 1] - qpos[address],
        ))
    return float(min(values))


def _state_score(
    env, state, previous_vz, support_metadata, model,
    terminal_target, terminal_center, terminal_scale, action_energy,
):
    sample = sample_from_state(env, state, previous_vz)
    feature = np.asarray(sample["physical_feature"], float)
    entry = evaluate_entry("apex", sample, env._config, support_metadata)
    done = bool(float(np.asarray(state.done)) > .5)
    qpos = np.asarray(state.data.qpos)
    apex = bool(int(np.asarray(state.info.get("apex_seen", 0))))
    pose_ok = (
        abs(feature[3]) < np.deg2rad(35.)
        and abs(feature[4]) < np.deg2rad(75.)
        and np.linalg.norm(feature[9:12]) < 4.
    )
    stable = bool(apex and feature[8] < -.05 and pose_ok and not done)
    target_distance = _terminal_distance(
        feature, terminal_target, terminal_center, terminal_scale
    )
    pose_cost = (
        (feature[3] / np.deg2rad(35.)) ** 2
        + (feature[4] / np.deg2rad(75.)) ** 2
        + np.sum(np.square(feature[9:12] / 4.))
    )
    margin = _joint_margin(model, qpos)
    score = (
        1_000_000. * int(entry["valid"])
        + 10_000. * int(stable)
        + 1_000. * int(apex)
        + 100. * int(not done)
        - 50. * pose_cost
        - .5 * target_distance
        - 10. * max(0., .03 - margin)
        - .2 * action_energy
    )
    if done:
        score -= 1_000_000.
    return score, {
        "feature": feature, "entry": entry, "done": done, "apex": apex,
        "stable": stable, "target_distance": target_distance,
        "joint_margin": margin, "pose_cost": float(pose_cost),
    }


def _choose_action(
    env, step, state, previous_vz, support_metadata, model,
    terminal_target, terminal_center, terminal_scale, lookahead,
):
    decisions = []
    for index, action in enumerate(_actions()):
        probe = state
        probe_previous_vz = previous_vz
        score = -float("inf"); diagnostic = None
        for _ in range(lookahead):
            probe = step(probe, action)
            score, diagnostic = _state_score(
                env, probe, probe_previous_vz, support_metadata, model,
                terminal_target, terminal_center, terminal_scale,
                float(np.sum(np.square(np.asarray(action)))),
            )
            probe_previous_vz = float(diagnostic["feature"][8])
            if diagnostic["done"] or diagnostic["entry"]["valid"]:
                break
        decisions.append((score, -float(np.sum(np.square(np.asarray(action)))),
                          -index, action, diagnostic))
    best = max(decisions, key=lambda row: row[:3])
    return best[3], {
        "selected_score": float(best[0]),
        "selected_terminal_distance": float(best[4]["target_distance"]),
        "candidate_scores": [float(row[0]) for row in decisions],
    }


def _downstream(
    runtime, state, key, support_metadata, horizon, *, allow_without_formal,
):
    env, step, dinfer, lenv, lstep, linfer = runtime
    previous_vz = float(np.asarray(state.data.qvel[2]))
    landing_snapshot = None; reason = "horizon_exhaustion"
    for tick in range(horizon):
        key, action_key = jax.random.split(key)
        action, _ = dinfer(state.obs, action_key)
        state = step(state, action)
        sample = sample_from_state(env, state, previous_vz)
        landing = evaluate_entry("descent", sample, env._config)
        if landing["valid"]:
            landing_snapshot = env.snapshot_record(state, "landing")
            reason = "valid_landing_entry"; break
        if float(np.asarray(state.done)) > .5:
            code = int(np.asarray(state.info["end_code"]))
            reason = END_REASON.get(code, f"unknown_{code}"); break
        previous_vz = float(sample["physical_feature"][8])
    final = False; landing_reason = None
    if landing_snapshot is not None:
        lkey = jax.random.fold_in(key, 50_000_000)
        _, outcome = frozen_rollout(
            lenv, linfer, restore_snapshot(lenv, landing_snapshot, lkey), lkey,
            horizon=horizon, step_fn=lstep,
        )
        final = bool(outcome["final"])
        landing_reason = END_REASON.get(outcome["end_code"], "unknown")
    return {
        "descent_controller_success": landing_snapshot is not None,
        "time_to_landing_entry": tick + 1 if landing_snapshot is not None else None,
        "final_landing_recovery": final,
        "descent_termination_reason": reason,
        "landing_termination_reason": landing_reason,
        "started_without_formal_support_entry": bool(allow_without_formal),
    }


def _run_bridge(
    runtime, start, seed, support_metadata, model,
    terminal_target, terminal_center, terminal_scale,
    horizon, lookahead, downstream_horizon,
):
    env, step, *_ = runtime
    key = jax.random.PRNGKey(seed)
    state = restore_snapshot(env, start, key)
    previous_vz = float(np.asarray(state.data.qvel[2]))
    trace = []; stable_count = 0; stable_snapshot = None
    formal_snapshot = None; reason = "horizon_exhaustion"
    for tick in range(horizon):
        action, decision = _choose_action(
            env, step, state, previous_vz, support_metadata, model,
            terminal_target, terminal_center, terminal_scale, lookahead,
        )
        state = step(state, action)
        score, diagnostic = _state_score(
            env, state, previous_vz, support_metadata, model,
            terminal_target, terminal_center, terminal_scale,
            float(np.sum(np.square(np.asarray(action)))),
        )
        feature = diagnostic["feature"]
        stable_count = stable_count + 1 if diagnostic["stable"] else 0
        if stable_count >= 4 and stable_snapshot is None:
            stable_snapshot = env.snapshot_record(state, "flight")
        trace.append({
            "tick": tick + 1, "action": np.asarray(action).tolist(),
            "roll": float(feature[3]), "pitch": float(feature[4]),
            "angular_velocity": feature[9:12].tolist(),
            "vx": float(feature[6]), "vz": float(feature[8]),
            "hip": float(feature[13]), "knee": float(feature[14]),
            "physical_apex_crossed": diagnostic["apex"],
            "stable_physical_descent": stable_count >= 4,
            "formal_descent_support_entry": bool(diagnostic["entry"]["valid"]),
            "support_distance": diagnostic["entry"].get("support_distance"),
            "terminal_cluster_distance": diagnostic["target_distance"],
            "joint_margin": diagnostic["joint_margin"],
            "decision": decision,
        })
        if diagnostic["entry"]["valid"]:
            formal_snapshot = env.snapshot_record(state, "flight")
            reason = "formal_descent_support_entry"; break
        if diagnostic["done"]:
            code = int(np.asarray(state.info["end_code"]))
            reason = END_REASON.get(code, f"unknown_{code}"); break
        previous_vz = float(feature[8])
    downstream = {
        "descent_controller_success": False, "final_landing_recovery": False,
        "descent_termination_reason": None, "landing_termination_reason": None,
        "started_without_formal_support_entry": False,
    }
    source = formal_snapshot or stable_snapshot
    if source is not None:
        downstream = _downstream(
            runtime, restore_snapshot(env, source, jax.random.fold_in(key, 70_000_000)),
            jax.random.fold_in(key, 80_000_000), support_metadata,
            downstream_horizon, allow_without_formal=formal_snapshot is None,
        )
    return {
        "physical_apex_crossed": any(row["physical_apex_crossed"] for row in trace),
        "stable_physical_descent": stable_snapshot is not None,
        "formal_descent_support_entry": formal_snapshot is not None,
        **downstream,
        "termination_reason": reason,
        "steps": len(trace), "trace": trace,
        "stable_snapshot": stable_snapshot, "formal_snapshot": formal_snapshot,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authority-bank", required=True)
    p.add_argument("--authority-report", required=True)
    p.add_argument("--start-bank", default="")
    p.add_argument("--support-bank", required=True)
    p.add_argument("--terminal-bank", required=True)
    p.add_argument("--descent-policy", required=True)
    p.add_argument("--landing-policy", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--parent", action="append", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--lookahead", type=int, default=3)
    p.add_argument("--downstream-horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=11_500_000)
    p.add_argument("--branch-mode", choices=("deterministic", "fresh", "all"),
                   default="all")
    p.add_argument("--nominal-report", default="")
    a = p.parse_args()
    root = Path(a.output_root); root.mkdir(parents=True, exist_ok=True)
    authority_bank = SnapshotBank.load(a.authority_bank)
    authority = json.loads(Path(a.authority_report).read_text())
    support = SnapshotBank.load(a.support_bank)
    support_metadata = dict(support.metadata)
    support_metadata["support_features"] = [
        row["physical_feature"] for row in support.records
    ]
    terminal = SnapshotBank.load(a.terminal_bank)
    terminal_center = np.asarray(terminal.metadata["normalization_center"], float)
    terminal_scale = np.asarray(terminal.metadata["normalization_scale"], float)
    terminal_target = np.asarray([
        [(row["physical_feature"][TERMINAL_INDEX[name]] - terminal_center[i])
         / terminal_scale[i] for i, name in enumerate(TERMINAL_FEATURES)]
        for row in terminal.records
    ], float)
    dp, dc, _ = load_bundle(a.descent_policy, verify_files=True)
    lp, lc, _ = load_bundle(a.landing_policy, verify_files=True)

    def make_runtime(variant):
        overrides = {k: v for k, v in variant.items() if k != "id"}
        cfg = load_config(a.config, {
            **dc, **overrides, "training_stage": "flight",
            "use_bank_resets": False, "domain_randomization": False,
            "obs_noise_enable": False, "stage_reachability_objective": "",
        })
        lcfg = load_config(a.config, {
            **lc, **overrides, "training_stage": "landing",
            "use_bank_resets": False, "domain_randomization": False,
            "obs_noise_enable": False,
        })
        env = OrangeBikeDVGC(
            cfg, snapshot_bank=SnapshotBank(), stage_support_bank=support
        )
        lenv = OrangeBikeDVGC(lcfg, snapshot_bank=SnapshotBank())
        return (
            env, jax.jit(env.step), build_inference(env, dp, deterministic=True),
            lenv, jax.jit(lenv.step),
            build_inference(lenv, lp, deterministic=True),
        )

    runtimes = {}
    model = mujoco.MjModel.from_xml_path(
        str(load_config(a.config).xml_path)
    )
    starts = []
    selected_start_bank = (
        SnapshotBank.load(a.start_bank) if a.start_bank else None
    )
    for requested in a.parent:
        info = authority["parent_results"][requested]
        if selected_start_bank is not None:
            start = next(
                row for row in selected_start_bank.records
                if row["display_parent"] == requested
            )
        else:
            rows = [
                row for row in authority_bank.records
                if row["trajectory_parent_id"] == info["parent_id"]
            ]
            if info["classification"] == "apex_local_correctable":
                start = max(rows, key=lambda row: row["relative_to_apex"])
            elif info["latest_effective_relative_to_apex"] is not None:
                start = next(
                    row for row in rows if row["relative_to_apex"]
                    == info["latest_effective_relative_to_apex"]
                )
            else:
                start = min(rows, key=lambda row: row["relative_to_apex"])
        starts.append((requested, start, info))
    outcomes = []; stable_records = []; formal_records = []
    nominal_variant = next(
        variant for variant in DYNAMICS_VARIANTS if variant["id"] == "nominal"
    )
    nominal_success = {}
    if a.nominal_report:
        nominal = json.loads(Path(a.nominal_report).read_text())
        nominal_success = {
            row["parent"]: bool(row["stable_physical_descent"])
            for row in nominal["outcomes"] if row["branch_kind"] == "deterministic"
        }
    for pi, (display_parent, start, info) in enumerate(starts):
        branch_results = []
        deterministic_stable = nominal_success.get(display_parent, False)
        if a.branch_mode in ("deterministic", "all"):
            deterministic_key = nominal_variant["id"]
            if deterministic_key not in runtimes:
                runtimes[deterministic_key] = make_runtime(nominal_variant)
            result = _run_bridge(
                runtimes[deterministic_key], start, a.seed + pi * 100_000,
                support_metadata, model, terminal_target, terminal_center,
                terminal_scale, a.horizon, a.lookahead, a.downstream_horizon,
            )
            branch_results.append(("deterministic", nominal_variant, 0, result))
            deterministic_stable = result["stable_physical_descent"]
        if a.branch_mode in ("fresh", "all") and deterministic_stable:
            for fresh in range(4):
                variant = DYNAMICS_VARIANTS[fresh % len(DYNAMICS_VARIANTS)]
                if variant["id"] not in runtimes:
                    runtimes[variant["id"]] = make_runtime(variant)
                fresh_result = _run_bridge(
                    runtimes[variant["id"]], start,
                    a.seed + pi * 100_000 + 10_000 + fresh,
                    support_metadata, model, terminal_target, terminal_center,
                    terminal_scale, a.horizon, a.lookahead,
                    a.downstream_horizon,
                )
                branch_results.append(("fresh_dynamics", variant, fresh, fresh_result))
        for kind, variant, branch, branch_result in branch_results:
            stable = branch_result.pop("stable_snapshot")
            formal = branch_result.pop("formal_snapshot")
            row = {
                "parent": display_parent, "parent_id": info["parent_id"],
                "start_snapshot_id": start["id"],
                "start_relative_to_apex": start["relative_to_apex"],
                "control_authority_class": info["classification"],
                "branch_kind": kind, "branch": branch,
                "dynamics_variant": variant["id"],
                "seed": a.seed + pi * 100_000 + (
                    0 if kind == "deterministic" else 10_000 + branch
                ),
                **branch_result,
            }
            outcomes.append(row)
            if stable is not None:
                stable.update({
                    "candidate_kind": "stable_physical_descent_proposal",
                    "trajectory_parent_id": info["parent_id"],
                    "bridge_parent": display_parent,
                    "bridge_seed": row["seed"],
                    "bridge_feedback": "receding_horizon_bounded_shooting",
                    "formal_descent_support_entry": formal is not None,
                })
                stable_records.append(stable)
            if formal is not None:
                formal.update({
                    "candidate_kind": "formal_descent_support_entry_proposal",
                    "trajectory_parent_id": info["parent_id"],
                    "bridge_parent": display_parent,
                    "bridge_seed": row["seed"],
                    "bridge_feedback": "receding_horizon_bounded_shooting",
                })
                formal_records.append(formal)
    SnapshotBank(stable_records, {
        "artifact_role": "stable_physical_descent_bridge_proposals",
        "certified_tube": False, "safe_claim_allowed": False,
        "authority_bank_sha256": file_sha256(a.authority_bank),
        "feedback_controller": "receding_horizon_bounded_shooting",
    }).save(root / "stable_physical_descent.pkl")
    SnapshotBank(formal_records, {
        "artifact_role": "formal_descent_support_entry_bridge_proposals",
        "certified_tube": False, "safe_claim_allowed": False,
        "support_bank_sha256": file_sha256(a.support_bank),
        "feedback_controller": "receding_horizon_bounded_shooting",
    }).save(root / "formal_descent_entries.pkl")
    fresh = [row for row in outcomes if row["branch_kind"] == "fresh_dynamics"]
    parent_fresh_stable = Counter(
        row["parent"] for row in fresh if row["stable_physical_descent"]
    )
    parent_fresh_stable_without_failure = Counter(
        row["parent"] for row in fresh
        if row["stable_physical_descent"] and _bridge_has_no_physical_failure(row)
    )
    parent_fresh_formal = Counter(
        row["parent"] for row in fresh if row["formal_descent_support_entry"]
    )
    gate_a = any(
        value >= 2 for value in parent_fresh_stable_without_failure.values()
    )
    gate_b = sum(value >= 1 for value in parent_fresh_formal.values()) >= 2
    payload = {
        "status": "PASS",
        "artifact_role": "apex_feedback_bridge_feasibility_pilot",
        "controller_type": "receding_horizon_bounded_shooting",
        "feedback_used": True, "replanning_interval_ticks": 1,
        "branch_mode": a.branch_mode,
        "lookahead_ticks": a.lookahead, "controller_horizon": a.horizon,
        "action_candidates": [np.asarray(x).tolist() for x in _actions()],
        "parents": list(a.parent), "branches": len(outcomes),
        "hierarchy": {
            "physical_apex_crossed": sum(
                row["physical_apex_crossed"] for row in outcomes
            ),
            "stable_physical_descent": sum(
                row["stable_physical_descent"] for row in outcomes
            ),
            "formal_descent_support_entry": sum(
                row["formal_descent_support_entry"] for row in outcomes
            ),
            "descent_controller_success": sum(
                row["descent_controller_success"] for row in outcomes
            ),
            "final_landing_recovery": sum(
                row["final_landing_recovery"] for row in outcomes
            ),
        },
        "fresh_seed_stable_by_parent": dict(parent_fresh_stable),
        "fresh_seed_stable_without_physical_failure_by_parent": dict(
            parent_fresh_stable_without_failure
        ),
        "fresh_seed_formal_by_parent": dict(parent_fresh_formal),
        "gate_a_local_physical_feasibility": gate_a,
        "gate_b_formal_interface_feasibility": gate_b,
        "gate_c_apex_ppo_authorized": False,
        "apex_ppo_authorization_reason": (
            "Gate C remains separately conditioned on 16-32 states, >=4 "
            "parents, >=2 formal-positive parents, fresh replay, reward and "
            "runtime gates"
        ),
        "termination_reasons": dict(Counter(
            row["termination_reason"] for row in outcomes
        )),
        "outcomes": outcomes,
        "stable_bank": str((root / "stable_physical_descent.pkl").resolve()),
        "stable_bank_sha256": file_sha256(root / "stable_physical_descent.pkl"),
        "formal_bank": str((root / "formal_descent_entries.pkl").resolve()),
        "formal_bank_sha256": file_sha256(root / "formal_descent_entries.pkl"),
    }
    save_json(root / "report.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
