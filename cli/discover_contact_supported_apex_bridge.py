"""Bounded last-support action scan followed by long-horizon feedback."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import mujoco
import numpy as np

from cli.audit_apex_mpc_horizons import _run
from cli.discover_apex_feedback_bridge import (
    TERMINAL_FEATURES,
    TERMINAL_INDEX,
    _actions,
)
from dvgc.bank import SnapshotBank
from dvgc.centroidal import replay_centroidal
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, save_json


PHYSICAL_FAILURES = {
    "roll_limit", "pitch_limit", "prohibited_contact",
    "invalid_wheel_step_contact", "backward", "platform_back_edge_exit",
    "nonfinite",
}


def _contact_rank(momentum, target_center, target_scale):
    h = np.asarray(momentum["centroidal_angular_momentum"])[:2]
    residual = (h - target_center[:2]) / target_scale[:2]
    return float(np.linalg.norm(residual))


def _lexicographic(row):
    return (
        int(row["final_landing_recovery"]),
        int(row["descent_controller_success"]),
        int(row["formal_descent_support_entry"]),
        int(row["stable_16_ticks"]),
        int(row["termination_reason"] not in PHYSICAL_FAILURES),
        int(row["physical_apex_tick"] is not None),
        row["max_stable_descent_ticks"],
        -row["contact_momentum_residual"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lineage-bank", required=True)
    p.add_argument("--lineage-report", required=True)
    p.add_argument("--support-bank", required=True)
    p.add_argument("--terminal-bank", required=True)
    p.add_argument("--descent-policy", required=True)
    p.add_argument("--landing-policy", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--parent", action="append", required=True)
    p.add_argument("--contact-candidates", type=int, default=5)
    p.add_argument("--prediction-horizon", type=int, default=12)
    p.add_argument("--controller-horizon", type=int, default=40)
    p.add_argument("--seed", type=int, default=11_800_000)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args()
    lineage = SnapshotBank.load(a.lineage_bank)
    lineage_report = json.loads(Path(a.lineage_report).read_text())
    support = SnapshotBank.load(a.support_bank)
    support_metadata = dict(support.metadata)
    support_metadata["support_features"] = [
        row["physical_feature"] for row in support.records
    ]
    terminal = SnapshotBank.load(a.terminal_bank)
    terminal_center = np.asarray(
        terminal.metadata["normalization_center"], float
    )
    terminal_scale = np.asarray(
        terminal.metadata["normalization_scale"], float
    )
    terminal_target = np.asarray([
        [(row["physical_feature"][TERMINAL_INDEX[name]] - terminal_center[i])
         / terminal_scale[i] for i, name in enumerate(TERMINAL_FEATURES)]
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
    step = runtime[1]
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    terminal_h = []
    for record in terminal.records:
        result = replay_centroidal(
            model, record["qpos"], record["qvel"], record.get("ctrl")
        )
        terminal_h.append(result["centroidal_angular_momentum"][:2])
    terminal_h = np.asarray(terminal_h)
    h_center = np.median(terminal_h, axis=0)
    h_scale = np.maximum(
        np.percentile(terminal_h, 75, axis=0)
        - np.percentile(terminal_h, 25, axis=0),
        np.asarray([.03, .03]),
    )
    outcomes = []
    contact_scans = {}
    for parent_index, parent in enumerate(a.parent):
        start = next((
            row for row in lineage.records
            if row["trajectory_parent_id"] == parent
            and row["event_label"] == "last_support"
        ), None)
        if start is None:
            contact_scans[parent] = {
                "status": "missing_real_last_support_snapshot"
            }
            continue
        scan = []
        for action_index, action in enumerate(_actions()):
            state = restore_snapshot(
                env, start,
                jax.random.PRNGKey(a.seed + parent_index * 1000 + action_index),
            )
            state = step(state, action)
            momentum = replay_centroidal(
                model, np.asarray(state.data.qpos), np.asarray(state.data.qvel),
                np.asarray(state.data.ctrl),
            )
            snapshot = env.snapshot_record(state, "flight")
            scan.append({
                "action_index": action_index,
                "action": np.asarray(action).tolist(),
                "contact_after_action": len(
                    momentum["robot_terrain_contacts"]
                ),
                "centroidal_angular_momentum":
                    momentum["centroidal_angular_momentum"],
                "momentum_residual": _contact_rank(
                    momentum, h_center, h_scale
                ),
                "_snapshot": snapshot,
            })
        ranked = sorted(scan, key=lambda row: (
            row["momentum_residual"], row["action_index"]
        ))
        selected = ranked[:a.contact_candidates]
        # Preserve the recorded nominal separation action even if not in top-k.
        nominal_action = np.asarray(
            lineage_report["parents"][parent]["captured_events"][
                "separation"
            ]["action"]
        )
        nominal = min(scan, key=lambda row: np.linalg.norm(
            np.asarray(row["action"]) - nominal_action
        ))
        if nominal not in selected:
            selected[-1] = nominal
        contact_scans[parent] = {
            "status": "PASS",
            "last_support_snapshot_id": start["id"],
            "nominal_separation_action": nominal_action.tolist(),
            "all_candidates": [{
                key: value for key, value in row.items() if key != "_snapshot"
            } for row in scan],
            "selected_action_indices": [
                row["action_index"] for row in selected
            ],
        }
        for branch, candidate in enumerate(selected):
            result = _run(
                runtime, model, candidate["_snapshot"],
                a.seed + parent_index * 100_000 + 10_000 + branch,
                a.prediction_horizon, support_metadata, terminal_target,
                terminal_center, terminal_scale, a.controller_horizon, 2, 200,
            )
            outcomes.append({
                "parent": parent, "branch": branch,
                "last_support_snapshot_id": start["id"],
                "contact_action_index": candidate["action_index"],
                "contact_action": candidate["action"],
                "contact_momentum": candidate[
                    "centroidal_angular_momentum"
                ],
                "contact_momentum_residual":
                    candidate["momentum_residual"],
                **result,
            })
    best_by_parent = {}
    for parent in a.parent:
        rows = [row for row in outcomes if row["parent"] == parent]
        if rows:
            best = max(rows, key=_lexicographic)
            best_by_parent[parent] = {
                key: value for key, value in best.items()
                if key not in ("trace", "plans")
            }
    clean_gate_a = [
        row for row in outcomes
        if (row["stable_16_ticks"] or row["formal_descent_support_entry"])
        and row["termination_reason"] not in PHYSICAL_FAILURES
    ]
    formal_parents = {
        row["parent"] for row in outcomes
        if row["formal_descent_support_entry"]
        and row["termination_reason"] not in PHYSICAL_FAILURES
    }
    payload = {
        "status": "PASS",
        "artifact_role": "contact_supported_segmented_bridge_diagnostic",
        "diagnostic_only": True, "apex_ppo_authorized": False,
        "segments": [
            "contact_supported_momentum_shaping",
            "ballistic_morphology_shaping", "descent_capture",
        ],
        "lexicographic_priority": [
            "final_landing_recovery", "descent_controller_success",
            "formal_descent_support_entry", "stable_16_ticks",
            "no_physical_failure", "physical_apex_crossed",
            "momentum_pose_corridor", "support_distance",
            "smoothness_energy",
        ],
        "prediction_horizon": a.prediction_horizon,
        "replanning_interval_ticks": 2,
        "lineage_bank_sha256": file_sha256(a.lineage_bank),
        "support_bank_sha256": file_sha256(a.support_bank),
        "terminal_bank_sha256": file_sha256(a.terminal_bank),
        "xml_sha256": file_sha256(cfg.xml_path),
        "terminal_h_center": h_center.tolist(),
        "terminal_h_scale": h_scale.tolist(),
        "contact_scans": contact_scans,
        "outcomes": outcomes,
        "best_by_parent": best_by_parent,
        "summary": {
            "branches": len(outcomes),
            "stable_16_ticks": sum(x["stable_16_ticks"] for x in outcomes),
            "formal_descent_support_entry": sum(
                x["formal_descent_support_entry"] for x in outcomes
            ),
            "descent_controller_success": sum(
                x["descent_controller_success"] for x in outcomes
            ),
            "final_landing_recovery": sum(
                x["final_landing_recovery"] for x in outcomes
            ),
            "termination_reasons": dict(Counter(
                x["termination_reason"] for x in outcomes
            )),
            "gate_a_nominal": bool(clean_gate_a),
            "gate_b_nominal_two_formal_parents": len(formal_parents) >= 2,
        },
    }
    save_json(a.output, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
