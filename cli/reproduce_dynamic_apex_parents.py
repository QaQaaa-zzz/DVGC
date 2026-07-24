"""Reproduce the three known Ascent->Apex parents under bounded variations."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from cli.acquire_ascent_apex_parents import _local_action
from cli.search_takeoff_actions import SEQUENCES, action_at
from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.certification import DYNAMICS_VARIANTS
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json
from dvgc.stage_reachability import evaluate_entry


def _variant_cfg(path, variant):
    return load_config(path, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
        **{k: v for k, v in variant.items() if k != "id"},
    })


def _branch_specs(parent, base, variants):
    specs = [
        {"kind": "deterministic", "seed": 0, "variant": variants[0],
         "parameters": dict(base)},
        {"kind": "deterministic_repeat", "seed": 1, "variant": variants[0],
         "parameters": dict(base)},
    ]
    for i, variant in enumerate(variants[:4]):
        specs.append({"kind": "fresh_dynamics", "seed": 100 + i,
                      "variant": variant, "parameters": dict(base)})
    for delta in (-2, 2):
        item = dict(base); item["duration"] = max(4, int(item["duration"]) + delta)
        specs.append({"kind": "duration_variation", "seed": 200 + delta,
                      "variant": variants[0], "parameters": item})
    for scale in (.95, 1.05):
        item = dict(base)
        item["hip_amplitude"] = float(np.clip(item["hip_amplitude"] * scale, -1, 1))
        item["knee_ratio"] = float(np.clip(item["knee_ratio"] * scale, 0, 1))
        specs.append({"kind": "amplitude_variation", "seed": int(scale * 1000),
                      "variant": variants[0], "parameters": item})
    return specs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-bank", required=True)
    p.add_argument("--entry-bank", required=True)
    p.add_argument("--acquisition-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_900_000)
    a = p.parse_args()
    reference = SnapshotBank.load(a.reference_bank)
    entries = SnapshotBank.load(a.entry_bank)
    acquisition = json.loads(Path(a.acquisition_report).read_text())
    ref = next(r for r in reference.records if r.get("reference_index") == 131)
    parents = [{
        "parent_id": "reference:131", "row": ref,
        "base": {"round": "known", "hip_amplitude": 1., "knee_ratio": .5,
                 "start_tick": 0, "duration": 50},
        "action_mode": "reference_parent",
    }]
    by_id = {r["trajectory_parent_id"]: r for r in entries.records}
    for parent_id in acquisition["successful_parent_ids"]:
        outcome = next(r for r in acquisition["search_outcomes"]
                       if r["success"] and r["trajectory_parent_id"] == parent_id)
        parents.append({"parent_id": parent_id, "row": by_id[parent_id],
                        "base": outcome["parameters"], "action_mode": "local"})
    variants = [dict(x) for x in DYNAMICS_VARIANTS[:4]]
    envs = {}
    for variant in variants:
        cfg = _variant_cfg(a.config, variant)
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
        envs[variant["id"]] = (cfg, env, jax.jit(env.step),
                               GroundSupportSolver(cfg.xml_path))
    model = mujoco.MjModel.from_xml_path(str(_variant_cfg(a.config, variants[0]).xml_path))
    hip_id = model.joint("hip_joint").id; knee_id = model.joint("knee_joint").id
    hip_q = int(model.jnt_qposadr[hip_id]); knee_q = int(model.jnt_qposadr[knee_id])
    hip_v = int(model.jnt_dofadr[hip_id]); knee_v = int(model.jnt_dofadr[knee_id])
    rows = []
    for parent in parents:
        for bi, spec in enumerate(_branch_specs(parent["parent_id"], parent["base"], variants)):
            cfg, env, step, geometry = envs[spec["variant"]["id"]]
            seed = a.seed + len(rows) * 100 + int(spec["seed"])
            state = restore_snapshot(env, parent["row"], jax.random.PRNGKey(seed))
            previous_vz = float(np.asarray(state.data.qvel[2]))
            trace, captured, reason = [], None, "horizon_exhaustion"
            for tick in range(a.horizon):
                action = (action_at(SEQUENCES["hip_full_knee_half"], tick)
                          if parent["action_mode"] == "reference_parent"
                          else _local_action(spec["parameters"], tick))
                state = step(state, action)
                sample = sample_from_state(env, state, previous_vz)
                entry = evaluate_entry("ascent", sample, cfg)
                qpos = np.asarray(state.data.qpos); qvel = np.asarray(state.data.qvel)
                contact = geometry.measure(qpos, qvel, np.asarray(state.data.ctrl))
                feature = np.asarray(sample["physical_feature"], float)
                trace.append({
                    "tick": tick + 1, "root_z": float(feature[2]),
                    "vertical_velocity": float(feature[8]),
                    "roll": float(feature[3]), "pitch": float(feature[4]),
                    "angular_velocity": feature[9:12].tolist(),
                    "hip": float(qpos[hip_q]), "knee": float(qpos[knee_q]),
                    "hip_velocity": float(qvel[hip_v]), "knee_velocity": float(qvel[knee_v]),
                    "hip_margin": float(min(qpos[hip_q] - model.jnt_range[hip_id, 0],
                                            model.jnt_range[hip_id, 1] - qpos[hip_q])),
                    "knee_margin": float(min(qpos[knee_q] - model.jnt_range[knee_id, 0],
                                             model.jnt_range[knee_id, 1] - qpos[knee_q])),
                    "action": np.asarray(action).tolist(),
                    "wheel_contacts": int(contact["wheel_contacts"]),
                    "body_contacts": int(contact["body_contacts"]),
                    "wheel_clearance": float(contact["wheel_min"]),
                    "valid_apex_entry": bool(entry["valid"]),
                })
                if entry["valid"]:
                    captured = (tick + 1, env.snapshot_record(state, "flight"),
                                np.asarray(action))
                    reason = "next_stage_entry"; break
                if float(np.asarray(state.done)) > .5:
                    code = int(np.asarray(state.info["end_code"]))
                    reason = END_REASON.get(code, f"unknown_{code}"); break
                previous_vz = float(feature[8])
            shock = None
            if captured:
                _, snapshot, continuation = captured
                probe = restore_snapshot(env, snapshot, jax.random.PRNGKey(seed + 500_000))
                shock = False
                for _ in range(5):
                    probe = step(probe, jp.asarray(continuation))
                    if float(np.asarray(probe.done)) > .5:
                        shock = True; break
            rows.append({
                "parent_id": parent["parent_id"], "branch_kind": spec["kind"],
                "seed": seed, "dynamics_variant": spec["variant"]["id"],
                "parameters": spec["parameters"], "success": captured is not None,
                "time_to_apex": captured[0] if captured else None,
                "failure_reason": None if captured else reason,
                "action_saturation_fraction": float(np.mean([
                    np.mean(np.abs(t["action"]) >= .999) for t in trace
                ])) if trace else 0.,
                "snapshot_five_step_reset_shock": shock, "trace": trace,
            })
    summary = {}
    for parent in parents:
        selected = [r for r in rows if r["parent_id"] == parent["parent_id"]]
        fresh = [r for r in selected if r["branch_kind"] == "fresh_dynamics"]
        fresh_success = sum(r["success"] and not r["snapshot_five_step_reset_shock"]
                            for r in fresh)
        deterministic = [r for r in selected if r["branch_kind"].startswith("deterministic")]
        if fresh_success >= 2:
            classification = "robust_dynamic_parent"
        elif fresh_success:
            classification = "seed_conditional_dynamic_parent"
        elif any(r["success"] for r in deterministic):
            classification = "deterministic_only_parent"
        else:
            classification = "not_reproduced"
        summary[parent["parent_id"]] = {
            "classification": classification, "branches": len(selected),
            "successful_branches": sum(r["success"] for r in selected),
            "fresh_dynamics_successes": fresh_success,
            "valid_reset_shock_successes": sum(
                r["success"] and not r["snapshot_five_step_reset_shock"] for r in selected
            ),
            "time_to_apex": [r["time_to_apex"] for r in selected if r["success"]],
            "termination_reasons": dict(Counter(
                "next_stage_entry" if r["success"] else r["failure_reason"] for r in selected
            )),
        }
    save_json(a.output, {
        "status": "PASS", "artifact_role": "known_dynamic_apex_parent_robustness",
        "reference_bank_sha256": file_sha256(a.reference_bank),
        "entry_bank_sha256": file_sha256(a.entry_bank),
        "acquisition_report_sha256": file_sha256(a.acquisition_report),
        "fresh_dynamics_variants": variants, "parents": summary, "rows": rows,
        "generic_apex_only": True, "not_descent_support": True,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
