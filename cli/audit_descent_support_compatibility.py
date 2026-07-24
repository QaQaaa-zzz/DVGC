"""Audit legacy Descent proposal support against the current frozen runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import jax
import numpy as np

from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, config_hash, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import frozen_rollout, restore_snapshot
from dvgc.runtime import build_inference, save_json
from dvgc.stage_reachability import evaluate_entry


def _schema_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _replay_label(branches):
    finals = sum(bool(row["final_landing_recovery"]) for row in branches)
    if finals == len(branches) and branches:
        return "final_safe_replay"
    if finals:
        return "boundary_replay"
    if branches and all(row["physical_failure"] for row in branches):
        return "dead_replay"
    return "unknown_replay"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support-bank", required=True)
    p.add_argument("--descent-policy", required=True)
    p.add_argument("--landing-policy", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--branches", type=int, default=4)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=11_000_000)
    p.add_argument(
        "--indices",
        help="Comma-separated support indices for a targeted runtime check.",
    )
    p.add_argument(
        "--historical-audit",
        help="Optional prior audit used only for label-agreement reporting.",
    )
    a = p.parse_args()
    bank = SnapshotBank.load(a.support_bank)
    dp, dc, dm = load_bundle(a.descent_policy, verify_files=True)
    lp, lc, lm = load_bundle(a.landing_policy, verify_files=True)
    cfg = load_config(a.config, {
        **dc, "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "descent_to_landing",
    })
    lcfg = load_config(a.config, {
        **lc, "training_stage": "landing", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    lenv = OrangeBikeDVGC(lcfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step); lstep = jax.jit(lenv.step)
    infer = build_inference(env, dp, deterministic=True)
    linfer = build_inference(lenv, lp, deterministic=True)
    sample_obs = env.reset(jax.random.PRNGKey(0)).obs
    observation_schema = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sample_obs.items()
    }
    selected_indices = (
        [int(value) for value in a.indices.split(",") if value.strip()]
        if a.indices else list(range(len(bank.records)))
    )
    if len(set(selected_indices)) != len(selected_indices) or any(
        index < 0 or index >= len(bank.records) for index in selected_indices
    ):
        raise SystemExit("indices must be unique valid support-bank indices")
    historical_rows = {}
    if a.historical_audit:
        prior = json.loads(Path(a.historical_audit).read_text())
        historical_rows = {
            row["candidate_id"]: row for row in prior.get("rows", [])
        }
    rows = []
    for i in selected_indices:
        row = bank.records[i]
        branches = []
        for branch in range(a.branches):
            seed = a.seed + i * 100 + branch
            key = jax.random.PRNGKey(seed)
            state = restore_snapshot(env, row, key)
            restored_qpos = np.asarray(state.data.qpos)
            restored_qvel = np.asarray(state.data.qvel)
            restore_qpos_error = float(np.max(np.abs(restored_qpos - np.asarray(row["qpos"]))))
            restore_qvel_error = float(np.max(np.abs(restored_qvel - np.asarray(row["qvel"]))))
            phase0 = int(np.asarray(state.info["phase"]))
            previous_vz = float(np.asarray(state.data.qvel[2]))
            landing_snapshot = None
            reason = "horizon_exhaustion"
            shock_failure = False
            stable_descent_ticks = 0
            maximum_abs_pitch = abs(float(np.asarray(
                env._physical_feature(state.data)[4]
            )))
            maximum_abs_pitch_rate = abs(float(np.asarray(
                env._physical_feature(state.data)[10]
            )))
            for tick in range(a.horizon):
                key, action_key, noise_key = jax.random.split(key, 3)
                action, _ = infer(state.obs, action_key)
                if branch:
                    action = np.clip(
                        np.asarray(action) + np.asarray(jax.random.normal(
                            noise_key, action.shape)) * .01, -1, 1
                    )
                state = step(state, action)
                feature = np.asarray(env._physical_feature(state.data))
                maximum_abs_pitch = max(
                    maximum_abs_pitch, abs(float(feature[4]))
                )
                maximum_abs_pitch_rate = max(
                    maximum_abs_pitch_rate, abs(float(feature[10]))
                )
                sample = sample_from_state(env, state, previous_vz)
                entry = evaluate_entry("descent", sample, cfg)
                if entry["valid"]:
                    landing_snapshot = env.snapshot_record(state, "landing")
                    reason = "valid_landing_entry"; break
                if float(np.asarray(state.done)) > .5:
                    code = int(np.asarray(state.info["end_code"]))
                    reason = END_REASON.get(code, f"unknown_{code}")
                    if tick < 5 and reason not in (
                        "recovery", "chain_entry", "next_stage_entry",
                    ):
                        shock_failure = True
                    break
                stable_descent_ticks += 1
                previous_vz = float(sample["physical_feature"][8])
            final = False; landing_reason = None
            if landing_snapshot is not None:
                lkey = jax.random.PRNGKey(seed + 50_000_000)
                _, outcome = frozen_rollout(
                    lenv, linfer, restore_snapshot(lenv, landing_snapshot, lkey),
                    lkey, horizon=a.horizon, step_fn=lstep,
                    action_noise_std=.01 if branch else 0.,
                )
                final = bool(outcome["final"])
                landing_reason = END_REASON.get(outcome["end_code"], "unknown")
            physical = reason not in (
                "valid_landing_entry", "horizon_exhaustion", "stage_timeout",
                "recovery", "next_stage_entry",
            )
            branches.append({
                "branch": branch, "seed": seed, "t0_phase": phase0,
                "restore_qpos_max_error": restore_qpos_error,
                "restore_qvel_max_error": restore_qvel_error,
                "five_step_reset_shock_failure": shock_failure,
                "descent_controller_success": landing_snapshot is not None,
                "time_to_landing_entry": tick + 1 if landing_snapshot else None,
                "stable_descent_ticks": stable_descent_ticks,
                "maximum_abs_pitch": maximum_abs_pitch,
                "maximum_abs_pitch_rate": maximum_abs_pitch_rate,
                "final_landing_recovery": final,
                "physical_failure": physical,
                "termination_reason": reason,
                "landing_termination_reason": landing_reason,
            })
        historical = historical_rows.get(row["id"], {})
        original = (
            historical.get("replay_label")
            or row.get("final", {}).get("label")
        )
        replay = _replay_label(branches)
        comparable = {
            "safe": "final_safe_replay", "boundary": "boundary_replay",
            "dead": "dead_replay",
        }
        rows.append({
            "candidate_id": row["id"],
            "support_index": i,
            "parent": row.get("origin_parent", row.get("trajectory_parent_id", row["id"])),
            "region": row.get("descent_support_region"),
            "source_provenance": row.get("support_provenance"),
            "original_final_label": original,
            "replay_label": replay,
            "original_label_agrees": (
                original == replay
                if historical
                else (
                    comparable.get(original) == replay
                    if original in comparable else None
                )
            ),
            "branches": branches,
        })
    flat = [branch for row in rows for branch in row["branches"]]
    comparable = [row["original_label_agrees"] for row in rows
                  if row["original_label_agrees"] is not None]
    parents = defaultdict(list)
    for row in rows:
        parents[str(row["parent"])].extend(row["branches"])
    parentwise = {
        parent: {
            "states": sum(str(row["parent"]) == parent for row in rows),
            "branches": len(branches),
            "descent_controller_success_rate": float(np.mean([
                b["descent_controller_success"] for b in branches
            ])),
            "final_recovery_rate": float(np.mean([
                b["final_landing_recovery"] for b in branches
            ])),
        } for parent, branches in parents.items()
    }
    matcher = bank.metadata.get("stage_entry_matcher", {})
    provenance = {
        "support_bank_sha256": file_sha256(a.support_bank),
        "support_xml_sha256": bank.metadata.get("xml_sha256"),
        "current_xml_sha256": file_sha256(cfg.xml_path),
        "support_config_hash": bank.metadata.get("config_hash"),
        "current_config_hash": config_hash(cfg),
        "support_action_mapping_version": bank.metadata.get("action_mapping_version"),
        "current_action_mapping_version": ACTION_MAPPING_VERSION,
        "descent_policy_version": dm["policy_version"],
        "descent_policy_hash": file_sha256(Path(a.descent_policy) / "params.pkl"),
        "landing_policy_version": lm["policy_version"],
        "landing_policy_hash": file_sha256(Path(a.landing_policy) / "params.pkl"),
        "observation_schema": observation_schema,
        "observation_schema_hash": _schema_hash(observation_schema),
        "action_schema": {"size": env.action_size,
                          "mapping_version": ACTION_MAPPING_VERSION,
                          "bounds": [-1., 1.]},
        "action_schema_hash": _schema_hash({
            "size": env.action_size, "mapping_version": ACTION_MAPPING_VERSION,
            "bounds": [-1., 1.],
        }),
        "phase_detector_source_hash": file_sha256("dvgc/env.py"),
        "feature_extractor_source_hash": file_sha256("dvgc/env.py"),
        "matcher_feature_names": matcher.get("feature_names"),
        "matcher_center": matcher.get("center"), "matcher_scale": matcher.get("scale"),
        "matcher_radius": matcher.get("radius"),
        "matcher_sha256": matcher.get("matcher_sha256"),
        "qpos_joint_order": [
            Path(cfg.xml_path).name, "floating_base_joint(7)", "frontwheel_joint",
            "steering_joint", "rearwheel_joint", "hip_joint", "knee_joint",
        ],
        "velocity_frame": "root linear velocity is world frame in physical_feature",
        "coordinate_frame": "root x/y/z are authoritative XML world coordinates",
    }
    reset_rate = float(np.mean([
        b["restore_qpos_max_error"] <= 1e-6 and b["restore_qvel_max_error"] <= 1e-6
        for b in flat
    ]))
    phase_rate = float(np.mean([b["t0_phase"] == 2 for b in flat]))
    shock_rate = float(np.mean([b["five_step_reset_shock_failure"] for b in flat]))
    payload = {
        "status": "PASS", "artifact_role": "descent_support_runtime_compatibility_audit",
        "provenance": provenance, "states": len(rows), "branches": len(flat),
        "reset_valid_rate": reset_rate,
        "t0_descent_phase_rate": phase_rate,
        "five_step_reset_shock_failure_rate": shock_rate,
        "descent_controller_success_rate": float(np.mean([
            b["descent_controller_success"] for b in flat
        ])),
        "landing_final_recovery_rate": float(np.mean([
            b["final_landing_recovery"] for b in flat
        ])),
        "physical_failure_rate": float(np.mean([b["physical_failure"] for b in flat])),
        "termination_reasons": dict(Counter(b["termination_reason"] for b in flat)),
        "replay_labels": dict(Counter(row["replay_label"] for row in rows)),
        "original_label_comparable_states": len(comparable),
        "original_label_agreement_rate": (
            float(np.mean(comparable)) if comparable else None
        ),
        "parentwise": parentwise, "rows": rows,
        "descent_support_runtime_stale": bool(
            reset_rate < .95 or phase_rate < .95 or shock_rate > .50
        ),
        "staleness_rule": "restore or phase validity below 95%, or more than half of states fail within the fixed five-step continuation window",
        "proposal_support_only": True,
        "selected_indices": selected_indices,
        "maximum_stable_descent_ticks": max(
            (branch["stable_descent_ticks"] for branch in flat), default=0
        ),
        "descent_asset_runtime_check": (
            "PASS"
            if any(branch["final_landing_recovery"] for branch in flat)
            else "FAIL"
        ),
    }
    save_json(a.output, payload)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("rows", "parentwise")}, indent=2))


if __name__ == "__main__":
    main()
