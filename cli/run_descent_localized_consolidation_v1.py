"""Absorb one verified Descent correction while preserving all working P1 nodes."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.freeze_descent_predecessor_assets import verify_frozen_assets
from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L, _load_record, _restore
from cli.run_backward_descent_rsi_pilot import certify_policy
from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_predecessor import expand_residual_knots
from dvgc.descent_supervised import build_actor_tools, train_supervised
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.runtime import save_json


SOURCE = Path("runs/descent_diverse_p1_predecessor_recovery_v1")
PRIOR = Path("runs/backward_recovery_tube_fast_track_v1/descent_cem3_tier2/descent_cem_pilot_report.json")
DEFAULT_RUN = Path("runs/descent_diverse_p1_predecessor_recovery_v2_localized_consolidation")
TARGET_NODE = "b30f84966afd6427c270afedf575f29c"
BRANCHES = {
    "head": {"learning_rate": 1e-5, "steps": (10, 25, 50)},
    "last_block": {"learning_rate": 3e-6, "steps": (10, 25, 50)},
}
TEACHER_WEIGHT = .15
ANCHOR_RMS_LIMIT = .005
ANCHOR_MAX_LIMIT = .015


def select_behavior_candidate(rows):
    eligible = [row for row in rows if row["anchor_gate"]]
    return None if not eligible else min(eligible, key=lambda row: (row["teacher_imitation_rms"], row["steps"]))


def physical_acceptance(summary, balanced_count, full_count, baseline_balanced_ids):
    return {
        "balanced_P0_complete": summary["balanced"]["P0"] == balanced_count,
        "balanced_P1_complete": summary["balanced"]["P1"] == balanced_count,
        "full_P0_complete": summary["full"]["P0"] == full_count,
        "full_P1_complete": summary["full"]["P1"] == full_count,
        "zero_forgetting": set(baseline_balanced_ids) <= set(summary["balanced"]["P1_ids"]),
    }


def _node_record(node, harvested):
    artifact = Path(node["physical_state"]["source_artifact"])
    if artifact == SOURCE / "trajectory_harvested_snapshots.pkl":
        return copy.deepcopy(harvested[int(node["physical_state"]["source_index"])]["snapshot_v4"])
    return copy.deepcopy(_load_record(node["physical_state"]))


def _cert_summary(cert, ids):
    rows = {row["node_id"]: row for row in cert["rows"]}
    return {
        "P0": sum(bool(rows[node]["P0"]["pass"]) for node in ids),
        "P1": sum(bool(rows[node]["P1"]["pass"]) for node in ids),
        "P0_ids": sorted(node for node in ids if rows[node]["P0"]["pass"]),
        "P1_ids": sorted(node for node in ids if rows[node]["P1"]["pass"]),
    }


def _action_delta(actor, base_policy, policy, observation):
    before = np.asarray(actor(base_policy, jnp.asarray(observation)))
    after = np.asarray(actor(policy, jnp.asarray(observation)))
    delta = after - before
    return {"rms": float(np.sqrt(np.mean(delta * delta))), "max": float(np.max(np.abs(delta)))}


def _collect_prefix(env, actor, policy, record, seed, *, targets=None, ticks=5):
    state = _restore(env, record, jax.random.PRNGKey(seed)); observations=[]; actions=[]
    for tick in range(ticks):
        base = np.asarray(actor(policy, state.obs["state"]))
        target = base if targets is None else np.clip(base + targets[tick], -1., 1.)
        observations.append(np.asarray(state.obs["state"], np.float32)); actions.append(np.asarray(target, np.float32))
        state = env.step(state, jnp.asarray(target))
        if bool(np.asarray(state.done)) or bool(np.asarray(state.info["chain_ever"] > 0)):
            break
    return observations, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args(); root = Path(args.run)
    if root.exists(): raise SystemExit(f"refusing overwrite {root}")
    valid, failed = verify_frozen_assets(SOURCE)
    if not valid: raise SystemExit(f"frozen asset mismatch: {failed}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text()); fingerprint = source_fingerprint(Path.cwd())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != fingerprint:
        raise SystemExit("runtime gate stale")
    cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {
        "use_bank_resets": False, "expert_chain_termination": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    if file_sha256(cfg.xml_path) != EXPECTED["xml"] or cfg.action_mapping_version != ACTION_MAPPING_VERSION:
        raise SystemExit("runtime model mismatch")
    root.mkdir(parents=True)
    save_json(root / "cost_estimate.json", {"experiment": "localized_consolidation_v1", "estimated_seconds": 1200,
        "dynamic_certifications": 3 * 18, "PPO_steps": 0, "new_CEM": False, "heldout_used": False})
    save_json(root / "preregistration.json", {"target_node": TARGET_NODE, "teacher_prefix_ticks": 5,
        "preservation_prefix_ticks": 5, "teacher_weight": TEACHER_WEIGHT, "branches": BRANCHES,
        "checkpoint_selection": "minimum_teacher_imitation_subject_to_anchor_gate_without_physical_results",
        "anchor_limits": {"rms": ANCHOR_RMS_LIMIT, "max": ANCHOR_MAX_LIMIT},
        "physical_gate": "balanced_16_of_16_and_full_18_of_18_P0_P1_with_zero_forgetting",
        "winner_priority": ["head", "last_block"], "PPO_authorization": False})
    dparams, _, _ = load_bundle(PI_D, verify_files=True); lparams, _, _ = load_bundle(PI_L, verify_files=True)
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(C_L))
    _, actor, _ = build_actor_tools(env, dparams)
    balanced = json.loads((SOURCE / "balanced_p1_launch_subset_v1_frozen.json").read_text())["nodes"]
    full = json.loads((SOURCE / "full_p1_bank_v1.json").read_text())["nodes"]
    prior_gate = json.loads((SOURCE / "descent_balanced_rsi_pilot_v1/prefit_physical_gate.json").read_text())
    baseline_ids = prior_gate["baseline_balanced"]["P1_ids"]
    if set(node["node_id"] for node in balanced) - set(baseline_ids) != {TARGET_NODE}:
        raise SystemExit("target identity no longer uniquely matches the frozen baseline gap")
    harvested = pickle.loads((SOURCE / "trajectory_harvested_snapshots.pkl").read_bytes())
    by_id = {node["node_id"]: node for node in balanced}; target_node = by_id[TARGET_NODE]
    prior = json.loads(PRIOR.read_text())
    target_row = next(row for row in prior["rows"] if row["proposal"]["physical_state_sha256"] == target_node["source_state_hash"] and row["controller"] == target_node["controller_type"])
    residual = expand_residual_knots(target_row["residual_knots"])
    teacher_obs, teacher_y = _collect_prefix(env, actor, dparams[1], _node_record(target_node, harvested), 81_000_000, targets=residual, ticks=5)
    anchor_obs=[]; anchor_y=[]; anchor_counts={}
    for index, node_id in enumerate(sorted(baseline_ids)):
        obs, action = _collect_prefix(env, actor, dparams[1], _node_record(by_id[node_id], harvested), 81_100_000 + index, ticks=5)
        anchor_obs.extend(obs); anchor_y.extend(action); anchor_counts[node_id] = len(obs)
    legacy = pickle.loads((SOURCE / "descent_recovery_teacher_dataset_v1_balanced.pkl").read_bytes())
    anchor_obs.extend(np.asarray(row["observation"], np.float32) for row in legacy["anchors"])
    anchor_y.extend(np.asarray(row["target_action"], np.float32) for row in legacy["anchors"])
    teacher_obs=np.asarray(teacher_obs); teacher_y=np.asarray(teacher_y); anchor_obs=np.asarray(anchor_obs); anchor_y=np.asarray(anchor_y)
    save_json(root / "localized_dataset_manifest.json", {"teacher_node": TARGET_NODE, "teacher_samples": len(teacher_obs),
        "preservation_nodes": len(anchor_counts), "preservation_samples_by_node": anchor_counts,
        "legacy_P0_anchor_samples": len(legacy["anchors"]), "teacher_residual_knots": target_row["residual_knots"],
        "teacher_entry_tick": target_node["entry_tick"], "input_hashes": {"pi_D": EXPECTED["pi_D"], "pi_L": EXPECTED["pi_L"],
        "C_L": EXPECTED["C_L"], "xml": EXPECTED["xml"], "balanced": file_sha256(SOURCE / "balanced_p1_launch_subset_v1_frozen.json")}})
    selected={}; behavior={}
    for mode, protocol in BRANCHES.items():
        rows=[]
        for steps in protocol["steps"]:
            policy, history = train_supervised(base_policy=dparams[1], actor_action=actor,
                teacher_observation=teacher_obs, teacher_target=teacher_y,
                anchor_observation=anchor_obs, anchor_target=anchor_y,
                learning_rate=protocol["learning_rate"], steps=steps, mode=mode, teacher_weight=TEACHER_WEIGHT)
            anchor_delta=_action_delta(actor,dparams[1],policy,anchor_obs)
            target=np.asarray(actor(policy,jnp.asarray(teacher_obs))); imitation=float(np.sqrt(np.mean((target-teacher_y)**2)))
            rows.append({"steps":steps,"policy":policy,"history":history,"anchor":anchor_delta,
                "teacher_imitation_rms":imitation,"anchor_gate":anchor_delta["rms"] <= ANCHOR_RMS_LIMIT and anchor_delta["max"] <= ANCHOR_MAX_LIMIT})
        choice=select_behavior_candidate(rows); behavior[mode]={"candidates":[{k:v for k,v in row.items() if k != "policy"} for row in rows],
            "selected_steps": None if choice is None else choice["steps"]}
        if choice is not None: selected[mode]=choice["policy"]
    save_json(root / "behavior_selection.json", behavior)
    if not selected: raise SystemExit("no branch passed preregistered behavior anchor gate")
    loader=lambda node:_node_record(node,harvested); full_ids=[node["node_id"] for node in full]; balanced_ids=[node["node_id"] for node in balanced]
    baseline=certify_policy(env,dparams,lparams,full,82_000_000,record_loader=loader)
    baseline_summary={"full":_cert_summary(baseline,full_ids),"balanced":_cert_summary(baseline,balanced_ids)}
    branch_reports={}; accepted=[]
    for offset,mode in enumerate(("head","last_block")):
        if mode not in selected: continue
        params=(dparams[0],selected[mode],dparams[2]); cert=certify_policy(env,params,lparams,full,83_000_000+offset*1_000_000,record_loader=loader)
        summary={"full":_cert_summary(cert,full_ids),"balanced":_cert_summary(cert,balanced_ids)}
        checks=physical_acceptance(summary,len(balanced_ids),len(full_ids),baseline_ids)
        branch_reports[mode]={"summary":summary,"checks":checks,"accepted":all(checks.values()),"policy_hash":cert["policy_hash"]}
        save_json(root / f"{mode}_construction_certification.json", cert)
        save_bundle(root / f"checkpoint_{mode}",params=params,config=cfg,xml_path=cfg.xml_path,
            candidate_bank=SOURCE / "balanced_p1_launch_subset_v1_frozen.json",downstream_bank=C_L,
            policy_version=f"descent-localized-consolidation-v1-{mode}",extra={"artifact_role":"bounded_descent_policy_consolidation","PPO_steps":0})
        if all(checks.values()): accepted.append(mode)
    winner=next((mode for mode in ("head","last_block") if mode in accepted),None)
    report={"status":"PASS" if winner else "FAIL","head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "baseline":baseline_summary,"branches":branch_reports,"winner":winner,"PPO_steps":0,"PPO_authorization":False,
        "formal_tube_or_jel":False,"heldout_used":False,"frozen_assets_unchanged":verify_frozen_assets(SOURCE)[0],"runtime_fingerprint":fingerprint}
    save_json(root / "DESCENT_LOCALIZED_CONSOLIDATION_V1_REPORT.json",report)
    save_json(root / "completed.json",{"status":report["status"],"winner":winner,"next":"fresh_recertification" if winner else "bounded_local_repair_round_2"})
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
