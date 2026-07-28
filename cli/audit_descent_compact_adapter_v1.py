"""Fresh 32-branch independent audit for the frozen compact Descent expert."""
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

from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L, _restore
from cli.run_descent_localized_consolidation_v1 import SOURCE, _node_record, verified_assets_allowing_runtime_gate_refresh
from cli.runtime_gate import source_fingerprint
from dvgc.audit import build_audit_report
from dvgc.backward_search import compact_observation_command_adapter, make_descent_landing_rollout
from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import DYNAMICS_VARIANTS, branch_seed, detailed_terminal_summary
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


DEFAULT_EXPERT=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
SEED_BASE=3_100_000_000
BRANCHES=32
SEED_NAMESPACE="descent-compact-expert-independent-audit-v1"


def audit_gate(report, minimum_precision=.95):
    return {"precision":float(report["tube_precision"])>=minimum_precision,
            "no_timeout":int(report["terminal_summary"]["timeouts"])==0,
            "no_horizon":int(report["terminal_summary"]["horizon_exhaustions"])==0,
            "no_nonfinite":int(report["terminal_summary"]["physical_end_reasons"]["nonfinite"])==0}


def _perturb_batch(env,record,seeds,deltas):
    """Vectorized reset preserves Warp contact-dimension metadata."""
    base=_restore(env,record,jax.random.PRNGKey(int(seeds[0])));info=base.info
    def reset(delta,key):
        qvel=base.data.qvel.at[env._qvel0].add(delta[0]).at[env._qvel0+2].add(delta[1])
        return env.reset_from_snapshot(base.data.qpos,qvel,base.data.ctrl,key,info["phase"],
            info["had_airborne"],info["had_valid_landing"],info["contact_age"],info["last_action"],
            estimated_phase=info["estimated_phase"],phase_probs=info["phase_probs"],airborne_count=info["airborne_count"],
            prelaunch_airborne_count=info["prelaunch_airborne_count"],landing_bounce_count=info["landing_bounce_count"],
            invalid_wheel_count=info["invalid_wheel_count"],recovery_count=info["recovery_count"],prev_acc_z=info["prev_acc_z"],
            prev_vz=info["prev_vz"],obs_history=info["actor_obs_history_pre"],obs_history_valid=jnp.asarray(True),
            stage_entry_ever=info["stage_entry_ever"],apex_seen=info["apex_seen"],jump_signal_latched=info["jump_signal_latched"],
            jump_window_start_x=info["jump_window_start_x"],jump_window_end_x=info["jump_window_end_x"])
    keys=jnp.stack([jax.random.PRNGKey(int(seed)) for seed in seeds])
    return jax.jit(jax.vmap(reset))(jnp.asarray(deltas),keys)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--expert-run",default=str(DEFAULT_EXPERT));args=parser.parse_args()
    expert=Path(args.expert_run);audit_root=expert/"independent_audit_v1";final_path=audit_root/"report.json"
    if final_path.exists():raise SystemExit(f"refusing overwrite {final_path}")
    source_report=json.loads((expert/"DESCENT_COMPACT_ADAPTER_V1_REPORT.json").read_text())
    if source_report["status"]!="PASS":raise SystemExit("construction gate did not pass")
    valid,failed,raw_failed=verified_assets_allowing_runtime_gate_refresh()
    if not valid:raise SystemExit(f"frozen scientific asset mismatch: {failed}; raw={raw_failed}")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text());fingerprint=source_fingerprint(Path.cwd())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=fingerprint:raise SystemExit("runtime gate stale")
    artifact=pickle.loads((expert/"adapter.pkl").read_bytes())
    if file_sha256(expert/"adapter.pkl")!=source_report["adapter_sha256"] or artifact["policy_identity_hash"]!=source_report["policy_identity_hash"]:
        raise SystemExit("adapter identity mismatch")
    dparams,_,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    if artifact["base_policy_sha256"]!=EXPECTED["pi_D"]:raise SystemExit("base policy mismatch")
    full=json.loads((SOURCE/"full_p1_bank_v1.json").read_text())["nodes"]
    harvested=pickle.loads((SOURCE/"trajectory_harvested_snapshots.pkl").read_bytes())
    audit_root.mkdir(parents=True,exist_ok=True);partial=audit_root/"partial.json"
    input_identity={"policy_identity_hash":artifact["policy_identity_hash"],"adapter_sha256":file_sha256(expert/"adapter.pkl"),
        "candidate_sha256":file_sha256(SOURCE/"full_p1_bank_v1.json"),"C_L":EXPECTED["C_L"],"pi_L":EXPECTED["pi_L"],
        "xml":EXPECTED["xml"],"seed_base":SEED_BASE,"branches":BRANCHES,"seed_namespace":SEED_NAMESPACE}
    save_json(audit_root/"cost_estimate.json",{"estimated_seconds":1800,"states":len(full),"branches_per_state":BRANCHES,
        "rollouts":len(full)*BRANCHES,"PPO_steps":0,"labels_reused_for_training":False})
    save_json(audit_root/"manifest.json",{"status":"FROZEN_BEFORE_AUDIT","inputs":input_identity,
        "dynamics_variants":DYNAMICS_VARIANTS,"perturbation":"independent uniform vx/vz in [-0.02,0.02]","safe_precision_gate":.95})
    variants=[]
    base=load_config("configs/backward_descent_rsi_pilot_v1.json",{"use_bank_resets":False,"expert_chain_termination":False,
        "domain_randomization":False,"obs_noise_enable":False})
    if file_sha256(base.xml_path)!=EXPECTED["xml"] or base.action_mapping_version!=ACTION_MAPPING_VERSION:raise SystemExit("runtime model mismatch")
    for spec in DYNAMICS_VARIANTS:
        cfg=load_config("configs/backward_descent_rsi_pilot_v1.json",{"use_bank_resets":False,"expert_chain_termination":False,
            "domain_randomization":False,"obs_noise_enable":False,**{k:v for k,v in spec.items() if k!="id"}})
        env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L))
        adapter=compact_observation_command_adapter(jnp.asarray(artifact["prototypes"]),jnp.asarray(artifact["targets"]),
            jnp.asarray(artifact["normalizer_mean"]),jnp.asarray(artifact["normalizer_std"]),float(artifact["radius"]),float(artifact["core_radius"]))
        variants.append((spec["id"],env,make_descent_landing_rollout(env,dparams,lparams,horizon=200,residual_ticks=8,
            descent_action_adapter=adapter)))
    rows=[]
    if partial.exists():
        saved=json.loads(partial.read_text())
        if saved.get("inputs")!=input_identity:raise SystemExit("partial audit input mismatch")
        rows=saved["rows"]
    for state_index in range(len(rows),len(full)):
        node=full[state_index];record=_node_record(node,harvested);branches=[]
        for variant_index,(variant_id,env,rollout) in enumerate(variants):
            branch_indices=list(range(variant_index,BRANCHES,len(variants)));seeds=[];deltas=[]
            for branch_index in branch_indices:
                seed=branch_seed(SEED_BASE,state_index,branch_index);rng=np.random.default_rng(seed)
                seeds.append(seed);deltas.append(rng.uniform(-.02,.02,size=2).astype(np.float32))
            batch=_perturb_batch(env,record,seeds,deltas);zero=jnp.zeros((len(seeds),2,4),jnp.float32)
            raw=jax.device_get(rollout(batch,zero,jax.random.PRNGKey(branch_seed(SEED_BASE,state_index,variant_index))))
            for local,branch_index in enumerate(branch_indices):
                seed=branch_seed(SEED_BASE,state_index,branch_index);code=int(np.asarray(raw["end_code"])[local])
                final=bool(np.asarray(raw["final_recovery"])[local]);chain=bool(np.asarray(raw["downstream_entry"])[local])
                if final:cause="final_recovery"
                elif code==8:cause="timeout"
                elif code==0:cause="horizon_exhausted"
                else:cause="physical_failure"
                branches.append({"branch_index":branch_index,"branch_seed":seed,"seed_namespace":SEED_NAMESPACE,
                    "dynamics_variant":variant_id,"chain_success":chain,"final_recovery":final,"terminal_cause":cause,
                    "end_code":code,"end_reason":END_REASON.get(code,"unknown"),"steps":int(np.asarray(raw["termination_tick"])[local])})
        branches.sort(key=lambda value:value["branch_index"]);finals=sum(row["final_recovery"] for row in branches)
        posterior=beta_posterior(finals,BRANCHES-finals);label=posterior_label(posterior,BRANCHES,
            min_branches=int(base.min_branches),safe_threshold=float(base.safe_threshold),dead_threshold=float(base.dead_threshold),
            boundary_max_width=float(base.boundary_max_width))
        rows.append({"state_index":state_index,"id":node["node_id"],"candidate_id":node["candidate_id"],
            "candidate_kind":node["region"],"layer":node["layer"],"region":node["region"],"predicted_label":"safe",
            "predicted_mean":1.0,"audit_final":finals/BRANCHES,"audit_label":label,"branches":branches})
        save_json(partial,{"inputs":input_identity,"completed":len(rows),"rows":rows})
        print(f"[descent-audit] {len(rows)}/{len(full)} final={finals}/{BRANCHES} label={label}",flush=True)
    report=build_audit_report(rows,policy_version=artifact["policy_identity_hash"],phase="descent",seed_namespace=SEED_NAMESPACE,
        branches_per_state=BRANCHES,safe_threshold=float(base.safe_threshold),dynamics_variants=DYNAMICS_VARIANTS)
    report["terminal_summary"]=detailed_terminal_summary([branch for row in rows for branch in row["branches"]])
    checks=audit_gate(report);passed=all(checks.values())
    report.update({"status":"PASS" if passed else "FAIL","checks":checks,"inputs":input_identity,
        "artifact_role":"certified_tube" if passed else "expert_conditioned_provisional_envelope_audit_failed",
        "formal_jel_eligible":False,"audit_labels_reused_for_training":False,"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()})
    if passed:
        records=[]
        for node,row in zip(full,rows,strict=True):
            record=copy.deepcopy(_node_record(node,harvested));record.update({"id":node["node_id"],"origin_phase":"descent",
                "artifact_role":"certified_tube","tube_metrics_eligible":True,"policy_identity_hash":artifact["policy_identity_hash"],
                "independent_audit":{"label":row["audit_label"],"final_rate":row["audit_final"],"branches":BRANCHES}});records.append(record)
        tube=SnapshotBank(records,{"artifact_role":"certified_tube","phase":"descent","policy_identity_hash":artifact["policy_identity_hash"],
            "adapter_sha256":file_sha256(expert/"adapter.pkl"),"audit_seed_namespace":SEED_NAMESPACE,"independent_audit":True,
            "formal_jel_eligible":False,"expert_conditioned":True})
        tube.save(audit_root/"descent_tube.pkl");report["tube_path"]=str(audit_root/"descent_tube.pkl");report["tube_sha256"]=file_sha256(audit_root/"descent_tube.pkl")
    save_json(final_path,report);save_json(audit_root/"completed.json",{"status":report["status"],"next":"tube_rsi_refinement" if passed else "audit_failure_analysis"})
    print(json.dumps({k:report[k] for k in ("status","tube_precision","recoverable_recall","candidate_mass_coverage","terminal_summary","checks")},indent=2))


if __name__=="__main__":main()
