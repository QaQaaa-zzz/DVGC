"""One-shot strictly gated student-state relabel audit for Descent v2."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_probe import formal_dynamic_margin
from dvgc.descent_supervised import build_actor_tools
from dvgc.descent_teacher import normalized_observation, relabel_support_gate
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


BANK=Path("runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl")
POLICY=Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy")
DATASET=Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v1/dataset_v3/teacher_dataset.pkl")


def precursor(snapshot, cfg):
    feature=np.asarray(snapshot["physical_feature"])
    roll=np.deg2rad(float(cfg.max_roll_deg))-abs(feature[3]);pitch=np.deg2rad(float(cfg.max_pitch_deg))-abs(feature[4])
    return "roll" if roll<=pitch else "pitch", bool(min(roll,pitch)>=0)


def four_tick(env, step, inference, initial, residuals, seed):
    state=initial;survival=0;minimum=float("inf")
    for offset in range(4):
        action,_=inference(state.obs,jax.random.PRNGKey(seed+offset));action=np.asarray(action)
        if residuals is not None and offset<len(residuals):action=np.clip(action+residuals[offset],-1,1)
        state=step(state,action);margin=float(formal_dynamic_margin(np.asarray(env._physical_feature(state.data)),env._config));minimum=min(minimum,margin)
        if float(state.done):break
        survival+=1
    return survival,minimum


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--cv-run",required=True);parser.add_argument("--student",required=True);parser.add_argument("--output",required=True);args=parser.parse_args();output=Path(args.output)
    if output.exists():raise SystemExit(f"refusing overwrite {output}")
    cv=Path(args.cv_run);authority=json.loads((cv/"teacher_authority_protocol_amendment.json").read_text());excluded=set(authority["folds"]["A"]);training=set(authority["teacher_candidates"])-excluded
    data=pickle.loads(DATASET.read_bytes());teacher=data["teacher"];by={(row["candidate_id"],row["tick"]):row for row in teacher}
    support=json.loads((cv/"teacher_representability_audit_v2.json").read_text())["teacher"]["candidate_support_p95"]
    bank=SnapshotBank.load(BANK);records={row["id"]:row for row in bank.records};cfg=load_config("configs/unified_descent_rsi_learnability_pilot_v1.json");env=OrangeBikeDVGC(cfg,snapshot_bank=bank);step=jax.jit(env.step)
    frozen,_,_=load_bundle(POLICY,verify_files=True);_,frozen_action,_=build_actor_tools(env,frozen)
    with Path(args.student).open("rb") as handle:student_policy=pickle.load(handle)
    student_params=(frozen[0],student_policy,frozen[2]);_,student_action,_=build_actor_tools(env,student_params)
    mean=np.asarray(frozen[0].mean["state"]);std=np.asarray(frozen[0].std["state"]);accepted=[];audits=[]
    for candidate_index,cid in enumerate(sorted(training)):
        state=restore_snapshot(env,records[cid],jax.random.PRNGKey(8100000+candidate_index))
        trajectory=[by[(cid,t)] for t in range(8)]
        for tick in range(8):
            teacher_row=trajectory[tick];snapshot=env.snapshot_record(state,"flight");obs=np.asarray(state.obs["state"]);norm=normalized_observation(obs,mean,std);distance=float(np.linalg.norm(norm-teacher_row["normalized_observation"])/np.sqrt(len(norm)))
            phase_equal=int(snapshot["oracle_phase"])==int(teacher_row["phase"]);contact=int(np.argmax(snapshot["policy_state"]["contact_probs"]));contact_equal=contact==int(teacher_row["contact_mode"])
            delay_equal=bool(np.allclose(snapshot["policy_state"]["delay_buffer"],teacher_row["delay_buffer"],atol=1e-6,rtol=0))
            precursor_equal=precursor(snapshot,cfg)==precursor(teacher_row["snapshot"],cfg)
            base_surv,base_margin=four_tick(env,step,lambda obs,key:(frozen_action(frozen[1],obs["state"]),None),state,None,8200000+candidate_index*100+tick*10)
            residuals=[trajectory[j]["residual"] for j in range(tick,min(8,tick+4))]
            cf_surv,cf_margin=four_tick(env,step,lambda obs,key:(frozen_action(frozen[1],obs["state"]),None),state,residuals,8300000+candidate_index*100+tick*10)
            duplicate=distance<=1e-8
            allowed,reasons=relabel_support_gate(normalized_distance=distance,support_p95=float(support[cid]),phase_equal=phase_equal,contact_mode_equal=contact_equal,delay_buffer_equal=delay_equal,precursor_equal=precursor_equal,counterfactual_survival_gain=cf_surv-base_surv,counterfactual_margin_gain=cf_margin-base_margin,excluded_or_heldout=False)
            if duplicate:allowed=False;reasons.append("duplicate_teacher_state")
            item={"candidate_id":cid,"tick":tick,"normalized_distance":distance,"support_p95":support[cid],"phase_equal":phase_equal,"contact_mode_equal":contact_equal,"delay_buffer_equal":delay_equal,"precursor_equal":precursor_equal,"baseline_4tick":{"survival":base_surv,"minimum_margin":base_margin},"counterfactual_4tick":{"survival":cf_surv,"minimum_margin":cf_margin},"accepted":allowed,"reasons":reasons}
            audits.append(item)
            if allowed:
                base=np.asarray(frozen_action(frozen[1],state.obs["state"]));residual=np.asarray(teacher_row["residual"]);accepted.append({"kind":"student_relabel","candidate_id":cid,"tick":tick,"snapshot":snapshot,"observation":obs,"normalized_observation":norm,"frozen_action":base,"residual":residual,"target_action":np.clip(base+residual,-1,1),"support_audit":item})
            action=np.asarray(student_action(student_policy,state.obs["state"]));state=step(state,action)
            if float(state.done):break
    output.mkdir(parents=True);report={"status":"PASS" if accepted else "COVARIATE_SHIFT_OR_TEACHER_SUPPORT_GAP","student":str(Path(args.student).resolve()),"excluded_fold":sorted(excluded),"training_candidates":sorted(training),"audited":len(audits),"accepted":len(accepted),"rejected":len(audits)-len(accepted),"heldout_used":False,"excluded_used":False,"audits":audits}
    save_json(output/"student_relabel_support_audit.json",report)
    with (output/"accepted_student_relabels.pkl").open("wb") as handle:pickle.dump(accepted,handle,pickle.HIGHEST_PROTOCOL)
    save_json(output/"accepted_student_relabels_manifest.json",{"status":report["status"],"records":len(accepted),"artifact_sha256":file_sha256(output/"accepted_student_relabels.pkl"),"one_shot_relabel_round":True})
    print(json.dumps({key:value for key,value in report.items() if key!="audits"},indent=2))


if __name__=="__main__":main()
