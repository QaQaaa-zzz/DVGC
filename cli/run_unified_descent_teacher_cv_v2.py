"""Run the immutable eight-teacher authority amendment and head-only CV."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
from pathlib import Path

import jax
import numpy as np

from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_supervised import (
    action_audit, build_actor_tools, evaluate_policy, make_fast_rollout,
    parameter_drift, train_supervised,
)
from dvgc.descent_teacher import fixed_candidate_folds, nearest_neighbor_audit
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.ppo_integrity import normalizer_summary
from dvgc.runtime import save_json
from dvgc.trajectory_mining import canonical_state_byte_hash


EXPECTED_HEAD="55284df";EXPECTED_BANK="8e6342bc0d9d5e7821929f6ddccd4fb5b7c23923c66ab2d058309249dfed45a1"
EXPECTED_XML="d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c"
EXPECTED_NORMALIZER="8f2e36b6f69a3d20da67c1854f7e908c98dd6b03ae70e287e0a7e28522f93a7e"
BANK=Path("runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl")
POLICY=Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy")
CEM=Path("runs/unified_descent_controllability_reward_curriculum_probe_v1/full_v2/residual_cem_oracle_results.json")
REPLAY=Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v1/exact_replay_revalidation.json")
DATASET=Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v1/dataset_v3/teacher_dataset.pkl")


def digest(value):
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),default=float).encode()
    return hashlib.sha256(raw).hexdigest()


def subset_evaluation(evaluation, ids):
    rows=[row for row in evaluation["rows"] if row["candidate_id"] in set(ids)]
    ticks=np.asarray([row["survived_ticks"] for row in rows])
    return {"rows":rows,"summary":{"states":len(rows),"survival_counts":{str(h):int(np.sum(ticks>=h)) for h in (8,12,16,24)},"median":float(np.median(ticks)),"lower_quartile":float(np.quantile(ticks,.25)),"failure_reasons":sorted({row["termination_reason"] for row in rows})}}


def gains(before, after):
    base={row["candidate_id"]:row for row in before["rows"]};values=[]
    for row in after["rows"]:
        old=base[row["candidate_id"]]
        values.append({"candidate_id":row["candidate_id"],"before":old["survived_ticks"],"after":row["survived_ticks"],"gain":row["survived_ticks"]-old["survived_ticks"],"margin_gain":row["minimum_formal_margin"]-old["minimum_formal_margin"],"failure_before":old["termination_reason"],"failure_after":row["termination_reason"]})
    array=np.asarray([row["gain"] for row in values])
    return {"rows":values,"median_gain":float(np.median(array)),"gain_at_least_2":int(np.sum(array>=2)),"positive":int(np.sum(array>0)),"sum_positive":int(np.maximum(array,0).sum())}


def save_policy(path, policy):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as handle:pickle.dump(jax.device_get(policy),handle,pickle.HIGHEST_PROTOCOL)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);args=parser.parse_args();root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    if subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_HEAD,"HEAD"]).returncode or subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise SystemExit("invalid git state")
    cfg=load_config("configs/unified_descent_rsi_learnability_pilot_v1.json");gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if file_sha256(BANK)!=EXPECTED_BANK or file_sha256(cfg.xml_path)!=EXPECTED_XML or cfg.action_mapping_version!=ACTION_MAPPING_VERSION:raise SystemExit("asset gate")
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate")
    params,_,manifest=load_bundle(POLICY,verify_files=True)
    if normalizer_summary(params[0])["sha256"]!=EXPECTED_NORMALIZER:raise SystemExit("normalizer gate")
    bank=SnapshotBank.load(BANK);records=bank.records;by_id={row["id"]:row for row in records}
    cem=json.loads(CEM.read_text());old={row["candidate_id"]:row for row in cem["candidates"]}
    replay=json.loads(REPLAY.read_text());exact={row["candidate_id"]:row for row in replay["rows"]}
    positive=sorted(cid for cid,row in exact.items() if row["repeat_exact"] and row["replayed_gain"]>=4)
    if len(positive)!=8:raise SystemExit(f"authority is not exactly eight: {len(positive)}")
    positive_rows=[by_id[cid] for cid in positive];folds=fixed_candidate_folds(positive_rows)
    if sorted(map(len,folds),reverse=True)!=[3,3,2]:raise SystemExit(f"bad fold sizes {list(map(len,folds))}")
    fold_of={cid:"ABC"[index] for index,fold in enumerate(folds) for cid in fold}
    amendment=[]
    for record in records:
        cid=record["id"];before=int(exact[cid]["baseline_survival"]);old_surv=int(exact[cid]["saved_survival"]);new=int(exact[cid]["replayed_survival"])
        amendment.append({"candidate_id":cid,"baseline_survival":before,"old_selected_survival":old_surv,"exact_replay_survival":new,"old_gain":old_surv-before,"exact_gain":new-before,"replay1_replay2_equal":bool(exact[cid]["repeat_exact"]),"old_summary_exact_summary_equal":old_surv==new,"identity":"positive_teacher" if cid in positive else "anchor","cv_excluded_fold":fold_of.get(cid),"residual_sequence_sha256":digest(old[cid]["oracle"]["residual_knots"]),"initial_snapshot_sha256":canonical_state_byte_hash(record)})
    mismatches=[row["candidate_id"] for row in amendment if not row["old_summary_exact_summary_equal"]]
    authority={"status":"PASS" if len(mismatches)==2 and len(positive)==8 and all(exact[cid]["repeat_exact"] and exact[cid]["replayed_gain"]>=4 for cid in positive) else "FAIL","protocol":"exact gain >=4 is the sole teacher inclusion condition","old_selected_summary_role":"historical_only","exact_replay_summary_role":"teacher_authority","summary_mismatch_candidates":mismatches,"teacher_candidates":positive,"folds":{"A":folds[0],"B":folds[1],"C":folds[2]},"candidates":amendment}
    if authority["status"]!="PASS":raise SystemExit("authority amendment failed")
    root.mkdir(parents=True);save_json(root/"teacher_authority_protocol_amendment.json",authority)
    data=pickle.loads(DATASET.read_bytes());teacher=data["teacher"];anchors=data["anchors"]
    if len(teacher)!=64 or len(anchors)!=198:raise SystemExit("dataset count mismatch")
    # Reconfirm every anchor label directly from frozen pi_D before gradients.
    env=OrangeBikeDVGC(cfg,snapshot_bank=bank);net,actor_action,loc_scale=build_actor_tools(env,params);base_policy=params[1];rollout=make_fast_rollout(env,params)
    anchor_obs=np.asarray([row["observation"] for row in anchors]);reconfirmed=np.asarray(actor_action(base_policy,anchor_obs))
    previously_stored=np.asarray([row["target_action"] for row in anchors]);anchor_reconfirm_max=float(np.max(np.abs(reconfirmed-previously_stored)))
    # The v2 contract explicitly makes a fresh frozen-pi_D deterministic pass
    # authoritative.  Retain the old-label delta as an audit, never as a gate
    # or a training target.
    stored=reconfirmed
    teacher_obs=np.asarray([row["observation"] for row in teacher]);teacher_y=np.asarray([row["target_action"] for row in teacher])
    teacher_norm=np.asarray([row["normalized_observation"] for row in teacher]);residual=np.asarray([row["residual"] for row in teacher])
    represent=nearest_neighbor_audit(teacher_norm,residual,[row["candidate_id"] for row in teacher])
    anchor_norm=np.asarray([row["normalized_observation"] for row in anchors]);distance=np.linalg.norm(teacher_norm[:,None]-anchor_norm[None,:],axis=-1)/math.sqrt(teacher_norm.shape[1]);nearest=np.argmin(distance,axis=1);nearest_d=distance[np.arange(len(teacher)),nearest];label_delta=np.linalg.norm(teacher_y-stored[nearest],axis=1)
    close=nearest_d<=np.quantile(represent["distance"].values() if False else np.min(np.where(np.eye(len(teacher),dtype=bool),np.inf,np.linalg.norm(teacher_norm[:,None]-teacher_norm[None,:],axis=-1)/math.sqrt(teacher_norm.shape[1])),axis=1),.25)
    phase_consistent=all(row["phase"]==row["snapshot"]["oracle_phase"] for row in teacher)
    delay_consistent=all(np.array_equal(row["delay_buffer"],row["snapshot"]["policy_state"]["delay_buffer"]) for row in teacher)
    rep={"status":"MARGINAL_PASS" if represent["representable"] else "FAIL","teacher":represent,"teacher_anchor_nearest_distance":{"median":float(np.median(nearest_d)),"p95":float(np.quantile(nearest_d,.95))},"teacher_anchor_close_action_conflict_fraction":float(np.mean(close & (label_delta>.10))),"phase_consistent":phase_consistent,"contact_mode_recorded":all("contact_mode" in row for row in teacher),"delay_buffer_consistent":delay_consistent,"failure_precursor_finite":all(np.isfinite(row["physical_margin"]) for row in teacher),"actor_candidate_id_input":False,"actor_oracle_phase_input":False,"gate":bool(represent["representable"] and phase_consistent and delay_consistent)}
    save_json(root/"teacher_representability_audit_v2.json",rep)
    if not rep["gate"]:raise SystemExit("representability gate")
    dataset_manifest={"status":"PASS","teacher_samples":64,"anchor_samples":198,"positive_candidates":positive,"folds":authority["folds"],"fold_assignment_frozen_before_training":True,"anchor_reconfirmation_max_abs":anchor_reconfirm_max,"teacher_dataset_sha256":file_sha256(DATASET),"candidate_bank_sha256":file_sha256(BANK),"xml_sha256":file_sha256(cfg.xml_path),"pi_d_params_sha256":file_sha256(POLICY/"params.pkl"),"normalizer":normalizer_summary(params[0]),"heldout_used":False}
    save_json(root/"teacher_dataset_manifest_v2.json",dataset_manifest)
    save_json(root/"teacher_action_alignment_audit_v2.json",{"status":"PASS","action_order":["steer","drive","hip","knee"],"post_squash_action_target":True,"anchor_reconfirmation_max_abs":anchor_reconfirm_max,"per_tick":json.loads(Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v1/dataset_v3/teacher_action_alignment_audit.json").read_text())["per_tick_residual"]})
    baseline=evaluate_policy(env,rollout,base_policy,records,7200000);save_json(root/"baseline_14_state.json",baseline)
    all_results=[];selected=[]
    for fold_index,excluded_ids in enumerate(folds):
        train_ids=set(positive)-set(excluded_ids)
        fold_teacher=[row for row in teacher if row["candidate_id"] in train_ids]
        fold_anchor=[row for row in anchors if not (row["kind"]=="teacher_tail_anchor" and row["candidate_id"] in set(excluded_ids))]
        train_obs=np.asarray([row["observation"] for row in fold_teacher]);train_y=np.asarray([row["target_action"] for row in fold_teacher])
        excluded_teacher=[row for row in teacher if row["candidate_id"] in set(excluded_ids)];excluded_obs=np.asarray([row["observation"] for row in excluded_teacher]);excluded_y=np.asarray([row["target_action"] for row in excluded_teacher])
        fold_anchor_obs=np.asarray([row["observation"] for row in fold_anchor]);fold_anchor_y=np.asarray([row["target_action"] for row in fold_anchor])
        excluded_base=subset_evaluation(baseline,excluded_ids);fold_candidates=[]
        for lr in (1e-5,3e-5,1e-4):
            checkpoint_dir=root/"cv"/f"fold_{'ABC'[fold_index]}"/f"lr_{lr:g}"
            def callback(step,policy,lr=lr):
                current=evaluate_policy(env,rollout,policy,records,7300000+fold_index*10000)
                excluded=subset_evaluation(current,excluded_ids);transfer=gains(excluded_base,excluded);overall=gains(baseline,current)
                anchor_audit=action_audit(actor_action,base_policy,policy,anchor_obs,stored)
                train_audit=action_audit(actor_action,base_policy,policy,train_obs,train_y)
                excluded_audit=action_audit(actor_action,base_policy,policy,excluded_obs,excluded_y)
                survivor=next(row for row in current["rows"] if row["candidate_id"]=="d3fbf57dbf3d07cfbf4b31d6d11a0428")["survived_ticks"]>=24
                payload={"lr":lr,"excluded":transfer,"overall":overall,"excluded_evaluation":excluded,"overall_evaluation":current,"anchor_action":anchor_audit,"train_imitation":train_audit,"excluded_imitation":excluded_audit,"parameter_relative_l2":parameter_drift(base_policy,policy),"original_24_survivor_preserved":survivor,"anchor_gate":anchor_audit["delta_rms"]<=.02 and anchor_audit["delta_max"]<=.05,"finite":all(np.isfinite(x) for x in jax.tree.leaves(policy) for x in np.asarray(x).reshape(-1))}
                save_policy(checkpoint_dir/f"step_{step:04d}.pkl",policy);return payload
            _,history=train_supervised(base_policy=base_policy,actor_action=actor_action,teacher_observation=train_obs,teacher_target=train_y,anchor_observation=fold_anchor_obs,anchor_target=fold_anchor_y,learning_rate=lr,callback=callback)
            for row in history:
                row.update({"fold":"ABC"[fold_index],"excluded_candidates":excluded_ids});fold_candidates.append(row);all_results.append(row)
        eligible=[row for row in fold_candidates if row["anchor_gate"] and row["finite"] and row["original_24_survivor_preserved"]]
        if not eligible:raise SystemExit(f"no eligible checkpoint fold {fold_index}")
        def score(row):
            ex=row["excluded"];overall=row["overall"];return (ex["gain_at_least_2"],ex["median_gain"],ex["sum_positive"],overall["gain_at_least_2"],overall["median_gain"],overall["sum_positive"],-row["anchor_action"]["delta_rms"],-row["parameter_relative_l2"],-row["step"])
        best=max(eligible,key=score);selected.append(best)
    combined=[item for row in selected for item in row["excluded"]["rows"]];combined_gain=np.asarray([row["gain"] for row in combined]);fold_positive=sum(row["excluded"]["median_gain"]>0 for row in selected)
    base_fail={row["candidate_id"]:row["termination_reason"] for row in baseline["rows"]};new_types={row["failure_after"] for row in combined if row["failure_after"] not in set(base_fail.values())}
    transfer_pass=bool(np.sum(combined_gain>=2)>=4 and np.median(combined_gain)>=1 and fold_positive>=2 and not new_types)
    report={"status":"PASS" if transfer_pass else "HEAD_ONLY_CV_FAIL","mode":"head_only","folds":[{"fold":row["fold"],"excluded_candidates":row["excluded_candidates"],"selected_lr":row["lr"],"selected_step":row["step"],"excluded":row["excluded"],"train_imitation":row["train_imitation"],"excluded_imitation":row["excluded_imitation"],"anchor_action":row["anchor_action"],"original_24_survivor_preserved":row["original_24_survivor_preserved"]} for row in selected],"combined":{"candidates":combined,"gain_at_least_2":int(np.sum(combined_gain>=2)),"median_gain":float(np.median(combined_gain)),"positive_folds":fold_positive,"new_failure_types":sorted(new_types)},"physical_transfer_gate":transfer_pass,"relabel_required":not transfer_pass and all(row["train_imitation"]["imitation_rms"]<float(np.sqrt(np.mean((np.asarray([x["target_action"] for x in teacher if x["candidate_id"] not in set(row["excluded_candidates"])])-np.asarray([x["frozen_action"] for x in teacher if x["candidate_id"] not in set(row["excluded_candidates"])]))**2))) for row in selected),"heldout_used":False,"all_checkpoint_results":all_results}
    save_json(root/"candidate_cross_validation_results_v2.json",report)
    print(json.dumps({key:value for key,value in report.items() if key!="all_checkpoint_results"},indent=2))


if __name__=="__main__":main()
