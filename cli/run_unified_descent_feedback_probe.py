"""Run the fixed-snapshot local feedback-teacher support probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_feedback import distinct_top_sequences, local_support_gate, select_three_ticks, tick_region
from dvgc.descent_probe import batched_base_state, lexicographic_order, make_residual_rollout
from dvgc.descent_supervised import build_actor_tools
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.ppo_integrity import normalizer_summary
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json
from dvgc.trajectory_mining import canonical_state_byte_hash


EXPECTED_HEAD="439aa8f"
EXPECTED_BANK="8e6342bc0d9d5e7821929f6ddccd4fb5b7c23923c66ab2d058309249dfed45a1"
EXPECTED_XML="d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c"
EXPECTED_NORMALIZER="8f2e36b6f69a3d20da67c1854f7e908c98dd6b03ae70e287e0a7e28522f93a7e"
BANK=Path("runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl")
POLICY=Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy")
DATASET=Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v1/dataset_v3/teacher_dataset.pkl")
AUTHORITY=Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v2_8teacher/cv_relabel_v1/teacher_authority_protocol_amendment.json")
OLD_RELABEL=Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v2_8teacher/relabel_round_v1/student_relabel_support_audit.json")
STUDENT=Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v2_8teacher/cv_head_v2/cv/fold_A/lr_1e-05/step_0025.pkl")
BOUNDS=(.05,.10,.20);GENERATIONS=5;POPULATION=256;ELITE=32;TOP_PER_GENERATION=16


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=float).encode()).hexdigest()


def _assets():
    if subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_HEAD,"HEAD"]).returncode:
        raise SystemExit("unexpected history")
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise SystemExit("worktree must be clean")
    cfg=load_config("configs/unified_descent_rsi_learnability_pilot_v1.json")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if file_sha256(BANK)!=EXPECTED_BANK or file_sha256(cfg.xml_path)!=EXPECTED_XML or cfg.action_mapping_version!=ACTION_MAPPING_VERSION:
        raise SystemExit("asset gate")
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate")
    params,_,_=load_bundle(POLICY,verify_files=True)
    if normalizer_summary(params[0])["sha256"]!=EXPECTED_NORMALIZER:raise SystemExit("normalizer gate")
    bank=SnapshotBank.load(BANK);env=OrangeBikeDVGC(cfg,snapshot_bank=bank)
    return cfg,bank,env,params


def _snapshot_manifest(root,bank,env,params):
    authority=json.loads(AUTHORITY.read_text());positive=authority["teacher_candidates"]
    old=json.loads(OLD_RELABEL.read_text());old_by=defaultdict(list)
    for row in old["audits"]:old_by[row["candidate_id"]].append(row)
    data=pickle.loads(DATASET.read_bytes());teacher={(row["candidate_id"],int(row["tick"])):row for row in data["teacher"]}
    records={row["id"]:row for row in bank.records};training=sorted(old["training_candidates"])
    with STUDENT.open("rb") as handle:student=pickle.load(handle)
    _,student_action,_=build_actor_tools(env,(params[0],student,params[2]));_,frozen_action,_=build_actor_tools(env,params)
    mean=np.asarray(params[0].mean["state"]);std=np.asarray(params[0].std["state"])
    snapshots=[];manifest=[]
    for candidate_index,cid in enumerate(sorted(positive)):
        seed=8100000+training.index(cid) if cid in training else 8110000+candidate_index
        state=restore_snapshot(env,records[cid],jax.random.PRNGKey(seed));ticks=select_three_ticks(old_by[cid])
        for tick in range(8):
            if tick in ticks:
                snapshot=env.snapshot_record(state,"flight");prior=next((row for row in old_by[cid] if int(row["tick"])==tick),None)
                trow=teacher[(cid,tick)];obs=np.asarray(state.obs["state"]);frozen=np.asarray(frozen_action(params[1],state.obs["state"]))
                item={"candidate_id":cid,"tick":tick,"tick_region":tick_region(tick),"generation_seed":seed,
                      "snapshot_hash":canonical_state_byte_hash(snapshot),"snapshot":snapshot,"normalized_actor_observation":((obs-mean)/std).tolist(),
                      "frozen_pi_d_action":frozen.tolist(),"original_cem_residual":np.asarray(trow["residual"]).tolist(),
                      "old_relabel_audited":prior is not None,"old_relabel_accepted":bool(prior and prior["accepted"]),
                      "old_relabel_rejection_reasons":[] if prior is None else prior["reasons"]}
                snapshots.append(item);manifest.append({k:v for k,v in item.items() if k!="snapshot"})
            action=np.asarray(student_action(student,state.obs["state"]));state=jax.jit(env.step)(state,action)
            if float(state.done) and tick<max(ticks):raise SystemExit(f"student terminated before selected tick: {cid} {tick}")
    if len(snapshots)!=24 or Counter(row["candidate_id"] for row in snapshots)!={cid:3 for cid in positive}:
        raise SystemExit("snapshot quota gate")
    accepted=sum(row["old_relabel_accepted"] for row in manifest)
    payload={"status":"PASS","selection_frozen_before_new_search":True,"selection_rule":"per candidate: lexicographically earliest three old accepted ticks, then fixed region representatives [1,4,7], then remaining ticks","protocol_constraint_note":"The historical 11 accepts include four ticks for 173ee307, so the exact 3-per-candidate quota can retain at most 10/11; deterministic quota selection retains the maximum 10 and excludes tick 4 for that candidate.","snapshot_count":24,"candidate_count":8,"per_candidate":dict(Counter(row["candidate_id"] for row in manifest)),"region_counts":dict(Counter(row["tick_region"] for row in manifest)),"old_audit_accepts_retained":accepted,"old_audit_accepts_available":11,"old_audit_rejects_retained":sum(row["old_relabel_audited"] and not row["old_relabel_accepted"] for row in manifest),"generated_for_previously_excluded_candidates":sum(not row["old_relabel_audited"] for row in manifest),"heldout_used":False,"student_policy_sha256":file_sha256(STUDENT),"bank_sha256":file_sha256(BANK),"pi_d_params_sha256":file_sha256(POLICY/"params.pkl"),"selection_protocol_sha256":_digest([(row["candidate_id"],row["tick"],row["snapshot_hash"]) for row in manifest]),"records":manifest}
    root.mkdir(parents=True);save_json(root/"feedback_probe_state_manifest.json",payload)
    with (root/"feedback_probe_snapshots.pkl").open("wb") as handle:pickle.dump(snapshots,handle,pickle.HIGHEST_PROTOCOL)
    save_json(root/"feedback_probe_snapshots_manifest.json",{"records":24,"sha256":file_sha256(root/"feedback_probe_snapshots.pkl"),"heldout_used":False})
    return snapshots


def _payload(result,index,knots,generation,sample,bound):
    return {"generation":generation,"sample":sample,"bound":bound,"residual_knots":knots.tolist(),
            "survival":int(result["survival"][index]),"minimum_margin":float(result["minimum_margin"][index]),
            "terminal_margin":float(result["terminal_margin"][index]),"residual_rms":float(result["residual_rms"][index]),
            "end_code":int(result["end_code"][index]),"actions":np.asarray(result["actions"][:,index]).tolist(),
            "features":np.asarray(result["features"][:,index]).tolist()}


def _exact(rollout,state_factory,row,seed):
    knots=jnp.asarray(np.asarray(row["residual_knots"],np.float32)[None]);state1=state_factory(1);state2=state_factory(1)
    one=jax.device_get(rollout(state1,knots,jax.random.PRNGKey(seed)));two=jax.device_get(rollout(state2,knots,jax.random.PRNGKey(seed)))
    keys=("survival","minimum_margin","terminal_margin","end_code","actions","features")
    repeated=all(np.array_equal(np.asarray(one[key]),np.asarray(two[key])) for key in keys)
    summary=all(np.array_equal(np.asarray(one[key])[0],np.asarray(row[key])) for key in ("survival","minimum_margin","terminal_margin","end_code"))
    summary &= np.array_equal(np.asarray(one["actions"])[:,0],np.asarray(row["actions"]))
    replay={"survival":int(one["survival"][0]),"minimum_margin":float(one["minimum_margin"][0]),
            "terminal_margin":float(one["terminal_margin"][0]),"end_code":int(one["end_code"][0]),
            "actions":np.asarray(one["actions"])[:,0].tolist(),"features":np.asarray(one["features"])[:,0].tolist()}
    return {"repeat1_repeat2_exact":bool(repeated),"search_batch_summary_equal":bool(summary),"replay":replay}


def _search(root,snapshots,env,params):
    rollout=make_residual_rollout(env,params,horizon=24,ticks_per_knot=4,residual_ticks=8);rows=[]
    for snapshot_index,item in enumerate(snapshots):
        state_factory=lambda count,item=item,snapshot_index=snapshot_index:batched_base_state(env,item["snapshot"],9100000+snapshot_index,count)
        zero=np.zeros((1,2,4),np.float32);baseline_raw=jax.device_get(rollout(state_factory(1),jnp.asarray(zero),jax.random.PRNGKey(9200000+snapshot_index)))
        baseline={"survival":int(baseline_raw["survival"][0]),"minimum_margin":float(baseline_raw["minimum_margin"][0]),"terminal_margin":float(baseline_raw["terminal_margin"][0]),"end_code":int(baseline_raw["end_code"][0]),"failure":END_REASON.get(int(baseline_raw["end_code"][0]),"horizon")}
        candidates=[]
        for level,bound in enumerate(BOUNDS):
            seed=9300000+snapshot_index*100+level;rng=np.random.default_rng(seed);mean=np.zeros((2,4),np.float32);std=np.full((2,4),bound*.5,np.float32)
            for generation in range(GENERATIONS):
                knots=np.clip(rng.normal(mean,std,size=(POPULATION,2,4)),-bound,bound).astype(np.float32)
                result=jax.device_get(rollout(state_factory(POPULATION),jnp.asarray(knots),jax.random.PRNGKey(seed+generation)))
                order=lexicographic_order(result);elite=knots[order[:ELITE]];mean=elite.mean(0);std=np.maximum(elite.std(0),bound*.02)
                for rank in order[:TOP_PER_GENERATION]:candidates.append(_payload(result,int(rank),knots[int(rank)],generation,int(rank),bound))
        top=distinct_top_sequences(candidates,5)
        for rank,row in enumerate(top):
            row["rank"]=rank+1;exact=_exact(rollout,state_factory,row,9400000+snapshot_index*10+rank);row["exact_replay"]=exact
            replay=exact["replay"];row["gain"]=replay["survival"]-baseline["survival"];row["failure"]=END_REASON.get(replay["end_code"],"horizon")
            row["no_new_failure_type"]=row["failure"] in {baseline["failure"],"horizon"}
            row["action_delay_command_alignment"]="same OrangeBikeDVGC step lineage; residual applied at command ticks 0..7 before the unchanged delay model"
        authoritative=next((row for row in top if row["exact_replay"]["repeat1_repeat2_exact"] and row["gain"]>=2 and row["no_new_failure_type"]),None)
        rows.append({"snapshot_index":snapshot_index,"candidate_id":item["candidate_id"],"tick":item["tick"],"tick_region":item["tick_region"],"snapshot_hash":item["snapshot_hash"],"old_relabel_accepted":item["old_relabel_accepted"],"old_relabel_rejection_reasons":item["old_relabel_rejection_reasons"],"baseline":baseline,"top5":top,"authoritative_correction":authoritative is not None,"authoritative_rank":None if authoritative is None else authoritative["rank"]})
        save_json(root/"partial_local_cem_authority_results.json",{"completed":len(rows),"total":24,"rows":rows})
    gate=local_support_gate(rows);report={"status":"PASS" if gate["gate"] else "BRITTLE_OPEN_LOOP_TEACHER","protocol":{"search_horizon_ticks":8,"continuation_evaluation_horizon_ticks":24,"knots":2,"ticks_per_knot":4,"bounds":list(BOUNDS),"generations_per_bound":GENERATIONS,"population_per_generation":POPULATION,"elite":ELITE,"same_budget_every_snapshot":True,"candidate_specific_tuning":False,"restart_count":0,"objective":"lexicographic survival, minimum formal margin, terminal formal margin, residual effort"},"gate":gate,"rows":rows,"heldout_used":False,"ppo_authorization":False}
    save_json(root/"local_cem_authority_results.json",report)
    print(json.dumps({"status":report["status"],"gate":gate},indent=2));return report


def _reanalyze(root,source,env,params):
    source_report=json.loads((source/"local_cem_authority_results.json").read_text())
    with (source/"feedback_probe_snapshots.pkl").open("rb") as handle:snapshots=pickle.load(handle)
    rollout=make_residual_rollout(env,params,horizon=24,ticks_per_knot=4,residual_ticks=8);rows=[]
    for snapshot_index,(item,old) in enumerate(zip(snapshots,source_report["rows"],strict=True)):
        state_factory=lambda count,item=item,snapshot_index=snapshot_index:batched_base_state(env,item["snapshot"],9100000+snapshot_index,count)
        top=[]
        for rank,old_top in enumerate(old["top5"]):
            row={key:value for key,value in old_top.items() if key not in {"exact_replay_twice","gain","failure","no_new_failure_type"}}
            exact=_exact(rollout,state_factory,row,9400000+snapshot_index*10+rank);row["exact_replay"]=exact
            replay=exact["replay"];row["gain"]=replay["survival"]-old["baseline"]["survival"]
            row["failure"]=END_REASON.get(replay["end_code"],"horizon");row["no_new_failure_type"]=row["failure"] in {old["baseline"]["failure"],"horizon"};top.append(row)
        authoritative=next((row for row in top if row["exact_replay"]["repeat1_repeat2_exact"] and row["gain"]>=2 and row["no_new_failure_type"]),None)
        rows.append({**{key:value for key,value in old.items() if key not in {"top5","authoritative_correction","authoritative_rank"}},"top5":top,"authoritative_correction":authoritative is not None,"authoritative_rank":None if authoritative is None else authoritative["rank"]})
    gate=local_support_gate(rows);root.mkdir(parents=True)
    for name in ("feedback_probe_state_manifest.json","feedback_probe_snapshots_manifest.json"):
        save_json(root/name,json.loads((source/name).read_text()))
    report={"status":"PASS" if gate["gate"] else "BRITTLE_OPEN_LOOP_TEACHER","protocol":source_report["protocol"],"authority_correction":"CEM discovery used batch 256; authority uses two bit-exact batch-1 replays from the same frozen snapshot. Search-batch summary equality is audited but is not the authority label.","supersedes_for_authority":str(source/"local_cem_authority_results.json"),"gate":gate,"rows":rows,"heldout_used":False,"ppo_authorization":False}
    save_json(root/"local_cem_authority_results.json",report);print(json.dumps({"status":report["status"],"gate":gate},indent=2));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);parser.add_argument("--reanalyze-source");args=parser.parse_args();root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    _,bank,env,params=_assets()
    if args.reanalyze_source:
        result=_reanalyze(root,Path(args.reanalyze_source),env,params)
    else:
        snapshots=_snapshot_manifest(root,bank,env,params);result=_search(root,snapshots,env,params)
    if not result["gate"]["gate"]:
        save_json(root/"successful_action_multimodality_audit.json",{"status":"NOT_EXECUTED","reason":"local support gate failed before multimodality authority stage"})
        save_json(root/"receding_horizon_feedback_oracle_results.json",{"status":"NOT_EXECUTED","reason":"local support gate failed","heldout_used":False})
        save_json(root/"head_vs_last_block_candidate_cv.json",{"status":"NOT_EXECUTED","reason":"feedback oracle not authorized","heldout_used":False})


if __name__=="__main__":main()
