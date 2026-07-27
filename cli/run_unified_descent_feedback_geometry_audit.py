"""Audit transfer and support geometry of immutable local CEM corrections."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cli.run_unified_descent_feedback_probe import DATASET, _assets, _exact
from dvgc.descent_probe import batched_base_state, make_residual_rollout
from dvgc.env import END_REASON
from dvgc.runtime import save_json
from dvgc.support_diagnostic import candidate_grouped_diagnostic, weak_components


EXPECTED_HEAD="eff25d5"
SOURCE=Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1_replay_corrected")
SNAPSHOTS=Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1/feedback_probe_snapshots.pkl")
MULTI=Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1_multimodality/successful_action_multimodality_audit.json")
EXPECTED_COUNTS=[3,0,1,2,1,1,1,3]


def _sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _precursor(snapshot,cfg):
    feature=np.asarray(snapshot["physical_feature"]);roll=np.deg2rad(float(cfg.max_roll_deg))-abs(feature[3]);pitch=np.deg2rad(float(cfg.max_pitch_deg))-abs(feature[4])
    return "roll" if roll<=pitch else "pitch"


def _contact(snapshot):return int(np.argmax(np.asarray(snapshot["policy_state"]["contact_probs"])))


def _delay_semantics(snapshot):
    value=np.asarray(snapshot["policy_state"]["delay_buffer"])
    return {"shape":list(value.shape),"dtype":value.dtype.str,"fifo_order":"oldest_to_newest","shared_runtime_contract":True}


def _medoid_map(multimodality):return {int(row["snapshot_index"]):int(row["successful_medoid"]["rank"]) for row in multimodality["rows"]}


def _origin(item):
    if item["old_relabel_accepted"]:return "historical_accepted"
    if item["old_relabel_audited"]:return "historical_rejected"
    return "newly_frozen_excluded_candidate"


def _authoritative_top(row,rank):return next(item for item in row["top5"] if int(item["rank"])==int(rank))


def _reconcile(root,authority,snapshots,multimodality,cfg):
    medoids=_medoid_map(multimodality);rows=[]
    for index,(row,item) in enumerate(zip(authority["rows"],snapshots,strict=True)):
        top=None if not row["authoritative_correction"] else _authoritative_top(row,medoids[index]);replay=None if top is None else top["exact_replay"]["replay"]
        rows.append({"snapshot_index":index,"snapshot_hash":item["snapshot_hash"],"candidate_id":row["candidate_id"],"tick":row["tick"],"tick_region":row["tick_region"],"origin":_origin(item),"old_relabel":{"audited":item["old_relabel_audited"],"accepted":item["old_relabel_accepted"],"rejection_reasons":item["old_relabel_rejection_reasons"]},"local_authority_pass":row["authoritative_correction"],"baseline_survival":row["baseline"]["survival"],"baseline_minimum_margin":row["baseline"]["minimum_margin"],"exact_local_survival":None if replay is None else replay["survival"],"exact_gain":None if top is None else top["gain"],"failure_before":row["baseline"]["failure"],"failure_after":None if top is None else top["failure"],"correction_sequence_sha256":None if top is None else _sha(np.asarray(top["residual_knots"],np.float32)),"phase":int(item["snapshot"]["oracle_phase"]),"contact_mode":_contact(item["snapshot"]),"delay_semantics":_delay_semantics(item["snapshot"]),"failure_precursor":_precursor(item["snapshot"],cfg)})
    origin={name:{"total":sum(row["origin"]==name for row in rows),"passed":sum(row["origin"]==name and row["local_authority_pass"] for row in rows)} for name in ("historical_accepted","historical_rejected","newly_frozen_excluded_candidate")}
    counts=Counter(row["candidate_id"] for row in rows if row["local_authority_pass"]);candidate_order=sorted({row["candidate_id"] for row in rows});actual=[counts[cid] for cid in candidate_order]
    rejected_pass=[{"candidate_id":row["candidate_id"],"tick":row["tick"],"snapshot_hash":row["snapshot_hash"],"exact_gain":row["exact_gain"]} for row in rows if row["origin"]=="historical_rejected" and row["local_authority_pass"]]
    gate=origin=={"historical_accepted":{"total":10,"passed":4},"historical_rejected":{"total":5,"passed":3},"newly_frozen_excluded_candidate":{"total":9,"passed":5}} and sum(value["passed"] for value in origin.values())==12 and actual==EXPECTED_COUNTS
    report={"status":"PASS" if gate else "AUTHORITY_ACCOUNTING_INCONSISTENCY","gate":gate,"origin_counts":origin,"historical_accepted_not_reproduced":origin["historical_accepted"]["total"]-origin["historical_accepted"]["passed"],"historical_rejected_passes":rejected_pass,"candidate_order":candidate_order,"candidate_pass_counts":actual,"rows":rows,"heldout_used":False,"ppo_authorization":False}
    save_json(root/"feedback_authority_origin_reconciliation.json",report)
    if not gate:raise SystemExit("AUTHORITY_ACCOUNTING_INCONSISTENCY")
    return report,medoids


def _geometry(root,authority,snapshots,reconciliation,medoids):
    data=pickle.loads(DATASET.read_bytes());teacher=defaultdict(list)
    for row in data["teacher"]:teacher[row["candidate_id"]].append(np.asarray(row["normalized_observation"]))
    counts=dict(zip(reconciliation["candidate_order"],reconciliation["candidate_pass_counts"]));layer_name={3:"robust-core",2:"frontier",1:"sparse-support",0:"unsupported"};rows=[]
    for index,(row,item,account) in enumerate(zip(authority["rows"],snapshots,reconciliation["rows"],strict=True)):
        normalized=np.asarray(item["normalized_actor_observation"]);distance=min(float(np.linalg.norm(normalized-old)/np.sqrt(len(normalized))) for old in teacher[row["candidate_id"]])
        top=None if not row["authoritative_correction"] else _authoritative_top(row,medoids[index]);residual=np.zeros((2,4)) if top is None else np.asarray(top["residual_knots"]);bound=None if top is None else float(top["bound"])
        actions=np.asarray([candidate["exact_replay"]["replay"]["actions"][:8] for candidate in row["top5"]]);pair=np.linalg.norm(actions[:,None]-actions[None,:],axis=(2,3))/np.sqrt(32);upper=pair[np.triu_indices(len(actions),1)]
        snapshot=item["snapshot"];policy=snapshot["policy_state"]
        rows.append({"snapshot_index":index,"candidate_id":row["candidate_id"],"snapshot_hash":item["snapshot_hash"],"support_layer":layer_name[counts[row["candidate_id"]]],"normalized_actor_observation_history":item["normalized_actor_observation"],"privileged_physical_state":{"qpos":np.asarray(snapshot["qpos"]).tolist(),"qvel":np.asarray(snapshot["qvel"]).tolist(),"ctrl":np.asarray(snapshot["ctrl"]).tolist(),"physical_feature":np.asarray(snapshot["physical_feature"]).tolist()},"delay_buffer":np.asarray(policy["delay_buffer"]).tolist(),"last_action":np.asarray(policy["last_action"]).tolist(),"phase":account["phase"],"contact_mode":account["contact_mode"],"failure_precursor":account["failure_precursor"],"baseline_time_to_failure":account["baseline_survival"],"normalized_distance_to_teacher_trajectory":distance,"local_correction_residual_rms":None if top is None else float(np.sqrt(np.mean(residual*residual))),"local_correction_residual_max":None if top is None else float(np.max(np.abs(residual))),"residual_bound":bound,"action_dimension_touches_bound":None if top is None else dict(zip(("steer","drive","hip","knee"),np.any(np.abs(residual)>=bound-1e-6,axis=0).tolist())),"top5_action_pairwise_distance":{"mean":float(upper.mean()),"max":float(upper.max())},"local_exact_gain":account["exact_gain"],"authority_pass":account["local_authority_pass"]})
    layer_counts=Counter(layer_name[value] for value in counts.values());report={"status":"PASS","artifact_role":"feedback_support_geometry_not_tube","candidate_layers":{cid:layer_name[count] for cid,count in counts.items()},"layer_counts":dict(layer_counts),"expected_layer_counts":{"robust-core":2,"frontier":1,"sparse-support":4,"unsupported":1},"rows":rows,"heldout_used":False}
    report["gate"]=report["layer_counts"]==report["expected_layer_counts"];save_json(root/"feedback_support_geometry_table.json",report)
    if not report["gate"]:raise SystemExit("support geometry layer mismatch")
    return report


def _cross_transfer(root,authority,snapshots,reconciliation,medoids,env,params):
    rollout=make_residual_rollout(env,params,horizon=24,ticks_per_knot=4,residual_ticks=8);sources=[]
    for index,row in enumerate(authority["rows"]):
        if row["authoritative_correction"]:
            top=_authoritative_top(row,medoids[index]);sources.append((index,row["candidate_id"],np.asarray(top["residual_knots"],np.float32),top))
    account=reconciliation["rows"];pairs=[]
    for source_number,(source_index,source_candidate,residual,source_top) in enumerate(sources):
        for target_index,target in enumerate(account):
            compatible=target["phase"]==account[source_index]["phase"] and target["contact_mode"]==account[source_index]["contact_mode"] and target["delay_semantics"]==account[source_index]["delay_semantics"] and target["failure_precursor"]==account[source_index]["failure_precursor"]
            if not compatible:continue
            factory=lambda count,target_index=target_index:batched_base_state(env,snapshots[target_index]["snapshot"],9100000+target_index,count)
            probe={"residual_knots":residual.tolist(),"survival":0,"minimum_margin":0,"terminal_margin":0,"end_code":0,"actions":[],"features":[]};exact=_exact(rollout,factory,probe,9800000+source_number*100+target_index);replay=exact["replay"];failure=END_REASON.get(replay["end_code"],"horizon");gain=replay["survival"]-target["baseline_survival"];no_new=failure in {target["failure_before"],"horizon"};transfer=exact["repeat1_repeat2_exact"] and gain>=2 and no_new
            pairs.append({"source_snapshot_index":source_index,"source_candidate_id":source_candidate,"target_snapshot_index":target_index,"target_candidate_id":target["candidate_id"],"same_snapshot":source_index==target_index,"same_candidate":source_candidate==target["candidate_id"],"source_layer":None,"target_layer":None,"repeat1_repeat2_exact":exact["repeat1_repeat2_exact"],"survival":replay["survival"],"baseline_survival":target["baseline_survival"],"gain":gain,"minimum_physical_margin":replay["minimum_margin"],"failure":failure,"no_new_failure_type":no_new,"physical_transfer":transfer,"margin_only_improvement":gain<2 and replay["minimum_margin"]>target["baseline_minimum_margin"]})
        save_json(root/"partial_feedback_correction_cross_snapshot_transfer_matrix.json",{"completed_sources":source_number+1,"total_sources":12,"pairs":pairs})
    layers={cid:({3:"robust-core",2:"frontier",1:"sparse-support",0:"unsupported"}[count]) for cid,count in zip(reconciliation["candidate_order"],reconciliation["candidate_pass_counts"])}
    for pair in pairs:pair["source_layer"]=layers[pair["source_candidate_id"]];pair["target_layer"]=layers[pair["target_candidate_id"]]
    diagonal=[p for p in pairs if p["same_snapshot"]];same=[p for p in pairs if p["same_candidate"] and not p["same_snapshot"]];cross=[p for p in pairs if not p["same_candidate"]]
    if not all(p["physical_transfer"] for p in diagonal):raise SystemExit("authority diagonal failed to reproduce")
    edges=sorted({(p["source_candidate_id"],p["target_candidate_id"]) for p in cross if p["physical_transfer"]});components=weak_components(reconciliation["candidate_order"],edges)
    def summary(rows):return {"eligible":len(rows),"gain_at_least_2":sum(p["physical_transfer"] for p in rows),"margin_only":sum(p["margin_only_improvement"] for p in rows)}
    categories={"diagonal":summary(diagonal),"same_candidate_off_diagonal":summary(same),"cross_candidate":summary(cross),"robust_core_internal":summary([p for p in cross if p["source_layer"]==p["target_layer"]=="robust-core"]),"robust_core_to_frontier_sparse":summary([p for p in cross if p["source_layer"]=="robust-core" and p["target_layer"] in {"frontier","sparse-support"}]),"frontier_sparse_to_robust_core":summary([p for p in cross if p["source_layer"] in {"frontier","sparse-support"} and p["target_layer"]=="robust-core"])}
    report={"status":"PASS","eligibility":"phase, contact mode, delay-buffer semantics, and failure precursor all equal","source_corrections":"12 immutable successful medoids","categories":categories,"successful_cross_candidate_edges":edges,"weak_candidate_components":components,"cross_target_candidates":sorted({p["target_candidate_id"] for p in cross if p["physical_transfer"]}),"pairs":pairs,"heldout_used":False,"ppo_authorization":False};save_json(root/"feedback_correction_cross_snapshot_transfer_matrix.json",report);return report


def _diagnostic(root,geometry,cross):
    labels=np.asarray([row["authority_pass"] for row in geometry["rows"]]);groups=[row["candidate_id"] for row in geometry["rows"]]
    actor=np.asarray([np.concatenate([row["normalized_actor_observation_history"],row["last_action"],np.asarray(row["delay_buffer"]).reshape(-1)]) for row in geometry["rows"]])
    privileged=np.asarray([np.concatenate([row["privileged_physical_state"][key] for key in ("qpos","qvel","ctrl","physical_feature")]+[[row["phase"],row["contact_mode"],int(row["failure_precursor"]=="pitch")]]) for row in geometry["rows"]])
    actor_result=candidate_grouped_diagnostic(actor,labels,groups);priv_result=candidate_grouped_diagnostic(privileged,labels,groups)
    def separable(result):
        metric=result["linear"];return metric["balanced_accuracy"]>=.70 and metric["positive_precision"]>=.60 and metric["positive_recall"]>=.60 and metric["balanced_accuracy"]>metric["permutation_balanced_accuracy_p95"]
    actor_sep=separable(actor_result);priv_sep=separable(priv_result);report={"status":"PASS","protocol":{"split":"leave-one-candidate-out","linear":"dual ridge alpha=1, threshold=0","knn":"k=3 Euclidean after train-fold standardization","permutations":256,"separability_rule":"primary linear balanced accuracy >=0.70, precision/recall >=0.60, and balanced accuracy above permutation p95"},"actor_visible":actor_result,"privileged_diagnostic":priv_result,"actor_visible_separable":actor_sep,"privileged_separable":priv_sep,"uncertainty":"n=24 with only eight candidate groups; estimates are diagnostic and not a policy, gate, or Tube classifier","candidate_groups":groups,"labels":labels.tolist(),"heldout_used":False};save_json(root/"actor_visible_vs_privileged_support_diagnostic.json",report);return report


def _classification(cross,diagnostic):
    same=cross["categories"]["same_candidate_off_diagonal"];other=cross["categories"]["cross_candidate"];same_rate=same["gain_at_least_2"]/max(same["eligible"],1);cross_rate=other["gain_at_least_2"]/max(other["eligible"],1);cross_rare=other["gain_at_least_2"]<=max(1,math.ceil(.05*other["eligible"]));targets=len(cross["cross_target_candidates"])
    if cross_rare:
        result="CANDIDATE_LOCAL_FEEDBACK_SUPPORT" if same_rate>=.25 and same_rate>=2*cross_rate else "POINTWISE_OPEN_LOOP_CORRECTIONS_ONLY"
    elif targets>=4 and diagnostic["actor_visible_separable"]:result="OBSERVABLE_TRANSFERABLE_FEEDBACK_CORE"
    elif diagnostic["privileged_separable"] and not diagnostic["actor_visible_separable"]:result="ACTOR_OBSERVATION_INFORMATION_GAP"
    else:result="NONCOHERENT_SEARCH_SUPPORT"
    return result,{"same_candidate_rate":same_rate,"cross_candidate_rate":cross_rate,"cross_rare_threshold":max(1,math.ceil(.05*other["eligible"])),"cross_target_candidates":targets}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);args=parser.parse_args();root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    if subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_HEAD,"HEAD"]).returncode or subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise SystemExit("invalid git state")
    cfg,_,env,params=_assets();authority=json.loads((SOURCE/"local_cem_authority_results.json").read_text());multimodality=json.loads(MULTI.read_text())
    with SNAPSHOTS.open("rb") as handle:snapshots=pickle.load(handle)
    root.mkdir(parents=True);reconciliation,medoids=_reconcile(root,authority,snapshots,multimodality,cfg);geometry=_geometry(root,authority,snapshots,reconciliation,medoids);cross=_cross_transfer(root,authority,snapshots,reconciliation,medoids,env,params);diagnostic=_diagnostic(root,geometry,cross);classification,decision=_classification(cross,diagnostic)
    closed=classification in {"POINTWISE_OPEN_LOOP_CORRECTIONS_ONLY","NONCOHERENT_SEARCH_SUPPORT","ACTOR_OBSERVATION_INFORMATION_GAP"}
    report={"experiment":"unified_descent_feedback_correction_transfer_and_support_geometry_audit_v1","status":"PASS","causal_classification":classification,"decision_metrics":decision,"authority_origin":reconciliation["origin_counts"],"support_layers":geometry["layer_counts"],"transfer_categories":cross["categories"],"cross_candidate_components":cross["weak_candidate_components"],"actor_visible_separable":diagnostic["actor_visible_separable"],"privileged_separable":diagnostic["privileged_separable"],"formal_tube_claim":False,"artifact_role":"provisional_feedback_support_only","cem_action_regression_bootstrap_route_closed_under_current_observation":closed,"next_permitted_work":"separate observation/history sufficiency audit" if classification=="ACTOR_OBSERVATION_INFORMATION_GAP" else None,"heldout_used":False,"ppo_authorization":False}
    save_json(root/"UNIFIED_DESCENT_FEEDBACK_CORRECTION_TRANSFER_AND_SUPPORT_GEOMETRY_AUDIT_V1_REPORT.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
