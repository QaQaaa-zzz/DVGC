"""No-training Descent policy displacement, controllability and CEM probe."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import losses as ppo_losses

from cli.calibrate_unified_descent_optimizer_trust_region import _load_saved_rollout
from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_pilot import REWARD_KEYS
from dvgc.descent_probe import batched_base_state, cem_search, exact_replay_matches, make_residual_rollout
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.ppo_integrity import logprob_audit, normalizer_summary, tree_delta
from dvgc.runtime import save_json

EXPECTED_HEAD="3aa5758";EXPECTED_BANK="8e6342bc0d9d5e7821929f6ddccd4fb5b7c23923c66ab2d058309249dfed45a1"
EXPECTED_XML="d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c"
EXPECTED_NORMALIZER="8f2e36b6f69a3d20da67c1854f7e908c98dd6b03ae70e287e0a7e28522f93a7e"
TRUST_RUN=Path("runs/unified_descent_rsi_learnability_pilot_v1_trust_region_rerun_20260727")


def _rank(x):
    x=np.asarray(x);order=np.argsort(x,kind="mergesort");ranks=np.empty(len(x),float)
    start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        ranks[order[start:end]]=(start+end-1)/2;start=end
    return ranks


def _spearman(x,y):
    rx,ry=_rank(x),_rank(y)
    if np.std(rx)<1e-12 or np.std(ry)<1e-12:return 0.0
    return float(np.corrcoef(rx,ry)[0,1])


def _displacement(env,rows,initial,checkpoint_eval):
    network,base_params,data,_=_load_saved_rollout(initial)
    dist=network.parametric_action_distribution
    old_logits=network.policy_network.apply(initial[0],base_params.policy,data.observation)
    old_action=np.asarray(dist.mode(old_logits));old_value=np.asarray(network.value_network.apply(initial[0],base_params.value,data.observation))
    output={};base_eval=checkpoint_eval["checkpoint_0000"]["evaluation"]
    for step in (0,1600,3200,4800,6400):
        name=f"checkpoint_{step:04d}";params,_,_=load_bundle(TRUST_RUN/"checkpoints"/name,verify_files=True)
        learner=type(base_params)(policy=params[1],value=params[2]);audit=logprob_audit(network,learner,initial[0],data)
        logits=network.policy_network.apply(initial[0],learner.policy,data.observation)
        action=np.asarray(dist.mode(logits));value=np.asarray(network.value_network.apply(initial[0],learner.value,data.observation))
        trajectory=[]
        for before,after in zip(base_eval["rows"],checkpoint_eval[name]["evaluation"]["rows"]):
            a0,a1=np.asarray(before["actions"]),np.asarray(after["actions"]);shared=min(len(a0),len(a1))
            windows={}
            for horizon in (4,8,12):
                n=min(shared,horizon);delta=a1[:n]-a0[:n] if n else np.zeros((1,4))
                windows[str(horizon)]={"max":float(np.max(np.abs(delta))),"rms":float(np.sqrt(np.mean(delta*delta)))}
            trajectory.append({"candidate_id":before["candidate_id"],"action_windows":windows,
                "time_to_failure_delta":after["survived_ticks"]-before["survived_ticks"],
                "minimum_margin_delta":{k:after["minimum_margins"][k]-before["minimum_margins"][k] for k in before["minimum_margins"]},
                "reward_component_delta":{k:after["reward_components"].get(k,0)-before["reward_components"].get(k,0) for k in REWARD_KEYS}})
        output[name]={"analytic_kl":audit["analytic_distribution_kl_mean"],"sample_kl":audit["sample_mean_kl"],
            "deterministic_action_delta_max":float(np.max(np.abs(action-old_action))),
            "deterministic_action_delta_rms":float(np.sqrt(np.mean((action-old_action)**2))),
            "actor_delta":tree_delta(params[1],initial[1]),"value_delta":tree_delta(params[2],initial[2]),
            "value_prediction_delta_max":float(np.max(np.abs(value-old_value))),
            "value_prediction_delta_rms":float(np.sqrt(np.mean((value-old_value)**2))),"trajectories":trajectory}
    # Exact saved-batch GAE and stratification.
    swap=lambda x:jnp.swapaxes(x,0,1);obs=jax.tree_util.tree_map(swap,data.observation);nobs=jax.tree_util.tree_map(swap,data.next_observation)
    values=network.value_network.apply(initial[0],base_params.value,obs);bootstrap=network.value_network.apply(initial[0],base_params.value,jax.tree_util.tree_map(lambda x:x[-1],nobs))
    trunc=swap(data.extras["state_extras"]["truncation"]);discount=swap(data.discount)
    returns,advantages=ppo_losses.compute_gae(truncation=trunc,termination=(1-discount)*(1-trunc),rewards=swap(data.reward)*.1,values=values,bootstrap_value=bootstrap,lambda_=.97,discount=.995)
    adv=np.asarray(advantages).T;ret=np.asarray(returns).T
    asset=np.load("runs/unified_descent_rsi_update_integrity_repair_v1_phase_a_v4_20260727/first_rollout_read_only.npz");candidate=np.asarray(asset["candidate_index"])
    by_candidate={}
    for index,row in enumerate(rows):
        mask=candidate==index;x=adv[mask]
        by_candidate[row["id"]]={"samples":int(mask.sum()),"advantage_mean":float(x.mean()),"advantage_std":float(x.std()),"positive_fraction":float(np.mean(x>0)),"negative_fraction":float(np.mean(x<0))}
    tick=[{"tick":t,"mean":float(adv[:,t].mean()),"std":float(adv[:,t].std()),"positive_fraction":float(np.mean(adv[:,t]>0))} for t in range(32)]
    output["training_batch"]={"advantage":{"mean":float(adv.mean()),"std":float(adv.std()),"positive_fraction":float(np.mean(adv>0)),"negative_fraction":float(np.mean(adv<0))},
        "candidate":by_candidate,"tick":tick,"policy_gradient_signal_to_noise_proxy":float(abs(adv.mean())/max(adv.std(),1e-12)),
        "return_reward_correlation":float(np.corrcoef(ret.reshape(-1),np.asarray(asset["reward"]).reshape(-1))[0,1])}
    return output


def _local_probe(rollout,state,candidate_id):
    zero=np.zeros((1,6,4),np.float32);baseline=jax.device_get(rollout(state,jnp.asarray(zero),jax.random.PRNGKey(1)))
    rows=[];central={}
    for scale in (.01,.025,.05):
        for dim in range(4):
            signed={}
            for sign in (-1,1):
                residual=np.zeros((1,6,4),np.float32);residual[:,:,dim]=sign*scale
                result=jax.device_get(rollout(state,jnp.asarray(residual),jax.random.PRNGKey(1)))
                features=np.asarray(result["features"])[:,0]
                item={"candidate_id":candidate_id,"scale":scale,"action_dim":dim,"sign":sign,
                    "survival":int(result["survival"][0]),"minimum_margin":float(result["minimum_margin"][0]),
                    "reward_return":float(result["reward_return"][0]),"end_code":int(result["end_code"][0]),
                    "feature_at":{"1":features[0].tolist(),"2":features[1].tolist(),"4":features[3].tolist(),"8":features[7].tolist()}}
                rows.append(item);signed[sign]=features
            if scale==.01:central[dim]=(signed[1]-signed[-1])/(2*scale)
    jacobians={}
    for h,index in ((1,0),(2,1),(4,3),(8,7)):
        matrix=np.stack([central[d][index,[3,9,4,10,8,2,14,15]] for d in range(4)],axis=1)
        singular=np.linalg.svd(matrix,compute_uv=False);rank=int(np.sum(singular>max(singular[0]*1e-5,1e-8)))
        jacobians[str(h)]={"matrix":matrix.tolist(),"singular_values":singular.tolist(),"effective_rank":rank,
            "condition_number":float(singular[0]/singular[rank-1]) if rank else float("inf")}
    return {"baseline":{"survival":int(baseline["survival"][0]),"minimum_margin":float(baseline["minimum_margin"][0]),"reward_return":float(baseline["reward_return"][0])},"perturbations":rows,"jacobians":jacobians}


def _alignment(all_rows):
    returns=np.asarray([r["reward_return"] for r in all_rows]);surv=np.asarray([r["survival"] for r in all_rows]);margin=np.asarray([r["minimum_margin"] for r in all_rows])
    by=defaultdict(list)
    for r in all_rows:by[r["candidate_id"]].append(r)
    spearman=[];correct=total=0;rng=np.random.default_rng(20260727)
    for rows in by.values():
        spearman.append(_spearman([r["reward_return"] for r in rows],[r["survival"] for r in rows]))
        for _ in range(min(10000,len(rows)*20)):
            a,b=rng.choice(len(rows),2,replace=False);ra,rb=rows[a],rows[b]
            pa=(ra["survival"],ra["minimum_margin"],ra["terminal_margin"]);pb=(rb["survival"],rb["minimum_margin"],rb["terminal_margin"])
            if pa==pb:continue
            correct+=int((ra["reward_return"]>rb["reward_return"])==(pa>pb));total+=1
    top=max(1,len(all_rows)//10);reward_top=np.argsort(returns)[-top:];physical_top=np.lexsort((-margin,-surv))[:top]
    components=np.asarray([r["reward_components"] for r in all_rows]);variance=np.var(components,axis=0);share=variance/max(float(variance.sum()),1e-12)
    return {"return_survival_spearman_global":_spearman(returns,surv),"return_minimum_margin_spearman":_spearman(returns,margin),
        "candidate_stratified_spearman_mean":float(np.mean(spearman)),"pairwise_ranking_accuracy":correct/max(total,1),
        "top_reward_survival":{"mean":float(surv[reward_top].mean()),"p10":float(np.quantile(surv[reward_top],.1))},
        "top_physical_reward_percentile_mean":float(np.mean(_rank(returns)[physical_top]/max(len(returns)-1,1))),
        "component_variance_share":dict(zip(REWARD_KEYS,share.tolist())),"sequence_count":len(all_rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True);p.add_argument("--max-candidates",type=int,default=14);p.add_argument("--samples",type=int,default=256);p.add_argument("--generations",type=int,default=5);p.add_argument("--bounds",default="0.05,0.10,0.20")
    a=p.parse_args();root=Path(a.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    if subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_HEAD,"HEAD"]).returncode or subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise SystemExit("invalid git state")
    bank_path=Path("runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl");cfg=load_config("configs/unified_descent_rsi_learnability_pilot_v1.json")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if file_sha256(bank_path)!=EXPECTED_BANK or file_sha256(cfg.xml_path)!=EXPECTED_XML or cfg.action_mapping_version!=ACTION_MAPPING_VERSION or gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("provenance gate")
    initial,_,_=load_bundle("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy",verify_files=True)
    if normalizer_summary(initial[0])["sha256"]!=EXPECTED_NORMALIZER:raise SystemExit("normalizer gate")
    root.mkdir(parents=True);bank=SnapshotBank.load(bank_path);records=bank.records[:a.max_candidates];env=OrangeBikeDVGC(cfg,snapshot_bank=bank)
    checkpoint_eval=json.loads((TRUST_RUN/"trust_region_rerun_checkpoint_evaluations.json").read_text())
    displacement=_displacement(env,bank.records,initial,checkpoint_eval);save_json(root/"checkpoint_policy_displacement_audit.json",displacement)
    rollout=make_residual_rollout(env,initial);local={};cem_out=[];all_rows=[];bounds=[float(x) for x in a.bounds.split(",")]
    for index,record in enumerate(records):
        state_factory=lambda count,record=record,index=index:batched_base_state(env,record,202607270+index,count)
        state=state_factory(1);local[record["id"]]=_local_probe(rollout,state,record["id"])
        baseline=local[record["id"]]["baseline"];best=None
        for level,bound in enumerate(bounds):
            knots,summary,rows=cem_search(rollout,state_factory,bound=bound,seed=20260727+index*100+level,generations=a.generations,samples=a.samples,elite_count=max(4,a.samples//8))
            for row in rows:row.update({"candidate_id":record["id"],"bound":bound})
            all_rows.extend(rows);replay_state=state_factory(1);replay1=jax.device_get(rollout(replay_state,jnp.asarray(knots[None]),jax.random.PRNGKey(77)));replay2=jax.device_get(rollout(replay_state,jnp.asarray(knots[None]),jax.random.PRNGKey(77)))
            exact=exact_replay_matches(replay1,replay2,summary)
            item={"bound":bound,"best":summary,"exact_replay":exact,"residual_knots":knots.tolist()};
            if best is None or summary["survival"]>best["best"]["survival"] or (summary["survival"]==best["best"]["survival"] and summary["minimum_margin"]>best["best"]["minimum_margin"]):best=item
            if int(summary["survival"])>=24:break
        cem_out.append({"candidate_id":record["id"],"baseline":baseline,"oracle":best,
            "time_to_failure_delta":int(best["best"]["survival"])-int(baseline["survival"]),
            "failure_before":None,"failure_after":END_REASON.get(int(best["best"]["end_code"]),"none")})
    save_json(root/"local_action_controllability_audit.json",{"status":"PASS","candidates":local,"action_space":"bounded policy action residual after tanh/mode, before env action-to-control mapping","scales":[.01,.025,.05],"horizons":[1,2,4,8]})
    save_json(root/"residual_cem_oracle_results.json",{"status":"PASS","protocol":{"knots":6,"ticks_per_knot":4,"horizon":24,"samples":a.samples,"generations":a.generations,"bounds":bounds,"objective":"lexicographic survival, min margin, terminal margin, effort"},"candidates":cem_out})
    save_json(root/"cem_sequence_summaries.json",{"rows":all_rows})
    alignment=_alignment(all_rows);save_json(root/"reward_physics_alignment_audit.json",alignment)
    improved16=sum(int(row["oracle"]["best"]["survival"])>=16 for row in cem_out);deltas=[row["time_to_failure_delta"] for row in cem_out]
    gain4=sum(delta>=4 for delta in deltas);median_gain=float(np.median(deltas))
    classification="control_authority_or_candidate_problem"
    if not (improved16<4 or median_gain<2 or gain4==0):
        if alignment["candidate_stratified_spearman_mean"]<.5 or alignment["pairwise_ranking_accuracy"]<.70:classification="reward_misalignment"
        else:
            effective=[row for row in cem_out if row["time_to_failure_delta"]>0];within=sum(float(row["oracle"]["best"]["residual_rms"])<=.10 for row in effective)>=max(1,math.ceil(len(effective)/2))
            classification="curriculum_gap_with_local_learnability" if (improved16>=4 or median_gain>=4) and within else "initialization_or_exploration_gap"
    diagnosis={"status":"PASS","classification":classification,"states_reaching_16":improved16,"median_time_to_failure_gain":median_gain,"states_gain_at_least_4":gain4,
        "reward_spearman":alignment["candidate_stratified_spearman_mean"],"pairwise_accuracy":alignment["pairwise_ranking_accuracy"],"phase_b_authorized":classification=="curriculum_gap_with_local_learnability","ppo_authorization":classification=="curriculum_gap_with_local_learnability"}
    save_json(root/"diagnosis_classification.json",diagnosis);print(json.dumps(diagnosis,indent=2))

if __name__=="__main__":main()
