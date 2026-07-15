"""Fixed-bank evaluation and drift probes for stage-expert discovery."""
from __future__ import annotations
from collections import Counter
import jax,numpy as np
from .bank import SnapshotBank
from .composite import CanonicalEntryMatcher,composite_rollout
from .config import load_config
from .env import END_REASON,OrangeBikeDVGC
from .rollout import restore_snapshot
from .runtime import build_inference,build_policy_distribution


def evaluate_flight_composite(flight_params,flight_cfg,landing_params,records,entry_path,*,seed,controller_stack_hash):
    cfg=load_config(overrides={**flight_cfg,"training_stage":"flight","expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
    entry=SnapshotBank.load(entry_path); env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=entry); step=jax.jit(env.step)
    inference={"flight":build_inference(env,flight_params,deterministic=True),"landing":build_inference(env,landing_params,deterministic=True)}; matcher=CanonicalEntryMatcher(env,"flight",entry_path); rows=[]
    for i,row in enumerate(records):
        key=jax.random.PRNGKey(seed+i); _,out=composite_rollout(env,("flight","landing"),inference,{"flight":matcher},restore_snapshot(env,row,key),key,horizon=int(cfg.branch_horizon),step_fn=step)
        rows.append({**out,"candidate_id":row["id"],"candidate_kind":row.get("candidate_kind"),"flight_subinterval":row.get("flight_subinterval"),"termination_reason":END_REASON.get(out["end_code"],"unknown")})
    def rates(part): return {"episodes":len(part),"chain_rate":float(np.mean([x["chain"] for x in part])) if part else 0.0,"composite_final_rate":float(np.mean([x["final"] for x in part])) if part else 0.0,"chain_missed_final_rate":float(np.mean([x["chain_missed_final"] for x in part])) if part else 0.0,"physical_failure_rate":float(np.mean([x["terminated"] and not x["final"] for x in part])) if part else 0.0,"timeout_rate":float(np.mean([x["truncated"] for x in part])) if part else 0.0}
    report=rates(rows); report.update({"controller_stack_hash":controller_stack_hash,"termination_reason_counts":dict(Counter(x["termination_reason"] for x in rows)),"subintervals":{name:rates([x for x in rows if x["flight_subinterval"]==name]) for name in ("ascent","apex","descent")},"rows":rows}); return report


def action_drift(params,reference_params,cfg_dict,records,seed):
    cfg=load_config(overrides={**cfg_dict,"training_stage":"flight","expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False}); env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank()); ref=build_policy_distribution(env,reference_params); cur=build_policy_distribution(env,params); rows=[]
    for i,row in enumerate(records):
        state=restore_snapshot(env,row,jax.random.PRNGKey(seed+i)); m0,s0,a0=map(np.asarray,ref(state.obs)); m1,s1,a1=map(np.asarray,cur(state.obs)); kl=float(np.sum(np.log(s1/s0)+(s0*s0+(m0-m1)**2)/(2*s1*s1)-.5)); rows.append({"id":row["id"],"kl":kl,"action_l2":float(np.linalg.norm(a0-a1))})
    def stats(key):
        x=np.asarray([r[key] for r in rows]); return {"mean":float(x.mean()),"p95":float(np.quantile(x,.95)),"max":float(x.max())}
    return {"states":len(rows),"kl":stats("kl"),"action_l2":stats("action_l2"),"rows":rows}
