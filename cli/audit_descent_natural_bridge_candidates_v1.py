"""Independent 32-branch audit and non-overwriting Tube v3 extension."""
from __future__ import annotations
import argparse,copy,hashlib,json,pickle,subprocess
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from cli.audit_descent_compact_adapter_v1 import _perturb_batch
from cli.finalize_descent_compact_tube_v2 import certified_outcome
from cli.run_backward_descent_nominal_pilot import C_L,EXPECTED,PI_D,PI_L
from cli.run_descent_localized_consolidation_v1 import verified_assets_allowing_runtime_gate_refresh
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter,make_descent_landing_rollout
from dvgc.bank import SnapshotBank,beta_posterior,posterior_label
from dvgc.certification import DYNAMICS_VARIANTS,branch_seed,detailed_terminal_summary
from dvgc.config import file_sha256,load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json

SOURCE=Path('runs/descent_natural_bridge_candidates_v1/pilot_12x_p1/natural_bridge_proposal_support.pkl');BASE=Path('runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1/descent_tube_v2.pkl')
EXPERT=Path('runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter');DEFAULT_RUN=Path('runs/descent_natural_bridge_candidates_v1/independent_audit_2x32');SEED=3_700_000_000;BRANCHES=32;NAMESPACE='descent-natural-targeted-independent-audit-v1'

def union_records(base,new):
 by={row['id']:copy.deepcopy(row) for row in base}
 for row in new:
  if row['id'] in by:raise ValueError('Tube extension id collision')
  by[row['id']]=copy.deepcopy(row)
 return list(by.values())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default=str(DEFAULT_RUN));a=p.parse_args();root=Path(a.run)
 if root.exists():raise SystemExit(f'refusing overwrite {root}')
 valid,failed,raw=verified_assets_allowing_runtime_gate_refresh()
 if not valid:raise SystemExit(f'frozen asset mismatch: {failed}; raw={raw}')
 gate=json.loads(Path('docs/RUNTIME_GATE.json').read_text())
 if gate.get('status')!='PASS' or gate.get('source_fingerprint')!=source_fingerprint(Path.cwd()):raise SystemExit('runtime gate stale')
 source=SnapshotBank.load(SOURCE);base_bank=SnapshotBank.load(BASE);artifact=pickle.loads((EXPERT/'adapter.pkl').read_bytes());dp,_,_=load_bundle(PI_D,verify_files=True);lp,_,_=load_bundle(PI_L,verify_files=True)
 cfg0=load_config('configs/backward_descent_rsi_pilot_v1.json',{'use_bank_resets':False,'expert_chain_termination':False,'domain_randomization':False,'obs_noise_enable':False});root.mkdir(parents=True)
 inputs={'candidate_sha256':file_sha256(SOURCE),'base_tube_sha256':file_sha256(BASE),'adapter_sha256':file_sha256(EXPERT/'adapter.pkl'),'policy_identity_hash':artifact['policy_identity_hash'],'C_L':file_sha256(C_L),'xml':EXPECTED['xml'],'seed':SEED,'namespace':NAMESPACE}
 save_json(root/'manifest.json',{'status':'FROZEN_BEFORE_AUDIT','inputs':inputs,'states':len(source.records),'branches':BRANCHES,'dynamics_variants':DYNAMICS_VARIANTS,'labels_reused_for_training':False});save_json(root/'cost_estimate.json',{'estimated_seconds':600,'rollouts':len(source.records)*BRANCHES,'PPO_steps':0})
 variants=[]
 for spec in DYNAMICS_VARIANTS:
  cfg=load_config('configs/backward_descent_rsi_pilot_v1.json',{'use_bank_resets':False,'expert_chain_termination':False,'domain_randomization':False,'obs_noise_enable':False,**{k:v for k,v in spec.items() if k!='id'}});env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L));adapter=compact_observation_command_adapter(jnp.asarray(artifact['prototypes']),jnp.asarray(artifact['targets']),jnp.asarray(artifact['normalizer_mean']),jnp.asarray(artifact['normalizer_std']),float(artifact['radius']),float(artifact['core_radius']));variants.append((spec['id'],env,make_descent_landing_rollout(env,dp,lp,horizon=200,residual_ticks=8,descent_action_adapter=adapter)))
 rows=[]
 for state_index,record in enumerate(source.records):
  evidence=[]
  for variant_index,(variant,env,rollout) in enumerate(variants):
   indices=list(range(variant_index,BRANCHES,len(variants)));seeds=[];deltas=[]
   for b in indices:
    seed=branch_seed(SEED,state_index,b);seeds.append(seed);deltas.append(np.random.default_rng(seed).uniform(-.02,.02,2).astype(np.float32))
   batch=_perturb_batch(env,record,seeds,deltas);raw_out=jax.device_get(rollout(batch,jnp.zeros((len(seeds),2,4),jnp.float32),jax.random.PRNGKey(branch_seed(SEED,state_index,variant_index))))
   for local,b in enumerate(indices):
    code=int(np.asarray(raw_out['end_code'])[local]);chain=bool(np.asarray(raw_out['downstream_entry'])[local]);final=bool(np.asarray(raw_out['final_recovery'])[local]);qualified=chain and final;cause='final_recovery' if qualified else ('timeout' if code==8 else ('horizon_exhausted' if code==0 else 'physical_failure'))
    evidence.append({'branch_index':b,'branch_seed':seeds[local],'seed_namespace':NAMESPACE,'dynamics_variant':variant,'chain_success':chain,'final_recovery':qualified,'raw_final_recovery':final,'terminal_cause':cause,'end_code':code,'end_reason':END_REASON.get(code,'unknown'),'steps':int(np.asarray(raw_out['termination_tick'])[local])})
  evidence.sort(key=lambda r:r['branch_index']);finals=sum(r['final_recovery'] for r in evidence);posterior=beta_posterior(finals,BRANCHES-finals);label=posterior_label(posterior,BRANCHES,min_branches=cfg0.min_branches,safe_threshold=cfg0.safe_threshold,dead_threshold=cfg0.dead_threshold,boundary_max_width=cfg0.boundary_max_width);rows.append({'id':record['id'],'final':finals,'branches':BRANCHES,'label':label,'evidence':evidence});print(f'[targeted-audit] {state_index+1}/{len(source.records)} final={finals}/32 {label}',flush=True)
 summary=detailed_terminal_summary([b for r in rows for b in r['evidence']]);passed=len(rows)>=2 and all(r['label']=='safe' for r in rows) and summary['timeouts']==summary['horizon_exhaustions']==summary['physical_end_reasons']['nonfinite']==0
 new=[];version='descent-compact-'+hashlib.sha256((file_sha256(BASE)+file_sha256(SOURCE)+NAMESPACE).encode()).hexdigest()[:12]
 if passed:
  by={r['id']:r for r in rows}
  for record in source.records:
   audit=by[record['id']];item=copy.deepcopy(record);final=audit['final'];chain=sum(b['chain_success'] for b in audit['evidence']);item.update({'source_phase':'flight','origin_phase':'descent','entry_feature':descent_entry_feature(item['physical_feature'],cfg0).astype(np.float32),'chain':certified_outcome(chain,BRANCHES,cfg0),'final':certified_outcome(final,BRANCHES,cfg0),'policy_version':artifact['policy_identity_hash'],'estimator_version':'event_filter_v1','tube_version':version,'certification_branches':audit['evidence'],'artifact_role':'certified_tube','certified_safe':True,'tube_metrics_eligible':True,'safe_claim_allowed':True});new.append(item)
  combined=union_records(base_bank.records,new);metadata=copy.deepcopy(base_bank.metadata);metadata.update({'last_tube_version':version,'last_policy_version':artifact['policy_identity_hash'],'supersedes':str(BASE.resolve()),'extension_audit_namespace':NAMESPACE,'extension_source_sha256':file_sha256(SOURCE),'branches_per_extension_state':BRANCHES,'continuous_matcher_active':False});SnapshotBank(combined,metadata).save(root/'descent_tube_v3.pkl');handoff=copy.deepcopy(metadata);handoff.update({'entry_bank_role':'canonical_descent_exact_handoff_set','membership_type':'exact_snapshot_identity','radius':None});SnapshotBank(copy.deepcopy(combined),handoff).save(root/'canonical_descent_exact_entry_v2.pkl')
 report={'status':'PASS' if passed else 'FAIL','head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'states':len(rows),'safe_states':sum(r['label']=='safe' for r in rows),'rows':rows,'terminal_summary':summary,'base_states':len(base_bank.records),'combined_states':len(base_bank.records)+len(new),'tube_version':version if passed else None,'tube_path':str(root/'descent_tube_v3.pkl') if passed else None,'tube_sha256':file_sha256(root/'descent_tube_v3.pkl') if passed else None,'exact_handoff_sha256':file_sha256(root/'canonical_descent_exact_entry_v2.pkl') if passed else None,'PPO_authorization':False,'next':'natural_bridge_reprobe_against_tube_v3' if passed else 'targeted_extension_audit_failure'}
 save_json(root/'DESCENT_NATURAL_BRIDGE_INDEPENDENT_AUDIT_V1_REPORT.json',report);save_json(root/'completed.json',{'status':report['status'],'next':report['next']});print(json.dumps({k:report[k] for k in ('status','safe_states','terminal_summary','combined_states','tube_version','tube_sha256','next')},indent=2))
if __name__=='__main__':main()
