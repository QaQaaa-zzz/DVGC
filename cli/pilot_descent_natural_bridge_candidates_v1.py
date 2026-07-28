"""Outcome-blind candidate pilot near the natural Descent bridge state."""
from __future__ import annotations
import argparse,copy,json,pickle,subprocess
from pathlib import Path
import jax.numpy as jnp,numpy as np
from cli.run_backward_descent_nominal_pilot import C_L,PI_D,PI_L,_load_record
from cli.run_backward_descent_rsi_pilot import certify_policy
from cli.run_descent_localized_consolidation_v1 import verified_assets_allowing_runtime_gate_refresh
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256,load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.entry import robust_normalization
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json

TUBE=Path('runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1/descent_tube_v2.pkl');EXPERT=Path('runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter')
INDEX=Path('runs/backward_recovery_tube_fast_track_v1/proposal_state_index.json');NATURAL=Path('runs/natural_compact_descent_bridge_v1/fixed_handoff_probe/NATURAL_COMPACT_DESCENT_BRIDGE_V1_REPORT.json')
PRIOR=Path('runs/descent_compact_matcher_neighborhood_v1/construction_24_adaptive/construction_bank.pkl');DEFAULT_RUN=Path('runs/descent_natural_bridge_candidates_v1/pilot_12x_p1')

def select_targeted(rows,per_region=4):
 selected=[];parents=set()
 for region in ('early','middle','late'):
  for row in sorted((r for r in rows if r['region']==region),key=lambda r:(r['target_distance'],r['candidate_id'],r['proposal_id'])):
   if row['candidate_id'] in parents:continue
   selected.append(row);parents.add(row['candidate_id'])
   if sum(r['region']==region for r in selected)==per_region:break
  if sum(r['region']==region for r in selected)!=per_region:raise ValueError(f'insufficient {region} parent diversity')
 return selected

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default=str(DEFAULT_RUN));a=p.parse_args();root=Path(a.run)
 if root.exists():raise SystemExit(f'refusing overwrite {root}')
 valid,failed,raw=verified_assets_allowing_runtime_gate_refresh()
 if not valid:raise SystemExit(f'frozen asset mismatch: {failed}; raw={raw}')
 gate=json.loads(Path('docs/RUNTIME_GATE.json').read_text())
 if gate.get('status')!='PASS' or gate.get('source_fingerprint')!=source_fingerprint(Path.cwd()):raise SystemExit('runtime gate stale')
 cfg=load_config('configs/backward_descent_rsi_pilot_v1.json',{'use_bank_resets':False,'expert_chain_termination':False,'domain_randomization':False,'obs_noise_enable':False});tube=SnapshotBank.load(TUBE);prior=SnapshotBank.load(PRIOR)
 target=np.asarray(json.loads(NATURAL.read_text())['nominal']['closest']['feature'],float);features=np.asarray([r['entry_feature'] for r in tube.records],float);_,scale=robust_normalization(features,cfg.descent_entry_scale_floors);excluded={r['id'] for r in prior.records}
 pool=[]
 for row in json.loads(INDEX.read_text())['rows']:
  if row['proposal_id'] in excluded:continue
  record=_load_record(row);feature=descent_entry_feature(record['physical_feature'],cfg);pool.append({**row,'target_distance':float(np.linalg.norm((feature-target)/scale))})
 selected=select_targeted(pool);root.mkdir(parents=True);artifact=pickle.loads((EXPERT/'adapter.pkl').read_bytes())
 save_json(root/'manifest.json',{'status':'FROZEN_BEFORE_OUTCOMES','target_source':str(NATURAL),'target_source_sha256':file_sha256(NATURAL),'target_feature':target.tolist(),'selection':'four nearest globally parent-distinct states per region; prior 24 excluded','rows':[{k:r[k] for k in ('proposal_id','candidate_id','region','target_distance','physical_state_sha256')} for r in selected]})
 save_json(root/'cost_estimate.json',{'estimated_seconds':900,'states':12,'rollouts_per_state':'2 exact + 4 P1 micro','PPO_steps':0,'new_search':False})
 dp,_,_=load_bundle(PI_D,verify_files=True);lp,_,_=load_bundle(PI_L,verify_files=True);env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L))
 adapter=compact_observation_command_adapter(jnp.asarray(artifact['prototypes']),jnp.asarray(artifact['targets']),jnp.asarray(artifact['normalizer_mean']),jnp.asarray(artifact['normalizer_std']),float(artifact['radius']),float(artifact['core_radius']))
 nodes=[{'node_id':r['proposal_id'],'candidate_id':r['candidate_id'],'layer':r['shell_layer'],'region':r['region'],'source_state_hash':r['physical_state_sha256'],'physical_state':r,'parent_node_id':r['nearest_downstream_node_id']} for r in selected]
 cert=certify_policy(env,dp,lp,nodes,3_600_000_000,record_loader=lambda node:_load_record(node['physical_state']),descent_action_adapter=adapter,policy_identity_hash=artifact['policy_identity_hash']);save_json(root/'certification.json',cert)
 p1={r['node_id'] for r in cert['rows'] if r['P1']['pass']};p0={r['node_id'] for r in cert['rows'] if r['P0']['pass']};accepted=[]
 for r in selected:
  if r['proposal_id'] not in p1:continue
  item=copy.deepcopy(_load_record(r));item.update({'id':r['proposal_id'],'artifact_role':'proposal_support_bank','safe_claim_allowed':False,'bootstrap_eligible':True,'target_distance':r['target_distance'],'policy_identity_hash':artifact['policy_identity_hash']});accepted.append(item)
 out=root/'natural_bridge_proposal_support.pkl';SnapshotBank(accepted,{'artifact_role':'proposal_support_bank','safe_claim_allowed':False,'policy_identity_hash':artifact['policy_identity_hash'],'source_target_sha256':file_sha256(NATURAL)}).save(out)
 parents=len({r['candidate_id'] for r in selected if r['proposal_id'] in p1});status='PASS' if len(p1)>=2 and parents>=2 else 'FAIL';report={'status':status,'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'states':12,'P0':len(p0),'P1':len(p1),'P1_parents':parents,'regions':{region:{'P0':sum(r['region']==region and r['proposal_id'] in p0 for r in selected),'P1':sum(r['region']==region and r['proposal_id'] in p1 for r in selected)} for region in ('early','middle','late')},'minimum_target_distance':min(r['target_distance'] for r in selected),'proposal_bank':str(out),'proposal_bank_sha256':file_sha256(out),'PPO_authorization':False,'next':'adaptive_natural_bridge_candidate_expansion' if status=='PASS' else 'candidate_pool_bridge_support_exhausted'}
 save_json(root/'DESCENT_NATURAL_BRIDGE_CANDIDATE_PILOT_V1_REPORT.json',report);save_json(root/'completed.json',{'status':status,'next':report['next']});print(json.dumps(report,indent=2))
if __name__=='__main__':main()
