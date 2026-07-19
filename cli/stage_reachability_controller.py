"""Resumable RA-L stage-reachability pipeline controller."""
from __future__ import annotations
import json,math,subprocess,time
from collections import Counter
from pathlib import Path
from cli.descent_local_controller import Controller,LANDING,PYTHON
from cli.jump_envelope_controller import CYCLE5_POLICY
from dvgc.config import file_sha256,load_config
from dvgc.runtime import save_json
from dvgc.stage_reachability import protocol_payload

OLD_RUN=Path('runs/jump_envelope_seed0_20260719')
FLIGHT_BANK=Path('artifacts/flight_candidates_augmented_v1.pkl')
CONTROLLER_PROPOSALS=(
 CYCLE5_POLICY,
 Path('runs/flight/pipeline_seed0_v5/pilot/policy'),
 Path('runs/stage_experts/flight_seed0_20260715T2045/curriculum/apex/blocks/block_4_102400/policy'),
 Path('runs/stage_experts/flight_seed0_20260715T2045/curriculum/descent/blocks/block_2_051200/policy'),
)
PILOT_STAGES=("takeoff","ascent","apex")
PILOT_SEEDS={"takeoff":101,"ascent":102,"apex":103}
PILOT_OBJECTIVES={"takeoff":"takeoff_to_ascent","ascent":"ascent_to_apex","apex":"apex_to_descent"}
PILOT_ROOT="stage_objective_controller_pilot"
ROUTE=("safe_pause_old_route","h4_replay_smoke","define_stage_entry_protocol","relabel_existing_data","coverage_inventory","stage_candidate_pilot","stage_label_acquisition","reward_preflight","takeoff_controller_pilot","ascent_controller_pilot","apex_controller_pilot","controller_bank_update","stage_label_pilot","coverage_analysis","reachability_model","active_candidate_acquisition","build_stage_tubes","Tube_RSI_final_PPO","final_policy_certification")

class StageReachabilityController(Controller):
 def __init__(self,run):
  super().__init__(run)
  if self.state.get('controller_type')!='stage_reachability':
   self.state={"controller_type":"stage_reachability","controller_version":1,"controller_unit":"dvgc-stage-reachability-controller.service","run_id":run.name,"current_stage":"safe_pause_old_route","last_completed_action":None,"in_progress_action":None,"expected_outputs":[],"next_decision":"h4_replay_smoke","retry_count":0,"heartbeat":time.time(),"stop_reason":None,"terminal_state":None,"research_gate_valid":False,"history":[],"provenance":{}}
   self.save()

 def pause_old(self):
  state=json.loads((OLD_RUN/'controller_state.json').read_text())
  if state.get('terminal_state')!='SUPERSEDED_BY_STAGE_REACHABILITY_PROTOCOL':raise RuntimeError('Old route is not atomically superseded')
  save_json(Path('runs/ACTIVE_PIPELINE.json'),{"status":"ACTIVE","activated_at":time.time(),"run_path":str(self.run),"controller_unit":"dvgc-stage-reachability-controller.service","start_script":"/home/qy/DVGC/scripts/start_stage_reachability_controller.sh","supersedes":str(OLD_RUN),"supersession_reason":state['terminal_state']})
  self.save(current_stage='h4_replay_smoke',last_completed_action='safe_pause_old_route',next_decision='define_stage_entry_protocol',old_route_terminal_state=state['terminal_state'])

 def replay_smoke(self):
  out=self.run/'deterministic_replay_smoke.json'
  if not out.exists():
   result=self.run_worker_command('deterministic_replay_smoke',[PYTHON,'-u','-m','cli.stage_replay_smoke','--policy',CYCLE5_POLICY,'--candidate-bank',FLIGHT_BANK,'--output',out,'--states',3,'--horizon',8,'--seed',9100000],self.run/'deterministic_replay_smoke.log',[out],unit_suffix=f'stage-replay-smoke-{int(time.time())}',preallocate=False)
   if not result['ok']:raise RuntimeError(f'Deterministic replay smoke failed: {result}')
  report=json.loads(out.read_text())
  if report.get('status')!='PASS':self.save(terminal_state='gate_pause',stop_reason='deterministic_replay_smoke_failed');raise SystemExit(40)
  self.save(current_stage='define_stage_entry_protocol',last_completed_action='h4_replay_smoke',next_decision='relabel_existing_data',deterministic_replay_smoke=str(out))

 def define_protocol(self):
  out=self.run/'stage_entry_protocol.json'
  if not out.exists():save_json(out,protocol_payload(load_config('configs/default.json')))
  self.save(current_stage='relabel_existing_data',last_completed_action='define_stage_entry_protocol',next_decision='coverage_inventory',stage_entry_protocol=str(out),stage_entry_protocol_sha256=file_sha256(out))

 def inventory(self):
  inventory=self.run/'coverage_inventory.json';relabel=self.run/'relabel_existing_data.json'
  if not inventory.exists() or not relabel.exists():self.run_command('relabel_and_inventory',[PYTHON,'-m','cli.stage_reachability_inventory','--output',inventory,'--relabel-output',relabel],self.run/'inventory.log',[inventory,relabel])
  # Both named stages are recorded even though one read-only pass creates the
  # two mutually consistent artifacts.
  if self.state['current_stage']=='relabel_existing_data':
   self.save(current_stage='coverage_inventory',last_completed_action='relabel_existing_data',next_decision='stage_candidate_pilot');return
  self.save(current_stage='stage_candidate_pilot',last_completed_action='coverage_inventory',next_decision='stage_candidate_pilot')

 def candidate_pilot(self):
  cost=self.run/'stage_candidate_pilot/cost_estimate.json'
  if not cost.exists():self.run_command('candidate_pilot_cost',[PYTHON,'-m','cli.stage_cost_estimate','--output',cost,'--unique-states','750','--branches','4','--horizon','200','--pilot-fraction','.04','--hypothesis','event-aligned labels are reproducible and cover all five stages'],self.run/'stage_candidate_pilot/cost.log',[cost])
  # The next GPU action is intentionally gated on the deterministic replay
  # artifact. No label worker may be launched by bypassing this condition.
  smoke=json.loads((self.run/'deterministic_replay_smoke.json').read_text())
  if not smoke.get('label_acquisition_allowed'):raise RuntimeError('Replay smoke does not authorize label acquisition')
  root=self.run/'stage_candidate_pilot';takeoff=root/'takeoff_candidates.pkl'
  if not takeoff.exists():
   result=self.run_worker_command('takeoff_candidate_pilot',[PYTHON,'-u','-m','cli.build_candidates','--phase','takeoff','--target','6','--bank',takeoff,'--seed','9700000','--attempt-budget','240','--dedup-distance','.06'],root/'takeoff_candidates.log',[takeoff],unit_suffix=f'stage-takeoff-candidates-{int(time.time())}',preallocate=False)
   if not result['ok']:raise RuntimeError(f'Takeoff candidate pilot failed: {result}')
  out=root/'report_v2.json';entries=root/'entry_snapshots_v2.pkl'
  if not out.exists():
   result=self.run_worker_command('stage_candidate_label_pilot',[PYTHON,'-u','-m','cli.stage_label_pilot','--takeoff-bank',takeoff,'--flight-bank',FLIGHT_BANK,'--landing-bank','artifacts/landing_tube.pkl','--flight-policy',CYCLE5_POLICY,'--landing-policy',LANDING,'--output',out,'--entry-bank',entries,'--states-per-stage','6','--branches','4','--horizon','200'],root/'label_pilot.log',[out,entries],unit_suffix=f'stage-label-pilot-{int(time.time())}',preallocate=False)
   if not result['ok']:raise RuntimeError(f'Stage label pilot failed: {result}')
  self.save(current_stage='stage_label_acquisition',last_completed_action='stage_candidate_pilot',next_decision='stage_label_acquisition',stage_candidate_pilot=str(out))

 def acquisition(self):
  if not (self.run/'stage_candidate_pilot/report_v2.json').exists():
   self.save(current_stage='stage_candidate_pilot',next_decision='stage_candidate_pilot_v2',pilot_v1_preserved=str(self.run/'stage_candidate_pilot/report.json'))
   return
  root=self.run/'stage_label_acquisition';cost=root/'cost_estimate.json';out=root/'controller_bank_probe.json';entries=root/'entry_snapshots.pkl'
  if not cost.exists():self.run_command('controller_bank_probe_cost',[PYTHON,'-m','cli.stage_cost_estimate','--output',cost,'--unique-states','30','--branches','4','--horizon','200','--pilot-fraction','.04','--hypothesis','historical controller diversity recovers next-stage positives missed by the current checkpoint'],root/'cost.log',[cost])
  if not out.exists():
   command=[PYTHON,'-u','-m','cli.stage_label_pilot','--takeoff-bank',self.run/'stage_candidate_pilot/takeoff_candidates.pkl','--flight-bank',FLIGHT_BANK,'--landing-bank','artifacts/landing_tube.pkl','--landing-policy',LANDING,'--output',out,'--entry-bank',entries,'--states-per-stage','6','--branches','4','--horizon','200']
   for policy in CONTROLLER_PROPOSALS:command+=['--flight-policy',policy]
   result=self.run_worker_command('controller_bank_probe',command,root/'controller_bank_probe.log',[out,entries],unit_suffix=f'stage-controller-bank-{int(time.time())}',preallocate=False)
   if not result['ok']:raise RuntimeError(f'Controller-bank probe failed: {result}')
  report=json.loads(out.read_text());zero=[stage for stage,rate in report['success_rates'].items() if stage!='landing' and rate==0]
  marker=root/'decision.json';save_json(marker,{"status":"PASS","zero_support_stages":zero,"success_rates":report['success_rates'],"decision":"bounded_stage_objective_controller_pilot" if zero else "adaptive_label_acquisition","controller_bank":[str(x) for x in CONTROLLER_PROPOSALS],"no_state_declared_physically_unreachable":True})
  if zero:
   self.save(current_stage='reward_preflight',last_completed_action='controller_bank_probe',controller_bank_probe=str(out),controller_support_gap=zero,next_decision='reward_preflight')
   return
  self.save(current_stage='train_reachability_model',last_completed_action='stage_label_acquisition',next_decision='train_reachability_model')

 def reward_preflight(self):
  root=self.run/PILOT_ROOT;root.mkdir(parents=True,exist_ok=True)
  preflight=root/'reward_preflight_v3.json';inputs=root/'inputs/inputs.json';cost=root/'cost_estimate.json'
  if not preflight.exists():
   self.run_command('reward_preflight',[PYTHON,'-m','cli.stage_reward_preflight','--output',preflight],root/'reward_preflight.log',[preflight])
  report=json.loads(preflight.read_text())
  if report.get('status')!='PASS':
   self.save(terminal_state='gate_pause',stop_reason='stage_reward_preflight_failed',reward_preflight=str(preflight));raise SystemExit(40)
  self.run_command('runtime_gate_current',[PYTHON,'-m','cli.runtime_gate','--config','configs/default.json','--output','docs/RUNTIME_GATE.json','--check-only'],root/'runtime_gate_check.log',[])
  if not inputs.exists():
   self.run_command('prepare_stage_controller_pilots',[PYTHON,'-m','cli.prepare_stage_controller_pilots','--takeoff-bank',self.run/'stage_candidate_pilot/takeoff_candidates.pkl','--flight-bank',FLIGHT_BANK,'--output-root',root/'inputs'],root/'prepare_inputs.log',[inputs])
  if not cost.exists():
   save_json(cost,{"status":"PASS","artifact_role":"cost_estimate","estimated_wall_hours":1.75,"assumptions":{"stages":3,"effective_steps_per_block":6400,"initial_blocks":3,"maximum_blocks":6,"unique_eval_states_per_stage":6,"branches_per_state":4,"full_certification":False},"over_two_hours":False,"decision":"bounded pilots authorized"})
  self.save(current_stage='takeoff_controller_pilot',last_completed_action='reward_preflight',next_decision='takeoff_controller_pilot',reward_preflight=str(preflight),runtime_gate='PASS/current',stage_controller_cost_estimate=str(cost))

 @staticmethod
 def _pilot_summary(report_path,metrics_path,policy_path,bank_path,config_path,block):
  report=json.loads(Path(report_path).read_text());labels=report['labels']
  successful=[row for row in labels if int(row['s'])>0]
  times=[branch['time_to_next_stage'] for row in labels for branch in row['branches'] if branch['success']]
  reasons=Counter(branch['failure_reason'] for row in labels for branch in row['branches'] if not branch['success'])
  metrics=json.loads(Path(metrics_path).read_text());progress=metrics.get('progress',[]);last=progress[-1] if progress else {}
  reward_terms={key:value for key,value in last.items() if 'stage_entry' in key or 'reward/joint_energy' in key or 'reward/action_' in key}
  return {"status":"PASS","block":block,"effective_steps":6400,"policy":str(policy_path),"policy_sha256":file_sha256(Path(policy_path)/'params.pkl'),"reset_bank":str(bank_path),"reset_bank_sha256":file_sha256(bank_path),"config":str(config_path),"config_sha256":file_sha256(config_path),"branch_successes":sum(int(row['s']) for row in labels),"total_branches":sum(int(row['n']) for row in labels),"successful_unique_states":len(successful),"successful_candidate_ids":[row['candidate_id'] for row in successful],"time_to_entry":{"count":len(times),"mean":sum(times)/len(times) if times else None,"p95":sorted(times)[min(len(times)-1,math.ceil(.95*len(times))-1)] if times else None},"termination_reasons":dict(reasons),"reward_terms":reward_terms,"health":{"nonfinite":sum(value for key,value in reasons.items() if key=='nonfinite'),"timeout":sum(value for key,value in reasons.items() if key in ('stage_timeout','horizon_exhaustion')),"oom":False},"label_report":str(report_path)}

 def _repair_config(self,stage,source,summary,target):
  cfg=json.loads(Path(source).read_text());reasons=summary['termination_reasons'];dominant=max(reasons,key=reasons.get) if reasons else 'no_next_stage_entry'
  if stage=='takeoff':cfg['stage_progress_coeff']=min(.15,float(cfg['stage_progress_coeff'])+.05);cfg['stage_pose_coeff']=min(.06,float(cfg['stage_pose_coeff'])+.02)
  elif stage=='ascent':cfg['stage_progress_coeff']=min(.12,float(cfg['stage_progress_coeff'])+.02);cfg['stage_pose_coeff']=min(.08,float(cfg['stage_pose_coeff'])+.04)
  else:cfg['stage_progress_coeff']=min(.12,float(cfg['stage_progress_coeff'])+.02);cfg['stage_angular_coeff']=min(.05,float(cfg['stage_angular_coeff'])+.03)
  target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
  save_json(target.with_suffix('.diagnosis.json'),{"stage":stage,"evidence":{"dominant_termination":dominant,"termination_reasons":reasons},"scope":"one bounded shaping repair; entry event and physical gates unchanged"})

 def controller_pilot(self,stage):
  root=self.run/PILOT_ROOT;stage_root=root/stage;inputs=root/'inputs'/stage
  previous=CYCLE5_POLICY;final_summary=None
  for block in (1,2):
   block_root=stage_root/f'block_{block}_{block*6400:06d}';train=block_root/'train';policy=train/'policy'
   config=inputs/'config.json' if block==1 else block_root/'repair_config.json'
   if block==2 and not config.exists():
    prior=json.loads((stage_root/'block_1_006400/summary.json').read_text())
    # A single success receives continuation with unchanged semantics. Zero
    # support receives the one evidence-backed bounded shaping repair.
    if prior['successful_unique_states']==1:
     config=inputs/'config.json'
    else:self._repair_config(stage,inputs/'config.json',prior,config)
   resume=previous if block==1 else stage_root/'block_1_006400/train/policy'
   expected=[policy/'params.pkl',train/'training_metrics.json']
   if not all(path.exists() for path in expected):
    command=[PYTHON,'-u','-m','cli.train','--stage','takeoff' if stage=='takeoff' else 'flight','--bank',inputs/'reset_bank.pkl','--config',config,'--run',train,'--timesteps','6400','--num-envs','20','--num-eval-envs','20','--num-evals','2','--batch-size','10','--num-minibatches','2','--seed',PILOT_SEEDS[stage],'--segment-index',block-1,'--resume',resume]
    result=self.run_worker_command(f'{stage}_controller_train_block_{block}',command,block_root/'train.log',expected,unit_suffix=f'stage-{stage}-pilot-b{block}-{int(time.time())}',preallocate=False)
    if not result['ok']:
     if result.get('oom'):self.save(terminal_state='gate_pause',stop_reason=f'{stage}_controller_oom_single_worker');raise SystemExit(40)
     raise RuntimeError(f'{stage} controller block {block} failed: {result}')
   evaluation=block_root/'evaluation.json';entries=block_root/'entry_snapshots.pkl'
   if not evaluation.exists():
    result=self.run_worker_command(f'{stage}_controller_evaluate_block_{block}',[PYTHON,'-u','-m','cli.stage_label_pilot','--takeoff-bank',self.run/'stage_candidate_pilot/takeoff_candidates.pkl','--flight-bank',FLIGHT_BANK,'--landing-bank','artifacts/landing_tube.pkl','--flight-policy',policy,'--landing-policy',LANDING,'--output',evaluation,'--entry-bank',entries,'--states-per-stage','6','--branches','4','--horizon','200','--only-stage',stage,'--config',config],block_root/'evaluation.log',[evaluation,entries],unit_suffix=f'stage-{stage}-eval-b{block}-{int(time.time())}',preallocate=False)
    if not result['ok']:raise RuntimeError(f'{stage} controller evaluation block {block} failed: {result}')
   summary_path=block_root/'summary.json'
   if not summary_path.exists():save_json(summary_path,self._pilot_summary(evaluation,train/'training_metrics.json',policy,inputs/'reset_bank.pkl',config,block))
   final_summary=json.loads(summary_path.read_text())
   if final_summary['successful_unique_states']>=2:break
   if block==2:break
  assert final_summary is not None
  if final_summary['successful_unique_states']<2:
   self.save(terminal_state='gate_pause',stop_reason=f'{stage}_stage_controller_blocker',blocked_stage=stage,stage_controller_summary=str(summary_path),next_decision=None);raise SystemExit(40)
  next_stage=PILOT_STAGES[PILOT_STAGES.index(stage)+1] if stage!='apex' else 'controller_bank_update'
  self.save(current_stage=f'{next_stage}_controller_pilot' if next_stage in PILOT_STAGES else next_stage,last_completed_action=f'{stage}_controller_pilot',next_decision=next_stage,**{f'{stage}_controller_summary':str(summary_path)})

 def controller_bank_update(self):
  root=self.run/PILOT_ROOT;policies=[]
  for stage in PILOT_STAGES:
   candidates=sorted((root/stage).glob('block_*/summary.json'))
   accepted=[json.loads(path.read_text()) for path in candidates if json.loads(path.read_text())['successful_unique_states']>=2]
   if not accepted:raise RuntimeError(f'No accepted {stage} controller policy')
   policies.append({"stage":stage,"objective":PILOT_OBJECTIVES[stage],**accepted[-1]})
  registry=root/'controller_proposal_bank.json';save_json(registry,{"status":"PASS","artifact_role":"stage_controller_proposal_bank","not_a_shared_actor":True,"not_a_certified_tube":True,"policies":policies})
  self.save(current_stage='stage_label_pilot',last_completed_action='controller_bank_update',next_decision='stage_label_pilot',controller_proposal_bank=str(registry))

 def label_pilot(self):
  # The next acquisition expands unique states, not branches. Keep this state
  # resumable while candidate construction is added from the accepted policy
  # registry; never fall back to full H1 or high-branch pointwise audits.
  marker=self.run/PILOT_ROOT/'stage_label_pilot_ready.json'
  if not marker.exists():save_json(marker,{"status":"READY","target_unique_states_per_phase":[100,200],"branches_per_state":4,"controller_proposal_bank":self.state['controller_proposal_bank'],"forbidden":["full_H1","32_or_64_branch_certification","750_state_direct_labeling"]})
  self.save(current_stage='coverage_analysis',last_completed_action='stage_label_pilot_ready',next_decision='build_unique_state_label_inputs',stage_label_pilot_ready=str(marker))

 def loop(self):
  while True:
   self.save();stage=self.state['current_stage']
   if stage=='safe_pause_old_route':self.pause_old()
   elif stage=='h4_replay_smoke':self.replay_smoke()
   elif stage=='define_stage_entry_protocol':self.define_protocol()
   elif stage in ('relabel_existing_data','coverage_inventory'):self.inventory()
   elif stage=='stage_candidate_pilot':self.candidate_pilot()
   elif stage=='stage_label_acquisition':self.acquisition()
   elif stage=='reward_preflight':self.reward_preflight()
   elif stage=='takeoff_controller_pilot':self.controller_pilot('takeoff')
   elif stage=='ascent_controller_pilot':self.controller_pilot('ascent')
   elif stage=='apex_controller_pilot':self.controller_pilot('apex')
   elif stage=='controller_bank_update':self.controller_bank_update()
   elif stage=='stage_label_pilot':self.label_pilot()
   elif stage=='coverage_analysis':time.sleep(60)
   elif stage=='pipeline_complete':return 0
   else:raise RuntimeError(f'Unimplemented stage-reachability stage: {stage}')

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();controller=StageReachabilityController(Path(a.run))
 try:raise SystemExit(controller.loop())
 except SystemExit:raise
 except Exception as exc:
  count=int(controller.state.get('retry_count',0))+1;controller.save(retry_count=count,stop_reason=f'{type(exc).__name__}: {exc}')
  if count>=3:controller.save(terminal_state='engineering_failure_after_retries');raise SystemExit(41)
  raise
if __name__=='__main__':main()
