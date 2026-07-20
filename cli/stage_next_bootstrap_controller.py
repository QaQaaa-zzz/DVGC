"""Resumable nonblocking stage-to-next-stage bootstrap controller."""
from __future__ import annotations
import json,subprocess,time
from pathlib import Path
from cli.descent_local_controller import Controller,PYTHON
from dvgc.config import file_sha256
from dvgc.runtime import save_json

RUN=Path('runs/stage_next_bootstrap_seed0_20260720')
SUPPORT=RUN/'support_v2/descent_proposal_support_v1.pkl'
PROTOCOL=RUN/'support_v2/apex_to_descent_protocol.json'
RELABEL=RUN/'relabel/apex_existing_5policies.json'
FLIGHT=Path('artifacts/flight_candidates_augmented_v1.pkl')
TAKEOFF=Path('runs/stage_reachability_seed0_20260719/stage_candidate_pilot/takeoff_candidates.pkl')
LANDING_BANK=Path('artifacts/landing_tube.pkl')
LANDING_POLICY=Path('runs/landing/refinement_seed0/policy')
INITIAL=Path('runs/decoupled_bootstrap_seed0_20260720/frozen/pi_f_init')
INPUTS=Path('runs/stage_reachability_seed0_20260719/stage_objective_controller_pilot/inputs')
OLD=Path('runs/decoupled_bootstrap_seed0_20260720/controller_state.json')
POLICIES=[INITIAL]+[Path(f'runs/decoupled_bootstrap_seed0_20260720/flight/curriculum/apex/train/blocks/block_{i}_{i*25600:06d}/policy') for i in range(1,5)]

class StageNextController(Controller):
 def __init__(self,run):
  super().__init__(run)
  if self.state.get('controller_type')!='stage_to_next_stage_bootstrap':
   self.state={'controller_type':'stage_to_next_stage_bootstrap','controller_version':1,'controller_module':'cli.stage_next_bootstrap_controller','controller_unit':'dvgc-stage-next-bootstrap-controller.service','run_id':run.name,'current_stage':'migrate_old_objective','last_completed_action':None,'in_progress_action':None,'expected_outputs':[],'next_decision':'migrate_old_objective','retry_count':0,'heartbeat':time.time(),'stop_reason':None,'terminal_state':None,'research_gate_valid':False,'history':[],'stage_status':{}};self.save()

 def migrate(self):
  old=json.loads(OLD.read_text())
  marker=self.run/'superseded_apex_to_c_l_objective.json'
  if not marker.exists():save_json(marker,{'status':'PRESERVED_SUPERSEDED','old_gate':old.get('stop_reason'),'old_terminal_state':old.get('terminal_state'),'interpretation':'Apex policies did not directly reach C_L; this is not Apex->Descent evidence','old_results_preserved':True})
  save_json(Path('runs/ACTIVE_PIPELINE.json'),{'status':'ACTIVE','activated_at':time.time(),'run_path':str(self.run),'controller_unit':'dvgc-stage-next-bootstrap-controller.service','start_script':'scripts/start_stage_next_bootstrap_controller.sh','supersedes':'runs/decoupled_bootstrap_seed0_20260720','supersession_reason':'STAGE_TO_NEXT_STAGE_DECOUPLED_BOOTSTRAP'})
  self.save(current_stage='verify_support_and_protocol',last_completed_action='migrate_old_objective',next_decision='relabel_existing_apex_policies',terminal_state=None,global_status='stage_reachability_acquisition',apex_old_objective_status='superseded_apex_to_c_l_objective',apex_current_status='pending_apex_to_descent_relabel')

 def verify(self):
  required=[SUPPORT,PROTOCOL,RELABEL,FLIGHT,TAKEOFF,LANDING_BANK,LANDING_POLICY/'params.pkl',INITIAL/'params.pkl',Path('docs/RUNTIME_GATE.json')]
  if any(not p.exists() for p in required):raise RuntimeError('Missing immutable stage-next input')
  subprocess.run([PYTHON,'-m','cli.runtime_gate','--config','configs/default.json','--output','docs/RUNTIME_GATE.json','--check-only'],check=True)
  proto=json.loads(PROTOCOL.read_text());rel=json.loads(RELABEL.read_text())
  if proto['status']!='PASS' or rel['protocol_sha256']!=proto['protocol_sha256']:raise RuntimeError('Frozen detector/relabel mismatch')
  self.save(current_stage='apex_local_decision',last_completed_action='verify_support_and_protocol',next_decision='apex_local_training_if_needed',provenance={'descent_support_sha256':file_sha256(SUPPORT),'protocol_sha256':proto['protocol_sha256'],'flight_bank_sha256':file_sha256(FLIGHT)},apex_relabel=str(RELABEL))

 def evaluate(self,stage,policy,output,seed):
  states='20' if stage=='apex' else '6';entry=output.with_suffix('.entries.pkl');args=[PYTHON,'-u','-m','cli.stage_label_pilot','--takeoff-bank',TAKEOFF,'--flight-bank',FLIGHT,'--landing-bank',LANDING_BANK,'--flight-policy',policy,'--landing-policy',LANDING_POLICY,'--output',output,'--entry-bank',entry,'--states-per-stage',states,'--branches','4','--horizon','200','--only-stage',stage]
  if stage=='apex':args+=['--stage-support-bank',SUPPORT]
  result=self.run_worker_command(f'{stage}_local_eval',args,output.with_suffix('.log'),[output,entry],unit_suffix=f'stage-next-{stage}-eval-{int(time.time())}',preallocate=False)
  if not result['ok']:raise RuntimeError(f'{stage} local evaluation failed: {result}')
  payload=json.loads(output.read_text());labels=payload['labels'];return {'successful_unique_states':sum(x['s']>0 for x in labels),'branch_successes':sum(x['s'] for x in labels),'branches':sum(x['n'] for x in labels),'report':str(output),'policy':str(policy),'policy_hash':file_sha256(Path(policy)/'params.pkl')}

 def train_block(self,stage,block,resume,support=None):
  root=self.run/f'{stage}/blocks/block_{block}_{block*25600:06d}';train=root/'train';evaluation=root/'evaluation.json'
  if not (train/'policy/params.pkl').exists():
   args=[PYTHON,'-u','-m','cli.train','--stage','takeoff' if stage=='takeoff' else 'flight','--bank',INPUTS/stage/'reset_bank.pkl','--config',INPUTS/stage/'config.json','--run',train,'--timesteps','25600','--num-envs','80','--num-eval-envs','40','--num-evals','2','--batch-size','40','--num-minibatches','4','--seed',str({'apex':101,'ascent':102,'takeoff':103}[stage]),'--segment-index',str(block-1),'--resume',resume]
   if support:args+=['--stage-support-bank',support]
   result=self.run_worker_command(f'{stage}_train_block_{block}',args,root/'train.log',[train/'policy/params.pkl',train/'training_metrics.json'],unit_suffix=f'stage-next-{stage}-b{block}-{int(time.time())}',preallocate=True)
   if not result['ok']:
    if result.get('oom'):
     args[args.index('--num-envs')+1]='40';args[args.index('--num-eval-envs')+1]='20';args[args.index('--batch-size')+1]='20'
     result=self.run_worker_command(f'{stage}_train_block_{block}_oom_retry',args,root/'train_retry.log',[train/'policy/params.pkl',train/'training_metrics.json'],unit_suffix=f'stage-next-{stage}-b{block}-retry-{int(time.time())}',preallocate=False)
    if not result['ok']:raise RuntimeError(f'{stage} training failed: {result}')
  summary=self.evaluate(stage,train/'policy',evaluation,9900000+block)
  save_json(root/'summary.json',summary);return summary,train/'policy'

 def apex(self):
  rel=json.loads(RELABEL.read_text());best=max(x['successful_unique_states'] for x in rel['policies'])
  if best>=2:
   self.state['stage_status']['apex']={'status':'existing_policy_support','successful_unique_states':best};self.save(current_stage='apex_label_pilot',last_completed_action='apex_existing_policy_selected',next_decision='apex_label_pilot');return
  resume=INITIAL;previous=best;stagnant=0;best_summary=None
  for block in range(1,5):
   summary,policy=self.train_block('apex',block,resume,SUPPORT);resume=policy;best_summary=summary
   improved=summary['successful_unique_states']>previous;stagnant=0 if improved else stagnant+1;previous=max(previous,summary['successful_unique_states'])
   if summary['successful_unique_states']>=2 or stagnant>=2:break
  self.state['stage_status']['apex']={'status':'proposal_ready' if previous>=2 else 'controller_support_gap','successful_unique_states':previous,'last_summary':best_summary}
  self.save(current_stage='apex_label_pilot',last_completed_action='apex_local_decision',next_decision='apex_label_pilot',apex_current_policy=str(resume))

 def apex_label(self):
  marker=self.run/'apex/label_pilot.json'
  if not marker.exists():
   rel=json.loads(RELABEL.read_text());save_json(marker,{'status':'PASS','artifact_role':'stage_label_pilot','source':str(RELABEL),'unique_states':rel['states'],'branches':len(rel['outcomes']),'controller_count':len(rel['policies']),'proposal_support_only':True,'not_certified_tube':True})
  self.save(current_stage='ascent_local_training',last_completed_action='apex_label_pilot',next_decision='ascent_local_training')

 def independent_stage(self,stage):
  start=INITIAL if stage=='ascent' else Path('runs/stage_reachability_seed0_20260719/stage_objective_controller_pilot/takeoff/block_1_006400/train/policy')
  previous=0;stagnant=0;resume=start;best=None
  for block in range(1,5):
   summary,policy=self.train_block(stage,block,resume);resume=policy;best=summary
   improved=summary['successful_unique_states']>previous;stagnant=0 if improved else stagnant+1;previous=max(previous,summary['successful_unique_states'])
   if previous>=2 or stagnant>=2:break
  self.state['stage_status'][stage]={'status':'proposal_ready' if previous>=2 else 'controller_support_gap','successful_unique_states':previous,'last_summary':best}
  next_stage='takeoff_local_training' if stage=='ascent' else 'controller_proposal_bank_update'
  self.save(current_stage=next_stage,last_completed_action=f'{stage}_local_training',next_decision=next_stage,**{f'{stage}_current_policy':str(resume)})

 def proposal_bank(self):
  output=self.run/'controller_proposal_bank_v2.json';policies=[];attempted=[]
  for stage in ('apex','ascent','takeoff'):
   status=self.state['stage_status'].get(stage,{});policy=self.state.get(f'{stage}_current_policy')
   if policy:
    row={'responsibility':{'apex':'apex_to_descent','ascent':'ascent_to_apex','takeoff':'takeoff_to_ascent'}[stage],'policy':policy,'policy_hash':file_sha256(Path(policy)/'params.pkl'),'evidence':status}
    (policies if status.get('status')=='proposal_ready' else attempted).append(row)
  save_json(output,{'status':'PASS','artifact_role':'controller_proposal_bank','active_policies':policies,'attempted_controller_evidence':attempted,'failed_controllers_excluded_from_teachers':True,'local_failures_nonblocking':True,'not_certified_tube':True})
  self.save(current_stage='takeoff_label_pilot',last_completed_action='controller_proposal_bank_update',next_decision='takeoff_label_pilot_120x4',controller_proposal_bank=str(output),terminal_state=None)

 def finish_takeoff_labels(self):
  report=self.run/'takeoff/label_support/takeoff_label_pilot_120x4.json'
  if not report.exists():time.sleep(30);self.save(in_progress_action='takeoff_label_pilot_120x4',active_worker_unit=None);return
  payload=json.loads(report.read_text());labels=payload['labels'];summary={'unique_states':len(labels),'branches':sum(x['n'] for x in labels),'successful_unique_states':sum(x['s']>0 for x in labels),'branch_successes':sum(x['s'] for x in labels),'label_counts':payload['label_counts'],'success_rates':payload['success_rates'],'termination_reasons':payload['termination_reasons'],'report':str(report),'entry_bank':payload['entry_bank']}
  save_json(self.run/'takeoff/label_support/coverage_analysis.json',summary)
  self.state['stage_status']['takeoff']['label_pilot']=summary
  blocked=[stage for stage in ('apex','ascent') if self.state['stage_status'].get(stage,{}).get('status')=='controller_support_gap']
  self.save(current_stage='gate_pause',last_completed_action='takeoff_label_pilot_120x4',next_decision=None,in_progress_action=None,active_worker_unit=None,terminal_state='gate_pause',research_gate_valid=True,blocked_stage=','.join(blocked),stop_reason='bounded local controller support exhausted for Apex and Ascent after independent Takeoff acquisition',takeoff_label_pilot=str(report))

 def loop(self):
  while True:
   self.save();stage=self.state['current_stage']
   if stage=='migrate_old_objective':self.migrate()
   elif stage=='verify_support_and_protocol':self.verify()
   elif stage=='apex_local_decision':self.apex()
   elif stage=='apex_label_pilot':self.apex_label()
   elif stage=='ascent_local_training':self.independent_stage('ascent')
   elif stage=='takeoff_local_training':self.independent_stage('takeoff')
   elif stage=='controller_proposal_bank_update':self.proposal_bank()
   elif stage=='takeoff_label_pilot':self.finish_takeoff_labels()
   elif stage=='gate_pause':return 40
   else:raise RuntimeError(f'Unknown stage {stage}')

def main():
 c=StageNextController(RUN)
 try:raise SystemExit(c.loop())
 except SystemExit:raise
 except Exception as exc:c.save(retry_count=int(c.state.get('retry_count',0))+1,stop_reason=f'{type(exc).__name__}: {exc}',terminal_state='engineering_failure_after_retries' if int(c.state.get('retry_count',0))+1>=3 else None);raise
if __name__=='__main__':main()
