"""Persistent controller for handoff-first seed-0 jump-envelope completion."""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path
from cli.descent_local_controller import Controller, ENTRY, LANDING, PYTHON, RUNTIME_GATE
from cli.handoff_decomposition_shard import _events
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json

ROOT=Path('runs/stage_experts/trajectory_mining_resume_seed0_20260718T194327')
CANDIDATES=ROOT/'trajectory_mining_corrected/candidate_pool.pkl'
PHYSICAL_AUDIT=ROOT/'trajectory_mining_corrected/physical_audit.json'
CYCLE4_REPORT=ROOT/'cycle_4/stable/report.json';CYCLE5_REPORT=ROOT/'cycle_5/stable/report.json'
CYCLE4_POLICY=Path('runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy')
CYCLE4_SOURCE_ORIGIN=Path('runs/stage_experts/descent_tube_seed0_20260716T2330/round_2/train/policy')
CYCLE5_POLICY=ROOT/'roll_targeted/train/policy'
LANDING_BANK=Path('artifacts/landing_tube.pkl');LANDING_AUDIT=Path('runs/landing/refinement_seed0/audit.json')
XML=Path('assets/orange_bike_4kg_horizontal.xml');SHARD=24

class JumpEnvelopeController(Controller):
 def __init__(self,run):
  super().__init__(run)
  if self.state.get('controller_type')!='jump_envelope':
   self.state={'controller_type':'jump_envelope','controller_version':1,'controller_unit':'dvgc-jump-envelope-controller.service','controller_module':'cli.jump_envelope_controller','run_id':run.name,'current_stage':'inspect','last_completed_action':None,'in_progress_action':None,'expected_outputs':[],'next_decision':'handoff_cycle4','retry_count':0,'heartbeat':time.time(),'stop_reason':None,'terminal_state':None,'research_gate_valid':False,'history':[],'provenance':{},'active_worker_unit':None}
   self.save()

 def inspect(self):
  if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise RuntimeError('Tracked worktree must be clean')
  required=[CANDIDATES,PHYSICAL_AUDIT,CYCLE4_REPORT,CYCLE5_REPORT,CYCLE4_POLICY/'params.pkl',CYCLE5_POLICY/'params.pkl',LANDING/'params.pkl',ENTRY,LANDING_BANK,LANDING_AUDIT,XML,RUNTIME_GATE]
  missing=[str(x) for x in required if not x.exists()]
  if missing:raise RuntimeError(f'Missing immutable inputs: {missing}')
  subprocess.run([PYTHON,'-m','cli.runtime_gate','--config','configs/default.json','--output',RUNTIME_GATE,'--check-only'],check=True,stdout=subprocess.DEVNULL)
  gate=json.loads(RUNTIME_GATE.read_text());
  if gate.get('status')!='PASS':raise RuntimeError('Runtime gate is not PASS/current')
  self.save(provenance={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'xml_sha256':file_sha256(XML),'candidate_bank_sha256':file_sha256(CANDIDATES),'cycle4_policy_hash':file_sha256(CYCLE4_POLICY/'params.pkl'),'cycle5_policy_hash':file_sha256(CYCLE5_POLICY/'params.pkl'),'landing_policy_hash':file_sha256(LANDING/'params.pkl'),'canonical_c_l_sha256':file_sha256(ENTRY),'canonical_c_l_radius':SnapshotBank.load(ENTRY).metadata['entry_matcher']['radius'],'runtime_source_fingerprint':gate['source_fingerprint']},current_entry_bank=str(ENTRY),current_stage='handoff_cycle4',last_completed_action='inspect',next_decision='handoff_cycle5')

 def handoff(self,label):
  policy,report,comparison,seed=(CYCLE4_POLICY,CYCLE4_REPORT,None,2_100_000_000) if label=='cycle4' else (CYCLE5_POLICY,CYCLE5_REPORT,CYCLE4_REPORT,2_200_000_000)
  payload=json.loads(report.read_text());base=json.loads(comparison.read_text()) if comparison else None;total=len(_events(payload,base));root=self.run/'handoff'/label;root.mkdir(parents=True,exist_ok=True)
  reports=[];banks=[]
  for start in range(0,total,SHARD):
   end=min(start+SHARD,total);out=root/f'shard_{start:05d}_{end:05d}.json';bank=root/f'shard_{start:05d}_{end:05d}.pkl';reports.append(out);banks.append(bank)
   if out.exists() and bank.exists():continue
   cmd=[PYTHON,'-u','-m','cli.handoff_decomposition_shard','--policy-label',label,'--policy',policy,'--stable-report',report,'--candidate-bank',CANDIDATES,'--landing-policy',LANDING,'--entry-bank',ENTRY,'--start-event',start,'--end-event',end,'--landing-seed',seed,'--output',out,'--proposal-bank',bank]
   if comparison:cmd+=['--comparison-report',comparison]
   result=self.run_worker_command(f'handoff_{label}_{start}_{end}',cmd,root/f'shard_{start:05d}_{end:05d}.log',[out,bank],unit_suffix=f'handoff-{label}-{start}-{end}-{int(time.time())}',preallocate=False)
   if not result['ok']:raise RuntimeError(f'Handoff shard failed: {result}')
  merged=root/'report.json';pending=root/'pending_entries.pkl'
  if not merged.exists():
   cmd=[PYTHON,'-m','cli.merge_handoff_decomposition','--entry-bank',ENTRY,'--output',merged,'--proposal-bank',pending]
   for x in reports:cmd+=['--shard',x]
   for x in banks:cmd+=['--proposal-shard',x]
   self.run_command(f'merge_handoff_{label}',cmd,root/'merge.log',[merged,pending])
  self.save(current_stage='handoff_cycle5' if label=='cycle4' else 'handoff_combine',last_completed_action=f'handoff_{label}',next_decision='handoff_combine' if label=='cycle5' else 'handoff_cycle5')

 def combine(self):
  root=self.run/'handoff';out=root/'report.json';pending=root/'pending_entries.pkl'
  if not out.exists():self.run_command('combine_handoff',[PYTHON,'-m','cli.combine_handoff_decomposition','--report',root/'cycle4/report.json','--report',root/'cycle5/report.json','--proposal-bank-input',root/'cycle4/pending_entries.pkl','--proposal-bank-input',root/'cycle5/pending_entries.pkl','--entry-bank',ENTRY,'--output',out,'--proposal-bank',pending],root/'combine.log',[out,pending])
  report=json.loads(out.read_text());self.save(handoff_report=str(out),pending_entry_bank=str(pending),handoff_classes=report['summary']['classes'],current_stage='entry_extension_certify' if report['new_entry_extension_eligible'] else 'landing_calibration_stage_a',next_decision='landing_calibration_stage_a')

 def fast_prepare(self):
  root=self.run/'handoff/fast_route';bank=root/'h1_selected.pkl';report=root/'selection.json';indices=root/'h4_indices.json'
  if not report.exists():self.run_command('prepare_fast_handoff',[PYTHON,'-m','cli.prepare_fast_handoff_route','--shard-root',self.run/'handoff/cycle4','--entry-bank',ENTRY,'--output-bank',bank,'--output-report',report,'--h4-indices',indices],root/'prepare.log',[bank,report,indices])
  self.save(pending_entry_bank=str(bank),fast_handoff_selection=str(report),current_stage='h4_formal_replay',last_completed_action='prepare_fast_handoff',next_decision='h4_analyze')

 def h4_replay(self):
  root=self.run/'handoff/fast_route';indices=json.loads((root/'h4_indices.json').read_text());out=root/'h4_formal_replay.json'
  if not indices:self.save(current_stage='h1_independent_certify',next_decision='h1_entry_decision');return
  if not out.exists():
   result=self.run_worker_command('h4_formal_replay',[PYTHON,'-u','-m','cli.certify_stable_descent_shard','--stage','stage_a','--descent-policy',CYCLE4_POLICY,'--candidate-source-policy',CYCLE4_SOURCE_ORIGIN,'--candidate-source-policy',CYCLE4_POLICY,'--landing-policy',LANDING,'--candidate-bank',CANDIDATES,'--landing-entry-set',ENTRY,'--indices-file',root/'h4_indices.json','--start-index',0,'--end-index',len(indices),'--seed',840000000,'--namespace','stable_cycle_4','--output',out],root/'h4_formal_replay.log',[out],unit_suffix=f'h4-replay-{int(time.time())}',preallocate=False)
   if not result['ok']:raise RuntimeError(f'H4 replay failed: {result}')
  self.save(current_stage='h4_analyze',next_decision='h1_independent_certify')

 def h4_analyze(self):
  root=self.run/'handoff/fast_route';out=root/'h4_analysis.json';shards=sorted((self.run/'handoff/cycle4').glob('shard_*.json'))
  if not out.exists():
   cmd=[PYTHON,'-m','cli.analyze_h4_replay','--formal-replay',root/'h4_formal_replay.json','--source-stable-report',CYCLE4_REPORT,'--output',out]
   for path in shards:cmd+=['--handoff-shard',path]
   self.run_command('analyze_h4_replay',cmd,root/'h4_analysis.log',[out])
  self.save(h4_analysis=str(out),current_stage='h1_independent_certify',last_completed_action='analyze_h4_replay',next_decision='h1_entry_decision')

 def h1_certify(self):
  root=self.run/'handoff/fast_route';cert=root/'h1_certified.pkl'
  if not cert.exists():
   result=self.run_worker_command('h1_independent_certification',[PYTHON,'-u','-m','cli.certify','--policy',LANDING,'--candidate-bank',self.state['pending_entry_bank'],'--phase','landing','--output-bank',cert,'--seed',2_500_000_000,'--namespace','fast-h1-independent','--allow-independent-candidates','--fixed-branches'],root/'h1_certify.log',[cert,cert.with_suffix('.cert.json')],unit_suffix=f'h1-cert-{int(time.time())}',preallocate=False)
   if not result['ok']:raise RuntimeError(f'H1 certification failed: {result}')
  self.save(h1_certified_bank=str(cert),current_stage='h1_entry_decision',next_decision='handoff_zero_training_ab')

 def h1_decision(self):
  root=self.run/'handoff/fast_route';cert=Path(self.state['h1_certified_bank']);bank=SnapshotBank.load(cert);safe=[r for r in bank.records_for_phase('landing',final_labels=['safe'],include_training_only=False)];parents={r.get('selection_parent') or r.get('entry_source_parent') or r.get('entry_source_id') for r in safe};decision=root/'h1_decision.json';extended=root/'entry_set.pkl'
  eligible=len(safe)>=4 and len(parents)>=2
  if eligible and not extended.exists():self.run_command('extend_c_l_fast',[PYTHON,'-m','cli.extend_landing_entries','--base-bank',ENTRY,'--certified-proposals',cert,'--output-bank',extended],root/'extend.log',[extended,extended.with_suffix('.extension.json')])
  save_json(decision,{'status':'PASS','safe_states':len(safe),'safe_parents':len(parents),'extension_eligible':eligible,'new_entry_bank':str(extended) if extended.exists() else None,'new_entry_bank_sha256':file_sha256(extended) if extended.exists() else None,'matcher_radius':SnapshotBank.load(ENTRY).metadata['entry_matcher']['radius'],'matcher_unchanged':True})
  self.save(new_c_l_generated=extended.exists(),current_entry_bank=str(extended if extended.exists() else ENTRY),h1_decision=str(decision),current_stage='handoff_zero_training_ab' if extended.exists() else 'landing_calibration_stage_a',last_completed_action='h1_entry_decision',next_decision='handoff_zero_training_ab' if extended.exists() else 'landing_calibration_stage_a')

 def handoff_ab(self):
  root=self.run/'handoff/fast_route';out=root/'handoff_ab.json'
  if not out.exists():
   result=self.run_worker_command('handoff_zero_training_ab',[PYTHON,'-u','-m','cli.evaluate_handoff_ab','--policy',CYCLE5_POLICY,'--landing-policy',LANDING,'--candidate-bank',CANDIDATES,'--entry-bank',f'canonical={ENTRY}','--entry-bank',f"extended={self.state['current_entry_bank']}",'--output',out,'--seed',2_600_000_000,'--branches-per-state',8],root/'handoff_ab.log',[out],unit_suffix=f'handoff-ab-{int(time.time())}',preallocate=False)
   if not result['ok']:raise RuntimeError(f'Handoff A/B failed: {result}')
  self.save(handoff_ab=str(out),current_stage='landing_calibration_stage_a',last_completed_action='handoff_zero_training_ab',next_decision='landing_calibration_stage_a')

 def entry_extension(self):
  root=self.run/'handoff/entry_extension';root.mkdir(parents=True,exist_ok=True);cert=root/'certified.pkl'
  if not cert.exists():
   self.run_worker_command('certify_pending_entries',[PYTHON,'-u','-m','cli.certify','--policy',LANDING,'--candidate-bank',self.state['pending_entry_bank'],'--phase','landing','--output-bank',cert,'--seed',2_500_000_000,'--namespace','handoff-entry-independent','--allow-independent-candidates','--fixed-branches'],root/'certify.log',[cert,cert.with_suffix('.cert.json')],unit_suffix=f'entry-cert-{int(time.time())}',preallocate=False)
  bank=SnapshotBank.load(cert);safe=[r for r in bank.records_for_phase('landing',final_labels=['safe'],include_training_only=False)];parents={r.get('entry_source_parent') or r.get('entry_source_id') for r in safe}
  extended=root/'entry_set.pkl'
  if len(safe)>=4 and len(parents)>=2 and not extended.exists():self.run_command('extend_c_l',[PYTHON,'-m','cli.extend_landing_entries','--base-bank',ENTRY,'--certified-proposals',cert,'--output-bank',extended],root/'extend.log',[extended,extended.with_suffix('.extension.json')])
  self.save(current_entry_bank=str(extended if extended.exists() else ENTRY),new_c_l_generated=extended.exists(),current_stage='landing_calibration_stage_a',next_decision='landing_calibration_stage_b')

 def landing_stage(self,name,seed,next_stage):
  root=self.run/'calibration'/name;out=root/'bank.pkl';report=out.with_suffix('.cert.json')
  if not report.exists():
   result=self.run_worker_command(f'landing_{name}',[PYTHON,'-u','-m','cli.certify','--policy',LANDING,'--candidate-bank',LANDING_BANK,'--phase','landing','--output-bank',out,'--seed',seed,'--namespace',f'certifier-calibration-{name}','--allow-independent-candidates','--fixed-branches'],root/'worker.log',[out,report],unit_suffix=f'landing-{name}-{int(time.time())}',preallocate=False)
   if not result['ok']:raise RuntimeError(f'Landing calibration {name} failed: {result}')
  self.save(current_stage=next_stage,last_completed_action=f'landing_{name}',next_decision=next_stage)

 def calibrate(self):
  root=self.run/'calibration';out=root/'protocol.json'
  if not out.exists():self.run_command('calibrate_certifier',[PYTHON,'-m','cli.calibrate_certifier_protocol','--landing-bank',LANDING_BANK,'--independent-audit',LANDING_AUDIT,'--stage-a-cert',root/'stage_a/bank.cert.json','--stage-b-cert',root/'stage_b/bank.cert.json','--output',out],root/'analyze.log',[out])
  payload=json.loads(out.read_text());self.save(certifier_protocol=str(out),certifier_protocol_hash=payload['protocol_hash'],certifier_selected_rule=payload['selected_rule'],certifier_overconservative=payload['selected_rule']!='stage_conjunction',current_stage='proposal_support',next_decision='full_jump_teacher_prepare')

 def support(self):
  root=self.run/'support';bank=root/'descent_proposal_support.pkl';report=root/'descent_proposal_support.json'
  if not report.exists():self.run_command('build_proposal_support',[PYTHON,'-m','cli.build_proposal_support_bank','--source-bank',CANDIDATES,'--stable-report',CYCLE5_REPORT,'--physical-audit',PHYSICAL_AUDIT,'--output-bank',bank,'--output-report',report],root/'build.log',[bank,report])
  self.save(proposal_support_bank=str(bank),current_stage='full_jump_teacher_prepare',last_completed_action='proposal_support',next_decision='full_jump_teacher_block_1')

 def teacher_prepare(self):
  marker=self.run/'teacher/protocol_ready.json'
  if marker.exists():self.save(current_stage='full_jump_teacher_block_1',next_decision='full_jump_teacher_block_1');return
  save_json(self.run/'teacher/implementation_pending.json',{'status':'ENGINEERING_CONTINUATION','research_gate':False,'reason':'full-jump teacher protocol implementation is the next automatic source milestone','inputs':{'proposal_support_bank':self.state.get('proposal_support_bank'),'entry_bank':self.state.get('current_entry_bank')}})
  self.save(last_completed_action='teacher_inputs_ready',next_decision='full_jump_teacher_protocol_implementation')
  time.sleep(60)

 def loop(self):
  while True:
   stage=self.state['current_stage'];self.save()
   if stage=='inspect':self.inspect()
   elif stage=='handoff_cycle4':self.handoff('cycle4')
   elif stage=='handoff_cycle5':self.handoff('cycle5')
   elif stage=='handoff_combine':self.combine()
   elif stage=='fast_handoff_prepare':self.fast_prepare()
   elif stage=='h4_formal_replay':self.h4_replay()
   elif stage=='h4_analyze':self.h4_analyze()
   elif stage=='h1_independent_certify':self.h1_certify()
   elif stage=='h1_entry_decision':self.h1_decision()
   elif stage=='handoff_zero_training_ab':self.handoff_ab()
   elif stage=='entry_extension_certify':self.entry_extension()
   elif stage=='landing_calibration_stage_a':self.landing_stage('stage_a',2_300_000_000,'landing_calibration_stage_b')
   elif stage=='landing_calibration_stage_b':self.landing_stage('stage_b',2_400_000_000,'certifier_calibration')
   elif stage=='certifier_calibration':self.calibrate()
   elif stage=='proposal_support':self.support()
   elif stage=='full_jump_teacher_prepare':self.teacher_prepare()
   elif stage=='pipeline_complete':return 0
   else:raise RuntimeError(f'Unknown jump-envelope stage {stage}')

def main():
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args();controller=JumpEnvelopeController(Path(a.run))
 try:raise SystemExit(controller.loop())
 except Exception as exc:
  count=int(controller.state.get('retry_count',0))+1;controller.save(retry_count=count,stop_reason=f'{type(exc).__name__}: {exc}')
  if count>=3:controller.save(terminal_state='engineering_failure_after_retries');raise SystemExit(41)
  raise
if __name__=='__main__':main()
