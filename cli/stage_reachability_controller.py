"""Resumable RA-L stage-reachability pipeline controller."""
from __future__ import annotations
import json,subprocess,time
from pathlib import Path
from cli.descent_local_controller import Controller,PYTHON
from cli.jump_envelope_controller import CYCLE5_POLICY
from dvgc.config import file_sha256,load_config
from dvgc.runtime import save_json
from dvgc.stage_reachability import protocol_payload

OLD_RUN=Path('runs/jump_envelope_seed0_20260719')
FLIGHT_BANK=Path('artifacts/flight_candidates_augmented_v1.pkl')
ROUTE=("safe_pause_old_route","h4_replay_smoke","define_stage_entry_protocol","relabel_existing_data","coverage_inventory","stage_candidate_pilot","stage_label_acquisition","train_reachability_model","active_candidate_acquisition","build_stage_tubes","Tube_RSI_final_PPO","final_policy_certification")

class StageReachabilityController(Controller):
 def __init__(self,run):
  super().__init__(run)
  if self.state.get('controller_type')!='stage_reachability':
   self.state={"controller_type":"stage_reachability","controller_version":1,"controller_unit":"dvgc-stage-reachability-controller.service","run_id":run.name,"current_stage":"safe_pause_old_route","last_completed_action":None,"in_progress_action":None,"expected_outputs":[],"next_decision":"h4_replay_smoke","retry_count":0,"heartbeat":time.time(),"stop_reason":None,"terminal_state":None,"research_gate_valid":False,"history":[],"provenance":{}}
   self.save()

 def pause_old(self):
  state=json.loads((OLD_RUN/'controller_state.json').read_text())
  if state.get('terminal_state')!='SUPERSEDED_BY_STAGE_REACHABILITY_PROTOCOL':raise RuntimeError('Old route is not atomically superseded')
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
  marker=self.run/'stage_candidate_pilot/implementation_ready.json'
  if not marker.exists():save_json(marker,{"status":"ENGINEERING_CONTINUATION","research_gate":False,"reason":"event-aligned candidate pilot worker is the next bounded implementation action","cost_estimate":str(cost),"no_rollout_started":True})
  self.save(last_completed_action='candidate_pilot_cost',next_decision='implement_stage_candidate_pilot_worker')
  time.sleep(60)

 def loop(self):
  while True:
   self.save();stage=self.state['current_stage']
   if stage=='safe_pause_old_route':self.pause_old()
   elif stage=='h4_replay_smoke':self.replay_smoke()
   elif stage=='define_stage_entry_protocol':self.define_protocol()
   elif stage in ('relabel_existing_data','coverage_inventory'):self.inventory()
   elif stage=='stage_candidate_pilot':self.candidate_pilot()
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
