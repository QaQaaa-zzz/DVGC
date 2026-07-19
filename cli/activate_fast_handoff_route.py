"""Move a stopped exhaustive controller to the bounded evidence-funnel route."""
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
from dvgc.runtime import save_json

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--completed-output',required=True);a=p.parse_args();run=Path(a.run);state_path=run/'controller_state.json';state=json.loads(state_path.read_text());output=Path(a.completed_output)
 if not output.exists():raise SystemExit('Current shard has not completed atomically')
 unit=state.get('active_worker_unit')
 if unit:
  status=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip()
  if status in ('active','activating','deactivating','reloading'):raise SystemExit(f'Current worker is still {status}')
 state.setdefault('history',[]).append({'action':state.get('in_progress_action'),'completed_at':time.time(),'outputs':[str(output)],'route_transition':'exhaustive_to_evidence_funnel'})
 state.update({'current_stage':'fast_handoff_prepare','last_completed_action':state.get('in_progress_action'),'in_progress_action':None,'active_worker_unit':None,'expected_outputs':[],'next_decision':'h4_formal_replay','retry_count':0,'stop_reason':None,'terminal_state':None,'research_gate_valid':False,'route_mode':'bounded_evidence_funnel','exhaustive_scan_stopped_after_event':json.loads(output.read_text())['end_event']})
 save_json(state_path,state);print(json.dumps({k:state.get(k) for k in ('current_stage','last_completed_action','route_mode','exhaustive_scan_stopped_after_event')},indent=2))
if __name__=='__main__':main()
