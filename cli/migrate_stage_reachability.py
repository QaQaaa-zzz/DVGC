"""Atomically supersede the retired handoff route after its worker boundary."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
from dvgc.config import file_sha256
from dvgc.runtime import save_json

SUPERSEDED="SUPERSEDED_BY_STAGE_REACHABILITY_PROTOCOL"

def inactive(unit:str)->bool:
 return subprocess.run(["systemctl","--user","is-active","--quiet",unit]).returncode!=0

def migrate(old_run:Path,new_run:Path,pointer:Path,*,check_systemd:bool=True):
 old_state_path=old_run/"controller_state.json";state=json.loads(old_state_path.read_text())
 worker=str(state.get("active_worker_unit") or "")
 if check_systemd and not inactive("dvgc-jump-envelope-controller.service"):raise RuntimeError("Old controller is still active")
 if check_systemd and worker and not inactive(worker):raise RuntimeError("Old atomic worker is still active")
 outputs=[Path(x) for x in state.get("expected_outputs",[])]
 missing=[str(x) for x in outputs if not x.exists()]
 if missing:raise RuntimeError(f"Atomic worker stopped without outputs: {missing}")
 preserved=[{"path":str(x),"sha256":file_sha256(x)} for x in outputs]
 state.update({"current_stage":"superseded","terminal_state":SUPERSEDED,"stop_reason":SUPERSEDED,"in_progress_action":None,"active_worker_unit":None,"expected_outputs":[],"next_decision":None,"superseded_at":time.time(),"superseded_artifacts":preserved})
 save_json(old_state_path,state)
 new_run.mkdir(parents=True,exist_ok=True)
 payload={"status":"PREPARING","activated_at":time.time(),"run_path":str(new_run),"controller_unit":"dvgc-stage-reachability-controller.service","start_script":"/home/qy/DVGC/scripts/start_stage_reachability_controller.sh","supersedes":str(old_run),"supersession_reason":SUPERSEDED}
 temporary=pointer.with_suffix(pointer.suffix+".tmp");temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(temporary,pointer)
 return payload

def main():
 p=argparse.ArgumentParser();p.add_argument('--old-run',required=True);p.add_argument('--new-run',required=True);p.add_argument('--pointer',default='runs/ACTIVE_PIPELINE.json');a=p.parse_args();print(json.dumps(migrate(Path(a.old_run),Path(a.new_run),Path(a.pointer)),indent=2))
if __name__=='__main__':main()
