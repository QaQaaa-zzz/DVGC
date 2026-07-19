"""Atomically activate the handoff-first seed-0 jump-envelope controller."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',default='runs/ACTIVE_PIPELINE.json');a=p.parse_args();run=Path(a.run)
    run.mkdir(parents=True,exist_ok=True)
    old_path=Path(a.output)
    if old_path.exists():
        old=json.loads(old_path.read_text());old_state=Path(old.get('run_path',''))/'controller_state.json'
        if old_state.exists():
            state=json.loads(old_state.read_text())
            if state.get('current_stage') not in ('gate_pause','pipeline_complete') and state.get('terminal_state') not in ('gate_pause','pipeline_complete'):
                raise SystemExit('Existing active pipeline is not terminal')
    payload={'status':'ACTIVE','activated_at':time.time(),'run_path':str(run),'controller_unit':'dvgc-jump-envelope-controller.service','start_script':'/home/qy/DVGC/scripts/start_jump_envelope_controller.sh'}
    temporary=old_path.with_suffix(old_path.suffix+'.tmp');temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(temporary,old_path);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
