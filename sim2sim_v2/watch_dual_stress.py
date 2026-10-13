"""Expose paired evaluation progress and produce report after both engines finish."""
import json,time,subprocess
from pathlib import Path
from dual_stress_protocol import write_json
root=Path(__file__).resolve().parent;out=root/'runs/dual_stress1000_20261013'
while True:
 states={}
 for e in ['mjx','physx']:
  p=out/e/'status.json'
  if p.exists():states[e]=json.loads(p.read_text())
 failed=any(s['state']=='error' for s in states.values());complete=len(states)==2 and all(s['state']=='complete' for s in states.values())
 write_json(out/'status.json',dict(state='error' if failed else 'analyzing' if complete else 'running',completed_mjx=states.get('mjx',{}).get('completed_cases',0),completed_physx=states.get('physx',{}).get('completed_cases',0),total_per_engine=1000,engine_states=states,updated_unix=time.time(),training_updates=0))
 if failed:raise RuntimeError('Engine error preserved')
 if complete:
  subprocess.run(['/home/qy/ISAAC——SIM/.venv/bin/python',str(root/'report_dual_stress.py')],check=True,cwd=root);break
 time.sleep(5)
