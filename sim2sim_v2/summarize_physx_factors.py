"""Derive contact-grounded P1 tables from immutable native report traces."""
import json,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from landing_evaluation import evaluate_landing
ROOT=Path(__file__).resolve().parent/'runs/p1_physx'
model=mujoco.MjModel.from_xml_path(str(ROOT.parent.parent/'model/source.xml'));kin=mujoco.MjData(model)
rows=[];repeat={};div=[]
from physx_probe_summary import summarize_probe
for d in sorted(ROOT.iterdir()):
 if (d/'budget.json').exists():rows.extend(summarize_probe(d))
for label in ['source','warm']:
 a=ROOT/f'{label}_hard_det_01/trace.npz';b=ROOT/f'{label}_hard_det_02/trace.npz'
 if not a.exists() or not b.exists():continue
 x=np.load(a);y=np.load(b);entry={}
 for field in ['trace','observations','actions']:
  T=min(len(x[field]),len(y[field]));err=np.abs(x[field][:T]-y[field][:T]);entry[field]={'max_abs':float(err.max()),'lengths':[len(x[field]),len(y[field])]}
  loc=np.argwhere(err>1e-5)
  if len(loc):tick,lane,channel=map(int,loc[0]);div.append(dict(comparison=label,field=field,first_tick=tick,channel=channel,abs_error=float(err[tick,lane,channel])))
  else:div.append(dict(comparison=label,field=field,first_tick='',channel='',abs_error=0))
 repeat[label]=entry
for policy in ['source','warm']:
 for suffix in ['after_episode','after_episode_corrected','stochastic_31','stochastic_47','stochastic_83']:
  if not (ROOT/f'{policy}_{suffix}/trace.npz').exists():continue
  x=np.load(ROOT/f'{policy}_hard_det_01/trace.npz');y=np.load(ROOT/f'{policy}_{suffix}/trace.npz')
  for field in ['trace','observations','actions']:
   T=min(len(x[field]),len(y[field]));err=abs(x[field][:T]-y[field][:T]);loc=np.argwhere(err>1e-5)
   if len(loc):tick,lane,channel=map(int,loc[0]);div.append(dict(comparison=f'{policy}_det_vs_{suffix}',field=field,first_tick=tick,channel=channel,abs_error=float(err[tick,lane,channel])))
   else:div.append(dict(comparison=f'{policy}_det_vs_{suffix}',field=field,first_tick='',channel='',abs_error=0))
capture=ROOT/'warm384_capture_01/trace.npz'
if capture.exists():
 x=np.load(capture)['trace']
 for lane in [0,383]:
  for kind in ['noise','action']:
   p=ROOT/f'warm_lane{lane}_{kind}_replay/trace.npz'
   if not p.exists():continue
   y=np.load(p)['trace'];T=min(len(x),len(y));err=abs(x[:T,lane]-y[:T,0]);loc=np.argwhere(err>1e-5)
   if len(loc):tick,channel=map(int,loc[0]);div.append(dict(comparison=f'384_vs_1_lane{lane}_{kind}',field='trace',first_tick=tick,channel=channel,abs_error=float(err[tick,channel])))
   else:div.append(dict(comparison=f'384_vs_1_lane{lane}_{kind}',field='trace',first_tick='',channel='',abs_error=0))
if rows:
 with open(ROOT/'eval_factorial.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with open(ROOT/'first_divergence.csv','w') as f:w=csv.DictWriter(f,fieldnames=['comparison','field','first_tick','channel','abs_error']);w.writeheader();w.writerows(div)
(ROOT/'repeatability.json').write_text(json.dumps(repeat,indent=2))
budgets=[json.loads(p.read_text()) for p in ROOT.glob('*/budget.json')];(ROOT/'actual_budget.json').write_text(json.dumps({'control_interactions':sum(b['control_interactions'] for b in budgets),'physics_interactions':sum(b['physics_interactions'] for b in budgets),'learner_transitions':0,'successful_probes':len(budgets)},indent=2))
captures=[json.loads((ROOT/n/'status.json').read_text()) for n in ['warm384_capture_01','fresh384_capture_01'] if (ROOT/n/'status.json').exists()]
combined={'single_probe_control_interactions':sum(b['control_interactions'] for b in budgets),'single_probe_physics_interactions':sum(b['physics_interactions'] for b in budgets),'prefix_control_interactions':sum(b.get('prefix_control_interactions',0) for b in budgets),'diagnostic_on_policy_transitions_collected':sum(b['control_interactions'] for b in captures),'diagnostic_capture_physics_interactions':sum(b['physics_interactions'] for b in captures),'total_control_interactions':sum(b['control_interactions'] for b in budgets+captures),'total_physics_interactions':sum(b['physics_interactions'] for b in budgets+captures),'accepted_learner_updates':0,'initialization_failures_before_physics':1,'single_probe_count':len(budgets),'full384_captures':len(captures)}
(ROOT/'total_budget.json').write_text(json.dumps(combined,indent=2))
