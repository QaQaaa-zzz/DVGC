"""Saved SOLVER_DEV and reset-pool XY. No physics, optimization, or TEST."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/T1_paired_ppo_002');out=r/'saved_review_002';p=json.loads((r/'plan.json').read_text());entries=json.loads(Path(p['solver_roots']).read_text());fig,axes=plt.subplots(2,4,figsize=(16,8))
for ordinal,entry in enumerate(entries):
 ax=axes.flat[ordinal];rid=entry['root']['root_id'];labels=[]
 for arm,c in zip(('PPO_keep','PPO_demo_keep'),('#2867b2','#d65f29')):
  res=json.loads((r/f'{arm}_SOLVER_DEV_repeat0_root{ordinal:02d}'/'results.json').read_text());a=res[entry['lane']]['attempts'][0];labels.append(str(a['label']))
  with np.load(a['trace']) as d:
   lane=a['trace_lane'];q=d['qpos'][d['mask'][:,lane].astype(bool),lane];ax.plot(q[:,0],q[:,1],color=c,label=arm);ax.plot(q[-1,0],q[-1,1],color=c,marker='o' if a['label'] else 'x')
 ax.set_title(rid+'\nkeep/demo labels '+ '/'.join(labels),fontsize=8);ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.grid(alpha=.2)
axes.flat[-1].set_visible(False);axes.flat[0].legend(fontsize=8);fig.suptitle('SOLVER_DEV all7 roots, first fixed evaluation only; both32k students\nOriginal world XY; true endpoints; circle=recovery, x=failure; not three-repeat stable labels');fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'xy_solver_roots.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,4,figsize=(16,4))
for arm,c in zip(('PPO_keep','PPO_demo_keep'),('#2867b2','#d65f29')):
 with np.load(r/'panel'/arm/'trace.npz') as data:
  for lane,ax in enumerate(axes):
   mask=data['active'][:,lane].astype(bool);q=data['qpos'][mask,lane];ax.plot(q[:,0],q[:,1],color=c,label=arm);ax.plot(q[-1,0],q[-1,1],color=c,marker='x')
for ax,name in zip(axes,('Nominal complete','Random complete','pi0 success snapshot','Teacher recoverable')):ax.set_title(name);ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.grid(alpha=.2)
axes[0].legend(fontsize=7);fig.suptitle('Four reset-pool TRAIN engineering diagnostic trajectories; excluded from success counts\nEndpoint markers indicate observed end only; no extra evaluation');fig.tight_layout(rect=(0,0,1,.85));fig.savefig(out/'xy_reset_pool_diagnostics.png',dpi=150);plt.close(fig)
print('Solver and four pool diagnostic XY complete')
