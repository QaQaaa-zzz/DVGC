"""Plot all saved DEV worlds and all fixed repeats; no physics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/T1_paired_ppo_002');out=r/'saved_review_002';p=json.loads((r/'plan.json').read_text());e=json.loads((r/'evaluation_results.json').read_text());datasets={}
for model in ('pi0','PPO_keep','PPO_demo_keep'):
 datasets[model]=[]
 for rep in range(3):
  directory=Path(p['reused_zero']['DEV']['pi0']) if model=='pi0' and rep==0 else r/f'{model}_DEV_repeat{rep}'
  with np.load(directory/'prefixes.npz') as f:datasets[model].append({k:f[k].copy() for k in ('qpos','prefix_mask','success')})
for gi,g in enumerate('ABCD'):
 fig,axes=plt.subplots(4,4,figsize=(16,14))
 for case,ax in enumerate(axes.flat):
  lane=gi*16+case;title=[]
  for model,color in zip(datasets,('#777777','#2867b2','#d65f29')):
   labels=[]
   for rep,data in enumerate(datasets[model]):
    labels.append((e['source_DEV'] if model=='pi0' else e['arms'][model]['DEV'])[rep][g][case]);mask=data['prefix_mask'][:,lane].astype(bool);q=data['qpos'][mask,lane];assert len(q)>0 and np.isfinite(q).all();assert int(np.any(data['success'][:,lane] & mask))==labels[-1]
    ax.plot(q[:,0],q[:,1],linestyle=('-','--',':')[rep],color=color,alpha=.5,linewidth=1,label=model if rep==0 and case==0 else None)
    ax.plot(q[-1,0],q[-1,1],color=color,marker='o' if labels[-1] else 'x',ms=3)
   title.append(model.replace('PPO_','')+':'+''.join(map(str,labels)))
  ax.set_title(g+f'_{case:02d}\n'+' / '.join(title),fontsize=7);ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
 axes.flat[0].legend(fontsize=7);fig.suptitle(f'DEV group {g}: all16 conditions, pi0 and both32k students, all3 fixed repeats\nOriginal world XY; true endpoints; solid/dashed/dotted=repeat0/1/2, circle=success, x=failure; no ideal geometric path defined',fontsize=11);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/f'xy_DEV_{g}.png',dpi=130);plt.close(fig)
print('All64 DEV x3 models x3 fixed repeats plotted')
