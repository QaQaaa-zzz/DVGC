from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent/'runs/p1_physx'
for policy in ['source','warm']:
 fig,axes=plt.subplots(2,3,figsize=(14,8));axes=axes.ravel()
 for label,name in [('deterministic',f'{policy}_hard_det_01')]+[(f'stochastic seed {s}',f'{policy}_stochastic_{s}') for s in [31,47,83]]:
  d=r/name;z=np.load(d/'landing_lane_0.npz');steps=json.loads((d/'steps.json').read_text());t=z['time_s'];p=z['position'];v=z['velocity'];rpy=z['rpy']
  color='C'+str(['deterministic','stochastic seed 31','stochastic seed 47','stochastic seed 83'].index(label))
  axes[0].plot(p[:,0],p[:,1],label=label,color=color);axes[0].plot(p[-1,0],p[-1,1],'x',color=color)
  axes[1].plot(t,v[:,0],label=label,color=color);axes[2].plot(t,p[:,2],label=label,color=color);axes[3].plot(t,np.rad2deg(rpy[:,0]),label=label,color=color)
  axes[4].plot(t,z['wheel_step_force'].sum(axis=1),label=label,color=color);axes[5].plot(t,[s[0]['reward'] for s in steps],label=label,color=color)
 axes[0].axhline(0,color='k',linestyle=':',label='nominal straight direction');axes[0].axvline(3.6,color='.5',linestyle='--');axes[0].set_aspect('equal',adjustable='datalim')
 for ax,title,y in zip(axes,['XY actual; x = true recorded endpoint','Forward speed','Root height','Roll','Native step wheel support force','Actual per-control reward'],['world y [m]','vx [m/s]','z [m]','roll [deg]','N','reward']):
  ax.set_title(title);ax.set_ylabel(y);ax.set_xlabel('world x [m]' if ax is axes[0] else 'time [s]');ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig.suptitle(f'{policy}: same natural initial state; deterministic vs three independent action noise seeds\n5 ms / 20 ms; kd=5; trajectories stop at actual legacy terminal. Timeout is not landing success.')
 fig.tight_layout();fig.savefig(r/f'{policy}_factor_comparison.png',dpi=160);plt.close(fig)
