import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/home/qy/ISAAC——SIM/worktrees/jump-landing-v2/sim2sim_v2/runs/p1_physx')
fig,axs=plt.subplots(2,3,figsize=(14,8));axs=axs.ravel()
for i,(label,name) in enumerate([('source9904128','source_hard_det_01'),('warmfinal','warm_hard_det_01'),('warm2.46M certified','warm_mid_02457600_certified'),('warm4.92M certified','warm_mid_04915200_certified')]):
 z=np.load(r/name/'landing_lane_0.npz');t=z['time_s'];p=z['position'];v=z['velocity'];color=f'C{i}'
 axs[0].plot(p[:,0],p[:,1],color=color,label=label);axs[0].plot(p[-1,0],p[-1,1],'x',color=color)
 axs[1].plot(t,v[:,0],color=color);axs[2].plot(t,p[:,2],color=color);axs[3].plot(t,np.rad2deg(z['rpy'][:,0]),color=color);axs[4].plot(t,z['wheel_step_force'].sum(axis=1),color=color)
 land=json.load(open(r/name/'landing_lane_0.json'));axs[5].plot(t,[x['stable_hold_s'] for x in land['details']],color=color)
axs[0].axhline(0,color='k',ls=':',label='nominal straight direction');axs[0].set_aspect('equal',adjustable='datalim');axs[0].axvline(3.6,color='.5',ls='--');axs[0].legend(fontsize=7);axs[1].axhline(2,color='k',ls=':');axs[5].axhline(1,color='k',ls=':')
for ax,title,y in zip(axs,['XY actual (x marks true endpoint)','Forward speed','Root height','Roll','Native step wheel support','Continuous strict hold'],['world y [m]','vx [m/s]','z [m]','roll [deg]','N','hold [s]']):
 ax.set_title(title);ax.set_ylabel(y);ax.set_xlabel('world x [m]' if ax is axs[0] else 'time [s]');ax.grid(alpha=.2)
fig.suptitle('Frozen checkpoint diagnostic: identical natural reset, kd=5, physics5ms/control20ms\nIntermediate checkpoints pass moving landing (>0.2m/s) on ONE case; not 2m/s tracking or robustness evidence.')
fig.tight_layout();fig.savefig(r/'checkpoint_candidates.png',dpi=160)
