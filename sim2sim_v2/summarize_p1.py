from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent;OUT=R/'runs/p1_analysis';OUT.mkdir(exist_ok=True)
source=R/'runs/p1_mjx_9904128_retry02';d=np.load(source/'trace.npz');series=[dict(name='9904128 MJX (nominal)',t=d['time_s'],pos=d['qpos'][:,:3],rpy=d['rpy'],vel=d['qvel'][:,:3],omega=d['qvel'][:,3:6],support=d['force_support_step'],reward=d['reward'])]
for folder,label in [('source_hard_det_01','9904128 PhysX det'),('warm_hard_det_01','warm final PhysX det'),('warm_stochastic_31','warm final PhysX sampled seed31'),('warm_mid_02457600_certified','warm 2.46M PhysX det'),('warm_mid_04915200_certified','warm 4.92M PhysX det')]:
 p=R/'runs/p1_physx'/folder;z=np.load(p/'trace.npz');tr=z['trace'][:,0];t=np.arange(1,len(tr)+1)*.02;ld=np.load(p/'landing_lane_0.npz')
 series.append(dict(name=label,t=t,pos=tr[:,1:4],rpy=Rotation.from_quat(tr[:,[5,6,7,4]]).as_euler('xyz'),vel=tr[:,18:21],omega=tr[:,21:24],support=ld['wheel_step_force'].sum(axis=1),reward=np.asarray([row[0]['reward'] for row in json.loads((p/'steps.json').read_text())])))
fig,axes=plt.subplots(3,2,figsize=(13,12))
for s in series:
 axes[0,0].plot(s['pos'][:,0],s['pos'][:,1],label=s['name']);axes[0,0].scatter(s['pos'][-1,0],s['pos'][-1,1],marker='x')
 for ax,y in [(axes[0,1],s['vel'][:,0]),(axes[1,0],s['pos'][:,2]),(axes[1,1],np.rad2deg(s['rpy'][:,0])),(axes[2,0],s['support']),(axes[2,1],s['reward'])]:ax.plot(s['t'],y,label=s['name'])
axes[0,0].plot([1.5,17.5],[0,0],'k--',label='2 m/s straight geometric reference');axes[0,0].set(xlabel='world x [m]',ylabel='world y [m]');axes[0,0].axis('equal');axes[0,0].legend(fontsize=7)
for ax,label in zip([axes[0,1],axes[1,0],axes[1,1],axes[2,0],axes[2,1]],['Forward velocity [m/s]','Root height [m]','Roll [deg]','Real wheel-step support [N]','Legacy per-control reward']):ax.set(xlabel='time [s]',ylabel=label)
for ax in axes.flat:ax.grid(alpha=.25)
fig.suptitle('P1: identical source checkpoint vs target; natural reset, 5/20 ms; endpoints not padded');fig.tight_layout();fig.savefig(OUT/'paired_diagnostics.png',dpi=170);plt.close(fig)
# Source strict criteria rejection diagnostics; all windows are descriptive, not new thresholds.
rows=json.loads((source/'landing_inputs.json').read_text());report={}
for start,end in [(1.34,3.),(1.34,8.)]:
 sel=[r for r in rows if start<=r['time_s']<=end]
 bad=dict(roll=sum(abs(r['roll'])>=np.deg2rad(10) for r in sel),pitch=sum(abs(r['pitch'])>=np.deg2rad(15) for r in sel),body_omega_x=sum(abs(r['roll_rate'])>=1 for r in sel),body_omega_y=sum(abs(r['pitch_rate'])>=1 for r in sel),world_lateral_speed=sum(abs(r['vy'])>=.25 for r in sel),forward_speed=sum(r['vx']<=.2 for r in sel),no_step_support=sum(r['support_step_N']<.1*float(np.asarray(ld['mass']).ravel()[0])*9.81 for r in sel))
 report[f'{start}-{end}s']={'n':len(sel),'violation_control_counts':bad,'counts_overlap':True}
(OUT/'source_landing_constraints.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item()))
with (OUT/'mjx_reward.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['time_s','legacy_reward','apex_seen','end_code']);w.writerows(zip(d['time_s'],d['reward'],d['apex_seen'],d['end_code']))
print(OUT)
