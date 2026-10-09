"""Report a matched actor's recorded dual-engine landing, without new simulation."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/no_speed_dual_landing'
mjx=OUT/'mjx_warm01966080';physx=OUT/'physx_warm01966080'
load=lambda p:json.loads(p.read_text())
md=load(mjx/'declaration.json');pb=load(physx/'budget.json');actor=Path(md['actor']);sha=hashlib.sha256(actor.read_bytes()).hexdigest()
assert sha==md['actor_sha256']==pb['actor_sha256']
mr=load(mjx/'runtime.json');pi=load(physx/'initial_state.json');pa=load(physx/'audit.json')
np.testing.assert_allclose(mr['initial_qpos'][:3],pi['world_position'][0],atol=1e-7)
np.testing.assert_allclose(mr['initial_qpos'][3:7],pi['quaternion_wxyz'][0],atol=1e-7)
assert mr['sim_dt']==pa['physics_dt']==.005 and mr['ctrl_dt']==pa['control_dt']==.02
rows=[];fig,axs=plt.subplots(3,2,figsize=(13,11),layout='constrained')
for label,path,color,style in [('MJX-Warp',mjx,'#0072B2','-'),('Isaac / PhysX',physx,'#D55E00','--')]:
 if path==mjx:
  trace=np.load(path/'trace.npz');t=trace['time_s'];pos=trace['qpos'][:,:3];vel=trace['qvel'][:,:3];rpy=trace['rpy'];land=load(path/'landing.json');support=trace['force_support_step'];end=int(trace['end_code'][-1]);inputs=load(path/'landing_inputs.json')
 else:
  trace=np.load(path/'landing_lane_0.npz');t=trace['time_s'];pos=trace['position'];vel=trace['velocity'];rpy=trace['rpy'];land=load(path/'landing_lane_0.json');support=trace['wheel_step_force'].sum(axis=1);end=load(path/'steps.json')[-1][0]['end_code'];inputs=load(path/'landing_inputs_lane_0.json')
 assert land['strict_landing_1p0s'] and land['contact_invalid_ticks']==0
 axs[0,0].plot(pos[:,0],pos[:,1],style,color=color,label=label);axs[0,0].scatter(pos[-1,0],pos[-1,1],color=color,marker='x')
 axs[0,1].plot(pos[:,0],pos[:,2],style,color=color,label=label)
 axs[1,0].plot(t,np.rad2deg(rpy[:,0]),style,color=color,label=label+' roll');axs[1,0].plot(t,np.rad2deg(rpy[:,1]),':',color=color,label=label+' pitch')
 axs[1,1].plot(t,vel[:,0],style,color=color,label=label)
 axs[2,0].plot(t,support,style,color=color,label=label)
 hold=[d['stable_hold_s'] for d in land['details']];axs[2,1].plot(t,hold,style,color=color,label=label)
 rows.append(dict(engine=label,checkpoint=actor.parent.name,actor_sha256=sha,takeoff_s=land['takeoff_time_s'],apex_s=land['apex_time_s'],touchdown_s=land['first_touchdown_time_s'],strict_landing_s=land['success_times']['1.0'],longest_hold_s=land['max_stable_hold_s'],end_s=float(t[-1]),end_code=end,final_x_m=float(pos[-1,0]),final_y_m=float(pos[-1,1]),final_z_m=float(pos[-1,2]),strict_pass=True,contact_invalid_ticks=0))
axs[0,0].axhline(0,color='.6',lw=1,label='original straight reference');axs[0,0].set_aspect('equal',adjustable='datalim')
axs[0,1].plot([0,3.6,3.6,7.6,7.6,20],[0,0,.16,.16,0,0],color='.5',lw=1,label='floor / platform top')
axs[2,1].axhline(1,color='.5',lw=1,label='1 s stable hold')
labels=[('XY trajectory (actual endpoint marked)','world x [m]','world y [m]'),('Full jump and post-landing motion','world x [m]','root height [m]'),('Attitude','time [s]','angle [deg]'),('Forward velocity — diagnostic only','time [s]','velocity [m/s]'),('Native wheel / platform support','time [s]','normal support [N]'),('Continuous stable landing interval','time [s]','hold duration [s]')]
for ax,(title,x,y) in zip(axs.ravel(),labels):ax.set(title=title,xlabel=x,ylabel=y);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Same frozen transition_01966080 on both engines | natural reset, sim 5 ms / control 20 ms\nOne development start per engine; no minimum forward speed; full 8 s retained',fontsize=13)
fig.savefig(OUT/'paired_landing.png',dpi=160);plt.close(fig)
with (OUT/'paired_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(OUT/'paired_identity.json').write_text(json.dumps(dict(actor=str(actor),sha256=sha,initial_root_pose_match=True,sim_dt=.005,ctrl_dt=.02,source_declaration=str(mjx/'declaration.json'),target_budget=str(physx/'budget.json'),results=rows),indent=2))
print(json.dumps(rows,indent=2))
