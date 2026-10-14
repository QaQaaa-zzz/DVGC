"""Evidence-backed paper figures and x<=5 display data; no simulation or policy edits."""
from pathlib import Path
import json,csv,math,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/publication/frontierbridge_sim2sim_20261014';NOM=ROOT/'runs/no_speed_dual_landing';STRESS=ROOT/'runs/dual_stress1000_20261013'
for n in ['figures','data','text','renders']:(OUT/n).mkdir(exist_ok=True)
load=lambda p:json.loads(p.read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'xtick.direction':'out','ytick.direction':'out','svg.fonttype':'none','pdf.fonttype':42,'savefig.bbox':'tight'})
identity=load(NOM/'paired_identity.json');data={};snapshots={}
def csvsave(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for engine,folder in [('mjx','mjx_warm01966080'),('physx','physx_warm01966080')]:
 p=NOM/folder;a=np.load(p/'trace.npz');land=load(p/('landing.json' if engine=='mjx' else 'landing_lane_0.json'))
 if engine=='mjx':
  q=a['qpos'];v=a['qvel'];obs=a['obs'];actions=a['action'];rows=load(p/'landing_inputs.json');times=a['time_s']
 else:
  raw=a['trace'][:,0];times=(np.arange(len(raw))+1)*.02;obs=a['observations'][:,0];actions=a['actions'][:,0];rows=load(p/'landing_inputs_lane_0.json')
  import mujoco
  m=mujoco.MjModel.from_xml_path(str(ROOT/'model/source.xml'));q=np.tile(m.key_qpos[0],(len(raw),1));v=np.zeros((len(raw),m.nv))
  q[:,:3]=raw[:,1:4];q[:,3:7]=raw[:,4:8];v[:,:3]=raw[:,18:21];v[:,3:6]=Rotation.from_quat(q[:,[4,5,6,3]]).inv().apply(raw[:,21:24])
  names=load(p/'audit.json')['dof_names']
  for j,name in enumerate(names):q[:,m.joint(name).qposadr[0]]=raw[:,8+j];v[:,m.joint(name).dofadr[0]]=raw[:,13+j]
 cut=int(np.flatnonzero(q[:,0]>=5)[0])+1;q=q[:cut];v=v[:cut];times=times[:cut];obs=obs[:cut];actions=actions[:cut];rows=rows[:cut]
 rpy=Rotation.from_quat(q[:,[4,5,6,3]]).as_euler('xyz');floor=np.array([r['support_floor_N'] for r in rows]);step=np.array([r['support_step_N'] for r in rows])
 data[engine]=dict(q=q,v=v,time=times,obs=obs,action=actions,rpy=rpy,floor=floor,step=step,folder=p,landing=land)
 np.savez_compressed(OUT/'data'/f'nominal_{engine}_x5.npz',qpos=q,qvel=v,time_s=times,actor_observation=obs,action=actions,force_floor_N=floor,force_step_N=step,rpy=rpy)
 table=[]
 for i,t in enumerate(times):
  r=dict(time_s=float(t),root_x_m=float(q[i,0]),root_y_m=float(q[i,1]),root_z_m=float(q[i,2]),roll_deg=float(np.rad2deg(rpy[i,0])),pitch_deg=float(np.rad2deg(rpy[i,1])),yaw_deg=float(np.rad2deg(rpy[i,2])),vx_mps=float(v[i,0]),vy_mps=float(v[i,1]),vz_mps=float(v[i,2]),body_wx_radps=float(v[i,3]),body_wy_radps=float(v[i,4]),body_wz_radps=float(v[i,5]),wheel_floor_force_N=float(floor[i]),wheel_step_force_N=float(step[i]))
  r.update({f'qpos_{j}':float(x) for j,x in enumerate(q[i])});r.update({f'qvel_{j}':float(x) for j,x in enumerate(v[i])});r.update({f'action_{j}':float(x) for j,x in enumerate(actions[i])});table.append(r)
 csvsave(OUT/'data'/f'nominal_{engine}_x5.csv',table)
 ids=[0,int(np.argmin(abs(times-(land['takeoff_time_s']-.02)))),int(np.argmin(abs(times-land['apex_time_s']))),int(np.argmin(abs(times-land['first_touchdown_time_s']))),len(times)-1]
 assert len(set(ids))==5
 snapshots[engine]=dict(indices=ids,times_s=times[ids].tolist(),qpos=q[ids].tolist(),events=['approach','takeoff','apex','touchdown','x5'],terminal_x=float(q[-1,0]),terminal_t=float(times[-1]),strict_hold_1s_completed_before_x5=bool(land['success_times']['1.0']<=times[-1]))
(OUT/'data/snapshot_states.json').write_text(json.dumps(snapshots,indent=2))
def lines(engine,variant):
 return dict(color=('#0072B2' if engine=='mjx' else '#D55E00') if variant=='color' else ('#1c1c1c' if engine=='mjx' else '#707070'),ls='-' if engine=='mjx' else '--',lw=1.35,label='MuJoCo/MJX' if engine=='mjx' else 'Isaac/PhysX')
def finish(fig,axes,name,variant):
 axes=np.atleast_1d(axes).ravel()
 for ax in axes:ax.grid(False);ax.tick_params(labelsize=8)
 handles,labels=axes[0].get_legend_handles_labels()
 if handles:fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.02))
 for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{name}_{variant}.{ext}',dpi=300)
 plt.close(fig)
for variant in ['color','grayscale']:
 fig,axs=plt.subplots(1,2,figsize=(7.0,2.55),layout='constrained')
 for e,d in data.items():
  kw=lines(e,variant);axs[0].plot(d['q'][:,0],d['q'][:,1],**kw);axs[1].plot(d['q'][:,0],d['q'][:,2],**kw)
 axs[0].set(xlabel='x (m)',ylabel='y (m)',xlim=(1.45,5.05));axs[0].set_aspect('equal',adjustable='box');axs[0].set_ylim(-.8,.8)
 axs[1].plot([1.45,3.6,3.6,5.05],[0,0,.16,.16],color='.6',lw=.8);axs[1].set(xlabel='x (m)',ylabel='Root height (m)',xlim=(1.45,5.05))
 finish(fig,axs,'02_trajectory',variant)
 fig,axs=plt.subplots(2,3,figsize=(7.0,4.05),layout='constrained')
 specs=[('Root height (m)',lambda d:d['q'][:,2]),('Roll (deg)',lambda d:np.rad2deg(d['rpy'][:,0])),('Pitch (deg)',lambda d:np.rad2deg(d['rpy'][:,1])),('Forward velocity (m/s)',lambda d:d['v'][:,0]),('Body ωx (rad/s)',lambda d:d['v'][:,3]),('Body ωy (rad/s)',lambda d:d['v'][:,4])]
 for ax,(label,extract) in zip(axs.ravel(),specs):
  for e,d in data.items():ax.plot(d['time'],extract(d),**lines(e,variant))
  ax.set(xlabel='Time (s)',ylabel=label,xlim=(0,2.7))
 finish(fig,axs,'03_states',variant)
 fig,axs=plt.subplots(1,2,figsize=(7.0,2.5),layout='constrained')
 for e,d in data.items():
  axs[0].plot(d['time'],d['floor'],**lines(e,variant));axs[1].plot(d['time'],d['step'],**lines(e,variant))
 axs[0].set(xlabel='Time (s)',ylabel='Wheel–floor force (N)',xlim=(0,2.7));axs[1].set(xlabel='Time (s)',ylabel='Wheel–platform force (N)',xlim=(0,2.7))
 finish(fig,axs,'04_contact',variant)
 fig,axs=plt.subplots(2,2,figsize=(7.0,3.65),layout='constrained')
 for j,(ax,title) in enumerate(zip(axs.ravel(),['Steering','Rear wheel','Hip','Knee'])):
  for e,d in data.items():ax.plot(d['time']-.02,d['action'][:,j],**lines(e,variant))
  ax.set(xlabel='Time (s)',ylabel=title+' action',xlim=(0,2.7),ylim=(-1.05,1.05))
 finish(fig,axs,'05_actions',variant)
 summary=load(STRESS/'summary.json');fig,axs=plt.subplots(1,2,figsize=(7.0,2.7),layout='constrained')
 for j,e in enumerate(['mjx','physx']):
  kw=lines(e,variant);ss=summary['totals'][e];ci=np.array(ss['ci95'])*100;rate=ss['rate']*100
  axs[0].bar(j,rate,color=kw['color'],width=.55,hatch=None if e=='mjx' else '///',label=kw['label']);axs[0].errorbar(j,rate,yerr=[[rate-ci[0]],[ci[1]-rate]],color='black',fmt='none',capsize=3,lw=.9)
  x=np.arange(4)+(-.18 if e=='mjx' else .18);rate=np.array([summary['onsets'][e][str(k)]['successes']/2.5 for k in [0,5,10,15]])
  axs[1].bar(x,rate,width=.34,color=kw['color'],hatch=None if e=='mjx' else '///',label=kw['label'])
 axs[0].set(xticks=[0,1],xticklabels=['MuJoCo/MJX','Isaac/PhysX'],ylabel='Success rate (%)',ylim=(0,40))
 axs[1].set(xticks=np.arange(4),xticklabels=[0,5,10,15],xlabel='Pulse onset (control steps)',ylabel='Success rate (%)',ylim=(0,40))
 finish(fig,axs,'06_robustness',variant)
import sys
if '--figures-only' in sys.argv:
 print('Rebuilt figures only');raise SystemExit(0)
# Export paired raw data through x=5; success labels retain the full observation window.
records=list(csv.DictReader((STRESS/'episodes.csv').open()));csvsave(OUT/'data/stress_episodes_full_window.csv',records)
for e in ['mjx','physx']:
 pieces={k:[] for k in ['qpos','qvel','force','action','requested','effective','time_s']};offsets=[0];case_ids=[]
 for f in sorted((STRESS/e).glob('batch_*.npz')):
  begin=int(f.stem.split('_')[1]);z=np.load(f);a={k:z[k] for k in ['qpos','qvel','force','action','requested','effective','mask']}
  for lane in range(a['qpos'].shape[1]):
   n=int(a['mask'][:,lane].sum());x=a['qpos'][:n,lane,0];cross=np.flatnonzero(x>=5);n=min(n,int(cross[0])+1) if len(cross) else n
   for k in ['qpos','qvel','force','action','requested','effective']:pieces[k].append(a[k][:n,lane])
   pieces['time_s'].append((np.arange(n)+1)*.02);offsets.append(offsets[-1]+n);case_ids.append(begin+lane)
 np.savez_compressed(OUT/'data'/f'stress_{e}_x5.npz',**{k:np.concatenate(v) for k,v in pieces.items()},offsets=offsets,case_ids=case_ids)
 table=[]
 for metric,key in [('stable_1s','successes'),('stable_1s_and_no_later_failure','landing_and_timeout_count')]:
  k=summary['totals'][e][key];n=1000;z=1.95996398454;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
  table.append(dict(engine=e,metric=metric,successes=k,episodes=n,percent=100*p,ci95_low=100*(c-h),ci95_high=100*(c+h)))
 csvsave(OUT/'data'/f'success_{e}.csv',table)
(OUT/'data/nominal_landings_full_window.json').write_text(json.dumps(identity,indent=2))
(OUT/'data/plot_contract.json').write_text(json.dumps(dict(cutoff_x_m=5,method='first measured endpoint at/after x5; no time interpolation or pose edits',snapshots=snapshots,full_window_statistics=str(STRESS/'summary.json'),labels_unchanged=True,warning='x5 display ends before full1s hold confirmation in nominal runs',actor_sha256=identity['sha256']),indent=2))
print('Created quantitative figures and raw display datasets',OUT)
