"""Saved-data diagnostic only. No environment imports, training, or simulation."""
import json,pickle,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_002')
OUT=ROOT/'diagnosis_001'
read=lambda x:json.loads(Path(x).read_text())
p=read(ROOT/'plan.json');rows=read(ROOT/'B_checkpoint_results.json');pair=read(ROOT/'B1000_warning_repeat.json')
normal=pickle.load((ROOT/'B/update_0000.pkl').open('rb'))[0]
std=np.maximum(np.asarray(normal.std['state'],dtype=np.float64),1e-8)
keep_receipt=read(p['keep_receipt']);keep=np.load(keep_receipt['path'])['actor_observation_before'].astype(np.float64)
# Distance is descriptive raw-observation RMSE divided by frozen std; no clipping, policy metric, or coverage guarantee.
def nearest(obs):
    obs=np.asarray(obs,dtype=np.float64)
    d=[]
    for o in obs:
        v=np.mean(((keep-o)/std)**2,axis=1)
        d.append(float(np.sqrt(v.min())))
    return dict(median=float(np.median(d)),maximum=float(np.max(d)),p90=float(np.quantile(d,.9)),first=float(d[0]))
def first(z,key,lane,mask):
    ids=np.flatnonzero(z[key][:,lane]&mask)
    return int(ids[0]) if len(ids) else None

def events(z,lane,mask):
    n=int(mask.sum());assert np.all(mask[:n]) and not np.any(mask[n:])
    return dict(steps=n,success=bool(np.any(z['success'][:,lane]&mask)),end_code=int(z['end_code'][n-1,lane]),**{k:first(z,k,lane,mask) for k in ('first_valid_contact','roll_limit','prohibited_contact','physical_failure')},recovery_ticks_end=int(z['recovery_ticks'][n-1,lane]),last_roll=float(z['roll'][n-1,lane]))
def compare(a,b,la,lb,mask_key):
    ma=a[mask_key][:,la];mb=b[mask_key][:,lb];n=min(int(ma.sum()),int(mb.sum()))
    obs=a['actor_observation_before'][:n,la];ob=b['actor_observation_before'][:n,lb]
    ac=a['normalized_action_executed'][:n,la]-b['normalized_action_executed'][:n,lb]
    def diff(x):
        v=np.max(np.abs(x).reshape(n,-1),axis=1);ix=np.flatnonzero(v>1e-6)
        return dict(first=int(ix[0]) if len(ix) else None,maxabs=float(v.max()))
    return dict(common_steps=n,observation=diff(obs-ob),executed_action=diff(ac),qpos_after=diff(a['qpos'][:n,la]-b['qpos'][:n,lb]),qvel_after=diff(a['qvel'][:n,la]-b['qvel'][:n,lb]),action_rmse_channels=np.sqrt(np.mean(ac**2,axis=0)).tolist(),initial_observation_maxabs=float(np.max(np.abs(obs[0]-ob[0]))))
loss=[]
for case in ('B_06','C_10'):
    lane=next(i for i,c in enumerate(p['batches'][0]['cases']) if c['case']==case)
    records=[]
    for run,ap,bp in [('first',p['reused_zero']['DEV']['pi0'],rows['1000']['full']['output']),('repeat',pair['repeated_pi0']['output'],pair['repeated']['output'])]:
        with np.load(Path(ap)/'prefixes.npz') as a,np.load(Path(bp)/'prefixes.npz') as b:
            ma=a['prefix_mask'][:,lane];mb=b['prefix_mask'][:,lane]
            records.append(dict(run=run,pi0=events(a,lane,ma),student=events(b,lane,mb),difference=compare(a,b,lane,lane,'prefix_mask'),pi0_keep_distance=nearest(a['actor_observation_before'][ma,lane]),student_keep_distance=nearest(b['actor_observation_before'][mb,lane]),pi0_trace=str(Path(ap)/'prefixes.npz'),student_trace=str(Path(bp)/'prefixes.npz')))
    loss.append(dict(case=case,lane=lane,runs=records,gradient_allowed=False))
train=[]
metadata={v['root']['root_id']:v for v in read(p['train_roots'])}
for entry in rows['1000']['train']:
    rid=entry['root_id'];meta=metadata[rid];student=entry['combinations']['independent']['attempt'];teacher=meta['teacher']['methods']['G']['verified_attempts'][-1]
    with np.load(student['trace']) as a,np.load(teacher['trace']) as b:
        la=student['trace_lane'];lb=teacher['trace_lane'];ma=a['mask'][:,la];mb=b['mask'][:,lb]
        am=a['actor_observation_before'][ma,la];bm=b['actor_observation_before'][mb,lb]
        rec=dict(root_id=rid,snapshot_control_step=meta['root']['snapshot_control_step'],student=events(a,la,ma),teacher=events(b,lb,mb),difference=compare(a,b,la,lb,'mask'),student_keep_distance=nearest(am),teacher_keep_distance=nearest(bm),student_trace=student['trace'],teacher_trace=teacher['trace'])
        contact=rec['student']['first_valid_contact']
        if contact is not None:
            rec['student_post_contact_keep_distance']=nearest(am[contact:])
        n=min(len(am),len(bm));rec['matched_tick_normalized_observation_rmse']=dict(prefix16=float(np.sqrt(np.mean(((am[:16]-bm[:16])/std)**2))),after16=float(np.sqrt(np.mean(((am[16:n]-bm[16:n])/std)**2))))
        train.append(rec)
with np.load(Path(p['keep_receipt']).parent/'keep_collection/prefixes.npz') as z:
    keep_phases=[]
    for e in keep_receipt['episodes']:
        lane=e['lane'];mask=z['prefix_mask'][:,lane];n=int(mask.sum());contact=first(z,'first_valid_contact',lane,mask)
        keep_phases.append(dict(ancestor=e['ancestor'],group=e['group'],steps=e['steps'],first_valid_contact=contact,post_contact_observations=0 if contact is None else n-contact))
inputs=[ROOT/'plan.json',ROOT/'B_checkpoint_results.json',ROOT/'B1000_warning_repeat.json',Path(p['keep_receipt']),Path(p['train_roots']),ROOT/'B/update_0000.pkl']
result=dict(schema='jit_saved_bc_diagnosis_v1',source_HEAD='aee1dbe741cf66cce01818009db0e2ac3c07e6d3',inputs={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs},new_physics=0,new_supervised_updates=0,old_losses=loss,train=train,keep_phase_records=keep_phases,keep_phase_summary={g:dict(trajectories=sum(x['group']==g for x in keep_phases),observations=sum(x['steps'] for x in keep_phases if x['group']==g),post_contact_observations=sum(x['post_contact_observations'] for x in keep_phases if x['group']==g)) for g in ('nominal','random')},distance_definition='Unclipped raw 76D observation nearest-neighbor RMSE scaled by fixed pi0 std; descriptive, not a membership threshold or causal proof.',budget=dict(charged_D2=388036,charged_remaining=2000000-388036,original_started_unix=p['original_D2_started_unix'],checked_unix=time.time(),wall_remaining_seconds=max(0,43200-(time.time()-p['original_D2_started_unix']))))
(OUT/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('losses',[(x['case'],[(y['run'],y['pi0'],y['student'],y['difference']['executed_action']['first']) for y in x['runs']]) for x in loss]);print('keep',result['keep_phase_summary']);print('TRAIN',[(x['root_id'],x['student']['success'],x['student']['first_valid_contact'],x['student']['physical_failure'],x['matched_tick_normalized_observation_rmse'],x['student_keep_distance']['median'],x['teacher_keep_distance']['median']) for x in train]);print('budget',result['budget'])
# Narrow same-observation evidence; exact observation equality does not assert equal complete hidden physical state.
for case in result['old_losses']:
 for r in case['runs']:
  with np.load(r['pi0_trace']) as a,np.load(r['student_trace']) as b:
   i=case['lane'];o=a['actor_observation_before'][0,i]
   r['initial_action_delta_student_minus_pi0']=(b['normalized_action_executed'][0,i]-a['normalized_action_executed'][0,i]).tolist()
   r['initial_exact_keep_observation_matches']=int(np.all(keep==o,axis=1).sum())
(OUT/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(14,8))
for row,c in enumerate(result['old_losses']):
 for repeat,r in enumerate(c['runs']):
  for name,path,col in [('pi0',r['pi0_trace'],'black'),('BC1000',r['student_trace'],'tab:orange')]:
   with np.load(path) as z:
    lane=c['lane'];m=z['prefix_mask'][:,lane];xy=z['qpos'][m,lane,:2];t=np.arange(len(xy));style='-' if repeat==0 else '--'
    label=name+' '+r['run'];axes[row,0].plot(*xy.T,style,color=col,label=label,lw=1);axes[row,0].scatter(*xy[-1],color=col,marker='o' if r[name if name=='pi0' else 'student']['success'] else 'x')
    axes[row,1].plot(t,z['roll'][m,lane],style,color=col,label=label,lw=1);axes[row,2].plot(t,z['recovery_ticks'][m,lane],style,color=col,label=label,lw=1)
 axes[row,0].set(title=c['case']+' actual XY',xlabel='World x (m)',ylabel='World y (m)');axes[row,0].set_aspect('equal',adjustable='datalim');axes[row,1].set(title='Roll after control step',xlabel='Original episode control step',ylabel='Roll (rad)');axes[row,2].set(title='Recorded recovery counter',xlabel='Original episode control step',ylabel='Recovery ticks')
 axes[row,0].legend(fontsize=7)
fig.suptitle('Two repeated paired old losses; frozen pi0 vs BC1000; true endpoints')
fig.tight_layout();fig.savefig(OUT/'xy_old_losses.png',dpi=140);plt.close(fig)
print('initial action evidence',[(c['case'],[(x['initial_action_delta_student_minus_pi0'],x['initial_exact_keep_observation_matches']) for x in c['runs']]) for c in result['old_losses']])
