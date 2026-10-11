"""Saved-only T1 review. No environment, optimizer, physics, or TEST execution."""
import argparse,csv,json,pickle,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jit_dvgc.retention_t0_diagnostics import SourceRegistry,load_inputs,make_predictors,error_statistics,tree_sha
from jit_dvgc.retention_t1 import mechanism_decision

parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
r=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(r/'plan.json');e=read(r/'evaluation_results.json');d=read(r/'mechanism_decision.json')
train=read(p['train_roots']);solver=read(p['solver_roots']);arms=('PPO_keep','PPO_demo_keep')
recomputed=mechanism_decision(e['source_DEV'],e['arms'],p['required_train_roots'],train_root_ids=[x['root']['root_id'] for x in train],solver_root_ids=[x['root']['root_id'] for x in solver]);assert d==recomputed
for f,h in p['locks'].items():assert sha(f)==h,('locked source drift',f)
summary={'schema':'jit_T1_saved_review_v1','run':str(r),'new_physical_steps':0,'new_optimizer_updates':0,'TEST_used':False,'gate_recomputed_equal':True,'evaluation_status':'complete','t1_passed':d['t1_passed'],'budget':read(r/'report_0001/summary.json')['costs'],'arms':{},'source_DEV':{}}
for name in ('pi0',)+arms:
 repeats=e['source_DEV'] if name=='pi0' else e['arms'][name]['DEV'];cls=d['source_DEV_classification'] if name=='pi0' else d['arms'][name]['DEV_classification']
 data={'per_repeat_success':[sum(sum(x) for x in row.values()) for row in repeats],'per_group_success':{g:[sum(row[g]) for row in repeats] for g in 'ABCD'},'stable_success':sum(v=='stable_success' for v in cls.values()),'stable_failure':sum(v=='stable_failure' for v in cls.values()),'ambiguous':sum(v=='ambiguous' for v in cls.values())}
 if name=='pi0':summary['source_DEV']=data
 else:summary['arms'][name]={'DEV':data,'stable_train':d['arms'][name]['stable_train_success'],'solver_first_success':d['arms'][name]['solver_success'],'stable_gained':d['arms'][name]['stable_gained'],'stable_lost':d['arms'][name]['stable_lost'],'stable_net':d['arms'][name]['stable_net'],'old_ambiguous':d['arms'][name]['old_ambiguous'],'TRAIN_labels':{k:[z[k] for z in e['arms'][name]['TRAIN']] for k in e['arms'][name]['TRAIN'][0]},'confirmed_regression':d['arms'][name]['confirmed_regression'],'warning_cells_by_repeat':d['arms'][name]['warning_cells_by_repeat']}
fig,axes=plt.subplots(2,4,figsize=(17,8));figz,az=plt.subplots(2,4,figsize=(17,8));figtime,at=plt.subplots(2,4,figsize=(17,8));rows=[];trace_receipts=[];failure_counts={arm:{} for arm in arms}
for ordinal,entry in enumerate(train):
 rid=entry['root']['root_id'];lane=entry['lane'];ax=axes.flat[ordinal];axz=az.flat[ordinal];axt=at.flat[ordinal]
 for arm,color in zip(arms,('#2867b2','#d65f29')):
  labels=[]
  for rep in range(3):
   root=r/f'{arm}_TRAIN_repeat{rep}_root{ordinal:02d}';res=read(root/'results.json');chosen=res[lane];attempt=chosen['attempts'][0];assert chosen['root_id']==rid and attempt['snapshot_context_sha256']==entry['root']['snapshot_context_sha256'];node=read(r/'training'/arm/'checkpoint_032000.json');assert attempt['actor_sha256']==node['actor_sha256'] and attempt['normalizer_sha256']==node['normalizer_sha256'];f=Path(attempt['trace']);assert sha(f)==attempt['trace_sha256'];trace_receipts.append({'path':str(f),'sha256':sha(f),'lane':attempt['trace_lane'],'root':rid,'arm':arm,'repeat':rep})
   with np.load(f) as data:
    idx=attempt['trace_lane'];mask=data['mask'][:,idx].astype(bool);valid=np.flatnonzero(mask);assert len(valid)==attempt['steps'] and len(valid)>0;last=valid[-1];q=data['qpos'][mask,idx];ts=data['time_after'][mask,idx];y=data['roll'][mask,idx];physical=bool(data['physical_failure'][last,idx]);label=int(bool(data['valid_contact'][last,idx]) and not physical);assert label==chosen['label']==e['arms'][arm]['TRAIN'][rep][rid];labels.append(label)
    reasons=[key.split('/')[-1] for key in ('metric/terminal/roll_limit','metric/terminal/pitch_limit','metric/terminal/yaw_limit','metric/terminal/stuck','metric/terminal/jump_zone_missed','metric/terminal/timeout') if key in data and data[key][last,idx]>0];reason='+'.join(reasons) if reasons else attempt['outcome'];failure_counts[arm][reason]=failure_counts[arm].get(reason,0)+1
    rows.append({'root':rid,'arm':arm,'repeat':rep,'label':label,'steps':len(valid),'end_time':float(ts[-1]),'failure_reason':reason,'end_code':int(data['end_code'][last,idx]),'first_contact_index':next((int(x) for x in valid if data['first_valid_contact'][x,idx]),None),'endpoint_recovery_ticks':int(data['recovery_ticks'][last,idx]),'max_abs_roll':float(np.max(np.abs(y))),'task_return':attempt['task_return'],'max_x':float(np.max(q[:,0])),'max_z':float(np.max(q[:,2]))})
    for axis,b in ((ax,1),(axz,2)):
     axis.plot(q[:,0],q[:,b],color=color,alpha=.65,linestyle=('-','--',':')[rep],linewidth=1.5,label=arm if rep==0 else None);axis.plot(q[-1,0],q[-1,b],marker='o' if label else 'x',color=color,ms=5)
    axt.plot(ts,y,color=color,alpha=.7,linestyle=('-','--',':')[rep],label=arm if rep==0 else None)
  for axis in (ax,axz,axt):axis.set_title(f'root {rid.rsplit("-",1)[1]} | {arm if False else "keep/demo labels below"}',fontsize=9)
 ax.text(.02,.98,'keep '+str(summary['arms'][arms[0]]['TRAIN_labels'][rid])+'\ndemo '+str(summary['arms'][arms[1]]['TRAIN_labels'][rid]),transform=ax.transAxes,va='top',fontsize=8)
 for axis in (ax,axz):axis.set_xlabel('World X (m)');axis.set_ylabel('World Y (m)' if axis==ax else 'World Z (m)');axis.set_aspect('equal',adjustable='datalim');axis.grid(alpha=.2)
 axt.set_xlabel('Original task time (s)');axt.set_ylabel('Roll (rad)');axt.grid(alpha=.2)
for f,axis,title,name in ((fig,axes,'All TRAIN roots: original world XY, 32k endpoint; three fixed repeats','xy_train_roots.png'),(figz,az,'All TRAIN roots: original world XZ, 32k endpoint; three fixed repeats','xz_train_roots.png'),(figtime,at,'Saved TRAIN roll traces at true endpoints; three fixed repeats','roll_train_roots.png')):
 axis.flat[0].legend(fontsize=7);f.suptitle(title+'\nSolid/dashed/dotted = repeats 0/1/2; circle = recovery, x = failure; no ideal geometric path defined',fontsize=11);f.tight_layout(rect=(0,0,1,.94));f.savefig(out/name,dpi=150);plt.close(f)
with (out/'train_trajectory_metrics.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
summary['TRAIN_endpoint_reasons']=failure_counts;write(out/'trajectory_provenance.json',trace_receipts)
reg=SourceRegistry();_,_,_,_,corpora,_=load_inputs(Path('/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_keep_coverage_003'),Path(p['parent']).parent,reg)
predict,_=make_predictors();cross={}
with open(r/'training'/'PPO_keep'/'inference_000000.pkl','rb') as stream:source_params=pickle.load(stream)
corpora[2].actions=predict(source_params,corpora[2].observations,independent=True)
assert np.max(np.abs(predict(source_params,corpora[2].observations)-corpora[2].actions))<1e-6
for arm in arms:
 for step in (0,16000,32000):
  node=read(r/'training'/arm/f'checkpoint_{step:06d}.json');assert sha(node['inference'])==node['inference_sha256']
  with open(node['inference'],'rb') as stream:params=pickle.load(stream)
  assert tree_sha(params[0])==node['normalizer_sha256'] and tree_sha(params[1])==node['actor_sha256']
  cross[f'{arm}_{step}']={}
  for corpus in corpora:
   prediction=predict(params,corpus.observations);cross[f'{arm}_{step}'][corpus.name]={'all':error_statistics(prediction,corpus.actions),'per_root':{rid:error_statistics(prediction[corpus.roots==rid],corpus.actions[corpus.roots==rid]) for rid in dict.fromkeys(corpus.roots)}}
reg.verify_unchanged();write(out/'common_observation_errors.json',cross);write(out/'common_observation_sources.json',reg.files)
fig,axs=plt.subplots(1,3,figsize=(14,4));summary['actual_update_diagnostics']={}
for arm,color in zip(arms,('#2867b2','#d65f29')):
 updates=sorted([json.loads(x) for x in (r/'training'/arm/'actual_updates.jsonl').read_text().splitlines()],key=lambda x:x['actual/optimizer_updates']);assert [int(x['actual/optimizer_updates']) for x in updates]==list(range(1,81))
 arrays={k:np.array([u[k] for u in updates]) for k in ('actual/gradient_norm_ppo','actual/gradient_norm_demo','actual/gradient_norm_keep','actual/critic_gradient_norm','actual/actor_gradient_norm','actual/global_clip_scale','actual/actor_adam_delta_norm','actual/critic_adam_delta_norm','actual/cosine_ppo_demo','effective_lambda_demo')};arrays['weighted_demo_to_ppo_norm']=arrays['actual/gradient_norm_demo']/arrays['actual/gradient_norm_ppo'];arrays['weighted_keep_to_ppo_norm']=arrays['actual/gradient_norm_keep']/arrays['actual/gradient_norm_ppo'];summary['actual_update_diagnostics'][arm]={k:{'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())} for k,a in arrays.items()};summary['actual_update_diagnostics'][arm]['negative_ppo_demo_cosine_updates']=int(np.sum(arrays['actual/cosine_ppo_demo']<0))
 x=np.arange(1,81);axs[0].plot(x,arrays['weighted_demo_to_ppo_norm'],color=color,label=arm);axs[1].plot(x,arrays['actual/global_clip_scale'],color=color);axs[2].plot(x,arrays['actual/actor_adam_delta_norm'],color=color)
for axis,title in zip(axs,('Weighted demo/PPO actor gradient norm','Global clip scale (actor and critic)','Actual actor Adam delta norm')):axis.set_title(title,fontsize=10);axis.set_xlabel('Actual optimizer update');axis.grid(alpha=.2)
axs[0].legend();fig.tight_layout();fig.savefig(out/'actual_update_diagnostics.png',dpi=150);plt.close(fig)
write(out/'summary.json',summary)
print(json.dumps(summary,indent=2))
