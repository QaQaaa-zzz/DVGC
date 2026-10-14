"""Replot saved query prefixes and audit coverage; zero physics/optimization."""
import argparse,csv,json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jit_dvgc.skill_transfer_student_runtime import student_visit_choices
from jit_dvgc.skill_transfer_dual_teacher import observation_conflicts

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path(__file__).parent)
args=parser.parse_args();r=args.run;o=args.output;o.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
rows=[];curves=[];coverage=[];teacher_seen=Counter();all_labels=[]
for prefix,run in [('S4',r),('S4_fine',r/'refinement')]:
    if not (run/'query_banks.json').exists():continue
    for ref in read(run/'query_banks.json')['banks']:
        block=Path(ref['path']).parent;d=read(block/'declaration.json');c=d['config'];inv=read(c['student_reset_inventory']['path'])
        chosen=student_visit_choices(len(d['acquired_root_ids']),c['student_old_offset'])
        def rootid(group,index):
            if group==3:return d['acquired_root_ids'][index]
            if group==0:return 'TRAIN_NOMINAL'
            if group==1:return 'TRAIN_RANDOM-'+str(inv['random_complete']['lanes'][index%len(inv['random_complete']['lanes'])])
            return inv['pi0_success_snapshots']['entries'][index%len(inv['pi0_success_snapshots']['entries'])]['ancestor']
        with np.load(block/'student_visitation.npz') as a:
            mask=a['mask']
            for lane,(group,index) in enumerate(chosen):
                id=rootid(group,index);live=mask[:,lane] if len(mask) else np.ones(1,bool)
                q=a['qpos'][:,lane]
                curves.append(dict(root=id,group=group,offset=c['student_old_offset'],lane=lane,visit_steps=d['visit_steps'],q=q[live]))
                for step,v in enumerate(q):rows.append(dict(stage=prefix,block=block.name,root=id,group=group,lane=lane,offset=c['student_old_offset'],query_visit_step=d['visit_steps'],pre_action_step=step,alive=int(live[step]),x=float(v[0]),y=float(v[1]),z=float(v[2])))
        bank=read(ref['path'])
        all_labels.extend(bank['labels'])
        for label in bank['labels']:teacher_seen[(label['task_group'],label['root_id'],label['chosen_teacher'])]+=1
def export(name,values):
    with (o/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values)
export('S4_query_visitation.csv',rows)
conflicts=observation_conflicts(all_labels);excluded={id for v in conflicts for id in v['sample_ids']}
candidate=[v for v in all_labels if v['sample_id'] not in excluded]
remaining=Counter((v['task_group'],v['root_id'],v['chosen_teacher']) for v in candidate)
for group,id in sorted(set((g,id) for g,id,t in teacher_seen)):
    coverage.append(dict(group=group,root=id,block_pi0_labels=teacher_seen[(group,id,'pi0')],block_R_labels=teacher_seen[(group,id,'R')],candidate_pi0_labels=remaining[(group,id,'pi0')],candidate_R_labels=remaining[(group,id,'R')]))
if coverage:export('S4_label_coverage.csv',coverage)
global_audit=dict(block_eligible=len(all_labels),cross_block_conflicts=conflicts,quarantined_sample_ids=sorted(excluded),candidate_labels_after_global_quarantine=len(candidate),candidate_teachers=dict(Counter(v['chosen_teacher'] for v in candidate)),candidate_groups=dict(Counter(v['task_group'] for v in candidate)),bank_exported=False,student_updates=0,final_TEST_opened=False,adopted=False)
(o/'S4_global_label_audit.json').write_text(json.dumps(global_audit,indent=2)+'\n')
longest={}
for curve in curves:
    key=(curve['root'],curve['offset'],curve['lane'])
    if key not in longest or curve['visit_steps']>longest[key]['visit_steps']:longest[key]=curve
roots=list(dict.fromkeys(v['root'] for v in curves));ncols=4;nrows=math.ceil(len(roots)/ncols)
with (o/'trajectories.csv').open() as f:pi0=[v for v in csv.DictReader(f) if v['model']=='pi0' and v['repeat']=='0']
fig,axes=plt.subplots(nrows,ncols,figsize=(16,3*nrows),squeeze=False)
for ax,id in zip(axes.flat,roots):
    limit=max(v['visit_steps'] for v in longest.values() if v['root']==id)
    reference=[v for v in pi0 if v['root']==id and int(v['step'])<=limit]
    if reference:ax.plot([float(v['x']) for v in reference],[float(v['y']) for v in reference],color='black',linewidth=1,label='archived pi0')
    for v in longest.values():
        if v['root']!=id:continue
        q=v['q'];ax.plot(q[:,0],q[:,1],color='tab:blue' if v['offset'] in (7,21) else 'tab:orange',alpha=.65,linewidth=1.2)
        if len(q):ax.scatter(q[-1,0],q[-1,1],s=8,color='tab:blue' if v['offset'] in (7,21) else 'tab:orange')
    ax.set_title(id,fontsize=9);ax.set(xlabel='world X (m)',ylabel='world Y (m)');ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim')
for ax in list(axes.flat)[len(roots):]:ax.set_visible(False)
fig.suptitle('Frozen S4, actual TRAIN visitation prefixes, checkpoint 1405bc02\nBlue: through40 steps; orange: fine1/2/4; black: archived pi0 for7 recovery roots only.\nDots: last saved active pre-state, not a failure label; no saved old-root teacher trajectories.',fontsize=10)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(o/'S4_query_XY.png',dpi=140);plt.close(fig)
identity=read(o/'identities.json');identity['S4_actual_query_20261015']={key:read(run/'declaration.json') for key,run in [('primary',r),('refinement',r/'refinement')] if (run/'declaration.json').exists()};identity['S4_actual_query_20261015']['report_runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(o/'identities.json').write_text(json.dumps(identity,indent=2)+'\n')
print('Saved',len(rows),'pre-state rows;',len(roots),'root identities;',len(coverage),'labeled roots')
