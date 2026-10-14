import json,csv,hashlib,zipfile,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jit_dvgc.skill_transfer_dual_teacher import audit_online_label

r=Path('/home/qy/DVGC/JIT/runs/experiments/frontierbridge_student_dense_20261014')
old=Path('/home/qy/DVGC/JIT/runs/experiments/frontierbridge_skill_transfer_20261014')
o=r/'results';o.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvout(name,rows):
    with (o/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
bank=read(r/'bank/labels.json');labels=bank['labels']
for row in labels:audit_online_label(row,bank['teachers'])
queryrows=[];contexts=Counter();visits=0;charged=0
for ref in read(r/'completed_query_receipts.json')['banks']:
    assert sha(ref['path'])==ref['sha256']
    block=Path(ref['path']).parent
    receipt=read(block/'query_receipts.json');charged+=receipt['charged_interactions']
    with np.load(block/'student_visitation.npz') as a:
        mask=a['mask'];visits+=int(mask.sum())
        assert not np.any(mask[1:] & ~mask[:-1]),'terminal world resurrected'
        for k in range(1,len(mask)):
            inactive=~mask[k-1]
            np.testing.assert_array_equal(a['observations'][k,inactive],a['observations'][k-1,inactive])
            np.testing.assert_array_equal(a['qpos'][k,inactive],a['qpos'][k-1,inactive])
    for item in receipt['queries']:
        q=item['query'];ctx=read(q['context_receipt']['path'])
        assert sha(q['context_receipt']['path'])==q['context_receipt']['sha256']
        for f in ctx['files']:assert sha(f['path'])==f['sha256']
        identity=read(Path(q['context_receipt']['path']).parent/'identity.json')
        phase=identity['active_phase'];contexts[phase]+=1
        branches={}
        for teacher,ref in item['branch_receipts'].items():
            assert sha(ref['path'])==ref['sha256'];branches[teacher]=read(ref['path'])
        target=next((v['chosen_teacher'] for v in labels if v['sample_id']==q['sample_id']),'abstain')
        queryrows.append(dict(sample_id=q['sample_id'],group=q['task_group'],root=q['root_id'],visit_step=read(block/'declaration.json')['visit_steps'],absolute_control_step=q['control_step'],time=q['time_before'],remaining_horizon=q['remaining_horizon'],active_phase=phase,branches_verified=int(item['labeled']),label_used=int(target!='abstain'),chosen_teacher=target,pi0_labels='/'.join(map(str,branches['pi0']['repeat_labels'])),R_labels='/'.join(map(str,branches['R']['repeat_labels'])),context_sha256=q['context_sha256']))
csvout('query_audit.csv',queryrows)
coverage=[]
for group,root in sorted(set((v['task_group'],v['root_id']) for v in labels)):
    rows=[v for v in labels if v['task_group']==group and v['root_id']==root]
    coverage.append(dict(group=group,root=root,labels=len(rows),pi0_labels=sum(v['chosen_teacher']=='pi0' for v in rows),R_labels=sum(v['chosen_teacher']=='R' for v in rows),absolute_steps='/'.join(map(str,sorted(set(v['control_step'] for v in rows))))))
csvout('label_coverage.csv',coverage)
ret=read(r/'retention_summary.json');s3=read(old/'S3_retention_DEV/results.json');s2=read(old/'S2_retention_DEV/results.json')
retainrows=[]
for v in ret['conditions']:
    id=v['case_id'];row=dict(case_id=id,group=v['group'],pi0_labels='/'.join(map(str,v['pi0_labels'])),S2_labels='/'.join(map(str,s2[id]['labels'])),S3_labels='/'.join(map(str,s3[id]['labels'])),S4_labels='/'.join(map(str,v['student_labels'])),old_positive_denominator=int(v['group']!='teacher_verified_expansion' and v['pi0']==1),S4_status=('retained' if v['student']==1 else 'lost' if v['student']==0 else 'unresolved') if v['pi0']==1 else 'baseline_unqualified')
    retainrows.append(row)
csvout('retention_by_case.csv',retainrows)
models={'pi0':read(old/'pi0_full_train_results.json'),'R_BC':read(old/'R1_full_train_summary.json')['per_root'],'S3':read(old/'S3_independent/summary.json')['per_root'],'S4':read(r/'TRAIN/summary.json')['per_root']}
traces=[];table=[]
for model,roots in models.items():
    for root,row in roots.items():
        table.append(dict(model=model,root=root,labels='/'.join(map(str,row['labels'])),stable_success=int(row['labels']==[1,1,1])))
        for repeat,receipt in enumerate(row['receipts']):
            source=next(v for v in read(Path(receipt)/'results.json') if v['root_id']==root);attempt=source['attempts'][0]
            path=Path(attempt['trace']);assert sha(path)==attempt['trace_sha256']
            with np.load(path) as a:
                lane=attempt['trace_lane'];valid=np.flatnonzero(a['mask'][:,lane]);q=a['qpos'][valid,lane]
                for i,t in enumerate(valid):traces.append(dict(model=model,root=root,repeat=repeat,step=i+1,time=float(a['time_after'][t,lane]),x=float(q[i,0]),y=float(q[i,1]),z=float(q[i,2]),reward=float(a['reward'][t,lane]),physical_failure=int(a['physical_failure'][t,lane]),success=int(a['success'][t,lane]),timeout=int(a['timeout'][t,lane])))
csvout('root_results.csv',table);csvout('trajectories.csv',traces)
colors=dict(pi0='black',R_BC='tab:blue',S3='tab:purple',S4='tab:green')
for kind in ('XY','reward'):
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    for ax,root in zip(axes.flat,models['S4']):
        for model in models:
            rows=[v for v in traces if v['root']==root and v['model']==model and v['repeat']==0]
            xx=[v['x' if kind=='XY' else 'step'] for v in rows];yy=[v['y' if kind=='XY' else 'reward'] for v in rows]
            ax.plot(xx,yy,label=model,color=colors[model],linewidth=1.2)
            ax.scatter(xx[-1],yy[-1],marker='x' if rows[-1]['physical_failure'] else 'o',s=20,color=colors[model])
        ax.set_title(root);ax.grid(alpha=.2)
        if kind=='XY':ax.set(xlabel='world X (m)',ylabel='world Y (m)');ax.set_aspect('equal',adjustable='datalim')
        else:ax.set(xlabel='actual control transitions',ylabel='original per-step reward')
    axes.flat[0].legend();fig.suptitle('Independent Actor, original TRAIN roots, numeric repeat 0\npi0 reference; x physical failure / o other endpoint. All three repeats in CSV.')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(o/(kind+'.png'),dpi=140);plt.close(fig)
summary=dict(TRAIN={m:sum(v['stable_success'] for v in table if v['model']==m) for m in models},TRAIN_denominator=8,old_retention=ret['counts'],label_count=len(labels),group_counts=dict(Counter(v['task_group'] for v in labels)),chosen_teachers=dict(Counter(v['chosen_teacher'] for v in labels)),queries=len(queryrows),abstentions=sum(not v['label_used'] for v in queryrows),queried_phase_counts=dict(contexts),labeled_phase_counts=dict(Counter(v['active_phase'] for v in queryrows if v['label_used'])),unique_contexts=len(set(v['context_sha256'] for v in queryrows)),known_teacher_charged=charged,interrupted_block_unknown_reserved=33600,student_live_visitation_transitions=visits,BC_updates=2000,added_R_PPO_steps=0,previous_S1_S2_S3_labels_reused=0,finish_phase_seconds=read(r/'dense_status.json')['wall_seconds'],approximate_total_wall_seconds=read(r/'dense_status.json')['updated_unix']-(r/'declaration.json').stat().st_mtime,terminal_mask_audit='no resurrection; inactive observations/qpos frozen in all saved visitation traces',final_TEST_opened=False,adopted=False)
(o/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
identities=dict(config_sha256=sha(r/'config.json'),query_code_commit='38c28d10',process_isolation_commit='d8521e11',parent_student=read(r/'config.json')['student_initializer'],R_teacher=read(r/'config.json')['recovery_inference'],final_checkpoint=dict(path=str(r/'learner/update_2000.pkl'),sha256=sha(r/'learner/update_2000.pkl')),bank_receipt_sha256=sha(r/'bank/labels.json'),teacher_identities=bank['teachers'])
(o/'identities.json').write_text(json.dumps(identities,indent=2)+'\n')
print(json.dumps(summary,indent=2))
