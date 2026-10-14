"""Offline audit of the zero-update late-state query run; no physics execution."""
import json,csv,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from jit_dvgc.skill_transfer_dual_teacher import audit_online_label

r=Path('/home/qy/DVGC/JIT/runs/experiments/frontierbridge_student_query_coverage_20261014')
o=Path(__file__).parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];blocks=[];inactive=0;live=0
for ref in read(r/'query_banks.json')['banks']:
    assert sha(ref['path'])==ref['sha256']
    p=Path(ref['path']).parent;bank=read(ref['path']);labels={v['sample_id']:v for v in bank['labels']}
    for label in labels.values():audit_online_label(label,bank['teachers'])
    with np.load(p/'student_visitation.npz') as a:
        mask=a['mask'];assert not np.any(mask[1:] & ~mask[:-1])
        inactive+=int((~mask).sum());live+=int(mask.sum())
        for k in range(1,len(mask)):
            idx=~mask[k-1]
            np.testing.assert_array_equal(a['observations'][k,idx],a['observations'][k-1,idx])
            np.testing.assert_array_equal(a['qpos'][k,idx],a['qpos'][k-1,idx])
    receipt=read(p/'query_receipts.json') if (p/'query_receipts.json').exists() else dict(queries=[])
    step=read(p/'declaration.json')['visit_steps']
    blocks.append(dict(block=p.name,visit_step=step,queries=len(receipt['queries']),labels=len(labels),inactive_trace_steps=int((~mask).sum()),known_teacher_charged=receipt.get('charged_interactions',0)))
    for item in receipt['queries']:
        q=item['query'];assert sha(q['context_receipt']['path'])==q['context_receipt']['sha256']
        context=read(q['context_receipt']['path'])
        for f in context['files']:assert sha(f['path'])==f['sha256']
        identity=read(Path(q['context_receipt']['path']).parent/'identity.json')
        branches={}
        for teacher,ref in item['branch_receipts'].items():
            assert sha(ref['path'])==ref['sha256'];branches[teacher]=read(ref['path'])
        rows.append(dict(sample_id=q['sample_id'],group=q['task_group'],root=q['root_id'],visit_step=step,absolute_control_step=q['control_step'],active_phase=identity['active_phase'],time=q['time_before'],remaining_horizon=q['remaining_horizon'],label_used=int(q['sample_id'] in labels),chosen_teacher=labels[q['sample_id']]['chosen_teacher'] if q['sample_id'] in labels else 'abstain',pi0_labels='/'.join(map(str,branches['pi0']['repeat_labels'])),R_labels='/'.join(map(str,branches['R']['repeat_labels']))))
def export(name,data):
    with (o/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
export('late_query_blocks.csv',blocks)
if rows:export('late_query_audit.csv',rows)
def branch_class(row):
    a,b=row['pi0_labels'],row['R_labels']
    if a not in ('1/1/1','0/0/0') or b not in ('1/1/1','0/0/0'):return 'unresolved'
    return 'both_success' if a==b=='1/1/1' else 'both_failure' if a==b=='0/0/0' else 'pi0_only' if a=='1/1/1' else 'R_only'
summary=dict(status=read(r/'status.json'),blocks=blocks,queries=len(rows),verified_labels=sum(v['label_used'] for v in rows),query_phase_counts=dict(Counter(v['active_phase'] for v in rows)),label_phase_counts=dict(Counter(v['active_phase'] for v in rows if v['label_used'])),label_groups=dict(Counter(v['group'] for v in rows if v['label_used'])),chosen_teachers=dict(Counter(v['chosen_teacher'] for v in rows if v['label_used'])),branch_classes=dict(Counter(branch_class(v) for v in rows)),inactive_trace_steps_audited=inactive,live_visitation_steps=live,terminal_freeze_audit='no resurrection; exact frozen inactive observations/qpos',terminal_cause_limitation='mask combines success and done; saved visitation NPZ cannot separately count completed success and failure',student_updates=0,R_PPO_steps=0,labels_merged_into_training=False,final_TEST_opened=False,adopted=False)
(o/'late_query_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
