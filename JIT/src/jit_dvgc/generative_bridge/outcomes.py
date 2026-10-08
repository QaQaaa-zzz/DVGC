"""Teacher evidence, independent student results and adoption are separate axes."""
from collections import Counter

TEACHER_FOUND = {'not_scheduled':None, 'searched_no_solution':False,
                 'verified_solution':True, 'replay_rejected':None, 'incomplete':None, 'invalid':None}


def root_outcome(raw, *, round_demo_samples_used=0):
    r=dict(raw)
    status=r['teacher_status']
    if status not in TEACHER_FOUND: raise ValueError('unknown teacher status')
    found=TEACHER_FOUND[status]
    if 'teacher_found' in r and r['teacher_found'] is not found:
        raise ValueError('teacher evidence cannot be rewritten by student success')
    label=r.get('student_label'); adopted=r.get('student_adopted')
    if label not in (0,1,None) or (adopted is not None and type(adopted) is not bool):
        raise ValueError('nullable independent student label/adoption required')
    available=r.get('n_direct_demo_examples_available',0)
    used=r.get('n_direct_demo_samples_used',0)
    if any(type(v) is not int or v<0 for v in (available,used,round_demo_samples_used)):
        raise ValueError('nonnegative actual sample counts required')
    if used and not available: raise ValueError('used demo without available examples')
    if status=='replay_rejected' and (used or available):raise ValueError('rejected replay cannot supply demo samples')
    if round_demo_samples_used < used: raise ValueError('round demo count below root count')
    kind='unknown_or_incomplete'
    if label is not None and status not in ('invalid','incomplete'):
        prefix={'verified_solution':'teacher_solved','searched_no_solution':'teacher_unsolved',
                'not_scheduled':'teacher_not_scheduled','replay_rejected':'teacher_replay_rejected'}[status]
        kind=prefix+'_student_'+('succeeded' if label==1 else 'failed')
    r.update(schema='jit_bidirectional_root_outcome_v1_1',teacher_found=found,
        outcome_class=kind,student_adopted=adopted,
        n_direct_demo_examples_available=available,n_direct_demo_samples_used=used,
        has_direct_teacher_demo=available>0,round_used_teacher_demo=round_demo_samples_used>0,
        causal_attribution='not_identified_in_single_run')
    return r


def result_matrix(rows):
    counts=Counter((r['outcome_class'],str(r.get('student_adopted'))) for r in rows)
    return {'total_roots':len(rows),'cells':[{'outcome_class':k[0],
        'adoption':k[1],'count':v} for k,v in sorted(counts.items())],
        'teacher_status_counts':dict(Counter(r['teacher_status'] for r in rows))}
