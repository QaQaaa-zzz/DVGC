"""Read-only fixed-data coefficient comparison, including rejected trained nodes."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
read = lambda path: json.loads(Path(path).read_text())
root, out = args.run.resolve(), args.output.resolve()
plan = read(root/'plan.json')
parent = Path(read(plan['source_B'])['output'])
rows, controls = read(root/'B_checkpoint_results.json'), read(parent/'B_checkpoint_results.json')
finish, control_finish = read(root/'B_completed.json'), read(parent/'B_completed.json')
assert plan['schema'] == 'jit_retention_B_weight_ratio_v1'
assert read(root/'single_factor_verification.json')['first_raw_loss_parts_equal']


def paired(pair):
    result = {}
    for group in 'ABCD':
        a, b, ar, br = [pair[k]['labels'][group] for k in
                        ('baseline', 'original', 'repeated_pi0', 'repeated')]
        result[group] = dict(
            first_gain=sum(not x and y for x,y in zip(a,b)),
            first_loss=sum(x and not y for x,y in zip(a,b)),
            repeat_gain=sum(not x and y for x,y in zip(ar,br)),
            repeat_loss=sum(x and not y for x,y in zip(ar,br)),
            both_pair_loss=sum(x and xr and not y and not yr for x,y,xr,yr in zip(a,b,ar,br)),
            source_flips=sum(x!=xr for x,xr in zip(a,ar)),
            student_flips=sum(y!=yr for y,yr in zip(b,br)))
    return dict(cells=result, totals={k:sum(v[k] for v in result.values()) for k in result['A']},
                warned=pair['warned'], confirmed=pair['confirmed'])


matched=[]
for step,row in rows.items():
    if step=='0':continue
    before=controls[step]
    entry=dict(update=int(step), control_KEEP_DEMO=[1,1], intervention_KEEP_DEMO=[1.5,.5],
        control_TRAIN=before['train_success'], intervention_TRAIN=row['train_success'],
        control_SOLVER_DEV=before['solver_success'], intervention_SOLVER_DEV=row['solver_success'],
        control_cells=before['cells'], intervention_cells=row['cells'],
        control_eligible=before['eligible'], intervention_eligible=row['eligible'])
    for name,path in [('control',parent),('intervention',root)]:
        f=path/f'B{int(step):04d}_warning_repeat.json'
        entry[name+'_paired']=paired(read(f)) if f.exists() else None
    entry['teacher_student_matrix']=[dict(
        role=e['role'],root_id=e['root_id'],pi0_independent_labels=e['pi0_labels'],
        G_verified_solution=1 if e['G_verified'] else None,
        student_independent=e['combinations']['independent']['label'],
        frozen_R5=e['R5_label'], TRAIN_repeated_at_this_node=False)
        for role in ('train','solver') for e in row[role]]
    matched.append(entry)
result=dict(schema='jit_weight_ratio_saved_comparison_v1', only_changed_factor=plan['only_changed_factor'],
    selected_update=finish['selected_update'],learner_last_update=finish['completed_updates'],
    control_selected_update=control_finish['selected_update'], matched_updates=matched,
    rejected_trained_updates=[int(k) for k,v in rows.items() if k!='0' and not v['eligible']],
    selected_zero_is_fallback=finish['selected_update']==0,
    independent_TRAIN_repeats_apply_to_update=read(root/'selected_diagnostics.json')['TRAIN_confirmation_update'],
    reliable_recovery_improvement='NOT_ESTABLISHED',
    source_provenance=read(root/'single_factor_verification.json'),
    interpretation='Compare identical update counts first. A zero-update fallback is not learned retention improvement. First/repeat numerical differences are not independent training seeds. TRAIN absorption, unseen SOLVER_DEV transfer and complete-task gains remain separate.',
    next_experiment_proposal=dict(choice='B',question='TRAIN student-visited-state closed-loop imitation diagnosis',
        source_student='fixed accepted control BC1000; no rejected BC100 adoption',
        source_data='new TRAIN visits only; no DEV failure reuse',
        charged_upper_bound=544000,wall_hours=3,student_updates=0,PPO_updates=0,E_G_updates=0,
        execution_authorized=False,
        distinguishable_outcomes=['teacher rescue works on own-visited states: supports guidance/distribution hypothesis',
            'teacher cannot rescue: coverage/teacher limitation; do not blame imitation capacity',
            'verified guidance later fits offline but fails closed loop: motivates imitation-path diagnosis; no automatic training']))
with (out/'weight_comparison.json').open('x') as stream:json.dump(result,stream,indent=2)
# Prevent a selected-zero retention count from being read as a learned improvement.
summary=read(out/'summary.json')
summary.update(weight_comparison='weight_comparison.json',selected_zero_is_fallback=result['selected_zero_is_fallback'],
               learned_retention_improvement='NOT_ESTABLISHED',
               selected_retention_decrease_applies_to_trained_policy=finish['selected_update']!=0)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'INDEX.md').open('a') as stream:
    stream.write('\nWeight comparison: [same-update control, teacher/student matrix, paired repeats](weight_comparison.json). '
                 'If selected update is0, it is the original pi0 fallback and cannot count as learned retention improvement. '
                 'TRAIN repeats belong only to the declared selected node; rejected trained nodes have first-pass absorption, not confirmed absorption.\n')
print(out/'weight_comparison.json')
