"""Opt-in finite promotion coordinator; simulator/PPO callbacks must be supplied.

Callbacks accept an isolated context and return JSON-compatible results plus actual
``physics`` and ``updates`` costs. This module does not claim to implement GPU
collection or optimization. Current-source publication is one atomic operation,
only after all five callback receipts have passed their contracts.
"""
from copy import deepcopy
import math
from pathlib import Path
import time

from .contracts import digest
from .protocol import atomic_json

STAGES=('collect_fresh','teacher_student','accept','update_explorer','update_generator')
REWARD_WEIGHTS=dict(adoption_bonus=2.,conversion=0.,failure=.1,pulse_failure=2.,novelty=.02,repeat=.02,teacher_success_bonus=0.)


def _validate_plan(plan, callbacks):
    if plan.get('schema')!='jit_bridge_promotion_series_v1_2' or plan.get('execute') is not True:
        raise ValueError('explicit promotion v1.2 execute=true required')
    if type(plan.get('rounds')) is not int or not 1<=plan['rounds']<=2:
        raise ValueError('promotion series capped at two declared rounds')
    if any(plan.get('gates',{}).get(k) is not True for k in
           ('source_validated','absorption','adopted_candidate','demo_control')):
        raise ValueError('all Stage B prerequisite gates required')
    if set(callbacks)!=set(STAGES) or not all(callable(callbacks[s]) for s in STAGES):
        raise ValueError('all production callbacks must be explicitly supplied')
    initial=plan['initial']
    for key in ('actor','generator','explorer'):
        if not initial.get(key,{}).get('sha256'):raise ValueError('initial frozen identity missing')
    if len(initial['demo_bank']['root_ids'])!=len(set(initial['demo_bank']['root_ids'])):
        raise ValueError('duplicate initial demo roots')
    budget=plan['budgets']
    if not isinstance(budget['max_wall_seconds'],(int,float)) or not math.isfinite(budget['max_wall_seconds']) or budget['max_wall_seconds']<=0:
        raise ValueError('finite positive wall budget required')
    for key in ('max_physics','max_updates'):
        if type(budget[key]) is not int or budget[key]<0:raise ValueError('finite nonnegative budget required')
    for stage in STAGES:
        for key in ('physics','updates'):
            value=budget['per_stage'][stage][key]
            if type(value) is not int or value<0:raise ValueError('nonnegative stage reservation required')
    for field,maximum in (('physics','max_physics'),('updates','max_updates')):
        if plan['rounds']*sum(budget['per_stage'][s][field] for s in STAGES)>budget[maximum]:
            raise ValueError('whole declared plan exceeds budget')


def _fresh_records(result, context, seen):
    records=result['records']
    if not records:raise ValueError('fresh collection cannot be empty')
    roots=set()
    for row in records:
        if row.get('round')!=context['round'] or row.get('behavior')!=context['frozen']:
            raise ValueError('collection must bind current round frozen behavior identities')
        if row.get('role')!='train' or row.get('mode') not in ('learned','uniform_random'):
            raise ValueError('current TRAIN behavior mode required')
        episode=row.get('episode_id');root=row.get('root_id')
        if episode is None or episode == '' or not root or episode in seen or root in roots:
            raise ValueError('unique fresh episode and root identities required')
        seen.add(episode);roots.add(root)
        logprob=row.get('raw_log_prob')
        if row['mode']=='learned':
            if type(logprob) not in (int,float) or not math.isfinite(logprob):
                raise ValueError('learned behavior requires finite original raw log probability')
        elif logprob is not None:raise ValueError('random branch cannot masquerade as on-policy PPO')
    return records


def run_promotion_series(plan, callbacks):
    """Run at most two rounds; defaults and old fixed-Actor series stay untouched.

    API: collect_fresh -> records; teacher_student -> candidate_actor/demo_bank,
    teacher_feedback/student_feedback; accept -> adopted/outcomes; update_explorer
    -> explorer; update_generator -> generator. Context includes immutable copies
    of frozen identities, prior committed bundle and all preceding stage results.
    E receives only this collection's learned, known-outcome rows. G receives only
    valid TRAIN teachers and, when adopted, valid TRAIN student feedback.
    """
    _validate_plan(plan,callbacks)
    root=Path(plan['output']);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'plan.json',plan)
    started=time.monotonic();ledger=[];seen=set();previous=deepcopy(plan['initial'])
    previous['baseline_actor']=deepcopy(plan['initial']['actor']);previous['completed_round']=0
    try:
        for index in range(1,plan['rounds']+1):
            directory=root/f'round_{index:04d}';directory.mkdir()
            context=dict(round=index,frozen={k:deepcopy(previous[k]) for k in ('actor','generator','explorer')},
                previous=deepcopy(previous),reward_weights=deepcopy(REWARD_WEIGHTS),results={})
            atomic_json(directory/'frozen.json',context)
            for stage in STAGES:
                if time.monotonic()-started>=plan['budgets']['max_wall_seconds']:
                    raise TimeoutError('promotion series wall budget exhausted')
                reservation=plan['budgets']['per_stage'][stage]
                entry=dict(round=index,stage=stage,**reservation,accounting='reserved',phase='running')
                ledger.append(entry);atomic_json(root/'costs.json',ledger)
                result=callbacks[stage](deepcopy(context))
                for key in ('physics','updates'):
                    if type(result.get(key)) is not int or not 0<=result[key]<=reservation[key]:
                        raise ValueError('callback cost exceeds stage reservation or is missing')
                if time.monotonic()-started>=plan['budgets']['max_wall_seconds']:
                    raise TimeoutError('promotion callback exceeded wall budget')
                # Persist outputs before subsequent validations; failed work stays inspectable.
                atomic_json(directory/(stage+'.json'),result)
                entry.update(physics=result['physics'],updates=result['updates'],accounting='measured',phase='completed')
                atomic_json(root/'costs.json',ledger)
                context['results'][stage]=deepcopy(result)
                if stage=='collect_fresh':
                    _fresh_records(result,context,seen)
                elif stage=='teacher_student':
                    old=set(previous['demo_bank']['root_ids']);new=result['demo_bank']['root_ids']
                    if not old<=set(new) or len(set(new))!=len(new):raise ValueError('cumulative demonstrations were lost or duplicated')
                    if not result['candidate_actor'].get('sha256'):raise ValueError('candidate actor identity missing')
                elif stage=='accept':
                    if type(result.get('adopted')) is not bool:raise ValueError('explicit acceptance decision required')
                    records=context['results']['collect_fresh']['records'];byroot={r['root_id']:r for r in records}
                    outcomes=result['outcomes']
                    if len(outcomes)!=len(byroot) or {r['root_id'] for r in outcomes}!=set(byroot):
                        raise ValueError('acceptance must preserve every fresh root denominator')
                    if any(r.get(k) not in (0,1,None) for r in outcomes for k in ('source_label','student_label')):
                        raise ValueError('unknown must remain unknown')
                    known={r['root_id'] for r in outcomes if r['source_label'] is not None and r['student_label'] is not None}
                    context['onpolicy_records']=[r for r in records if r['mode']=='learned' and r['root_id'] in known]
                    context['feedback']=[dict(**r,adopted_gain=result['adopted'] and r['source_label']==0 and r['student_label']==1)
                                         for r in outcomes if r['root_id'] in known]
                    teaching=context['results']['teacher_student']
                    candidates=teaching['teacher_feedback']+(teaching['student_feedback'] if result['adopted'] else [])
                    context['eligible_feedback']=[r for r in candidates if r.get('role')=='train' and r.get('valid') is True]
                elif stage=='update_explorer':
                    if not result.get('explorer',{}).get('sha256'):raise ValueError('explorer output identity missing')
                    legal={r['episode_id'] for r in context['onpolicy_records']}
                    used=result.get('training_episode_ids')
                    if (not isinstance(used,list) or len(set(used))!=len(used) or not set(used)<=legal
                            or result.get('behavior_sha256')!=digest(context['frozen'])):
                        raise ValueError('explorer update must receipt current on-policy behavior and episode IDs')
                elif stage=='update_generator':
                    if not result.get('generator',{}).get('sha256'):raise ValueError('generator output identity missing')
            results=context['results'];accepted=results['accept']['adopted']
            bundle=dict(actor=results['teacher_student']['candidate_actor'] if accepted else previous['actor'],
                baseline_actor=previous['baseline_actor'],generator=results['update_generator']['generator'],
                explorer=results['update_explorer']['explorer'],demo_bank=results['teacher_student']['demo_bank'],
                completed_round=index,adopted=accepted,previous_bundle_sha256=digest(previous),
                frozen=context['frozen'],stage_receipts={s:digest(results[s]) for s in STAGES})
            atomic_json(directory/'completed_bundle.json',bundle)
            # Sole publication point: no Actor/G/E pointer changes before this.
            atomic_json(root/'current_source.json',bundle)
            previous=deepcopy(bundle)
            atomic_json(root/'status.json',dict(phase='running',completed_rounds=index))
        atomic_json(root/'status.json',dict(phase='completed',completed_rounds=previous['completed_round']))
        return previous
    except BaseException as error:
        atomic_json(root/'status.json',dict(phase='failed',completed_rounds=previous['completed_round'],error=repr(error),automatic_retry=False))
        raise
