"""Legacy student credit and opt-in verified generated-teacher credit."""
from copy import deepcopy

from ..discovery_reward import conversion_feedback

WEIGHTS=dict(conversion=0.,adoption_bonus=2.,novelty=.02,repeat=.02,failure=.10,pulse_failure=2.)
TEACHER_WEIGHTS=dict(novelty=.02,repeat=.02,failure=.10,pulse_failure=2.,teacher_success_bonus=2.)


def feedback(rows, seen):
    checked=[]
    for raw in rows:
        r=dict(raw)
        direct=(r.get('prefix_terminal',False) and r.get('prefix_physical_failure',False)
                and r.get('pulse_applied_steps',0)>0)
        # Missing final adoption is unresolved, not a rejection or a failed attempt.
        if not direct and r.get('initial_label')==0 and type(r.get('successor_adopted')) is not bool:
            r['learning_attempted']=False
        checked.append(r)
    reward,mask,ledger,parts=conversion_feedback(checked,seen,WEIGHTS)
    for i,r in enumerate(checked):
        # A historical pending root already solved by the current source has no new 0-to-1 gain.
        if (mask[i] and r.get('source_recheck_label')==1 and r.get('initial_label')==0
                and r.get('learning_attempted') and r.get('label')==1 and r.get('successor_adopted') is True):
            reward[i]-=WEIGHTS['adoption_bonus'];parts['quality'][i]-=WEIGHTS['adoption_bonus']
        # A physical failure can occur before the first quantizable pulse cell.
        if (r.get('stage_reached',True) and r.get('prefix_terminal',False)
                and r.get('prefix_physical_failure',False) and r.get('pulse_applied_steps',0)>0):
            reward[i]=-2.;mask[i]=True;parts['novelty'][i]=0.;parts['quality'][i]=-2.
    return reward,mask,ledger,parts


def _teacher_evidence(root_id, teacher):
    """Classify durable search/review outcomes; scores and students are not labels."""
    status=teacher.get('teacher_status','not_scheduled')
    candidates=teacher.get('candidates',[])
    selected_id=teacher.get('selected_candidate_id')
    selected=[c for c in candidates if c.get('candidate_id')==selected_id]
    winner=selected[0] if len(selected)==1 else {}
    verification=teacher.get('verification',{})
    metadata=teacher.get('demo',{}).get('metadata',{})
    evidence=dict(teacher_status=status,teacher_quality_outcome=None,
        quality_exclusion_reason='unknown_teacher_outcome',
        selected_candidate_id=selected_id,selected_candidate_kind=winner.get('kind'),
        selected_action_delta_cost=winner.get('action_delta_cost'),
        selected_task_return=winner.get('task_return'),candidate_count=len(candidates),
        source_actor_sha256=teacher.get('source_actor_sha256'),
        source_recheck_label=teacher.get('source_recheck_label'),
        generator_checkpoint_manifest_sha256=teacher.get('generator',{}).get('checkpoint_manifest_sha256'),
        proposal_sha256=teacher.get('proposal_sha256'),
        verification_receipt_sha256=metadata.get('verification_receipt_sha256'),
        verification=deepcopy(verification),teacher_reason=teacher.get('reason'),
        generated_only_search=teacher.get('generated_only_search',False),
        source_control_consistent=teacher.get('source_control_consistent'))
    if teacher.get('root_id',root_id)!=root_id:
        evidence['quality_exclusion_reason']='teacher_root_identity_mismatch'
        return evidence
    if status not in ('verified_solution','searched_no_solution'):
        return evidence
    generated_only=teacher.get('generated_only_search') is True
    if (teacher.get('new_gain_eligible') is False
            or (generated_only and teacher.get('source_control_consistent') is not True)
            or (not generated_only and
                (('source_recheck_label' in teacher and teacher['source_recheck_label']!=0)
                 or any(c.get('kind')=='source_only' and c.get('full_success') is True for c in candidates)))):
        evidence['quality_exclusion_reason']='source_control_conflict'
        return evidence
    generated=[c for c in candidates if c.get('kind')=='diffusion']
    if (not generated or any(c.get('invalid') or c.get('incomplete')
                            or type(c.get('full_success')) is not bool for c in candidates)):
        evidence['quality_exclusion_reason']='incomplete_candidate_outcomes'
        return evidence
    if status=='searched_no_solution':
        if any(c['full_success'] for c in (generated if generated_only else candidates)):
            evidence['quality_exclusion_reason']='contradictory_unsolved_search'
            return evidence
        evidence.update(teacher_quality_outcome=0,quality_exclusion_reason=None)
        return evidence
    if (winner.get('kind')!='diffusion' or winner.get('full_success') is not True
            or verification.get('candidate_id')!=selected_id
            or verification.get('full_success') is not True
            or verification.get('repeat_labels')!=[1,1]
            or verification.get('repeat_complete')!=[True,True]):
        evidence['quality_exclusion_reason']='unverified_generated_full_task_winner'
        return evidence
    evidence.update(teacher_quality_outcome=1,quality_exclusion_reason=None)
    return evidence


def teacher_reward_rows(rows, teachers):
    """Return ordered, unmodified collection rows plus teacher-only audit evidence.

    ``teachers`` maps the exact root identity to production teacher results.
    Verified solutions additionally require two complete successful reviews in
    ``verification.repeat_labels`` and ``verification.repeat_complete``.
    Prefix paths/hashes, pulse provenance and source labels remain in each row;
    this function does not load traces, run physics or inspect student outcomes.
    """
    checked=[]
    for raw in rows:
        row=deepcopy(raw)
        root_id=row['root_id']
        row['teacher_reward_evidence']=_teacher_evidence(root_id,teachers.get(root_id,{}))
        checked.append(row)
    return checked


def teacher_feedback(rows, seen, teachers):
    """Opt-in teacher-only reward with the legacy four-value return interface.

    Use original collection/source rows and this round's root-indexed teacher
    results. Window novelty and its ledger have exactly the legacy semantics.
    Fresh verified generated recovery earns +2, searched-unsolved earns -.1,
    and direct pulse failure overrides all terms with -2. UNKNOWN teacher
    outcomes exclude the entire PPO sample; their arrivals still update the
    source ledger, matching the legacy unresolved-arrival treatment.
    """
    audited=teacher_reward_rows(rows,teachers)
    checked=[]
    for raw in audited:
        row=dict(raw)
        outcome=row['teacher_reward_evidence']['teacher_quality_outcome']
        # Reuse the exact window/duplicate weighting without student gain terms.
        row.update(initial_label=1 if outcome is not None else None,
                   label=1 if outcome is not None else None,learning_attempted=False)
        checked.append(row)
    reward,mask,ledger,parts=conversion_feedback(checked,seen,{**WEIGHTS,'adoption_bonus':0.})
    for i,row in enumerate(checked):
        direct=(row.get('stage_reached',True) and row.get('prefix_terminal',False)
                and row.get('prefix_physical_failure',False) and row.get('pulse_applied_steps',0)>0)
        if direct:
            reward[i]=-TEACHER_WEIGHTS['pulse_failure'];mask[i]=True
            parts['novelty'][i]=0.;parts['quality'][i]=-TEACHER_WEIGHTS['pulse_failure']
        elif mask[i]:
            outcome=row['teacher_reward_evidence']['teacher_quality_outcome']
            quality=(-TEACHER_WEIGHTS['failure'] if outcome==0 else
                     TEACHER_WEIGHTS['teacher_success_bonus'] if parts['novelty'][i]>0 else 0.)
            reward[i]+=quality;parts['quality'][i]=quality
    return reward,mask,ledger,parts
