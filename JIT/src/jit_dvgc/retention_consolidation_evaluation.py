"""Independent deterministic capability accounting; no environment or optimizer imports.

Three repeats test label stability, never three independent training seeds. Actor
witness eligibility remains owned by the existing frozen-policy admission rules.
"""
from copy import deepcopy
import math


_CATEGORIES = ('retained', 'lost', 'gained', 'unchanged_negative',
               'mixed', 'unknown', 'execution_error', 'timeout')


def stable_repeat_label(labels, expected_repeats=3):
    """Keep uncertainty and execution failures separate from physical failure."""
    if len(labels) != expected_repeats or expected_repeats < 1:
        raise ValueError('exact predeclared repeat count required')
    allowed = ('mixed', 'unknown', 'execution_error', 'timeout')
    for value in labels:
        if value is None:
            continue
        if type(value) is int and value in (0, 1):
            continue
        if type(value) is str and value in allowed:
            continue
        raise ValueError('labels must be integer 0/1 or explicit unresolved categories')
    for category in ('execution_error', 'timeout', 'unknown'):
        if category in labels or (category == 'unknown' and None in labels):
            return category
    if 'mixed' in labels or len(set(labels)) > 1:
        return 'mixed'
    return labels[0]


def paired_capability_counts(reference, student):
    if len(reference) != len(student):
        raise ValueError('equal paired condition counts required')
    result = {name: 0 for name in _CATEGORIES}
    result.update(total=len(reference), old_positive=sum(v == 1 for v in reference),
                  reference_positive=sum(v == 1 for v in reference),
                  student_positive=sum(v == 1 for v in student))
    for a, b in zip(reference, student):
        unresolved = next((k for k in ('execution_error', 'timeout', 'unknown', 'mixed')
                           if k in (a, b)), None)
        category = unresolved or {(1,1): 'retained', (1,0): 'lost',
                     (0,1): 'gained', (0,0): 'unchanged_negative'}.get((a,b), 'unknown')
        result[category] += 1
    result['unresolved'] = sum(result[k] for k in ('mixed','unknown','execution_error','timeout'))
    # Full group denominator; unresolved pairs never silently disappear.
    result['reference_success_rate'] = result['reference_positive']/len(reference) if reference else None
    result['student_success_rate'] = result['student_positive']/len(student) if student else None
    result['retention_rate'] = (result['retained']/result['old_positive']
                                if result['old_positive'] else None)
    return result


def _selection_config(config):
    groups = config.get('required_groups')
    if not isinstance(groups, dict) or not groups:
        raise ValueError('predeclared nonempty required_groups required')
    for group, limits in groups.items():
        if (type(limits.get('max_lost')) is not int or limits['max_lost'] < 0
                or type(limits.get('max_rate_drop')) not in (int,float)
                or not math.isfinite(limits['max_rate_drop'])
                or not 0 <= limits['max_rate_drop'] <= 1):
            raise ValueError('resolved loss-count and fractional rate-drop gates required: '+group)
    for key in ('mixed_policy','unknown_policy'):
        if config.get(key) not in ('block','report'):
            raise ValueError('explicit '+key+' required')
    if type(config.get('min_expansion_gain')) is not int or config['min_expansion_gain'] < 1:
        raise ValueError('positive predeclared min_expansion_gain required')
    if config.get('baseline_mixed_policy','block') not in ('block','diagnostic_only'):
        raise ValueError('baseline_mixed_policy must be block or diagnostic_only')
    return groups


def evaluate_capability_groups(rows, selection_config):
    """Audit paired DEV rows. An independent execution tag binds the replay receipt.

    Identity/snapshot/history validation belongs to stage preflight; this layer
    does not turn teacher composites or actor-witness admission into absorption.
    """
    required = _selection_config(selection_config)
    repeats = selection_config.get('expected_repeats',3)
    diagnostic_mixed=selection_config.get('baseline_mixed_policy','block')=='diagnostic_only'
    normalized = []; conditions = set(); cases = set()
    for row in rows:
        if row.get('role') not in ('DEV_RETENTION','DEV_EXPANSION','TRAIN_RETENTION'):
            raise ValueError('only DEV capability rows permitted')
        if row.get('role')=='TRAIN_RETENTION':
            proof=row.get('seen_condition_validation',{})
            if (row.get('group')!='old_nominal' or proof.get('schema')!='frontierbridge_seen_condition_validation_v1'
                    or proof.get('training_condition_revisit') is not True or proof.get('independent_dev') is not False
                    or proof.get('group')!='old_nominal' or proof.get('source_train_condition_identity')!=row.get('condition_identity')):
                raise ValueError('only explicit nominal training-condition revisit gate permitted')
        case = row['case_id']; condition = row.get('condition_identity',case)
        if case in cases or condition in conditions:
            raise ValueError('duplicate case or condition is not independent evidence')
        cases.add(case); conditions.add(condition)
        labels = {name: stable_repeat_label(row[name+'_labels'],repeats)
                  for name in ('parent','pi0','student')}
        if row.get('execution') != 'independent_complete':
            labels['student'] = 'unknown'
        normalized.append((row,labels))
    result = dict(parent={},pi0={},gates={},eligible=True,
                  repeat_semantics='label_stability_not_training_seeds',
                  policy_semantics='deterministic_policy_retention',seen_condition_validation={},
                  baseline_mixed_policy=selection_config.get('baseline_mixed_policy','block'),
                  baseline_mixed_diagnostics=[dict(case_id=r['case_id'],group=r['group'],role=r['role'],
                      parent_labels=list(r['parent_labels']),student_labels=list(r['student_labels']),
                      counts_as_gain_or_absorption=False) for r,l in normalized if l['parent']=='mixed'])
    groups = sorted(set(r['group'] for r,_ in normalized) | set(required))
    for group in groups:
        group_rows = [(r,l) for r,l in normalized if r['group']==group and r['role'] in ('DEV_RETENTION','TRAIN_RETENTION')]
        for reference in ('parent','pi0'):
            result[reference][group] = paired_capability_counts(
                [l[reference] for _,l in group_rows],[l['student'] for _,l in group_rows])
        revisit=[(r,l) for r,l in group_rows if r['role']=='TRAIN_RETENTION']
        if revisit:result['seen_condition_validation'][group]=paired_capability_counts([l['parent'] for _,l in revisit],[l['student'] for _,l in revisit])
        if group not in required:
            continue
        excluded=[(r,l) for r,l in group_rows if diagnostic_mixed and l['parent']=='mixed']
        gate_rows=[(r,l) for r,l in group_rows if not(diagnostic_mixed and l['parent']=='mixed')]
        counts=paired_capability_counts([l['parent'] for _,l in gate_rows],[l['student'] for _,l in gate_rows])
        limits = required[group]; reasons=[]
        if any(l['student'] in ('execution_error','timeout') for _,l in excluded):
            reasons.append('diagnostic_condition_execution_failed')
        if not group_rows: reasons.append('missing_required_group')
        if counts['old_positive'] == 0: reasons.append('no_stable_parent_success_to_protect')
        if counts['lost'] > limits['max_lost']: reasons.append('old_success_loss_limit')
        if counts['unresolved']: reasons.append('unresolved_required_group')
        if gate_rows and counts['reference_success_rate']-counts['student_success_rate'] > limits['max_rate_drop'] + 1e-12:
            reasons.append('group_success_rate_drop')
        result['gates'][group] = dict(passed=not reasons,reasons=reasons,limits=deepcopy(limits),
            gate_cohort=counts,gate_denominator=counts['total'],full_denominator=len(group_rows),
            baseline_mixed_diagnostic_count=len(excluded),
            cohort_semantics='stable_frozen_parent_reference' if diagnostic_mixed else 'full_declared_group')
        result['eligible'] &= not reasons
    expansion_rows = [(r,l) for r,l in normalized if r['role']=='DEV_EXPANSION']
    expansion = paired_capability_counts([l['parent'] for _,l in expansion_rows],
                                         [l['student'] for _,l in expansion_rows])
    expansion['independent_success'] = sum(l['student']==1 for _,l in expansion_rows)
    expansion['teacher_verified_absorption'] = sum(l['parent']==0 and l['student']==1
        and stable_repeat_label(r.get('teacher_repeat_labels',[None]*repeats),repeats)==1
        for r,l in expansion_rows)
    result['expansion'] = expansion
    result['total_old_lost'] = sum(v['lost'] for v in result['parent'].values())
    expansion_gate_rows=[(r,l) for r,l in expansion_rows if not(diagnostic_mixed and l['parent']=='mixed')]
    expansion_gate=paired_capability_counts([l['parent'] for _,l in expansion_gate_rows],[l['student'] for _,l in expansion_gate_rows])
    expansion['gate_cohort']=expansion_gate
    if diagnostic_mixed and any(l['parent'] in ('execution_error','timeout') or l['student'] in ('execution_error','timeout') for _,l in normalized):
        result['eligible']=False
    if (expansion['execution_error'] or expansion['timeout']
        or (expansion_gate['mixed'] and selection_config['mixed_policy']=='block')
        or (expansion_gate['unknown'] and selection_config['unknown_policy']=='block')):
        result['eligible'] = False
    return result


def select_checkpoint(candidates, parent, selection_config):
    """Validity -> required gates -> independent DEV expansion -> loss -> earliest.

    No automatic publication. update0 and no qualified gain retain the parent.
    """
    _selection_config(selection_config)
    audits=[]; eligible=[]
    for candidate in candidates:
        valid = all(candidate.get(k) is True for k in ('identity_valid','data_valid','numerical_valid'))
        audit = dict(checkpoint=candidate['checkpoint'],attempt=candidate['attempt'],valid=valid,accepted_updates=candidate.get('accepted_updates','UNKNOWN'))
        if type(candidate['attempt']) is not int or candidate['attempt'] < 0:
            raise ValueError('nonnegative integer attempt required')
        if valid:
            metrics=evaluate_capability_groups(candidate['rows'],selection_config)
            audit['evaluation']=metrics
            if metrics['eligible']:eligible.append((candidate,metrics))
        audits.append(audit)
    ranked = sorted(eligible,key=lambda pair:(-pair[1]['expansion']['independent_success'],
                    pair[1]['total_old_lost'],pair[0]['attempt']))
    improving=[pair for pair in ranked if pair[0]['attempt'] > 0
               and pair[1]['expansion']['gained'] >= selection_config['min_expansion_gain']
               and pair[1]['expansion']['teacher_verified_absorption'] >= selection_config['min_expansion_gain']]
    accepted=improving[0][0]['checkpoint'] if improving else parent
    return dict(phase='accepted_dev_improvement' if improving else 'no_accepted_improvement',
                learner_last=max(candidates,key=lambda c:c['attempt'])['checkpoint'] if candidates else None,
                stage_candidate=accepted,
                best_dev_candidate=ranked[0][0]['checkpoint'] if ranked else None,
                published_policy=None,retained_parent=not bool(improving),audits=audits)


def build_independent_replay_spec(base_spec, rows_path, initializer):
    """Prepare existing pulse-exploration evaluate input; dispatch stays authorized."""
    forbidden=('bridge_action_plan','closed_loop_prefix_policy','closed_loop_prefix_initializer',
               'phase_policy','actor_witness_eligible')
    if any(k in base_spec for k in forbidden):
        raise ValueError('independent replay must not use teacher/prefix overrides or witness overrides')
    spec=deepcopy(base_spec)
    spec.update(candidates=str(rows_path),order=[spec['proposer']],full_matrix=True,
                warmup_initializer=deepcopy(initializer),preserve_snapshot_episode_context=True,
                record_actor_preobservations=True,record_retention_diagnostics=True)
    return spec


def independent_replay_label(result):
    """Read actual independent attempt, without changing actor_witness_eligible.

    A witness=None warmup can be evaluated as a student; formal witness admission
    remains false. Composite/prefix results cannot supply independent success.
    """
    attempts=result.get('attempts',[])
    if len(attempts)!=1 or result.get('prefix_terminal'):
        return 'unknown'
    attempt=attempts[0]
    if (attempt.get('controller_kind') not in ('warmup_candidate','source_actor')
            and not (attempt.get('controller_kind') is None and attempt.get('actor_witness_eligible') is True)):
        return 'unknown'
    outcome=attempt.get('outcome')
    if outcome in ('timeout','horizon_exhausted','time_limit',
                   'timeout_before_landing','horizon_exhausted_before_landing'):
        return 'timeout'
    if outcome in ('execution_error','nonfinite'):
        return 'execution_error'
    if outcome in ('open_nonterminal_before_landing','simultaneous_landing_failure_unresolved'):
        return 'unknown'
    value=attempt.get('label')
    if value == 1 and outcome != 'stable_forward_recovery':return 'unknown'
    return value if type(value) is int and value in (0,1) else 'unknown'


def independent_collection_label(tape, lane, horizon):
    """Read existing full-episode collect tape, without treating horizon as failure.

    The caller must validate no residual/explorer prefix and locked deterministic
    policy identity before dispatch; this reader never creates a witness.
    """
    import numpy as np
    required=('prefix_mask','finite','success','terminal','physical_failure')
    if any(k not in tape for k in required):
        return 'unknown'
    mask=np.asarray(tape['prefix_mask'])
    if mask.ndim!=2 or not 0 <= lane < mask.shape[1] or horizon < 1:
        raise ValueError('valid full-episode lane and horizon required')
    indices=np.flatnonzero(mask[:,lane])
    if not len(indices) or not np.array_equal(indices,np.arange(len(indices))):
        return 'unknown'
    for key in required:
        if np.asarray(tape[key]).shape!=mask.shape:
            raise ValueError('collection label arrays must match prefix mask')
    if not np.asarray(tape['finite'])[indices,lane].all():
        return 'execution_error'
    if 'end_code' in tape:
        from .constants import END_NONFINITE, END_TIMEOUT
        codes=np.asarray(tape['end_code'])
        if codes.shape != mask.shape:
            raise ValueError('end-code array must match prefix mask')
        if np.any(codes[indices,lane] == END_NONFINITE):
            return 'execution_error'
        if codes[indices[-1],lane] == END_TIMEOUT:
            return 'timeout'
    successes=np.asarray(tape['success'])[indices,lane].astype(bool)
    failures=np.asarray(tape['physical_failure'])[indices,lane].astype(bool)
    terminals=np.asarray(tape['terminal'])[indices,lane].astype(bool)
    if terminals[:-1].any():return 'unknown'
    if np.any(successes & failures):
        return 'unknown'
    if successes.any():
        return 1 if successes[-1] and terminals[-1] and not successes[:-1].any() and not failures.any() else 'unknown'
    if failures.any():
        return 0 if failures[-1] and terminals[-1] and not failures[:-1].any() else 'unknown'
    if 'timeout' in tape and np.asarray(tape['timeout'])[indices,lane].any():
        return 'timeout'
    if np.asarray(tape['terminal'])[indices[-1],lane]:
        return 0
    return 'timeout' if len(indices)>=horizon else 'unknown'
