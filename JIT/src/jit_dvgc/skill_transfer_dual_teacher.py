"""TRAIN-only verified same-context pi0/R query labels for Actor-only S.

This module imports no simulator and performs no online teacher driving. The
caller supplies student-visited contexts and independent frozen-teacher branch
receipts; those receipts, not root-level success, authorize individual labels.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .generative_bridge.contracts import digest, file_sha
from .generative_bridge.protocol import atomic_json
from .retention_consolidation_evaluation import stable_repeat_label

SCHEMA='frontierbridge_online_labels_v1'
_GROUPS={'new_recovery','old_nominal','old_random','old_recovery'}
_CONTEXT_FIELDS=('context_sha256','root_id','ancestor_root_id','split','time_before',
    'remaining_horizon','model_sha256','protocol_sha256','control_step')


def merge_verified_labels(paths, output):
    """Aggregate only already queried, audited states under identical teachers."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];teachers=None
    for path in paths:
        bank=json.loads(Path(path).read_text())
        if bank['schema']!=SCHEMA:raise ValueError('online label bank required')
        if teachers is not None and bank['teachers']!=teachers:raise ValueError('teacher changed during aggregation')
        teachers=bank['teachers']
        rows.extend(audit_online_label(row,teachers) for row in bank['labels'])
    ids=[r['sample_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('duplicate queried state identities')
    conflicts=observation_conflicts(rows);excluded={s for c in conflicts for s in c['sample_ids']}
    rows=[r for r in rows if r['sample_id'] not in excluded]
    weights=consolidation_weights(rows)
    receipt=output/'labels.json';atomic_json(receipt,dict(schema=SCHEMA,teachers=teachers,labels=rows,conflicts=conflicts,
        source_banks=[dict(path=str(p),sha256=file_sha(p)) for p in paths]))
    data=output/'online_labels.npz';np.savez(data,observations=np.asarray([r['observation_before'] for r in rows],np.float32),
        actions=np.asarray([r['target_action'] for r in rows],np.float32),weights=weights)
    manifest=dict(schema=SCHEMA,path=str(data),sha256=file_sha(data),count=len(rows),
        sample_roots=[r['root_id'] for r in rows],sample_origins=['student_visited_query']*len(rows),
        sampling_weights=weights.tolist(),label_receipt=dict(path=str(receipt),sha256=file_sha(receipt)))
    atomic_json(output/'online_manifest.json',manifest)
    return manifest


def choose_verified_teacher(pi0_outcome, recovery_outcome):
    """Retention-priority tie; unresolved branches always abstain."""
    if type(pi0_outcome) is not int or type(recovery_outcome) is not int:
        return None
    if pi0_outcome not in (0,1) or recovery_outcome not in (0,1):
        raise ValueError('physical outcomes must be 0/1 or explicit unresolved categories')
    if pi0_outcome==1:return 'pi0'
    if recovery_outcome==1:return 'R'
    return None


def _reference(ref):
    if not isinstance(ref,dict) or file_sha(ref['path'])!=ref['sha256']:
        raise ValueError('source receipt hash mismatch')
    return json.loads(Path(ref['path']).read_text())


def _query(query):
    required=('sample_id','root_id','ancestor_root_id','student_actor_sha256','context_sha256',
        'model_sha256','protocol_sha256','timestamp')
    if any(not isinstance(query.get(k),str) or not query[k] for k in required):
        raise ValueError('complete query provenance required')
    if query.get('split')!='TRAIN':raise ValueError('only TRAIN queries may create supervision')
    if (query.get('source_type')!='student_visited_query' or query.get('student_closed_loop') is not True
            or query.get('teacher_used_as_driver') is not False):
        raise ValueError('only independently executed student-visited queries allowed')
    if query.get('task_group') not in _GROUPS:raise ValueError('explicit new/old task group required')
    if type(query.get('control_step')) is not int or query['control_step']<0:
        raise ValueError('real control step required')
    if type(query.get('remaining_horizon')) is not int or query['remaining_horizon']<=0:
        raise ValueError('positive original remaining horizon required')
    if not isinstance(query.get('time_before'),(int,float)) or not np.isfinite(query['time_before']) or query['time_before']<0:
        raise ValueError('finite original episode time required')
    observation=np.asarray(query.get('observation_before'),dtype=np.float32)
    if observation.shape!=(76,) or not np.isfinite(observation).all():
        raise ValueError('real finite pre-action 76D observation required')
    ref=query.get('context_receipt')
    if not isinstance(ref,dict) or file_sha(ref['path'])!=ref['sha256'] or ref['sha256']!=query['context_sha256']:
        raise ValueError('full context source hash mismatch')
    return observation


def label_verified_query(query, branch_receipts, teachers, *, expected_repeats=3):
    """Audit two complete frozen branches; return labeled row or None (abstain).

    Receipt schema: all _CONTEXT_FIELDS, observation_sha256, teacher Actor and
    normalizer hashes, independent_frozen_teacher execution_mode, formal success
    criterion, repeat_labels, and initial normalized_action. Receipt paths and
    their hashes must be preserved. Context files may be binary snapshots.
    """
    _query(query)
    if set(branch_receipts)!={'pi0','R'} or set(teachers)!={'pi0','R'}:
        raise ValueError('exact pi0 and R branches/descriptors required')
    outcomes={};records={}
    for teacher in ('pi0','R'):
        record=_reference(branch_receipts[teacher]);records[teacher]=record
        if (any(record.get(k)!=query[k] for k in _CONTEXT_FIELDS)
                or record.get('observation_sha256')!=digest(query['observation_before'])):
            raise ValueError('teacher branch must use same context/time/horizon/observation')
        if (record.get('teacher')!=teacher or record.get('execution_mode')!='independent_frozen_teacher'
                or record.get('success_criterion')!='stable_forward_recovery'):
            raise ValueError('formal independent frozen-teacher branch required')
        descriptor=teachers[teacher]
        for name in ('actor_sha256','normalizer_sha256'):
            if not descriptor.get(name) or record.get(name)!=descriptor[name]:
                raise ValueError('frozen teacher identity mismatch')
        outcomes[teacher]=stable_repeat_label(record.get('repeat_labels',[]),expected_repeats)
        action=np.asarray(record.get('normalized_action'),dtype=np.float32)
        if action.shape!=(4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('finite normalized branch initial action required')
    chosen=choose_verified_teacher(outcomes['pi0'],outcomes['R'])
    if chosen is None:return None
    return dict(deepcopy(query),schema=SCHEMA,chosen_teacher=chosen,
        chosen_teacher_actor_sha256=teachers[chosen]['actor_sha256'],
        chosen_teacher_normalizer_sha256=teachers[chosen]['normalizer_sha256'],
        target_action=deepcopy(records[chosen]['normalized_action']),pi0_outcome=outcomes['pi0'],R_outcome=outcomes['R'],
        expected_repeats=expected_repeats,verification_receipts=deepcopy(branch_receipts),
        verification_receipt_sha256=digest(branch_receipts),teacher_id_is_actor_input=False)


def audit_online_label(row, teachers):
    if row.get('schema')!=SCHEMA or row.get('teacher_id_is_actor_input') is not False:
        raise ValueError('versioned training-only oracle label required')
    rebuilt=label_verified_query(row,row['verification_receipts'],teachers,expected_repeats=row['expected_repeats'])
    if rebuilt is None:raise ValueError('abstained query cannot supply a target')
    for name in ('chosen_teacher','chosen_teacher_actor_sha256','chosen_teacher_normalizer_sha256',
            'target_action','pi0_outcome','R_outcome','verification_receipt_sha256'):
        if rebuilt[name]!=row.get(name):raise ValueError('label target/provenance mismatch: '+name)
    return deepcopy(row)


def observation_conflicts(rows, *, observation_tolerance=1e-6, action_tolerance=.1):
    """Identify near-identical deployable observations with conflicting actions.

    Every row in a conflicting pair is quarantined by bank export. Labels are
    not reconciled using privileged context, teacher IDs, or root IDs.
    """
    if not all(np.isfinite(v) and v>=0 for v in (observation_tolerance,action_tolerance)):
        raise ValueError('nonnegative finite conflict tolerances required')
    if not rows:return []
    obs=np.asarray([r['observation_before'] for r in rows],np.float64)
    actions=np.asarray([r['target_action'] for r in rows],np.float64)
    if obs.shape!=(len(rows),76) or actions.shape!=(len(rows),4) or not np.isfinite(obs).all() or not np.isfinite(actions).all():
        raise ValueError('finite aligned observation/actions required')
    conflicts=[]
    # KD tree avoids quadratic comparison across unrelated rollout states.
    from scipy.spatial import cKDTree
    for i,j in sorted(cKDTree(obs).query_pairs(observation_tolerance,p=np.inf)):
        action_distance=float(np.max(np.abs(actions[i]-actions[j])))
        if action_distance>action_tolerance:
            conflicts.append(dict(sample_ids=[rows[i]['sample_id'],rows[j]['sample_id']],
                observation_max_abs=float(np.max(np.abs(obs[i]-obs[j]))),action_max_abs=action_distance))
    return conflicts


def consolidation_weights(rows):
    """Half new/old mass; new roots equal, old categories then roots equal."""
    if not rows:raise ValueError('nonempty verified labels required')
    groups=np.asarray([r['task_group'] for r in rows]);roots=np.asarray([r['root_id'] for r in rows])
    if not set(groups.tolist())<=_GROUPS:raise ValueError('unknown consolidation category')
    families=[groups=='new_recovery',groups!='new_recovery'];families=[f for f in families if np.any(f)]
    weights=np.zeros(len(rows),np.float64)
    for family in families:
        categories=sorted(set(groups[family].tolist()))
        for category in categories:
            category_mask=family&(groups==category);members=sorted(set(roots[category_mask].tolist()))
            for root in members:
                mask=category_mask&(roots==root)
                weights[mask]=1./(len(families)*len(categories)*len(members)*mask.sum())
    return weights


def collect_verified_queries(queries, evaluate_branch, teachers, output, *, expected_repeats=3):
    """Call evaluator twice per queried state, preserving all abstentions.

    evaluate_branch(query_copy, teacher_name, frozen_descriptor_copy) must return
    an on-disk verified branch receipt reference. This orchestration itself does
    not manufacture contexts, execute physics, or extrapolate unqueried labels.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    labels=[];outcomes=[];seen=set()
    for query in queries:
        _query(query)
        if query['sample_id'] in seen:raise ValueError('duplicate query sample_id')
        seen.add(query['sample_id'])
        refs={teacher:evaluate_branch(deepcopy(query),teacher,deepcopy(teachers[teacher])) for teacher in ('pi0','R')}
        label=label_verified_query(query,refs,teachers,expected_repeats=expected_repeats)
        if label is not None:labels.append(label)
        outcomes.append(dict(sample_id=query['sample_id'],root_id=query['root_id'],
            labeled=label is not None,chosen_teacher=None if label is None else label['chosen_teacher'],
            branch_receipts=refs,query=deepcopy(query)))
        atomic_json(output/'query_receipts.json',dict(schema='frontierbridge_query_collection_v1',queries=outcomes,
            total_queries=len(outcomes),labeled_queries=len(labels),abstained_queries=len(outcomes)-len(labels),status='running'))
    atomic_json(output/'labels.json',dict(schema=SCHEMA,teachers=teachers,labels=labels))
    atomic_json(output/'query_receipts.json',dict(schema='frontierbridge_query_collection_v1',queries=outcomes,
        total_queries=len(outcomes),labeled_queries=len(labels),abstained_queries=len(outcomes)-len(labels),status='completed'))
    return labels
