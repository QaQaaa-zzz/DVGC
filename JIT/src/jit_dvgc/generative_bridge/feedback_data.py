"""TRAIN-only bidirectional success corpus; teacher labels are immutable."""
import numpy as np
from .data import build_action_windows, trajectory_identity

MIX={'history':.5,'teacher_new':.25,'actor_new':.25}


def realized_mix(counts):
    if set(counts)!=set(MIX) or any(type(n) is not int or n<0 for n in counts.values()):
        raise ValueError('three nonnegative source counts required')
    total=sum(w for k,w in MIX.items() if counts[k])
    return {k:w/total if total and counts[k] else 0. for k,w in MIX.items()}


def admit_trace(trace, *, adoption, expected, splits):
    m=trace['metadata'];row={**m,'eligible':False,'window_count':0,
        'bridge_verified_against_actor_sha256':None}
    def excluded(reason): return {**row,'reason':reason}
    if m.get('role')!='train': return excluded('excluded_role')
    ancestor=m.get('root_episode_id')
    if ancestor not in splits or splits[ancestor]!=m.get('inherited_split'):
        raise ValueError('ancestor split missing or changed')
    if m['inherited_split']!='generator_train':return excluded('excluded_split')
    if not m.get('complete') or not m.get('full_success'):return excluded('incomplete_or_no_full_success')
    origin=m.get('origin_type')
    if origin not in ('bootstrap_actor','verified_teacher','adopted_actor'):
        raise ValueError('unsupported provenance')
    if origin=='adopted_actor':
        if adoption.get('adopted') is not True:return excluded('student_not_adopted')
        for key in ('actor_sha256','normalizer_sha256'):
            if adoption.get(key)!=m.get(key):raise ValueError('adoption checkpoint identity mismatch: '+key)
    for key,value in expected.items():
        if m.get(key)!=value:raise ValueError('trace protocol identity mismatch: '+key)
    for k in ('root_id','root_context_sha256','actor_sha256','normalizer_sha256','model_sha256','protocol_sha256'):
        if not m.get(k):raise ValueError('missing trace identity: '+k)
    if 'actor_observation_before' not in trace['arrays']:
        return excluded('trace_unavailable_preobs_require_separate_bounded_recapture')
    windows=build_action_windows(trace)
    a=trace['arrays'];n=len(a['done'])
    if not n or not a['success'][-1] or np.any(a['failure']) or np.any(a['timeout']):
        return excluded('no_unambiguous_full_success')
    if origin in ('adopted_actor','bootstrap_actor') and set(a['action_origin'])!={'actor_only'}:
        raise ValueError('student trace must contain independent actor_only actions')
    if origin=='verified_teacher':
        if m.get('teacher_status')!='verified_solution' or not m.get('verification_receipt_sha256'):
            raise ValueError('teacher requires completed replay verification')
        row['bridge_verified_against_actor_sha256']=m['source_actor_sha256']
    if not len(windows['actions']):return excluded('no_full_H16_window')
    return {**row,'eligible':True,'reason':'admitted','window_count':len(windows['actions']),
            'trajectory_sha256':trajectory_identity(trace)}


def build_corpus(history, teachers, actors, *, adoption, expected, splits):
    groups={'history':[],'teacher_new':[],'actor_new':[]};seen=set();admission=[]
    for group,items in zip(groups,(history,teachers,actors)):
        for trace in items:
            required_origin={'teacher_new':'verified_teacher','actor_new':'adopted_actor'}.get(group)
            if required_origin and trace['metadata'].get('origin_type')!=required_origin:
                raise ValueError('source group provenance mismatch: '+group)
            # History carries its original, checkpoint-bound adoption receipt.
            if group=='history' and trace['metadata'].get('origin_type')=='adopted_actor':
                if not trace.get('adoption'):raise ValueError('historical adopted Actor requires its original adoption receipt')
                receipt=trace['adoption']
            else:receipt=adoption
            row=admit_trace(trace,adoption=receipt,expected=expected,splits=splits)
            if row['eligible']:
                key=row['trajectory_sha256']
                if key in seen:row={**row,'eligible':False,'reason':'duplicate_full_trajectory'}
                else:
                    seen.add(key)
                    stored=dict(trace)
                    if trace['metadata']['origin_type']=='adopted_actor':stored['adoption']=dict(receipt)
                    groups[group].append(stored)
            admission.append({**row,'source_group':group})
    counts={k:len(v) for k,v in groups.items()}
    return {'groups':groups,'admission':admission,'requested_mix':dict(MIX),
            'realized_mix':realized_mix(counts),'new_data':bool(groups['teacher_new'] or groups['actor_new'])}


def sample_corpus(corpus, rng, batch_size, *, return_metadata=False, include_trajectory_identity=False):
    """Choose source, ancestor, trajectory, then window; loss is not reweighted.

    Pass compile_corpus_index(corpus) to reuse an explicit immutable snapshot.
    The historical dict API builds a fresh index, never silently caches writes.
    """
    from .corpus_index import CompiledCorpusIndex, compile_corpus_index
    index = corpus if isinstance(corpus, CompiledCorpusIndex) else compile_corpus_index(corpus,include_trajectory_identity=include_trajectory_identity)
    if include_trajectory_identity and not index.includes_trajectory_identity:
        raise ValueError('compile corpus index with trajectory identities enabled')
    names=index.names;probs=index.probabilities
    if not any(probs):raise ValueError('empty corpus cannot be sampled')
    observations=[];actions=[];sources=[];metadata=[]
    for _ in range(batch_size):
        group_index=int(rng.choice(3,p=probs));group=names[group_index]
        ancestors=index.groups[group_index]
        ancestor,choices=ancestors[int(rng.integers(len(ancestors)))]
        trace=choices[int(rng.integers(len(choices)))]
        i=int(rng.integers(len(trace.starts)));start=trace.starts[i]
        observations.append(trace.observations[start]);actions.append(trace.actions[start:start+16]);sources.append(group)
        if return_metadata:
            origins=sorted(set(trace.origins[start:start+16].tolist()))
            offset=trace.trace_start_step
            metadata.append({'source_group':group,'recency':'history' if group=='history' else 'new',
                'root_episode_id':ancestor,'root_id':trace.root_id,'onset':trace.onset,
                'window_start':start,'window_end_exclusive':start+16,
                'window_start_step':None if offset is None else int(offset)+start,
                'window_end_step_exclusive':None if offset is None else int(offset)+start+16,
                'segment':origins[0] if len(origins)==1 else 'mixed','action_origins':origins})
            if include_trajectory_identity:metadata[-1]['trajectory_sha256']=trace.trajectory_sha256
    result=(np.asarray(observations),np.asarray(actions),sources)
    return (*result,metadata) if return_metadata else result


def trace_from_evaluation(attempt, metadata):
    """Read an existing lane; absence of preobs is explicit, never recaptured here."""
    from .verified_trace_cache import TRACE_CACHE
    if (attempt.get('recording_schema')!='jit_actor_success_trace_v1_1'
            or attempt.get('action_origin')!='actor_only'
            or attempt.get('outcome')!='stable_forward_recovery'
            or metadata.get('success_criterion')!='stable_forward_recovery'):
        raise ValueError('explicit actor-only full recovery recording required')
    for source,target in [('model_sha256','model_sha256'),('actor_sha256','actor_sha256'),('normalizer_sha256','normalizer_sha256'),
                          ('snapshot_context_sha256','root_context_sha256')]:
        if attempt.get(source)!=metadata.get(target):raise ValueError('evaluation identity mismatch: '+source)
    if attempt.get('controller_kind')=='composite_teacher':raise ValueError('teacher cannot become actor_only')
    raw=TRACE_CACHE.load(attempt['trace'],attempt['trace_sha256'])
    if 'actor_observation_before' not in raw:
        return {'metadata':dict(metadata),'arrays':{}}
    lane=attempt['trace_lane'];mask_key='valid_mask' if 'valid_mask' in raw else 'mask'
    valid=np.asarray(raw[mask_key][:,lane],bool)
    ids=np.flatnonzero(valid)
    if len(ids) and not np.array_equal(ids,np.arange(len(ids))):raise ValueError('noncontiguous real evaluation lane')
    arrays={k:value[:len(ids),lane] for k,value in raw.items()}
    if 'failure' not in arrays:arrays['failure']=arrays['physical_failure']
    arrays['valid_mask']=np.ones(len(ids),bool)
    if 'action_origin' in arrays and set(arrays['action_origin'])!={'actor_only'}:
        raise ValueError('recorded action provenance conflicts with Actor-only evaluation')
    arrays['action_origin']=np.array(['actor_only']*len(ids))
    meta={**metadata,'origin_type':'adopted_actor','full_success':attempt['label']==1,
        'trace_uri':attempt['trace'],'trace_sha256':attempt['trace_sha256'],'trace_lane':lane}
    return {'metadata':meta,'arrays':arrays}
