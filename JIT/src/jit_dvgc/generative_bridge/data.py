"""Real pre-action observations only; never synthesize history or pad targets."""
import hashlib
import numpy as np
from .contracts import digest


def validate_trace(trace):
    a=trace['arrays']
    required=('actor_observation_before','actor_observation_after','normalized_action_executed',
              'valid_mask','done','success','failure','timeout','phase_before','action_origin')
    if any(k not in a for k in required): raise ValueError('missing trace timing fields')
    n=len(a['normalized_action_executed'])
    for key,width in [('actor_observation_before',76),('actor_observation_after',76),
                      ('normalized_action_executed',4)]:
        x=np.asarray(a[key])
        if x.shape!=(n,width) or not np.isfinite(x).all(): raise ValueError('invalid '+key)
    if np.any(np.abs(a['normalized_action_executed'])>1): raise ValueError('action outside source permission')
    for k in required[3:]:
        if np.asarray(a[k]).shape!=(n,): raise ValueError('invalid trace vector: '+k)
    for k in ('valid_mask','done','success','failure','timeout'):
        v=np.asarray(a[k])
        if not np.isfinite(v).all() or not np.isin(v,[0,1]).all():
            raise ValueError('finite binary trace flags required: '+k)
    phase=np.asarray(a['phase_before'])
    if not np.isfinite(phase).all() or not np.isin(phase,[0,1]).all():
        raise ValueError('finite integer up/down phase required')
    if not np.asarray(a['valid_mask']).all(): raise ValueError('trim padding before trace admission')
    if n and (np.any(np.asarray(a['done'])[:-1]) or np.any(np.asarray(a['success'])[:-1])):
        raise ValueError('trace crosses terminal/reset')
    if n>1 and not np.array_equal(a['actor_observation_after'][:-1],a['actor_observation_before'][1:]):
        raise ValueError('pre/post continuity mismatch; cannot infer FIFO')
    origins=set(np.asarray(a['action_origin']).tolist())
    if not origins <= {'actor_only','bridge_prefix','source_tail','source_only'}:
        raise ValueError('invalid action origin')
    return n


def trajectory_identity(trace):
    # Full timing/action content and root/producer/protocol, not URI or window index.
    m=trace['metadata']
    h=hashlib.sha256(digest({k:m[k] for k in ('root_context_sha256','actor_sha256',
        'normalizer_sha256','model_sha256','protocol_sha256')}).encode())
    for k in sorted(trace['arrays']):
        a=np.ascontiguousarray(trace['arrays'][k])
        h.update(k.encode());h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()


def build_action_windows(trace, horizon=16, *, max_per_phase=None):
    if horizon!=16: raise ValueError('H16 only')
    n=validate_trace(trace);a=trace['arrays'];indices=[];counts={}
    for t in range(max(0,n-horizon+1)):
        phase=int(a['phase_before'][t])
        if max_per_phase is not None and counts.get(phase,0)>=max_per_phase:continue
        indices.append(t);counts[phase]=counts.get(phase,0)+1
    return {'observations':np.asarray(a['actor_observation_before'])[indices].reshape(-1,76),
            'actions':np.asarray([a['normalized_action_executed'][t:t+horizon] for t in indices],
                                 dtype=np.float32).reshape(-1,16,4),
            'start_indices':np.asarray(indices,int)}


def export_student_demonstrations(teachers,output):
    """One verified representative per root, real steps only, prefix/tail balance."""
    from pathlib import Path
    from .contracts import file_sha
    from .protocol import atomic_json
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    obs=[];actions=[];sample_roots=[];sample_origins=[];entries=[]
    roots=set()
    for t in teachers:
        m=t['metadata'];a=t['arrays'];n=validate_trace(t)
        validate_student_demonstration(t)
        if m['root_id'] in roots:raise ValueError('one representative teacher per root')
        roots.add(m['root_id'])
        obs.extend(a['actor_observation_before']);actions.extend(a['normalized_action_executed'])
        sample_roots.extend([m['root_id']]*n);sample_origins.extend(a['action_origin'].tolist())
        entries.append({**m,'count':n,'trajectory_sha256':trajectory_identity(t)})
    manifest={'schema':'jit_bridge_demo_v1_1','count':len(obs),'entries':entries,
              'sample_roots':sample_roots,'sample_origins':sample_origins}
    if obs:
        weights=np.zeros(len(obs));origins=np.asarray(sample_origins);rids=np.asarray(sample_roots)
        segments=[s for s in ('bridge_prefix','source_tail') if np.any(origins==s)]
        for segment in segments:
            group_roots=sorted(set(rids[origins==segment]))
            for rid in group_roots:
                mask=(rids==rid)&(origins==segment)
                weights[mask]=1/(len(segments)*len(group_roots)*int(mask.sum()))
        path=root/'demonstrations.npz'
        np.savez_compressed(path,observations=np.asarray(obs,np.float32),
                            actions=np.asarray(actions,np.float32),weights=weights)
        manifest.update(path=str(path.resolve()),sha256=file_sha(path))
    else:manifest['status']='empty_valid_demo'
    atomic_json(root/'manifest.json',manifest)
    return manifest


def validate_student_demonstration(trace):
    """Admission shared by current and cumulative student demonstration views."""
    m=trace['metadata'];a=trace['arrays'];n=validate_trace(trace)
    if (m.get('origin_type')!='verified_teacher' or m.get('teacher_status')!='verified_solution'
        or not m.get('verification_receipt_sha256') or m.get('role')!='train'
        or m.get('inherited_split')!='generator_train' or not m.get('complete')
        or not m.get('full_success') or not n or not a['success'][-1]
        or np.any(a['failure']) or np.any(a['timeout'])):
        raise ValueError('only complete verified TRAIN teacher trajectories may be demonstrations')
    if not set(a['action_origin'])<={'bridge_prefix','source_tail'}:
        raise ValueError('student/source-only actions are not new external demonstrations')
    return n
