"""Cumulative real TRAIN demonstrations, independent of generator training data."""
import json
from pathlib import Path
import numpy as np
from .contracts import file_sha
from .data import export_student_demonstrations, trajectory_identity, validate_student_demonstration
from .protocol import atomic_json


def stratified_weights(sample_roots, sample_origins, new_roots):
    roots=np.asarray(sample_roots); origins=np.asarray(sample_origins)
    weights=np.zeros(len(roots),np.float64)
    groups=[set(roots)&set(new_roots),set(roots)-set(new_roots)]
    groups=[g for g in groups if g]
    for group in groups:
        for root in group:
            segments=[s for s in ('bridge_prefix','source_tail') if np.any((roots==root)&(origins==s))]
            for segment in segments:
                mask=(roots==root)&(origins==segment)
                weights[mask]=1/(len(groups)*len(group)*len(segments)*mask.sum())
    return weights


def build_student_demo_bank(teachers, output, *, source_identity, previous=None, round_id):
    """Retain first consistent representative; source/task changes require a new bank.

    previous is a hash-locked {path, sha256} manifest reference. A repeated root
    with a different complete trajectory is rejected, never silently averaged.
    """
    from .student import load_optional_demo
    required=('actor_sha256','normalizer_sha256','model_sha256','protocol_sha256')
    if any(not source_identity.get(k) for k in required):
        raise ValueError('complete source/task identity required')
    old=None; obs=[]; actions=[]; entries=[]; roots=[]; origins=[]
    if previous is not None:
        if file_sha(previous['path'])!=previous['sha256']:raise ValueError('previous bank hash drift')
        old=json.loads(Path(previous['path']).read_text())
        if old.get('schema')!='jit_bridge_demo_v1_2' or old['source_identity']!=source_identity:
            raise ValueError('previous bank source/task identity mismatch')
        for entry in old['entries']:
            if (entry.get('normalizer_sha256')!=source_identity['normalizer_sha256'] or
                entry.get('source_actor_sha256')!=source_identity['actor_sha256']):
                raise ValueError('previous bank teacher source/normalizer identity mismatch')
        data=load_optional_demo(old)
        if data is not None:obs.extend(data[0]);actions.extend(data[1])
        entries.extend(old['entries']);roots.extend(old['sample_roots']);origins.extend(old['sample_origins'])
    known={e['root_id']:e['trajectory_sha256'] for e in entries}; fresh=[]
    for trace in teachers:
        validate_student_demonstration(trace)
        m=trace['metadata']
        # The producer and tail identity are retained separately; source binding
        # may change only by creating an explicitly new bank.
        for k in ('model_sha256','protocol_sha256'):
            if m.get(k)!=source_identity[k]:raise ValueError('teacher task identity mismatch')
        if m.get('normalizer_sha256')!=source_identity['normalizer_sha256']:
            raise ValueError('teacher source normalizer identity mismatch')
        if m.get('source_actor_sha256')!=source_identity['actor_sha256']:
            raise ValueError('teacher tail source identity mismatch')
        rid=m['root_id']; identity=trajectory_identity(trace)
        if rid in known:
            if known[rid]!=identity:raise ValueError('conflicting representative for root')
            continue
        known[rid]=identity;fresh.append(trace)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    new=export_student_demonstrations(fresh,output/'new_verified')
    data=load_optional_demo(new)
    if data is not None:obs.extend(data[0]);actions.extend(data[1])
    entries.extend({**e,'admitted_round':round_id} for e in new['entries'])
    roots.extend(new['sample_roots']);origins.extend(new['sample_origins'])
    new_roots=[t['metadata']['root_id'] for t in fresh]
    manifest=dict(schema='jit_bridge_demo_v1_2',count=len(obs),entries=entries,
        sample_roots=roots,sample_origins=origins,source_identity=source_identity,
        round_id=round_id,new_roots=new_roots,previous=previous,
        sampling='new/history 50:50; roots equal; root prefix/tail 50:50; empty groups renormalized',
        student_adoption_required=False)
    if obs:
        path=output/'demonstrations.npz'
        np.savez_compressed(path,observations=np.asarray(obs,np.float32),actions=np.asarray(actions,np.float32),
            weights=stratified_weights(roots,origins,new_roots))
        manifest.update(path=str(path.resolve()),sha256=file_sha(path))
    else:manifest['status']='empty_valid_demo'
    atomic_json(output/'manifest.json',manifest)
    return manifest
