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



def _tail_key(identity):
    if not isinstance(identity,dict) or any(not identity.get(k) for k in ('actor_sha256','normalizer_sha256')):
        raise ValueError('complete teacher tail Actor and normalizer identity required')
    return identity['actor_sha256'],identity['normalizer_sha256']


def _allowed_teacher_tails(source_identity, baseline_identity, lineage):
    """Verify an explicit accepted-checkpoint chain, never infer adoption."""
    from .student import load_retention_reference
    required=('actor_sha256','normalizer_sha256','model_sha256','protocol_sha256')
    if not isinstance(baseline_identity,dict) or any(not baseline_identity.get(k) for k in required):
        raise ValueError('complete immutable baseline identity required')
    if not isinstance(lineage,list):raise ValueError('explicit teacher tail lineage list required')
    if any(source_identity[k]!=baseline_identity[k] for k in ('model_sha256','protocol_sha256')):
        raise ValueError('promotion cannot change physical/task protocol identity')
    head=_tail_key(baseline_identity);allowed={head};seen=set()
    for ref in lineage:
        if not isinstance(ref,dict) or file_sha(ref['path'])!=ref['sha256']:
            raise ValueError('teacher tail adoption receipt hash drift')
        if ref['sha256'] in seen:raise ValueError('duplicate teacher tail adoption receipt')
        seen.add(ref['sha256'])
        receipt=json.loads(Path(ref['path']).read_text())
        if (receipt.get('schema')!='jit_bridge_teacher_tail_adoption_v1_2' or
            receipt.get('adopted') is not True):
            raise ValueError('explicit adopted teacher tail receipt required')
        if receipt.get('baseline_identity')!=baseline_identity:
            raise ValueError('teacher tail baseline/task identity drift')
        if _tail_key(receipt['previous_tail'])!=head:
            raise ValueError('teacher tail adoption lineage is not contiguous')
        _,policy=load_retention_reference(receipt['frozen_policy'])
        candidate=_tail_key(receipt['candidate_tail'])
        if candidate!=_tail_key(policy):raise ValueError('adopted teacher tail checkpoint identity mismatch')
        if policy.get('xml_sha256')!=baseline_identity['model_sha256']:
            raise ValueError('adopted teacher tail physical model identity mismatch')
        if candidate in allowed:raise ValueError('teacher tail adoption must advance to a new checkpoint identity')
        head=candidate;allowed.add(head)
    if _tail_key(source_identity)!=head:
        raise ValueError('changed teacher source requires a verified accepted-tail receipt')
    return allowed

def build_student_demo_bank(teachers, output, *, source_identity, previous=None, round_id,
                            baseline_identity=None, teacher_tail_lineage=None):
    """Retain first consistent real TRAIN representatives without relabeling tails.

    Default remains strict same-source identity. Opt-in migration takes immutable
    baseline_identity plus an ordered hash-locked teacher_tail_lineage. Every new
    tail must have an explicit adoption receipt and verified frozen checkpoint;
    task/physics remain fixed. Old v1.1 banks are never automatically imported.
    """
    from .student import load_optional_demo
    required=('actor_sha256','normalizer_sha256','model_sha256','protocol_sha256')
    if any(not source_identity.get(k) for k in required):
        raise ValueError('complete source/task identity required')
    promotion=baseline_identity is not None or teacher_tail_lineage is not None
    allowed=(_allowed_teacher_tails(source_identity,baseline_identity,teacher_tail_lineage)
        if promotion else {_tail_key(source_identity)})
    old=None; obs=[]; actions=[]; entries=[]; roots=[]; origins=[]
    if previous is not None:
        if file_sha(previous['path'])!=previous['sha256']:raise ValueError('previous bank hash drift')
        old=json.loads(Path(previous['path']).read_text())
        if old.get('schema')!='jit_bridge_demo_v1_2':
            raise ValueError('previous bank source/task identity mismatch; no automatic old-protocol migration')
        if promotion:
            if old.get('baseline_identity',old['source_identity'])!=baseline_identity:
                raise ValueError('previous bank immutable baseline identity mismatch')
            old_lineage=old.get('teacher_tail_lineage',[])
            if teacher_tail_lineage[:len(old_lineage)]!=old_lineage:
                raise ValueError('previous teacher tail adoption lineage must be retained unchanged')
            # The previous bank head must match the end of its own declared chain.
            history_allowed=_allowed_teacher_tails(old['source_identity'],baseline_identity,old_lineage)
        elif old['source_identity']!=source_identity or 'baseline_identity' in old:
            raise ValueError('previous bank source/task identity mismatch')
        else:history_allowed=allowed
        for entry in old['entries']:
            if (entry.get('source_actor_sha256'),entry.get('normalizer_sha256')) not in history_allowed:
                raise ValueError('previous bank teacher source/normalizer identity mismatch')
            if any(entry.get(k)!=source_identity[k] for k in ('model_sha256','protocol_sha256')):
                raise ValueError('previous bank teacher task identity mismatch')
        data=load_optional_demo(old)
        if data is not None:obs.extend(data[0]);actions.extend(data[1])
        entries.extend(old['entries']);roots.extend(old['sample_roots']);origins.extend(old['sample_origins'])
    known={e['root_id']:e['trajectory_sha256'] for e in entries}; fresh=[]
    for trace in teachers:
        validate_student_demonstration(trace)
        m=trace['metadata']
        # Keep original producer/tail hashes: historical solutions are not
        # relabeled as witnesses for the currently adopted source.
        for k in ('model_sha256','protocol_sha256'):
            if m.get(k)!=source_identity[k]:raise ValueError('teacher task identity mismatch')
        if (m.get('source_actor_sha256'),m.get('normalizer_sha256')) not in allowed:
            raise ValueError('teacher tail source/normalizer identity mismatch')
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
    if promotion:
        manifest.update(baseline_identity=dict(baseline_identity),teacher_tail_lineage=list(teacher_tail_lineage),
            historical_tail_semantics='original verified tail only; no claim of success under current tail')
    if obs:
        path=output/'demonstrations.npz'
        np.savez_compressed(path,observations=np.asarray(obs,np.float32),actions=np.asarray(actions,np.float32),
            weights=stratified_weights(roots,origins,new_roots))
        manifest.update(path=str(path.resolve()),sha256=file_sha(path))
    else:manifest['status']='empty_valid_demo'
    atomic_json(output/'manifest.json',manifest)
    return manifest
