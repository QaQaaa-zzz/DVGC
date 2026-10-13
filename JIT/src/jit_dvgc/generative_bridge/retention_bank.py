"""Hash-bound real-frame parent retention data; never open locked TEST arrays.

No simulator, JAX backend or optimizer is initialized by this module. Guard data
is training information. Ancestor/condition reuse within TRAIN is reported,
while any overlap with development or between development splits fails closed.
"""
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import numpy as np
from .contracts import file_sha
from .data import validate_trace, trajectory_identity

SCHEMA = 'frontierbridge_retention_bank_v1'
TRAIN_ROLES = ('TRAIN_FIT', 'TRAIN_GUARD')
DEV_ROLES = ('DEV_RETENTION', 'DEV_EXPANSION')
ROLES = TRAIN_ROLES + DEV_ROLES + ('TEST',)
IDENTITY_FIELDS = ('parent_actor_sha256', 'normalizer_sha256', 'protocol_sha256',
                   'actor_observation_dim', 'action_dim')


def _identity(identity):
    if any(not identity.get(k) for k in IDENTITY_FIELDS):
        raise ValueError('unresolved parent/normalizer/protocol identity')
    if identity['actor_observation_dim'] != 76 or identity['action_dim'] != 4:
        raise ValueError('identity must bind actual 76D Actor and 4D action schemas')
    for k in IDENTITY_FIELDS[:3]:
        _sha(identity[k], k)


def _sha(value, field):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid SHA256 identity: '+field)


def _roles(records, allowed_roles):
    if 'TEST' in allowed_roles:
        raise ValueError('locked TEST data cannot be opened by this interface')
    if not set(allowed_roles) <= set(TRAIN_ROLES + DEV_ROLES):
        raise ValueError('unsupported allowed roles')
    for r in records:
        if r.get('role') not in allowed_roles:
            raise ValueError('data role not permitted: '+str(r.get('role')))


def audit_split_isolation(records):
    """Metadata-only cross-split check, including unopened TEST declarations."""
    seen=[];reuse=False
    for r in records:
        if r.get('role') not in ROLES or not r.get('ancestor_ids') or not r.get('condition_identity') or not r.get('root_id'):
            raise ValueError('role, root, ancestor and condition identity required')
        ancestors=set(r['ancestor_ids'])
        for old in seen:
            overlap=(bool(ancestors & set(old['ancestor_ids'])) or
                     r['condition_identity']==old['condition_identity'] or r['root_id']==old['root_id'])
            if not overlap:continue
            if r['role']!=old['role']:
                if r['role'] in TRAIN_ROLES and old['role'] in TRAIN_ROLES:reuse=True
                else:raise ValueError('ancestor/root/condition split leakage')
        seen.append(r)
    return dict(train_guard_reuse=reuse, independent_guard=not reuse,
                roles=sorted({r['role'] for r in records}))


def _balanced_weights(records, episode_ids, group_weights):
    groups={r['capability_group'] for r in records}
    if set(group_weights)!=groups or not groups:
        raise ValueError('each declared capability group must be nonempty')
    values=np.asarray(list(group_weights.values()),float)
    if not np.isfinite(values).all() or np.any(values<=0) or not np.isclose(values.sum(),1.):
        raise ValueError('group weights must be positive finite and normalized')
    weights=np.zeros(len(episode_ids),float)
    for group,mass in group_weights.items():
        rows=[r for r in records if r['capability_group']==group]
        # Correlated ancestor sets form connected components; multi-ancestor
        # trajectories cannot gain mass simply by naming another ancestor.
        components=[]
        for row in rows:
            linked=[c for c in components if any(set(row['ancestor_ids']) & set(x['ancestor_ids']) for x in c)]
            merged=[row]+[x for c in linked for x in c]
            components=[c for c in components if c not in linked]+[merged]
        for component in components:
            for row in component:
                mask=episode_ids==row['root_id'];count=int(mask.sum())
                if not count:raise ValueError('empty trajectory')
                weights[mask]=mass/len(components)/len(component)/count
    return weights


def validate_retention_bank(manifest, arrays, *, allowed_roles=TRAIN_ROLES,
                            expected_identity=None, split_records=()):
    if manifest.get('schema')!=SCHEMA:raise ValueError('retention bank schema mismatch')
    identity=manifest.get('identity',{});_identity(identity)
    if expected_identity is not None and identity!=expected_identity:raise ValueError('bank identity mismatch')
    if (manifest.get('reference_label_policy')!='frozen_parent_deterministic' or
        manifest.get('reference_identity')!=identity):
        raise ValueError('reference labels must bind frozen parent identity')
    records=manifest.get('records',[]);_roles(records,allowed_roles)
    split=audit_split_isolation(list(records)+list(split_records))
    n=len(arrays['observations'])
    if not n:raise ValueError('empty retention bank')
    for key,width in [('observations',76),('reference_actions',4)]:
        x=np.asarray(arrays[key])
        if x.shape!=(n,width) or not np.isfinite(x).all():raise ValueError('invalid '+key)
    if np.any(np.abs(arrays['reference_actions'])>1):raise ValueError('reference action outside normalized range')
    for key in ('weights','group_ids','episode_ids'):
        if np.asarray(arrays[key]).shape!=(n,):raise ValueError('invalid '+key)
    weights=np.asarray(arrays['weights'])
    if not np.isfinite(weights).all() or np.any(weights<=0) or not np.isclose(weights.sum(),1):
        raise ValueError('invalid finite normalized weights')
    if len({r['root_id'] for r in records})!=len(records):raise ValueError('duplicate root representative')
    required=('root_id','ancestor_ids','condition_identity','source_policy_sha','normalizer_sha',
              'protocol_sha','role','capability_group','phase','trace_sha','valid_frame_range','validation_label')
    for r in records:
        if any(k not in r for k in required):raise ValueError('missing retention record provenance')
        for k in ('source_policy_sha','normalizer_sha','protocol_sha','trace_sha'):_sha(r[k],k)
        start,end=r['valid_frame_range']
        if not isinstance(start,int) or not isinstance(end,int) or start<0 or end<=start:raise ValueError('invalid valid frame interval')
        if r['validation_label']!='success':raise ValueError('retention requires validated success; UNKNOWN is not success')
        mask=np.asarray(arrays['episode_ids'])==r['root_id']
        if mask.sum()!=end-start or not np.all(np.asarray(arrays['group_ids'])[mask]==r['capability_group']):
            raise ValueError('record frame/group alignment mismatch')
    if set(arrays['episode_ids'])!={r['root_id'] for r in records}:raise ValueError('unknown episode samples')
    expected=_balanced_weights(records,np.asarray(arrays['episode_ids']),manifest['group_weights'])
    if not np.allclose(weights,expected,rtol=1e-10,atol=1e-12):raise ValueError('group/ancestor/trajectory/frame balance drift')
    return {**split,'count':n,'groups':sorted(manifest['group_weights']),
            'post_contact_observations':'UNKNOWN','deterministic_policy_retention':True}


def load_retention_bank(manifest_path, *, allowed_roles=TRAIN_ROLES,
                        expected_identity=None, split_records=()):
    path=Path(manifest_path);manifest=json.loads(path.read_text())
    # Reject forbidden split metadata before reading any array payload.
    _roles(manifest.get('records',[]),allowed_roles)
    audit_split_isolation(manifest.get('records',[])+list(split_records))
    payload=Path(manifest['path'])
    if not payload.is_absolute():payload=path.parent/payload
    if file_sha(payload)!=manifest.get('sha256'):raise ValueError('retention payload hash drift')
    with np.load(payload,allow_pickle=False) as raw:
        arrays={k:np.asarray(raw[k]) for k in ('observations','reference_actions','weights','group_ids','episode_ids')}
    audit=validate_retention_bank(manifest,arrays,allowed_roles=allowed_roles,
                                   expected_identity=expected_identity,split_records=split_records)
    return manifest,arrays,audit


def build_retention_bank(records, output, *, identity, reference_action_fn,
                         group_weights, split_records=()):
    """Compute frozen-parent labels on validated real pre-action observations.

    Caller owns the identity-bound frozen inference closure, including the fixed
    normalizer. Source traces are checked by the unchanged historical validator.
    No privileged observations or reconstructed FIFO are admitted.
    """
    _identity(identity);_roles(records,TRAIN_ROLES)
    audit_split_isolation(list(records)+list(split_records))
    entries=[];obs=[];episodes=[];groups=[]
    for r in records:
        trace=r['trace'];n=validate_trace(trace)
        if trace['metadata'].get('role') not in ('train','TRAIN') or trace['metadata'].get('inherited_split') in ('generator_dev','generator_test','test','dev'):
            raise ValueError('successful TRAIN source trace required')
        if (trace['metadata'].get('execution')!='independent_complete' or
            not set(trace['arrays']['action_origin'])<={'actor_only','source_only'}):
            raise ValueError('retention source requires independent complete Actor execution')
        if r['capability_group']=='accepted_parent_recovery' and (
            r['source_policy_sha']!=identity['parent_actor_sha256'] or r['normalizer_sha']!=identity['normalizer_sha256']):
            raise ValueError('accepted parent recovery source identity mismatch')
        if trace['metadata'].get('condition_identity')!=r['condition_identity']:
            raise ValueError('source condition identity must bind validated trace metadata')
        if trace['metadata'].get('root_id') != r['root_id']:
            raise ValueError('source root identity mismatch')
        if r.get('phase')!='UNKNOWN' and r.get('phase')!=trace['metadata'].get('validated_phase_label'):
            raise ValueError('phase label requires recorded validated source evidence')
        if not n or not trace['arrays']['success'][-1] or np.any(trace['arrays']['failure']) or np.any(trace['arrays']['timeout']):
            raise ValueError('complete successful real trajectory required')
        if trace['metadata'].get('complete') is not True or trace['metadata'].get('full_success') is not True:
            raise ValueError('complete source validation required')
        start,end=r['valid_frame_range']
        if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=n:raise ValueError('invalid source frame interval')
        entry={k:v for k,v in r.items() if k!='trace'}
        entry['trace_sha']=trajectory_identity(trace)
        for field,source in [('source_policy_sha','actor_sha256'),('normalizer_sha','normalizer_sha256'),('protocol_sha','protocol_sha256')]:
            if entry[field]!=trace['metadata'][source]:raise ValueError('source trace identity mismatch: '+field)
        entries.append(entry)
        obs.extend(np.asarray(trace['arrays']['actor_observation_before'])[start:end])
        episodes.extend([r['root_id']]*(end-start));groups.extend([r['capability_group']]*(end-start))
    observations=np.asarray(obs,np.float32).reshape(-1,76)
    arrays=dict(observations=observations,reference_actions=np.asarray(reference_action_fn(observations)),
                episode_ids=np.asarray(episodes),group_ids=np.asarray(groups))
    arrays['weights']=_balanced_weights(entries,arrays['episode_ids'],group_weights)
    manifest=dict(schema=SCHEMA,identity=identity,records=entries,group_weights=group_weights,
                  reference_label_policy='frozen_parent_deterministic',reference_identity=identity,
                  sampling_semantics='group/ancestor_component/trajectory/frame')
    validate_retention_bank(manifest,arrays,split_records=split_records)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    payload=root/'bank.npz';np.savez_compressed(payload,**arrays)
    manifest.update(path=str(payload.resolve()),sha256=file_sha(payload))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def audit_parent_manifest(manifest):
    """Audit declared artifacts read-only; parent remains an explicit checkpoint."""
    unresolved=[];artifacts={}
    from ..retention_consolidation import _audit_policy
    for name in ('source_pi0','parent_candidate'):
        policy=manifest.get(name)
        if not isinstance(policy,dict):unresolved.append(name);continue
        # Support the earlier explicit per-artifact fixture declaration while
        # production stage descriptors use the real checkpoint wrapper contract.
        if 'checkpoint' in policy and 'path' in policy:
            artifacts[name]=_audit_policy(policy,name,unresolved)
        else:
            artifacts[name]={}
            for key in ('payload','actor','normalizer','critic','config'):
                ref=policy.get(key)
                if not isinstance(ref,dict) or not ref.get('path') or not ref.get('sha256'):
                    unresolved.append(name+'.'+key);continue
                try:actual=file_sha(ref['path'])
                except (OSError,TypeError):unresolved.append(name+'.'+key);continue
                if actual!=ref['sha256']:raise ValueError('parent artifact hash drift: '+name+'.'+key)
                artifacts[name][key]=actual
    for key in ('initializer','retention_reference','task_identity','execution_code_identity'):
        if not manifest.get(key):unresolved.append(key)
    if manifest.get('initializer')!=manifest.get('retention_reference'):
        unresolved.append('initializer_reference_mismatch')
    return dict(status='unresolved' if unresolved else 'identity_verified',training_ready=not unresolved,
                unresolved=unresolved,artifacts=artifacts,reported_results=manifest.get('reported_results','UNKNOWN'),
                pi0_is_independent_reference=True)


def environment_provenance():
    """Package metadata and precision environment, with no backend device query."""
    versions={}
    for name in ('jax','jaxlib','brax','optax','numpy','mujoco','mujoco-mjx','nvidia-cuda-runtime-cu12'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='UNKNOWN_NOT_INSTALLED'
    return dict(python=platform.python_version(),packages=versions,devices='UNKNOWN_NOT_INITIALIZED',
                cuda='UNKNOWN_NOT_INITIALIZED',precision_environment={
                    k:os.environ.get(k,'UNSET') for k in ('JAX_ENABLE_X64','JAX_DEFAULT_MATMUL_PRECISION','JAX_PLATFORMS')})
