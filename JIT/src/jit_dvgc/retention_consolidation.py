"""Independent, opt-in Actor-only P0/P1/P2 stage. Preparation never simulates."""
from pathlib import Path
import json
import math
import subprocess
import sys
import time
from .generative_bridge.contracts import digest, file_sha
from .generative_bridge.protocol import atomic_json

SCHEMA='frontierbridge_retention_consolidation_v1'
REQUIRED_GROUPS=('old_nominal','old_random','old_recovery','accepted_parent_recovery')


def read(path):
    return json.loads(Path(path).read_text())


def validate_config(c):
    if c.get('schema')!=SCHEMA:raise ValueError('new consolidation schema required')
    if c.get('automatic_next_stage') is not False:raise ValueError('automatic next stage prohibited')
    roles=c.get('data_roles',{})
    if not isinstance(roles,dict) or set(roles)-{'TRAIN_FIT','TRAIN_GUARD','TEACHER'}:
        raise ValueError('only declared TRAIN_FIT/TRAIN_GUARD/TEACHER data inputs permitted; TEST remains closed')
    n=c.get('max_attempted_updates')
    if type(n) is not int or not 0<n<=1000:raise ValueError('attempt budget must be 1..1000')
    if type(c.get('seed')) is not int:raise ValueError('explicit integer seed required')
    steps=c.get('checkpoint_steps')
    if not isinstance(steps,list) or steps!=sorted(set(steps)) or not steps or steps[0]!=0 or any(type(x) is not int or x<0 or x>n for x in steps) or steps[-1]!=n:
        raise ValueError('checkpoint schedule must include zero and final attempt')
    if c.get('arms')!=['P0','P1','P2']:raise ValueError('matched P0/P1/P2 required')
    if c.get('tensorboard',{}).get('host') not in ('127.0.0.1','localhost'):
        raise ValueError('TensorBoard must use loopback')
    cost=c.get('cost_limits')
    if cost is not None:
        for k in ('environment_transitions','wall_seconds','supervised_attempts'):
            v=cost.get(k)
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('finite positive budget '+k)
        if type(cost['environment_transitions']) is not int or type(cost['supervised_attempts']) is not int or cost['supervised_attempts']!=2*n:
            raise ValueError('P1/P2 attempted update budget must equal 2*max_attempted_updates')
    return c


def validate_panel(panel):
    cases=panel.get('cases',[])
    if not cases:raise ValueError('nonempty locked evaluation panel required')
    seen=set();physical_seen=set()
    for c in cases:
        key=c.get('case_id')
        if not key or key in seen:raise ValueError('unique condition case_id required')
        seen.add(key)
        if c.get('role') not in ('DEV_RETENTION','DEV_EXPANSION','TRAIN_RETENTION'):raise ValueError('DEV or scoped seen nominal panel only; final TEST closed')
        if c.get('role')=='TRAIN_RETENTION':
            seen_contract=c.get('seen_condition_validation',{})
            if (c.get('group')!='old_nominal' or seen_contract.get('schema')!='frontierbridge_seen_condition_validation_v1'
                or seen_contract.get('training_condition_revisit') is not True or seen_contract.get('independent_dev') is not False
                or seen_contract.get('group')!='old_nominal' or seen_contract.get('source_train_condition_identity')!=c.get('condition_identity')):
                raise ValueError('only explicit nominal training-condition revisit permitted')
        if not c.get('condition_identity') or not c.get('group'):raise ValueError('condition/group identities required')
        seeds=c.get('repeat_seeds')
        if not isinstance(seeds,list) or len(seeds)!=3 or any(type(s) is not int for s in seeds):raise ValueError('three declared numerical repeats required')
        if c.get('mode') not in ('collect','evaluate') or c.get('horizon')!=400 or type(c.get('capacity')) is not int or c['capacity']<=0 or type(c.get('lane')) is not int or not 0<=c['lane']<c['capacity']:
            raise ValueError('invalid evaluation layout')
        s=c.get('spec',{})
        if any(k in s for k in ('bridge_action_plan','closed_loop_prefix_policy','warmup_initializer','closed_loop_prefix_initializer','reuse_results')):
            raise ValueError('student must independently execute complete state')
        if s.get('success_criterion')!='stable_forward_recovery' or s.get('horizon')!=400:
            raise ValueError('unchanged complete success protocol required')
        if c['mode']=='collect':
            from .retention_consolidation_runtime import validate_collection
            validate_collection(s)
            if s.get('num_envs')!=c['capacity']:raise ValueError('collect capacity must equal num_envs')
            if len(set(seeds))!=1:raise ValueError('collect numerical repeats must preserve identical condition seed')
            if s.get('frozen_request_table'):
                from .retention_consolidation_runtime import validate_collection
                validate_collection(s)
        if s.get('budget')!=c['capacity']*c['horizon']:
            raise ValueError('panel budget must exactly reserve actual capacity*horizon')
        if c['mode']=='evaluate' and (s.get('preserve_snapshot_episode_context') is not True or not s.get('candidates')):
            raise ValueError('full snapshot context required')
        if c['mode']=='evaluate':
            rows=read(s['candidates'])
            if not isinstance(rows,list) or len(rows)!=c['capacity'] or not rows:
                raise ValueError('evaluate capacity must equal complete candidate rows')
            if len({r.get('candidate_id') for r in rows})!=len(rows):raise ValueError('duplicate candidate ids')
            for row in rows:
                if row.get('role') not in ('DEV_RETENTION','DEV_EXPANSION','dev','acceptance'):
                    raise ValueError('snapshot candidate must have explicit DEV role')
                from .unified_envelope_snapshot import load_unified_envelope_snapshot,snapshot_context_sha256
                snapshot=load_unified_envelope_snapshot(Path(row['snapshot']))
                if row.get('snapshot_context_sha256')!=snapshot_context_sha256(snapshot):
                    raise ValueError('snapshot context must bind recorded complete context')
                if not math.isfinite(float(row.get('snapshot_time',float('nan')))):raise ValueError('finite snapshot time required')
                time_source=row.get('time_provenance')
                if not isinstance(time_source,dict) or not time_source.get('path') or not time_source.get('sha256'):
                    raise ValueError('recorded snapshot time provenance required')
                if file_sha(time_source['path'])!=time_source['sha256']:raise ValueError('snapshot time source hash drift')
                tick=time_source.get('tick');lane=time_source.get('lane')
                if type(tick) is not int or type(lane) is not int or tick<0 or lane<0 or time_source.get('time_field')!='time_after':
                    raise ValueError('explicit actual source tick/lane/time_after required')
                import numpy as np
                with np.load(time_source['path'],allow_pickle=False) as tape:
                    mask=tape['prefix_mask'] if 'prefix_mask' in tape else (tape['mask'] if 'mask' in tape else tape['valid_mask'])
                    if mask.ndim not in (1,2) or tick>=len(mask) or (mask.ndim==1 and lane!=0) or (mask.ndim==2 and lane>=mask.shape[1]):
                        raise ValueError('snapshot time source frame/lane outside actual tape')
                    select=lambda array:array[tick,lane] if mask.ndim==2 else array[tick]
                    if select(mask)!=1 or float(select(tape['time_after']))!=row['snapshot_time']:
                        raise ValueError('snapshot time differs from active recorded source frame')
                    if not np.array_equal(select(tape['qpos']),snapshot.qpos) or not np.array_equal(select(tape['qvel']),snapshot.qvel):
                        raise ValueError('snapshot time source physical frame mismatch')
            physical=digest({'snapshot_context':rows[c['lane']]['snapshot_context_sha256'],'snapshot_time':rows[c['lane']]['snapshot_time']})
        else:physical=digest({'spec':s,'lane':c['lane'],'condition_seed':seeds[0]})
        if physical in physical_seen:raise ValueError('duplicate actual physical condition despite renamed identity')
        physical_seen.add(physical)
    # Repeated nominal conditions are not independent cases.
    identities=[c['condition_identity'] for c in cases]
    if len(set(identities))!=len(identities):raise ValueError('duplicate condition_identity')
    return cases


def code_identity(repository):
    """Commit plus dirty tracked patch and untracked source hashes, never HEAD alone."""
    code=Path(repository).resolve()
    def git(*args):return subprocess.check_output(['git',*args],cwd=code)
    import hashlib
    patch=git('diff','HEAD','--binary')
    untracked=git('ls-files','--others','--exclude-standard','JIT/src','JIT/cli','JIT/configs').decode().splitlines()
    extra={p:file_sha(code/p) for p in untracked if (code/p).is_file()}
    return dict(commit=git('rev-parse','HEAD').decode().strip(),
        branch=git('branch','--show-current').decode().strip(),
        patch_sha256=hashlib.sha256(patch).hexdigest(),untracked_source_sha256=extra,
        dirty=bool(patch or extra))


def _lock_descriptors(value,locks,missing,prefix='',base=None,visited=None,repository_root=None):
    """Traverse manifests and known input paths, including their JSON references.

    Descriptor hashes are checked when supplied. Undeclared referenced payloads
    are hashed into the lock; this freezes bytes without inventing provenance.
    TEST manifests may be locked as metadata, but their data is never opened.
    """
    visited=set() if visited is None else visited
    base=Path.cwd() if base is None else Path(base)
    path_keys={'checkpoint','candidates','snapshot','spec_path','config_path',
               'source_frozen_policy','model_xml','xml_path','payload_path','bank','spec',
               'up_config_path','down_config_path','formal_config','initial_state_bank'}
    def lock(path,expected=None,label=prefix,repository_asset=False):
        original=Path(path)
        path=original if original.is_absolute() else base/original
        if (not path.exists() and repository_asset and repository_root is not None
            and not original.is_absolute() and original.parts and original.parts[0]=='assets'
            and '..' not in original.parts):
            # Official model.xml_path is repository-relative; adjacent manifests
            # and all generic references retain their original resolution rules.
            path=Path(repository_root)/original
        path=path.resolve()
        if path.is_dir():
            names=('identity.json','snapshot.pkl') if (path/'snapshot.pkl').exists() else ('identity.json','payload.pkl')
            for name in names:lock(path/name,label=label+'/'+name)
            return
        if not path.is_file():missing.append(label+':missing_file');return
        if str(path) in locks:
            if expected is not None and locks[str(path)]!=expected:raise ValueError('input hash drift '+str(path))
            return
        actual=file_sha(path)
        if expected is not None and actual!=expected:raise ValueError('input hash drift '+str(path))
        locks[str(path)]=actual
        if path.suffix=='.json' and path not in visited:
            visited.add(path)
            content=read(path)
            if isinstance(content,dict) and content.get('role')=='TEST':return
            _lock_descriptors(content,locks,missing,label,path.parent,visited,repository_root)
    if isinstance(value,dict):
        if value.get('role')=='TEST':return
        if 'path' in value:
            path=value.get('path');h=value.get('sha256')
            if path is None or h is None:missing.append(prefix+':unbound_path')
            else:lock(path,h)
        for key,child in value.items():
            if key in ('path','sha256'):continue
            if key in path_keys and isinstance(child,str):
                model_asset=key=='xml_path' and prefix.endswith('.model') and child.startswith('assets/')
                expected=value.get('xml_sha256') if model_asset else None
                if model_asset and not expected:missing.append(prefix+'.xml_path:unbound_asset_identity')
                lock(child,expected,label=prefix+'.'+key,repository_asset=model_asset and expected is not None)
            else:_lock_descriptors(child,locks,missing,prefix+'.'+key,base,visited,repository_root)
    elif isinstance(value,list):
        for i,child in enumerate(value):_lock_descriptors(child,locks,missing,prefix+f'[{i}]',base,visited,repository_root)


def _audit_policy(descriptor,name,missing):
    """Cross-bind serialized trees, checkpoint sidecar and source protocol."""
    from .generative_bridge.worker import source_payload
    from .handoff_bank import pytree_sha256
    _descriptor_json(descriptor)
    from .unified_policy_freeze import load_frozen_unified_manifest
    authoritative=load_frozen_unified_manifest(Path(descriptor['path']))['policy']
    policy,payload=source_payload(descriptor['path'])
    if policy!=authoritative:raise ValueError(name+' authoritative frozen wrapper mismatch')
    actual={key:pytree_sha256(tree) for key,tree in (
        ('actor_sha256',payload.actor_params),('normalizer_sha256',payload.observation_normalizer),
        ('critic_sha256',payload.critic_params))}
    for key,value in actual.items():
        if descriptor.get(key) is None:missing.append(name+'.'+key)
        elif descriptor[key]!=value:raise ValueError(name+' '+key+' payload mismatch')
        if key in policy and policy[key]!=value:raise ValueError(name+' wrapper '+key+' mismatch')
    for key in ('checkpoint','config','physics','actor_schema','critic_schema','action_transform'):
        if not descriptor.get(key):missing.append(name+'.'+key)
    checkpoint=descriptor.get('checkpoint')
    if checkpoint:
        expected=Path(policy['checkpoint']).resolve()/'payload.pkl'
        if not isinstance(checkpoint,dict) or Path(checkpoint.get('path','')).resolve()!=expected:
            raise ValueError(name+' checkpoint must bind wrapper payload.pkl')
        if file_sha(expected)!=checkpoint.get('sha256'):raise ValueError(name+' checkpoint payload drift')
    identity=payload.identity
    for field,source in (('config','config_sha256'),('physics','xml_sha256')):
        bound=descriptor.get(field)
        if bound:
            if not isinstance(bound,dict) or not bound.get('path') or file_sha(bound['path'])!=bound.get('sha256'):
                raise ValueError(name+' '+field+' descriptor mismatch')
            if field=='config':
                from .config import canonical_sha256
                semantic=canonical_sha256(read(bound['path']))
                declared=policy.get('formal_config_sha256',policy.get('config_sha256'))
                if policy.get('formal_config_file_sha256',bound['sha256'])!=bound['sha256']:
                    raise ValueError(name+' config file identity mismatch')
            else:semantic=bound['sha256'];declared=policy.get(source)
            if semantic!=getattr(identity,source) or declared!=semantic:
                raise ValueError(name+' '+field+' checkpoint identity mismatch')
    actor=descriptor.get('actor_schema')
    if actor:
        if (actor.get('observation_dim')!=76 or actor.get('frame_fields')!=list(identity.actor_frame_fields)
            or actor.get('task_fields')!=list(identity.actor_task_fields)
            or type(actor.get('history_window')) is not int or actor['history_window']<1
            or len(identity.actor_frame_fields)*actor['history_window']+len(identity.actor_task_fields)!=76):
            raise ValueError(name+' Actor observation/history identity mismatch')
    critic=descriptor.get('critic_schema')
    if critic and critic.get('observation_dim')!=106:raise ValueError(name+' critic observation identity mismatch')
    transform=descriptor.get('action_transform')
    if transform:
        raw_config=read(descriptor['config']['path'])
        upstream=raw_config.get('inputs',{}).get('up_config_path')
        action_contract=read(upstream).get('action') if upstream else None
        if not action_contract:missing.append(name+'.action_contract')
        elif transform.get('transform_identity')!=digest(action_contract):
            raise ValueError(name+' action transform identity mismatch')
        if transform.get('action_order')!=list(identity.action_order) or not transform.get('transform_identity'):
            raise ValueError(name+' action transform/order identity mismatch')
    return {**actual,'policy':policy,'identity':{
        'config_sha256':identity.config_sha256,'xml_sha256':identity.xml_sha256,
        'actor_schema':actor,'critic_schema':critic,'action_transform':transform}}


def _audit_teacher_sources(teacher,missing):
    import numpy as np
    from .generative_bridge.data import validate_student_demonstration,trajectory_identity
    data=teacher['path']
    with np.load(data,allow_pickle=False) as raw:
        observed=np.asarray(raw['observations']);actions=np.asarray(raw['actions'])
    offset=0
    for row in teacher.get('entries',[]):
        source=row.get('source_trace');verification=row.get('verification_receipt')
        if not source or not verification:
            missing.append('data_roles.TEACHER.source_trace_verification');continue
        if not isinstance(source,dict) or not source.get('path') or file_sha(source['path'])!=source.get('sha256'):
            raise ValueError('teacher source trace hash drift')
        receipt=_descriptor_json(verification)
        if row.get('verification_receipt_sha256') not in (digest(receipt),verification['sha256']):
            raise ValueError('teacher verification receipt does not bind source metadata')
        if (receipt.get('root_id')!=row['root_id'] or receipt.get('root_context_sha256')!=row['root_context_sha256']
            or receipt.get('source_trace_sha256')!=source['sha256']):
            raise ValueError('teacher receipt source root/context/trace mismatch')
        if receipt.get('label')!=1 or not receipt.get('attempts') or any(a.get('label')!=1 or a.get('outcome')!='stable_forward_recovery' for a in receipt['attempts']):
            raise ValueError('teacher receipt requires actual complete successful verification attempts')
        with np.load(source['path'],allow_pickle=False) as raw:
            trace={'metadata':row,'arrays':{key:np.asarray(raw[key]) for key in raw.files}}
        n=validate_student_demonstration(trace)
        if trajectory_identity(trace)!=row.get('trajectory_sha256') or row.get('count')!=n:
            raise ValueError('teacher trajectory identity/count drift')
        if (not np.array_equal(observed[offset:offset+n],trace['arrays']['actor_observation_before']) or
            not np.array_equal(actions[offset:offset+n],trace['arrays']['normalized_action_executed'])):
            raise ValueError('teacher BC dataset differs from verified real executed frames')
        if teacher.get('sample_roots',[])[offset:offset+n]!=[row['root_id']]*n or teacher.get('sample_origins',[])[offset:offset+n]!=trace['arrays']['action_origin'].tolist():
            raise ValueError('teacher frame provenance differs from source trace')
        offset+=n
    if offset!=len(observed) and not any('source_trace_verification' in reason for reason in missing):
        raise ValueError('teacher dataset has unaccounted frames')


def _audit_thresholds(guard,guard_manifest,parent_identity,missing):
    import numpy as np
    limit=guard.get('max_consecutive_rejections')
    if limit is None:missing.append('guard_config.max_consecutive_rejections')
    elif type(limit) is not int or limit<1:raise ValueError('positive explicit max_consecutive_rejections required')
    for key in ('mse_thresholds','p99_thresholds'):
        thresholds=guard.get(key)
        if not isinstance(thresholds,dict) or set(thresholds)!=set(REQUIRED_GROUPS):
            missing.append('guard_config.'+key);continue
        for group,values in thresholds.items():
            if values is None or (isinstance(values,list) and any(v is None for v in values)):
                missing.append('guard_config.'+key+'.'+group);continue
            try:array=np.asarray(values,dtype=float)
            except (TypeError,ValueError):raise ValueError('invalid guard threshold '+group)
            if array.shape!=(4,) or not np.isfinite(array).all() or np.any(array<0):
                raise ValueError('four finite nonnegative channel thresholds required: '+group)
    if guard.get('calibrated') is not True or not guard.get('calibration_receipt'):
        missing.append('guard_config.calibration');return
    receipt=_descriptor_json(guard['calibration_receipt'])
    if (receipt.get('schema')!='frontierbridge_guard_calibration_v1'
        or receipt.get('calibrated') is not True or receipt.get('data_role') not in ('TRAIN_GUARD','DEV_RETENTION')
        or not receipt.get('cost_ledger') or not receipt.get('candidate_thresholds')
        or receipt.get('guard_bank_sha256')!=guard_manifest.get('sha256')
        or receipt.get('parent_actor_sha256')!=parent_identity.get('actor_sha256')
        or any(receipt.get(k)!=guard.get(k) for k in ('mse_thresholds','p99_thresholds'))):
        raise ValueError('actual calibrated receipt/parent/bank/threshold identity mismatch')


def prepare(config,output,repository):
    c=validate_config(read(config));root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    locks={str(Path(config).resolve()):file_sha(config)};missing=[]
    _lock_descriptors(c,locks,missing,base=Path(config).resolve().parent,repository_root=repository)
    identity=code_identity(repository)
    from .generative_bridge.retention_bank import environment_provenance
    parent_manifest=dict(schema='frontierbridge_parent_manifest_v1',source_pi0=c.get('pi0_reference'),
        parent_candidate=c.get('parent'),initializer=c.get('parent'),retention_reference=c.get('parent'),
        task_identity=c.get('task_identity') or ({key:c['parent'].get(key) for key in
            ('config','physics','actor_schema','critic_schema','action_transform')} if c.get('parent') else None),
        execution_code_identity=identity,
        status='unresolved' if c.get('parent') is None else 'declared_not_yet_runtime_verified')
    atomic_json(root/'parent_manifest.json',parent_manifest)
    atomic_json(root/'environment.json',environment_provenance())
    atomic_json(root/'input_lock.json',dict(locks=locks,missing=missing,code=identity))
    p=dict(schema=SCHEMA,config=c,config_path=str(Path(config).resolve()),output=str(root),
        repository=str(Path(repository).resolve()),input_locks=locks,execution_code_identity=identity,
        preparation_missing=missing,automatic_next_stage=False)
    atomic_json(root/'plan.json',p)
    atomic_json(root/'status.json',dict(phase='prepared',attempted_updates=0,accepted_updates=0,
        environment_transitions=0,published_policy=None))
    atomic_json(root/'cost_ledger.json',dict(old_acquisition_cost='UNKNOWN',new_data_transitions=0,
        teacher_search_transitions=0,training_environment_transitions=0,eval_reserved_transitions=0,
        supervised_attempts=0,accepted_updates=0,entries=[]))
    return root/'plan.json'


def _descriptor_json(d):
    if not isinstance(d,dict) or not d.get('path') or not d.get('sha256'):raise ValueError('bound path/sha256 required')
    if file_sha(d['path'])!=d['sha256']:raise ValueError('descriptor drift '+d['path'])
    return read(d['path'])


def audit(plan):
    p=read(plan);c=validate_config(p['config']);missing=list(p.get('preparation_missing',[]))
    if p.get('schema')!=SCHEMA or read(p['config_path'])!=c:raise ValueError('embedded plan config drift')
    expected_locks={str(Path(p['config_path']).resolve()):file_sha(p['config_path'])};current_missing=[]
    _lock_descriptors(c,expected_locks,current_missing,base=Path(p['config_path']).parent,repository_root=p['repository'])
    if expected_locks!=p['input_locks']:raise ValueError('recursive input lock membership/hash drift')
    for f,h in p['input_locks'].items():
        if file_sha(f)!=h:raise ValueError('input drift '+f)
    if code_identity(p['repository'])!=p['execution_code_identity']:raise ValueError('execution code drift; prepare a new plan')
    for k in ('parent','pi0_reference','guard_config','selection_config','cost_limits','evaluation_panel','task_identity'):
        if c.get(k) is None:missing.append(k)
    roles=c.get('data_roles',{})
    for k in ('TRAIN_FIT','TRAIN_GUARD','TEACHER'):
        if roles.get(k) is None:missing.append('data_roles.'+k)
    port=c['tensorboard'].get('port')
    if type(port) is not int or not 1024<=port<=65535:missing.append('tensorboard.port')
    if c.get('execution_code_identity') is not None and c['execution_code_identity']!=p['execution_code_identity']:
        raise ValueError('declared execution code identity differs')
    details={}
    if c.get('evaluation_panel'):
        panel=_descriptor_json(c['evaluation_panel']);cases=validate_panel(panel)
        # Parent + independent pi0 three repeats; P1/P2 checkpoints (zero reused).
        jobs={}
        for case in cases:
            key=digest(dict(mode=case['mode'],spec=case['spec'],repeat_seeds=case['repeat_seeds']))
            jobs[key]=case['capacity']*case['horizon']*3
        baseline_count=1 if c.get('parent')==c.get('pi0_reference') and c.get('parent') else 2
        reserve=sum(jobs.values())*(baseline_count+2*(len(c['checkpoint_steps'])-1))
        details['evaluation_reservation']=reserve
        if c.get('cost_limits') and reserve>c['cost_limits']['environment_transitions']:missing.append('cost_limits.evaluation_reserve')
        dev_records=[row for row in panel.get('split_records',[]) if row.get('role')!='TRAIN_RETENTION']
        if not any(case['role']=='DEV_EXPANSION' for case in cases):missing.append('evaluation_panel.DEV_EXPANSION')
        if set(REQUIRED_GROUPS)-{case['group'] for case in cases if case['role'] in ('DEV_RETENTION','TRAIN_RETENTION')}:
            missing.append('evaluation_panel.required_groups')
        if any(not any(row.get('condition_identity')==case['condition_identity'] and row.get('role')==case['role'] for row in dev_records) for case in cases if case['role']!='TRAIN_RETENTION'):
            missing.append('evaluation_panel.split_records')
    else:dev_records=[]
    identities={}
    for name in ('parent','pi0_reference'):
        if c.get(name):
            identities[name]=_audit_policy(c[name],name,missing)
            details[name+'_identity']=identities[name]
    if len(identities)==2 and identities['parent']['identity']!=identities['pi0_reference']['identity']:
        raise ValueError('parent/pi0 physics/config/observation/action protocols differ')
    task={}
    if c.get('task_identity'):
        task=_descriptor_json(c['task_identity'])
        if task.get('schema')!='frontierbridge_task_identity_v1' or any(not task.get(key) for key in
            ('physics_sha256','task','reward','termination','actor_schema','critic_schema','action_transform')):
            raise ValueError('complete physical/task/reward/termination/schema identity required')
        parent=identities.get('parent')
        if parent and (task['physics_sha256']!=parent['identity']['xml_sha256'] or
            any(task[key]!=parent['identity'][key] for key in ('actor_schema','critic_schema','action_transform'))):
            raise ValueError('task identity differs from frozen physical/action/observation contract')
    banks={}
    if roles.get('TRAIN_FIT') and roles.get('TRAIN_GUARD'):
        from .generative_bridge.retention_bank import load_retention_bank,audit_split_isolation
        all_records=[]
        for role in ('TRAIN_FIT','TRAIN_GUARD'):
            _descriptor_json(roles[role])
            manifest,arrays,result=load_retention_bank(roles[role]['path'],allowed_roles=(role,),split_records=dev_records)
            if set(result['groups'])!=set(REQUIRED_GROUPS):missing.append('data_roles.'+role+'.required_groups')
            parent=identities.get('parent')
            if parent and (manifest['identity']['parent_actor_sha256']!=parent['actor_sha256'] or
                manifest['identity']['normalizer_sha256']!=parent['normalizer_sha256']):
                raise ValueError('bank labels frozen parent/normalizer mismatch')
            if task and manifest['identity']['protocol_sha256']!=digest(task):
                raise ValueError('bank protocol identity mismatch')
            banks[role]=manifest;all_records.extend(manifest['records']);details[role]=result
        details['split_isolation']=audit_split_isolation(all_records+dev_records)
        for case in cases if c.get('evaluation_panel') else []:
            if case['role']!='TRAIN_RETENTION':continue
            if c.get('seen_condition_validation')!=case['seen_condition_validation']:
                raise ValueError('configuration must explicitly bind seen nominal condition scope')
            if not any(row['capability_group']=='old_nominal' and row['condition_identity']==case['condition_identity'] for row in all_records):
                raise ValueError('seen nominal revisit must bind an actual TRAIN bank condition')
            details['nominal_seen_condition']=case['seen_condition_validation']
    if roles.get('TEACHER'):
        from .generative_bridge.student import load_optional_demo
        teacher=_descriptor_json(roles['TEACHER'])
        if load_optional_demo(teacher) is None:missing.append('data_roles.TEACHER.empty')
        for row in teacher.get('entries',[]):
            if row.get('role')!='train' or row.get('teacher_status')!='verified_solution' or not row.get('full_success') or not row.get('complete'):
                raise ValueError('teacher must retain verified complete TRAIN provenance')
        if not teacher.get('entries'):missing.append('data_roles.TEACHER.provenance')
        _audit_teacher_sources(teacher,missing)
        for case in cases if c.get('evaluation_panel') else []:
            if case['role']!='DEV_EXPANSION':continue
            proof=case.get('teacher_evidence')
            if not proof:missing.append('evaluation_panel.expansion_teacher_evidence');continue
            evidence=_descriptor_json(proof)
            row=evidence.get('metadata',{})
            if row.get('role')!='DEV_EXPANSION' or case['mode']!='evaluate' or row.get('condition_identity')!=case['condition_identity']:
                raise ValueError('expansion requires independent DEV teacher condition evidence')
            source=evidence['source_trace'];receipt=_descriptor_json(evidence['verification_receipt'])
            if file_sha(source['path'])!=source['sha256']:raise ValueError('DEV teacher source trace drift')
            if (receipt.get('root_id')!=row.get('root_id') or receipt.get('root_context_sha256')!=row.get('root_context_sha256')
                or receipt.get('source_trace_sha256')!=source['sha256'] or receipt.get('label')!=1
                or row.get('parent_repeat_labels')!=[0,0,0] or row.get('teacher_repeat_labels')!=[1,1,1]):
                raise ValueError('DEV expansion stable parent/teacher/root/context evidence mismatch')
            from .generative_bridge.data import validate_trace,trajectory_identity
            import numpy as np
            with np.load(source['path'],allow_pickle=False) as raw:trace=dict(metadata=row,arrays={k:np.asarray(raw[k]) for k in raw.files})
            n=validate_trace(trace)
            if (not n or not trace['arrays']['success'][-1] or np.any(trace['arrays']['failure']) or np.any(trace['arrays']['timeout'])
                or row.get('complete') is not True or row.get('full_success') is not True or row.get('teacher_status')!='verified_solution'
                or row.get('origin_type')!='verified_teacher' or not set(trace['arrays']['action_origin'])<={'bridge_prefix','source_tail'}
                or trajectory_identity(trace)!=evidence.get('trajectory_sha256')):
                raise ValueError('DEV teacher complete real trajectory identity invalid')
            snapshot_rows=read(case['spec']['candidates'])
            if any(r.get('snapshot_context_sha256')!=row['root_context_sha256'] for r in snapshot_rows):
                raise ValueError('expansion must independently replay verified teacher complete context')
        teacher_splits=[]
        for row in teacher.get('entries',[]):
            if not row.get('root_id') or not row.get('ancestor_ids') or not row.get('condition_identity'):
                missing.append('data_roles.TEACHER.split_provenance');continue
            teacher_splits.append(dict(root_id=row['root_id'],ancestor_ids=row['ancestor_ids'],
                condition_identity=row['condition_identity'],role='TRAIN_FIT'))
        from .generative_bridge.retention_bank import audit_split_isolation
        details['teacher_split_isolation']=audit_split_isolation(teacher_splits+dev_records+
            [row for manifest in banks.values() for row in manifest['records']])
        # Repeat labels are retained as declarations, never converted to expansion.
        if any('parent_repeat_labels' not in row or 'teacher_repeat_labels' not in row for row in teacher.get('entries',[])):
            missing.append('data_roles.TEACHER.repeat_labels')
    guard=c.get('guard_config')
    if guard:
        _audit_thresholds(guard,banks.get('TRAIN_GUARD',{}),identities.get('parent',{}),missing)
    selection=c.get('selection_config')
    if selection:
        from .retention_consolidation_evaluation import _selection_config
        _selection_config(selection)
        if set(selection.get('required_groups',{}))!=set(REQUIRED_GROUPS):missing.append('selection_config.required_groups')
        if selection.get('min_expansion_gain',0)<=0:missing.append('selection_config.min_expansion_gain')
        if not selection.get('parent_pi0_gap_decision'):missing.append('selection_config.parent_pi0_gap_decision')
    return dict(status='blocked' if missing else 'ready',unresolved=sorted(set(missing)),
        environment_transitions=0,optimizer_updates=0,teacher_searches=0,
        deterministic_policy_retention=True,details=details)


def dry_run(plan):
    result=audit(plan);atomic_json(Path(read(plan)['output'])/'dry_run.json',result);return result


def require_execution(plan,authorization):
    a=audit(plan)
    if a['status']!='ready':raise ValueError('execution blocked: '+', '.join(a['unresolved']))
    if authorization is None:raise ValueError('explicit new-stage authorization required')
    grant=read(authorization);p=read(plan)
    if grant.get('schema')!='frontierbridge_execution_authorization_v1' or grant.get('execute') is not True or grant.get('plan_sha256')!=file_sha(plan) or grant.get('cost_limits')!=p['config']['cost_limits'] or grant.get('arms')!=['P0','P1','P2'] or not grant.get('user_authorization_record'):
        raise ValueError('authorization must bind new plan, arms, budgets and actual user record')
    return p,grant


def run(plan,authorization=None):
    p,grant=require_execution(plan,authorization)
    root=Path(p['output'])
    if read(root/'status.json')['phase']!='prepared':raise ValueError('no implicit rerun/resume')
    with (root/'execution.claim').open('x') as f:f.write(file_sha(plan))
    atomic_json(root/'execution_authorization.json',dict(path=str(Path(authorization).resolve()),sha256=file_sha(authorization),grant=grant))
    from .retention_consolidation_runtime import launch
    return launch(plan,authorization)


def worker(plan,authorization=None):
    p,grant=require_execution(plan,authorization)
    root=Path(p['output']);receipt=read(root/'execution_authorization.json')
    if receipt['sha256']!=file_sha(authorization) or receipt['grant']!=grant or not (root/'execution.claim').is_file() or read(root/'status.json')['phase']!='running':
        raise ValueError('worker authorization or stage state invalid')
    with (root/'worker.claim').open('x') as f:f.write(file_sha(plan))
    from .retention_consolidation_runtime import train_stage
    return train_stage(plan)


def report(plan):
    p=read(plan);root=Path(p['output']);status=read(root/'status.json')
    result=dict(schema=SCHEMA,training='not_run' if status['phase']=='prepared' else status['phase'],
        conclusion='engineering_blocked' if status['phase']!='completed' else 'no_accepted_improvement',
        published_policy=None,deterministic_policy_retention=True,
        stochastic_scale_retention='not_guaranteed',status=status,costs=read(root/'cost_ledger.json'))
    if (root/'checkpoint_results.json').exists():
        from .retention_consolidation_evaluation import select_checkpoint
        rows=read(root/'checkpoint_results.json');result['selection']={arm:select_checkpoint(candidates,p['config']['parent'],p['config']['selection_config']) for arm,candidates in rows.items() if arm in ('P1','P2')}
    result['arms']={arm:read(root/arm/'warmup_status.json') for arm in ('P1','P2') if (root/arm/'warmup_status.json').exists()}
    p2=result['arms'].get('P2',{})
    if p2.get('status')=='blocked_by_guard':result['conclusion']='blocked_by_guard'
    elif p2.get('status')=='stopped_retention_guard':result['conclusion']='retention_gate_failed'
    elif result.get('selection',{}).get('P2',{}).get('phase')=='accepted_dev_improvement':result['conclusion']='absorption_and_retention_dev_gates_passed'
    elif p2.get('status')=='completed':result['conclusion']='no_accepted_improvement'
    if status['phase']=='failed':result['conclusion']='engineering_error'
    # Optional report assets are derived solely from completed saved observations.
    if (root/'checkpoint_results.json').exists():
        from .retention_consolidation_report import generate_figures
        result['figures']=generate_figures(p,result)
    atomic_json(root/'report.json',result)
    return result
