"""Read-only adapters from locked historical real traces to FrontierBridge inputs.

Derived receipts explicitly describe verified historical evidence: they do not
claim new simulation, repeat verification or source-tail generation. Sources are
never modified. Forward-only parent labeling initializes no physical environment.
"""
from pathlib import Path
import json
import numpy as np
from .generative_bridge.contracts import digest,file_sha
from .generative_bridge.protocol import atomic_json
from .generative_bridge.data import validate_trace,trajectory_identity,validate_student_demonstration


def descriptor(path):
    return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


def collection_condition_identity(qpos,qvel,requests,*,protocol_sha256):
    """Nominal duplicate detection excludes seed/episode names and group labels."""
    arrays=[np.asarray(x) for x in (qpos,qvel,requests)]
    if any(not np.isfinite(x).all() for x in arrays):raise ValueError('finite declared condition required')
    return digest(dict(protocol_sha256=protocol_sha256,qpos=arrays[0].tolist(),
        qvel=arrays[1].tolist(),requested=arrays[2].tolist()))


def prepare_parent_inputs(frozen,output,repository):
    from .unified_policy_freeze import load_frozen_unified_manifest
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    policy=load_frozen_unified_manifest(Path(frozen))['policy']
    config=Path(policy['formal_config']);formal=json.loads(config.read_text())
    upstream=Path(formal['inputs']['up_config_path']);up=json.loads(upstream.read_text())
    downstream=Path(formal['inputs']['down_config_path']);down=json.loads(downstream.read_text())
    xml=Path(up['model']['xml_path'])
    if not xml.is_absolute():xml=Path(repository)/xml
    history=(76-len(policy['actor_task_fields']))//len(policy['actor_frame_fields'])
    actor_schema=dict(observation_dim=76,frame_fields=policy['actor_frame_fields'],
        task_fields=policy['actor_task_fields'],history_window=history)
    critic_schema=dict(observation_dim=106,history_window=history,privileged=True)
    action_transform=dict(action_order=policy['action_order'],transform_identity=digest(up['action']))
    parent=dict(**descriptor(frozen),checkpoint=descriptor(Path(policy['checkpoint'])/'payload.pkl'),
        config=descriptor(config),physics=descriptor(xml),actor_schema=actor_schema,
        critic_schema=critic_schema,action_transform=action_transform,
        **{key:policy[key] for key in ('actor_sha256','normalizer_sha256','critic_sha256')})
    protocol=dict(schema='frontierbridge_task_identity_v1',physics_sha256=policy['xml_sha256'],
        task={'source_config':descriptor(config),'source_config_semantic_sha256':policy['formal_config_sha256'],
              'upstream':descriptor(upstream),'downstream':descriptor(downstream),
              'start':'fixed_complete_near_ground_x2.5','control_dt':.02,'simulation_dt':.005},
        reward={'upstream':up.get('reward',up.get('rewards')),'downstream':down.get('reward',down.get('rewards')),
                'formal_reward_mode':formal.get('reward_mode'),'formal_contract':formal.get('reward_contract')},
        termination={'criterion':formal.get('success_criterion'),'upstream_events':up.get('events'),
                     'downstream_descent':down.get('descent'),'horizon':400},
        actor_schema=actor_schema,critic_schema=critic_schema,action_transform=action_transform)
    path=root/'task_identity.json';atomic_json(path,protocol)
    result=dict(parent=parent,pi0_reference=parent,task_identity=descriptor(path),
        bank_identity=dict(parent_actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256'],
                           protocol_sha256=digest(protocol),actor_observation_dim=76,action_dim=4))
    atomic_json(root/'parent_inputs.json',result)
    return result


def _real_lane(raw,lane,*,mask_key='mask',steps=None):
    mask=np.asarray(raw[mask_key][:,lane])
    if not np.isin(mask,[0,1]).all():raise ValueError('binary real frame mask required')
    ids=np.flatnonzero(mask);n=len(ids)
    if not n or not np.array_equal(ids,np.arange(n)) or (steps is not None and n!=steps):
        raise ValueError('contiguous complete real-frame lane required')
    keys=('actor_observation_before','actor_observation_after','normalized_action_executed',
          'success','phase_before')
    arrays={key:np.asarray(raw[key])[ids,lane].copy() for key in keys}
    for target,sources in (('done',('done','terminal')),('failure',('failure','physical_failure')),
                           ('timeout',('timeout','metric/terminal/descent_timeout','metric/terminal/timeout'))):
        found=next((key for key in sources if key in raw),None)
        if found is None:raise ValueError('recorded source '+target+' flag required')
        arrays[target]=np.asarray(raw[found])[ids,lane].copy()
    arrays['valid_mask']=np.ones(n,bool)
    return arrays


def adapt_teacher_record(row,output,*,role,condition_identity,protocol_sha256,source_descriptor):
    """Keep the existing fixed G representative: final original confirmation."""
    if role not in ('TRAIN_FIT','DEV_EXPANSION'):raise ValueError('explicit teacher source role required')
    if (row.get('role')!=('TRAIN' if role=='TRAIN_FIT' else 'SOLVER_DEV') or
        row.get('pi0_labels')!=[0,0,0]):raise ValueError('original teacher split/source labels mismatch')
    if file_sha(source_descriptor['path'])!=source_descriptor['sha256']:
        raise ValueError('original teacher source receipt hash drift')
    source_rows=json.loads(Path(source_descriptor['path']).read_text())
    if not isinstance(source_rows,list) or row not in source_rows:
        raise ValueError('teacher metadata must match the original locked source row')
    method=row['methods']['G'];attempts=method['verified_attempts']
    if (method.get('status')!='verified_solution' or method.get('repeat_verified') is not True or
        method.get('source_labels')!=[0,0,0] or method.get('winner_labels')!=[1,1,1] or len(attempts)!=3 or
        not method.get('winner_lock',{}).get('fixed_before_replays')):
        raise ValueError('historical fixed winner/three confirmations required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    locks=[]
    for attempt in attempts:
        if (attempt.get('label')!=1 or attempt.get('outcome')!='stable_forward_recovery' or
            attempt.get('controller_kind')!='composite_teacher' or attempt.get('source_only') is not False or
            attempt.get('snapshot_context_sha256')!=row['snapshot_context_sha256'] or
            attempt.get('actor_sha256')!=row['tail']['actor_sha256'] or
            attempt.get('normalizer_sha256')!=row['tail']['normalizer_sha256'] or
            file_sha(attempt['trace'])!=attempt['trace_sha256']):
            raise ValueError('historical teacher attempt/tail/context/hash mismatch')
        locks.append(dict(path=attempt['trace'],sha256=attempt['trace_sha256']))
    selected=attempts[-1];lane=selected['trace_lane']
    with np.load(selected['trace'],allow_pickle=False) as raw:
        arrays=_real_lane(raw,lane,steps=selected['steps'])
        codes=np.asarray(raw['action_origin_code'])[:len(arrays['done']),lane]
    if not np.array_equal(codes,np.where(np.arange(len(codes))<16,1,2)):
        raise ValueError('locked real H16/source tail origin mismatch')
    arrays['action_origin']=np.where(codes==1,'bridge_prefix','source_tail')
    metadata=dict(root_id=row['root_id'],root_episode_id=row['root_episode_id'],
        ancestor_ids=[row['root_episode_id']],condition_identity=condition_identity,
        root_context_sha256=row['snapshot_context_sha256'],actor_sha256=selected['actor_sha256'],
        normalizer_sha256=selected['normalizer_sha256'],model_sha256=selected['model_sha256'],
        protocol_sha256=protocol_sha256,role='train' if role=='TRAIN_FIT' else 'DEV_EXPANSION',
        inherited_split='generator_train' if role=='TRAIN_FIT' else 'generator_dev',
        origin_type='verified_teacher',teacher_status='verified_solution',complete=True,full_success=True,
        parent_repeat_labels=row['pi0_labels'],teacher_repeat_labels=method['winner_labels'],
        source_tail_identity=row['tail'],post_contact='UNKNOWN')
    source_path=root/'canonical_trace.npz';np.savez_compressed(source_path,**arrays)
    receipt=dict(schema='frontierbridge_historical_teacher_adapter_v1',label=1,
        root_id=row['root_id'],root_context_sha256=row['snapshot_context_sha256'],
        source_trace_sha256=file_sha(source_path),source_role=row['role'],attempts=attempts,
        original_source=source_descriptor,original_trace_locks=locks,winner_lock=method['winner_lock'],
        new_environment_transitions=0,new_verification_repeats=0,
        representative='final_original_confirmation',source_tail_recomputed=False)
    receipt_path=root/'verification_adapter.json';atomic_json(receipt_path,receipt)
    metadata['verification_receipt_sha256']=digest(receipt)
    trace=dict(metadata=metadata,arrays=arrays)
    if role=='TRAIN_FIT':validate_student_demonstration(trace)
    else:validate_trace(trace)
    evidence=dict(metadata=metadata,source_trace=descriptor(source_path),verification_receipt=descriptor(receipt_path),
        trajectory_sha256=trajectory_identity(trace))
    evidence_path=root/'evidence.json';atomic_json(evidence_path,evidence)
    return trace,descriptor(evidence_path)


def independent_keep_records(collection,receipt,*,protocol_sha256,policy,role='TRAIN_FIT',group_for_episode):
    """Use raw complete Actor-only successes, preserving real post-contact frames."""
    source=Path(collection);declaration=json.loads((source.parent/'keep_spec.json').read_text())
    if (declaration.get('role')!='TRAIN' or declaration.get('controller_mode')!='fixed_random' or
        not declaration.get('frozen_request_table') or any(declaration.get(k) for k in
            ('warmup_initializer','explorer_checkpoint','episode_policy_mixture'))):
        raise ValueError('unchanged independent TRAIN pi0 collection required')
    request_ref=declaration['frozen_request_table']
    if file_sha(request_ref['path'])!=request_ref['sha256']:raise ValueError('historical fixed request hash drift')
    with np.load(request_ref['path'],allow_pickle=False) as requests_data:
        if np.any(requests_data['requested']):raise ValueError('zero-residual original keep requests required')
    from .probe_bank import load_probe_bank
    source_bank=load_probe_bank(Path(declaration['bank']))
    member=next((m for m in source_bank['members'] if m['name']==declaration['proposer']),None)
    if not member or any(member['policy'][key]!=policy[key] for key in ('actor_sha256','normalizer_sha256','xml_sha256')):
        raise ValueError('historical keep frozen policy identity differs from selected parent')
    original=json.loads(Path(receipt).read_text())
    if original.get('role')!='train' or original.get('full_success') is not True or original.get('DEV_used') is not False:
        raise ValueError('original successful TRAIN receipt required')
    records=[]
    with np.load(source/'prefixes.npz',allow_pickle=False) as raw:
        for entry in original['episodes']:
            # Merged receipts may refer to an old collection. Caller selects
            # source-native entries only, never guesses an episode's lane.
            if original.get('schema')=='jit_merged_keep_coverage_v1' and entry.get('source_kind')=='old_keep':continue
            lane=entry['lane'];decision=group_for_episode(entry)
            if decision is None:continue
            group,start_mode=decision
            arrays=_real_lane(raw,lane,mask_key='prefix_mask',steps=entry['steps'])
            n=len(arrays['done'])
            if not arrays['success'][-1] or np.any(arrays['failure']) or np.any(arrays['timeout']):
                raise ValueError('successful receipt disagrees with original complete source trace')
            if np.any(raw['requested_delta'][:n,lane]) or np.any(raw['effective_delta'][:n,lane]):
                raise ValueError('keep source includes exploration residual')
            arrays['action_origin']=np.full(n,'actor_only')
            requested=np.zeros((400,4),np.float32)
            condition=collection_condition_identity(raw['initial_qpos'][lane],raw['initial_qvel'][lane],requested,protocol_sha256=protocol_sha256)
            ancestor=entry['ancestor'];context=digest(dict(source=descriptor(source/'prefixes.npz'),lane=lane))
            metadata=dict(root_id=ancestor,root_context_sha256=context,condition_identity=condition,
                role='train',inherited_split='generator_train',execution='independent_complete',
                actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256'],
                model_sha256=policy['xml_sha256'],protocol_sha256=protocol_sha256,complete=True,full_success=True)
            trace=dict(metadata=metadata,arrays=arrays);validate_trace(trace)
            contact=np.flatnonzero(np.asarray(raw['first_valid_contact'])[:n,lane])
            if start_mode=='post_contact':
                if not len(contact):raise ValueError('recorded contact phase required; UNKNOWN not imputed')
                start=int(contact[0])+1;phase='post_contact';metadata['validated_phase_label']=phase
            else:start=0;phase='UNKNOWN'
            records.append(dict(root_id=ancestor,ancestor_ids=[ancestor,'condition:'+condition],condition_identity=condition,
                source_policy_sha=policy['actor_sha256'],normalizer_sha=policy['normalizer_sha256'],protocol_sha=protocol_sha256,
                role=role,capability_group=group,phase=phase,trace=trace,valid_frame_range=[start,n],validation_label='success',
                original_source=descriptor(source/'prefixes.npz'),original_receipt=descriptor(receipt),source_lane=lane,
                source_full_steps=n,recorded_contact_frame=int(contact[0]) if len(contact) else 'UNKNOWN'))
    return records


def materialize_parent_banks(sources,parent,output,*,require_backend='gpu'):
    """Only deterministic target-backend forward labels; zero step/update/search."""
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics as rs
    from .ppo import make_network_factory
    from .generative_bridge.worker import source_payload
    from .generative_bridge.retention_bank import build_retention_bank
    from .handoff_bank import pytree_sha256
    if jax.default_backend()!=require_backend:raise ValueError('explicit target inference backend required')
    source_path=Path(sources);manifest=json.loads(source_path.read_text())
    if manifest.get('schema')!='frontierbridge_real_retention_sources_v1':raise ValueError('real source schema required')
    policy,payload=source_payload(parent)
    bound=manifest['identity']
    if (pytree_sha256(payload.actor_params)!=bound['parent_actor_sha256'] or
        pytree_sha256(payload.observation_normalizer)!=bound['normalizer_sha256']):
        raise ValueError('labeling parent tree identity mismatch')
    network=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
    norm,actor=jax.tree.map(jax.lax.stop_gradient,(payload.observation_normalizer,payload.actor_params))
    @jax.jit
    def parent_prediction(x):
        return network.parametric_action_distribution.mode(network.policy_network.apply(norm,actor,{'state':jp.asarray(x)}))
    records=[]
    for row in manifest['records']:
        for key in ('source_trace','original_source','original_receipt'):
            ref=row[key]
            if file_sha(ref['path'])!=ref['sha256']:raise ValueError('real source hash drift: '+key)
        with np.load(row['source_trace']['path'],allow_pickle=False) as raw:
            trace=dict(metadata=row['trace_metadata'],arrays={key:np.asarray(raw[key]) for key in raw.files})
        validate_trace(trace)
        if trajectory_identity(trace)!=row['trajectory_sha256']:raise ValueError('source trajectory identity drift')
        records.append(dict(**{key:value for key,value in row.items() if key not in ('trace_metadata','source_trace','trajectory_sha256')},trace=trace))
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    result={}
    for role in ('TRAIN_FIT','TRAIN_GUARD'):
        rows=[{**row,'role':role} for row in records]
        bank=build_retention_bank(rows,root/role,identity=bound,
            reference_action_fn=lambda obs:np.asarray(parent_prediction(obs)),group_weights=manifest['group_weights'])
        result[role]=descriptor(root/role/'manifest.json')
    receipt=dict(schema='frontierbridge_forward_labeling_v1',backend=jax.default_backend(),
        devices=[dict(id=d.id,platform=d.platform,kind=d.device_kind) for d in jax.devices()],
        jax_enable_x64=bool(jax.config.jax_enable_x64),matmul_precision=str(jax.config.jax_default_matmul_precision),
        parent=descriptor(parent),source_manifest=descriptor(sources),bank_identity=bound,banks=result,
        guard_reuses_fit=True,guard_is_training_information=True,new_environment_transitions=0,
        optimizer_updates=0,teacher_searches=0,protocol='same jitted stop-gradient parent graph as warmup')
    atomic_json(root/'labeling_receipt.json',receipt)
    atomic_json(root/'bank_inputs.json',result)
    return result
