"""Sparse causal TRAIN exports from completed checkpoint comparisons, no replay.

Observations belong to the action origin; novelty and suffix success belong to a
strictly later saved state. These are correlated imitation examples, not evidence
that the residual is necessary, or that a learned explorer improves discovery.
"""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import math
import numpy as np
from .jump_evidence_validation import read, write, file_sha, verify_hash
from .residual_exploration import digest, _verify_files

DATASET_SCHEMA = 'residual_causal_dataset_v2'
SOURCE_SCHEMA = 'residual_causal_action_witnesses_v2'


def split_groups(groups, *, seed, development_fraction):
    groups=sorted(set(groups))
    if len(groups)<2 or not 0<development_fraction<1 or type(seed) is not int:
        raise ValueError('at least two trajectory groups and explicit split required')
    ranked=sorted(groups,key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    count=max(1,min(len(groups)-1,math.ceil(len(groups)*development_fraction)))
    return dict(fit=sorted(ranked[count:]),development=sorted(ranked[:count]))


def link_action_pairs(rows, witnesses, points, observations, baseline, base_sha, endpoint_sha):
    if len(rows)!=len(witnesses) or len(rows)!=len(points):
        raise ValueError('candidate/witness/projection count mismatch')
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['trajectory_id']].append((i,r))
    result=[];stats=dict(no_next_saved_action=0,no_later_novel_witness=0)
    keys=('nominal_actions','perturbed_actions','effective_deltas')
    for group_name,group in groups.items():
        group.sort(key=lambda x:x[1]['trajectory_step'])
        ticks=[r['trajectory_step'] for _,r in group]
        if any(type(t) is not int or t<1 for t in ticks) or len(set(ticks))!=len(ticks):
            raise ValueError('invalid/duplicate trajectory tick')
        last=group[-1][1];full=last['perturbation']
        for i,r in group:
            tick=r['trajectory_step'];p=r['perturbation']
            if r['protocol_sha256']!=last['protocol_sha256']:
                raise ValueError('trajectory protocol drift')
            for key in keys:
                values=np.asarray(p[key])
                if (len(p[key])!=tick or values.shape!=(tick,4) or not np.isfinite(values).all()
                    or p[key]!=full[key][:tick]):raise ValueError('action prefix mismatch')
            if not np.allclose(np.asarray(p['perturbed_actions'])-np.asarray(p['nominal_actions']),p['effective_deltas'],atol=1e-7,rtol=0):
                raise ValueError('effective action delta mismatch')
            if tick==ticks[-1]:stats['no_next_saved_action']+=1;continue
            eligible=[(j,s) for j,s in group if s['trajectory_step']>tick
                and witnesses[j].get('label')==1 and witnesses[j].get('witness_status')=='observed_landing'
                and points[j]['root_cell'] not in baseline]
            if not eligible:stats['no_later_novel_witness']+=1;continue
            j,target=eligible[0]
            goal=[r['jump_start_reachability']['perturbation_anchor_x_m'],
                  r['jump_start_reachability']['perturbation_anchor_x_m']-p['lookback_m'] if 'lookback_m' in p else r['jump_start_reachability']['perturbation_anchor_x_m']-r['jump_start_reachability']['lookback_m'],
                  *p['basis_vector'],p['strength']]
            # Index tick is the action executed AFTER the origin snapshot.
            pair=dict(role='TRAIN',base_sha256=base_sha,observation=list(observations[i]),history=[],
                base_action=full['nominal_actions'][tick],goal=goal,target_delta=full['effective_deltas'][tick],
                applied_action=full['perturbed_actions'][tick],previous_delta=full['effective_deltas'][tick-1],
                state_sha256=target['state_sha256'],context_sha256=target['snapshot_context_sha256'],
                acquisition_protocol_sha256=r['protocol_sha256'],root_cell=points[j]['root_cell'],
                forward_prefix_complete=True,continuation_status='success',endpoint='first_valid_landing',
                physical_failure=False,continuation_endpoint_protocol_sha256=endpoint_sha,
                continuation_state_sha256=target['state_sha256'],continuation_context_sha256=target['snapshot_context_sha256'],
                trajectory_group=group_name,
                action_origin=dict(candidate_index=i,tick=tick,state_sha256=r['state_sha256'],context_sha256=r['snapshot_context_sha256']),
                witness_origin=dict(candidate_index=j,tick=target['trajectory_step'],state_sha256=target['state_sha256'],context_sha256=target['snapshot_context_sha256']))
            result.append(pair)
    return result,stats


def link_tape_action_pairs(rows, witnesses, points, tapes, baseline, base_sha, endpoint_sha, *, condition_group):
    """Join verified complete tapes to nearest later clean, novel saved witnesses."""
    if len(rows)!=len(witnesses) or len(rows)!=len(points):
        raise ValueError('candidate/witness/projection count mismatch')
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['trajectory_id']].append((i,row))
    if set(groups)-set(tapes):raise ValueError('candidate missing action tape')
    result=[];stats=dict(raw_actions=0,exported_actions=0,no_later_novel_witness=0,channels={
        name:{sign:dict(raw_actions=0,exported_actions=0,no_later_novel_witness=0) for sign in ('positive','negative')}
        for name in ('steer','rear_wheel_drive','hip','knee')})
    for trajectory,tape in sorted(tapes.items()):
        if tape['trajectory_id']!=trajectory:raise ValueError('tape trajectory identity mismatch')
        axis=next(i for i,v in enumerate(tape['goal'][2:6]) if v)
        sign='positive' if tape['goal'][axis+2]>0 else 'negative'
        channel=stats['channels'][('steer','rear_wheel_drive','hip','knee')[axis]][sign]
        records=tape['records'];group=sorted(groups[trajectory],key=lambda item:item[1]['trajectory_step'])
        by_tick={}
        for i,row in group:
            tick=row['trajectory_step']
            if type(tick) is not int or not 1<=tick<=len(records) or tick in by_tick:
                raise ValueError('invalid/duplicate candidate tape tick')
            by_tick[tick]=i
            if row['protocol_sha256']!=tape['protocol_sha256']:
                raise ValueError('tape protocol mismatch')
            before=records[tick-1]
            if (row['state_sha256']!=before['next_state_sha256'] or
                row['snapshot_context_sha256']!=before['next_context_sha256']):
                raise ValueError('candidate tape post-action identity mismatch')
            if tick<len(records) and (row['state_sha256']!=records[tick]['state_sha256'] or
                row['snapshot_context_sha256']!=records[tick]['context_sha256']):
                raise ValueError('candidate tape pre-action identity mismatch')
            perturbation=row['perturbation'];reach=row['jump_start_reachability']
            goal=[reach['perturbation_anchor_x_m'],reach['perturbation_anchor_x_m']-
                  perturbation.get('lookback_m',reach.get('lookback_m')), *perturbation['basis_vector'],perturbation['strength']]
            if not np.allclose(goal,tape['goal'],rtol=0,atol=1e-12):raise ValueError('tape exogenous goal mismatch')
            for prefix,field in (('nominal_actions','base_action'),('perturbed_actions','applied_action'),('effective_deltas','effective_delta')):
                if perturbation[prefix]!=[record[field] for record in records[:tick]]:
                    raise ValueError('candidate action tape prefix mismatch')
        eligible=[(i,row) for i,row in group if witnesses[i].get('label')==1
                  and witnesses[i].get('witness_status')=='observed_landing'
                  and points[i]['root_cell'] not in baseline]
        for tick,record in enumerate(records):
            stats['raw_actions']+=1;channel['raw_actions']+=1
            later=next(((i,row) for i,row in eligible if row['trajectory_step']>tick),None)
            if later is None:
                stats['no_later_novel_witness']+=1;channel['no_later_novel_witness']+=1;continue
            j,target=later
            origin=dict(tick=tick,state_sha256=record['state_sha256'],context_sha256=record['context_sha256'],
                trajectory_id=trajectory,protocol_sha256=tape['protocol_sha256'],tape_sha256=tape['tape_sha256'],
                controller_history=record['controller_history'],controller_previous_delta=record['previous_delta'])
            if tick in by_tick:origin['candidate_index']=by_tick[tick]
            if 'source_binding' in tape:origin['action_tape']=tape['source_binding']
            result.append(dict(role='TRAIN',base_sha256=base_sha,observation=record['observation'],history=[],
                base_action=record['base_action'],goal=tape['goal'],target_delta=record['effective_delta'],
                applied_action=record['applied_action'],previous_delta=records[tick-1]['effective_delta'] if tick else [0.]*4,
                state_sha256=target['state_sha256'],context_sha256=target['snapshot_context_sha256'],
                acquisition_protocol_sha256=tape['protocol_sha256'],root_cell=points[j]['root_cell'],
                forward_prefix_complete=True,continuation_status='success',endpoint='first_valid_landing',physical_failure=False,
                continuation_endpoint_protocol_sha256=endpoint_sha,continuation_state_sha256=target['state_sha256'],
                continuation_context_sha256=target['snapshot_context_sha256'],trajectory_group=trajectory,
                condition_group=condition_group,action_origin=origin,
                witness_origin=dict(candidate_index=j,tick=target['trajectory_step'],state_sha256=target['state_sha256'],
                    context_sha256=target['snapshot_context_sha256'])))
            stats['exported_actions']+=1;channel['exported_actions']+=1
    return result,stats


def verify_observation_sources(source_files):
    """Only saved-Actor observation semantics must match; no physics replay gate."""
    for name in ('observation.py', 'constants.py'):
        path=Path(__file__).resolve().parent/name
        if source_files.get(str(path)) != file_sha(path):
            raise ValueError('historical observation source mismatch: '+name)


def _checked_comparison(root, *, require_action_tapes=False):
    """Validate completed data receipts, not restart authority under new sources."""
    from .probe_bank import load_probe_bank
    from .checkpoint_comparison import project_catalog, validate_witness_receipt
    from .unified_envelope_snapshot import load_unified_envelope_snapshot
    from .observation import actor_observation, HistoryState
    from .tube_rsi import PHASE_UPSTREAM
    from jax import numpy as jnp
    root=Path(root).resolve();plan=read(root/'plan.json');verify_hash(plan,'plan_sha256')
    verify_observation_sources(plan['source_files'])
    summary=read(root/'summary.json')
    if summary['status']!='completed' or summary['plan_sha256']!=plan['plan_sha256'] or summary['final_test_used'] is not False:
        raise ValueError('completed TRAIN comparison required')
    _verify_files([dict(path=p,sha256=s) for p,s in plan['input_files'].items()])
    spec=plan['spec'];bank=load_probe_bank(Path(spec['bank']));members={m['name']:m for m in bank['members']}
    endpoint=dict(model_sha256=bank['task']['xml_sha256'],physical_cell_schema_sha256=bank['task']['resolution_sha256'],success_criterion=bank['task']['success_criterion'],continuation_start_semantics=bank['task']['continuation_start_semantics'],horizon=bank['max_ticks'],conflict_policy='quarantine_contact_and_physical_failure_v1')
    with Path(spec['baseline_csv']).open() as f:baseline={r['root_cell'] for r in csv.DictReader(f) if r['witnessed']=='True'}
    pairs=[];stats={};total=0;inputs={root/'plan.json',root/'summary.json',Path(spec['baseline_csv']),Path(spec['bank'])}
    inputs.update(Path(p) for p in plan['input_files'])
    for name in spec['proposers']:
        arm=root/name;cp=arm/'acquire/result/catalog.json';catalog=read(cp);member_name=spec.get('proposer_members',{}).get(name,name);member=members[member_name]
        ep=read(arm/'existence_plan.json');verify_hash(ep,'plan_sha256')
        _verify_files([dict(path=p,sha256=s) for p,s in ep['input_files'].items()])
        if (ep['bank_sha256']!=bank['bank_sha256'] or Path(ep['catalog']).resolve()!=cp or ep['proposer']!=member_name
            or ep['role']!='train' or ep['final_test_used'] is not False or ep['conflict_policy']!=endpoint['conflict_policy']):
            raise ValueError('existence input binding mismatch')
        index=read(arm/'labels/result/witness_index.json');verify_hash(index,'index_sha256')
        validate_witness_receipt(index,ep,catalog,members,member,arm/'labels/result',ep['budget'])
        points=project_catalog(cp,member)
        total+=index['charged_interactions']+catalog['environment_interactions']
        if require_action_tapes:
            from .action_tape import load_action_tape
            tapes={}
            receipts=catalog.get('trajectory_receipts',[])
            if not receipts:raise ValueError('complete action tape receipts required')
            for receipt in receipts:
                binding=receipt.get('action_tape')
                if not binding:raise ValueError('complete action tape binding required')
                path=(cp.parent/binding['path']).resolve()
                if not path.is_relative_to(cp.parent):raise ValueError('action tape path escapes acquisition')
                tape=load_action_tape(path,expected_sha256=binding['sha256'])
                trajectory=receipt['trajectory_id']
                if trajectory in tapes:raise ValueError('duplicate trajectory tape')
                if (tape['trajectory_id']!=trajectory or tape['protocol_sha256']!=catalog['protocol_sha256'] or
                    tape['policy_actor_sha256']!=catalog['policy_actor_sha256'] or
                    tape['policy_payload_sha256']!=catalog['policy_payload_sha256'] or
                    tape['environment_interactions']!=receipt['environment_interactions']):
                    raise ValueError('action tape acquisition binding mismatch')
                tapes[trajectory]=dict(tape,source_binding=dict(path=str(path),sha256=binding['sha256']));inputs.add(path)
            if sum(t['environment_interactions'] for t in tapes.values())!=catalog['environment_interactions']:
                raise ValueError('complete action tape interaction ledger mismatch')
            new,counts=link_tape_action_pairs(catalog['entries'],index['entries'],points,tapes,baseline,
                member['frozen_file_sha256'],digest(endpoint),condition_group=str(root))
        else:
            obs={}
            for i,row in enumerate(catalog['entries']):
                snap=load_unified_envelope_snapshot(cp.parent/row['source_bank']/row['snapshot'])
                if snap.episode_step!=row['trajectory_step']:raise ValueError('snapshot action tick mismatch')
                signal=snap.up_events['jump_signal'] if snap.active_phase==PHASE_UPSTREAM else False
                obs[i]=np.asarray(actor_observation(HistoryState(jnp.asarray(snap.observation_fifo),jnp.asarray(snap.history_valid_count)),jnp.asarray(signal))).tolist()
            new,counts=link_action_pairs(catalog['entries'],index['entries'],points,obs,baseline,member['frozen_file_sha256'],digest(endpoint))
        for r in new:r['source_arm']=name;r['catalog']=str(cp)
        pairs.extend(new);stats[name]=counts
        inputs.update(p for p in arm.rglob('*') if p.is_file() and p.suffix in {'.json','.pkl','.npz'})
    if total!=summary['charged_interactions']:raise ValueError('completed comparison cost mismatch')
    return pairs,stats,baseline,endpoint,[dict(path=m['frozen_policy'],sha256=m['frozen_file_sha256']) for m in bank['members'] if m['name'] in {spec.get('proposer_members',{}).get(n,n) for n in spec['proposers']}],inputs


def build_export(spec):
    diagnostics={};partition_sources={};conditions={}
    if 'comparison_partitions' in spec:
        declared=spec['comparison_partitions']
        if set(declared)!={'fit','development'} or spec.get('require_action_tapes') is not True:
            raise ValueError('explicit fit/development partitions require complete action tapes')
        roots={}
        for partition,paths in declared.items():
            if not isinstance(paths,list) or not paths or any(not Path(p).is_absolute() for p in paths):
                raise ValueError('nonempty absolute comparison roots required')
            roots[partition]=[str(Path(p).resolve()) for p in paths]
        flat=[p for paths in roots.values() for p in paths]
        if len(set(flat))!=len(flat):raise ValueError('comparison conditions must be disjoint')
        partitions={p:[] for p in roots};stats={};inputs=set();bank=[];baseline=None;endpoint=None
        for partition,paths in roots.items():
            partition_sources[partition]=[]
            for root in paths:
                rows,counts,seen,ep,bases,files=_checked_comparison(root,require_action_tapes=True)
                if baseline is None:baseline=seen;endpoint=ep
                elif baseline!=seen or endpoint!=ep:raise ValueError('comparison baseline or endpoint drift')
                inputs.update(files);stats[root]=counts
                for base in bases:
                    if base not in bank:bank.append(base)
                for row in rows:
                    row=dict(row,trajectory_group=root+'::'+row['trajectory_group'],condition_group=root)
                    partitions[partition].append(row)
                partition_sources[partition].append(dict(comparison_root=root,rows=len(rows)))
            conditions[partition]=sorted({tuple(r['goal'][k] for k in (0,1,6)) for r in partitions[partition]})
            diagnostics[partition]={}
            for axis,name in enumerate(('steer','rear_wheel_drive','hip','knee')):
                counts={sign:sum(r['goal'][axis+2]*direction>0 and abs(r['target_delta'][axis])>1e-7
                    for r in partitions[partition]) for sign,direction in (('positive',1),('negative',-1))}
                if not all(counts.values()):raise ValueError('all signed action channel coverage required in each partition')
                diagnostics[partition][name]=counts
        if set(conditions['fit']) & set(conditions['development']):
            raise ValueError('whole perturbation conditions must be disjoint across partitions')
        split={p:sorted({r['trajectory_group'] for r in rows}) for p,rows in partitions.items()}
        pairs=[r for rows in partitions.values() for r in rows]
        bank=sorted(bank,key=lambda b:(b['path'],b['sha256']))
        for r in pairs:
            target=np.asarray(r['target_delta']);previous=np.asarray(r['previous_delta'])
            if (np.any(abs(target)>np.asarray(spec['delta_limit'])+1e-7) or
                np.any(abs(previous)>np.asarray(spec['delta_limit'])+1e-7) or
                np.any(abs(target-previous)>np.asarray(spec['slew_limit'])+1e-7)):
                raise ValueError('action tape training target exceeds residual limits')
    else:
        pairs,stats,baseline,endpoint,bank,inputs=_checked_comparison(spec['comparison_root'])
        # Preserve the legacy trajectory-condition grouping and seeded split.
        split=split_groups([r['trajectory_group'] for r in pairs],seed=spec['split_seed'],development_fraction=spec['development_fraction'])
        partitions={key:[r for r in pairs if r['trajectory_group'] in groups] for key,groups in split.items()}
    if any(not rows for rows in partitions.values()):raise ValueError('empty grouped partition')
    width=len(pairs[0]['observation']);goal_names=['anchor_x_m','window_start_x_m','direction_steer','direction_rear_wheel_drive','direction_hip','direction_knee','strength']
    features=np.asarray([r['observation']+r['base_action']+r['goal'] for r in partitions['fit']],dtype=np.float32)
    limit=spec['delta_limit'];slew=spec['slew_limit']
    contract=dict(observation_names=[f'actor_observation_{i:03d}' for i in range(width)],goal_names=goal_names,
        action_names=['steer','rear_wheel_drive','hip','knee'],history_steps=0,
        feature_mean=features.mean(axis=0).tolist(),feature_scale=np.maximum(features.std(axis=0),1e-3).tolist(),
        delta_limit=limit,slew_limit=slew,action_low=[-1.]*4,action_high=[1.]*4,control_dt=.02,
        model_sha256=endpoint['model_sha256'],physical_cell_schema_sha256=endpoint['physical_cell_schema_sha256'],
        observation_contract_sha256=digest({name:file_sha(Path(__file__).parent/name) for name in ('observation.py','constants.py')}),
        endpoint_protocol_sha256=digest(endpoint),observation_contract='canonical saved FIFO plus active-phase jump signal; actor history already included',
        goal_contract='exogenous fixed perturbation schedule; no future outcome features',normalization_fit_groups=split['fit'])
    mode=spec.get('normalization_mode','fit_std_v1')
    if mode == 'actor_fit_std_goal_units_v1':
        # Action coordinates and exogenous goals have known physical units.
        # Constant training directions must not amplify unseen unit directions.
        contract['feature_mean'][width:]=[0.]*(4+len(goal_names))
        contract['feature_scale'][width:]=[1.]*(4+len(goal_names))
        contract['normalization_mode']=mode
    elif mode != 'fit_std_v1':
        raise ValueError('unknown normalization mode')
    from .residual_exploration import validate_contract
    validate_contract(contract)
    inputs.update(Path(__file__).parent/name for name in ('observation.py','constants.py'))
    return dict(contract=contract,partitions=partitions,split=split,stats=stats,baseline=sorted(baseline),endpoint=endpoint,bank=bank,
                inputs=[dict(path=str(p.resolve()),sha256=file_sha(p)) for p in sorted(inputs)],
                channel_diagnostics=diagnostics,partition_sources=partition_sources,condition_sets=conditions)


def export_dataset(spec, output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('preserve existing export')
    built=build_export(spec);output.mkdir(parents=True)
    write(output/'spec.json',spec)
    for partition,records in built['partitions'].items():
        source=output/f'{partition}_records.json'
        doc=dict(schema=SOURCE_SCHEMA,role='TRAIN',partition=partition,export_spec=spec,
            contract_sha256=digest(built['contract']),records=records,split=built['split'],inputs=built['inputs'])
        write(source,doc)
        dataset=dict(schema=DATASET_SCHEMA,role='TRAIN',partition=partition,contract=built['contract'],
            frozen_base_bank=built['bank'],cumulative_baseline=dict(name=str(spec.get('comparison_root',spec.get('comparison_partitions'))),root_cells=built['baseline']),
            sources=[dict(path=str(source),sha256=file_sha(source))])
        write(output/f'{partition}_dataset.json',dataset)
    write(output/'summary.json',dict(schema='jit_residual_export_summary_v2',environment_interactions=0,ppo_transitions=0,
        counts={p:len(r) for p,r in built['partitions'].items()},nonzero_counts={p:int(sum(np.any(np.abs(r['target_delta'])>1e-7) for r in rows)) for p,rows in built['partitions'].items()},
        split=built['split'],exclusions=built['stats'],channel_diagnostics=built['channel_diagnostics'],
        partition_sources=built['partition_sources'],condition_sets=built['condition_sets'],independent_repetitions=False,final_test_used=False))
    return read(output/'summary.json')


def validate_causal_export(doc,dataset):
    if doc.get('schema')!=SOURCE_SCHEMA or doc.get('role')!='TRAIN' or doc.get('partition')!=dataset.get('partition'):
        raise ValueError('causal export schema/partition mismatch')
    _verify_files(doc['inputs'])
    built=build_export(doc['export_spec']);partition=doc['partition']
    if (partition not in built['partitions'] or doc['records']!=built['partitions'][partition]
        or doc['split']!=built['split'] or doc['inputs']!=built['inputs'] or dataset['contract']!=built['contract']
        or dataset['frozen_base_bank']!=built['bank'] or dataset['cumulative_baseline']['root_cells']!=built['baseline']):
        raise ValueError('causal action origin/witness/partition or contract drift')
