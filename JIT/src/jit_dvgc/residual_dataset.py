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


def verify_observation_sources(source_files):
    """Only saved-Actor observation semantics must match; no physics replay gate."""
    for name in ('observation.py', 'constants.py'):
        path=Path(__file__).resolve().parent/name
        if source_files.get(str(path)) != file_sha(path):
            raise ValueError('historical observation source mismatch: '+name)


def _checked_comparison(root):
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
        arm=root/name;cp=arm/'acquire/result/catalog.json';catalog=read(cp);member=members[name]
        ep=read(arm/'existence_plan.json');verify_hash(ep,'plan_sha256')
        _verify_files([dict(path=p,sha256=s) for p,s in ep['input_files'].items()])
        if (ep['bank_sha256']!=bank['bank_sha256'] or Path(ep['catalog']).resolve()!=cp or ep['proposer']!=name
            or ep['role']!='train' or ep['final_test_used'] is not False or ep['conflict_policy']!=endpoint['conflict_policy']):
            raise ValueError('existence input binding mismatch')
        index=read(arm/'labels/result/witness_index.json');verify_hash(index,'index_sha256')
        validate_witness_receipt(index,ep,catalog,members,member,arm/'labels/result',ep['budget'])
        points=project_catalog(cp,member)
        total+=index['charged_interactions']+catalog['environment_interactions']
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
    return pairs,stats,baseline,endpoint,[dict(path=m['frozen_policy'],sha256=m['frozen_file_sha256']) for m in bank['members'] if m['name'] in spec['proposers']],inputs


def build_export(spec):
    pairs,stats,baseline,endpoint,bank,inputs=_checked_comparison(spec['comparison_root'])
    # Group before partition selection; same trajectory condition across policies
    # stays together. Shared policy ancestors still prohibit independence claims.
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
    from .residual_exploration import validate_contract
    validate_contract(contract)
    inputs.update(Path(__file__).parent/name for name in ('observation.py','constants.py'))
    return dict(contract=contract,partitions=partitions,split=split,stats=stats,baseline=sorted(baseline),endpoint=endpoint,bank=bank,
                inputs=[dict(path=str(p.resolve()),sha256=file_sha(p)) for p in sorted(inputs)])


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
            frozen_base_bank=built['bank'],cumulative_baseline=dict(name=str(spec['comparison_root']),root_cells=built['baseline']),
            sources=[dict(path=str(source),sha256=file_sha(source))])
        write(output/f'{partition}_dataset.json',dataset)
    write(output/'summary.json',dict(schema='jit_residual_export_summary_v2',environment_interactions=0,ppo_transitions=0,
        counts={p:len(r) for p,r in built['partitions'].items()},nonzero_counts={p:int(sum(np.any(np.abs(r['target_delta'])>1e-7) for r in rows)) for p,rows in built['partitions'].items()},
        split=built['split'],exclusions=built['stats'],independent_repetitions=False,final_test_used=False))
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
