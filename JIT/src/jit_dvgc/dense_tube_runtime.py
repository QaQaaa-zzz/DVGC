"""Production workers for the versioned, bounded dense Tube pilot."""
from __future__ import annotations
from pathlib import Path
import time
import traceback
from .jump_evidence_validation import read,write,file_sha,verify_hash
from .evidence_integrity import canonical_sha256
from .policy_comparison import verify_plan


def prepare(output):
    from .unified_policy_freeze import load_frozen_unified_manifest
    from .analysis.nominal_jump_centerline import load_nominal_jump_centerline
    from .analysis.capability_tube import resolution_contract
    from .probe_bank import lock_probe_bank, load_probe_bank
    import matplotlib
    request=read(output/'request.json');repo=Path(request['repo']);baseline=Path(request['baseline'])
    if (output/'plan.json').exists():
        p=verify_plan(output/'plan.json')
        if p['request']!=request:raise ValueError('dense request/plan drift')
        return {'status':'completed','environment_interactions':0}
    profile=request.get('frontier_profile')
    from .frontier_exploration import validate_profile
    validate_profile(profile)
    old=read(baseline/'plan.json');verify_hash(old,'plan_sha256')
    summary=read(baseline/'summary.json')
    if summary['status']!='completed' or summary['plan_sha256']!=old['plan_sha256']:
        raise ValueError('completed baseline comparison required')
    inputs={}
    def lock(path):
        path=Path(path).resolve();inputs[str(path)]=file_sha(path);return path
    for source, sha in (profile or {}).get('source_files', {}).items():
        if file_sha(source) != sha: raise ValueError('boundary source evidence changed')
        lock(source)
    lock(baseline/'plan.json');lock(baseline/'summary.json')
    members=old['members']
    if [m['policy']['name'] for m in members]!=['pi_0','pi_1','pi_2','pi_3']:raise ValueError('baseline bank mismatch')
    for m in members:
        if file_sha(m['path'])!=m['file_sha256'] or load_frozen_unified_manifest(Path(m['path']))['policy']!=m['policy']:
            raise ValueError('baseline frozen policy drift')
        lock(m['path']);lock(m['policy']['formal_config'])
    center=load_nominal_jump_centerline(lock(old['centerline']))
    original=Path(old['panels']['train']['catalog'])
    if file_sha(original)!=old['panels']['train']['catalog_sha256']:raise ValueError('baseline catalog changed')
    catalog=read(lock(original));protocol=read(lock(original.parent/'protocol.json'));verify_hash(protocol,'protocol_sha256')
    if catalog['protocol_sha256']!=protocol['protocol_sha256']:raise ValueError('baseline acquisition protocol drift')
    for role in ('train','calibration','acceptance'):
        lock(old['panels'][role]['projected'])
        path=lock(baseline/'figures'/role/'summary.json');verify_hash(read(path),'report_sha256')
    import numpy as np
    indices=[]
    for phase in ('upstream','downstream'):
        choices=[i for i,r in enumerate(catalog['entries']) if r['phase']==phase]
        if len(choices)<8:raise ValueError('benchmark needs eight existing states per phase')
        indices += [choices[i] for i in np.linspace(0,len(choices)-1,8,dtype=int)]
    rows=[]
    for i in indices:
        row=dict(catalog['entries'][i]);row['source_bank']=str((original.parent/row['source_bank']).resolve());rows.append(row)
    bench=output/'benchmark/catalog.json'
    write(bench,{**catalog,'entries':rows,'candidate_count':len(rows),'logical_role':'engineering_execution_comparison','training_admission_authorized':False})
    write(bench.parent/'protocol.json',protocol);lock(bench);lock(bench.parent/'protocol.json')
    start=output/'start_contract.json'
    write(start,{'jump_start_state_sha256':center['jump_start_state_sha256'],'xml_sha256':members[0]['policy']['xml_sha256'],
                 'continuation_start_semantics':'fresh_continuation_v1','initial_clearance_accepted':True,'extra_replay_validation':False})
    bank_spec={'version':profile['version'] if profile else 'dense_5cm_discovery_v2','task':{'xml_sha256':members[0]['policy']['xml_sha256'],
        'start_contract_sha256':file_sha(start),'centerline_sha256':center['centerline_sha256'],
        'resolution_sha256':resolution_contract()['resolution_sha256'],'success_criterion':'first_valid_landing',
        'continuation_start_semantics':'fresh_continuation_v1'},'max_ticks':old['horizon'],
        'label_interaction_budget':request['budget'],'max_candidates_per_process':128,
        'members':[{'frozen_policy':m['path'],'roles':['proposer','evaluator']} for i,m in enumerate(members)]}
    if (output/'bank.json').exists():
        bank=load_probe_bank(output/'bank.json')
        if any(bank[k]!=bank_spec[k] for k in ('version','task','max_ticks','label_interaction_budget','max_candidates_per_process')) or [m['frozen_policy'] for m in bank['members']] != [m['frozen_policy'] for m in bank_spec['members']]:
            raise ValueError('incomplete preflight bank does not match request')
    else:
        bank=lock_probe_bank(bank_spec,output/'bank.json')
    lock(start);lock(output/'bank.json')
    anchors=[]
    # New action windows differ from the historical scan; all four channels/signs.
    for target in (profile['targets'] if profile else (2.9,3.1)):
        for family in range(5):
            anchors.append({'phase':'upstream','x_target_m':target,'proposal_family_index':family,
                'state_sha256':center['jump_start_state_sha256'],
                'parent_group_id':f'dense_v1_train_x{target:.2f}_family{family}'})
    write(output/'anchors.json',anchors);lock(output/'anchors.json')
    spec={'bank':str(output/'bank.json'),'proposer':request.get('proposer','pi_0'),'role':'train','start_contract':str(start),
          'nominal_centerline':old['centerline'],'anchors':str(output/'anchors.json'),'seed':9841101,
          'strengths':[0.075],'action_names':['steer','rear_wheel_drive','hip','knee'],'signs':[-1,1],
          'lookbacks_m':[0.15],'max_forward_ticks':400,'interaction_ceiling':8000,
          'sampling_mode':'trajectory_slices_v2','slice_spacing_m':0.05,'max_candidates_per_attempt':32}
    if profile:
        spec.update(seed=profile['acquisition_seed'],strengths=profile['strengths'],
                    max_candidates_per_attempt=profile['max_candidates'],sampling_max_x_m=profile['sampling_max_x_m'],
                    interaction_ceiling=profile['acquisition_ceiling'])
    if profile and profile['version'] == 'knee_boundary_v1':
        if request.get('proposer') != 'pi_1': raise ValueError('boundary proposer drift')
        spec.update(action_names=profile['action_names'], signs=profile['signs'])
    write(output/'acquisition_spec.json',spec);lock(output/'acquisition_spec.json')
    if old['horizon']!=400:raise ValueError('pilot budget is declared for horizon 400')
    p={'schema':'jit_dense_tube_pilot_v2','proposer':request.get('proposer','pi_0'),'repo':str(repo),'request':request,'members':members,'names':[m['policy']['name'] for m in members],
       'bank_sha256':bank['bank_sha256'],'horizon':400,'centerline':old['centerline'],'baseline':str(baseline),'benchmark_catalog':str(bench),
       'baseline_plan_sha256':old['plan_sha256'],'input_files':inputs,
       'sources':{str(f.relative_to(repo)):file_sha(f) for f in (repo/'JIT').rglob('*.py') if 'runs' not in f.parts},
       'sampling_spacing_m':0.05,'physical_resolution':resolution_contract(),
       'max_trajectories':16,'max_candidates_per_trajectory':32,'first_attempt_interaction_ceiling':878400,
       'training_admission_authorized':False,'role':'train','new_replay_validation':False,
       'snapshot_replay_equivalence_verified':False,'matched_budget_discovery_claim':False}
    if profile:
        p.update(frontier_profile=profile,label_seed=profile['label_seed'],
                 acquisition_ceiling=profile['acquisition_ceiling'],
                 max_trajectories=profile['max_trajectories'],max_candidates_per_trajectory=profile['max_candidates'],
                 first_attempt_interaction_ceiling=profile['acquisition_ceiling']+profile['max_trajectories']*profile['max_candidates']*4*400)
    if p['first_attempt_interaction_ceiling']>request['budget']:raise ValueError('budget below declared first-attempt ceiling')
    p['plan_sha256']=canonical_sha256(p);write(output/'plan.json',p)
    return {'status':'completed','environment_interactions':0,'plan_sha256':p['plan_sha256']}


def project(plan,catalog_path,destination):
    from .probe_bank import validate_probe_arrivals
    from .unified_continuation_labels import validate_unified_boundary_catalog
    from .unified_envelope_snapshot import load_unified_envelope_snapshot,snapshot_context_sha256
    from .analysis.capability_tube import physical_coordinates_from_arrays
    from .analysis.policy_envelopes import project_row
    from .model import load_host_model
    from .config import load_config
    catalog=read(catalog_path);protocol=read(catalog_path.parent/'protocol.json');verify_hash(protocol,'protocol_sha256')
    m=next(m for m in plan['members'] if m['policy']['name']==plan.get('proposer','pi_0'))
    if catalog['status']!='completed' or catalog['protocol_sha256']!=protocol['protocol_sha256'] or protocol.get('sampling_mode')!='trajectory_slices_v2':
        raise ValueError('dense acquisition incomplete or protocol drift')
    if protocol.get("probe_bank_sha256") != plan["bank_sha256"] or protocol.get("logical_role") != "train":
        raise ValueError("dense bank/role drift")
    rows=validate_unified_boundary_catalog(catalog,policy_record=m['policy'],frozen_manifest_sha256=m['file_sha256'],allow_empty=True)
    validate_probe_arrivals(catalog_path,rows,m['policy'])
    config=read(m['policy']['formal_config']);bundle=load_host_model(load_config(Path(config['inputs']['up_config_path'])))
    if bundle.xml_sha256!=m['policy']['xml_sha256']:raise ValueError('projection XML mismatch')
    points=[]
    for row in rows:
        snap=load_unified_envelope_snapshot(catalog_path.parent/row['source_bank']/row['snapshot'])
        coordinates=physical_coordinates_from_arrays(snap.qpos,snap.qvel,bundle=bundle)
        point=project_row(row,coordinates,snapshot_context_sha256(snap),x_slice_width_m=0.05)
        point.update(trajectory_id=row['trajectory_id'],trajectory_step=row['trajectory_step']);points.append(point)
    write(destination/'projected.json',points)
    old=read(Path(plan['baseline'])/'plan.json')
    overlap={r:len({p['state_sha256'] for p in points}&{p['state_sha256'] for p in read(old['panels'][r]['projected'])}) for r in ('calibration','acceptance')}
    return {'status':'completed','environment_interactions':0,'candidate_count':len(points),
            'trajectory_count':len({p['trajectory_id'] for p in points}),'exact_physical_overlap_with_development_roles':overlap,
            'training_admission_authorized':False,'correlated_frames_not_independent_trials':True}


def analyze(plan,output,destination):
    from .analysis.policy_envelopes import summarize,render_comparison,write_csv
    from .policy_comparison_runtime import complete_output
    from .analysis.dense_coverage import compare_coverage
    manifest=read(output/'analysis_inputs.json');points=read(manifest['projected']);labels={}
    for member in plan['members']:
        name=member['policy']['name'];result=complete_output(manifest['merged'][name],manifest['catalog'],next(m for m in plan['members'] if m['policy']['name']==plan.get('proposer','pi_0')),member,400,plan.get('label_seed',9841201))
        if result is None:raise ValueError('missing dense labels')
        labels[name]=result[1]
    summary=summarize(points,labels,role='train',x_slice_width_m=0.05)
    summary.update(plan_sha256=plan['plan_sha256'],scope=f"{plan.get('proposer','pi_0')} bounded TRAIN pilot; not full physical envelope",
                   training_admission_authorized=False,trajectory_count=len({p['trajectory_id'] for p in points}))
    figures=output/'figures';render_comparison(points,summary,figures,centerline=read(plan['centerline'])['points'])
    baseline=Path(plan['baseline']);old=read(baseline/'plan.json')
    oldpoints=read(old['panels']['train']['projected']);oldsummary=read(baseline/'figures/train/summary.json')
    delta=compare_coverage(oldpoints,oldsummary,points,summary,figures)
    return {'status':'completed','environment_interactions':0,'coverage_delta':delta,'figures':str(figures)}


def worker(args):
    output=args.output_dir.resolve();destination=args.destination.resolve();start=time.perf_counter()
    try:
        if args.worker=='prepare':result=prepare(output)
        else:
            plan=verify_plan(output/'plan.json')
            if args.worker in {'benchmark','acquire','label'}:
                import jax
                if jax.default_backend()!='gpu' or len(jax.local_devices())!=1:raise RuntimeError('exactly one visible GPU required')
            if args.worker=='acquire':
                from .probe_bank import acquire_probe_catalog
                result=acquire_probe_catalog(output/'acquisition_spec.json',destination/'result')
                result={k:v for k,v in result.items() if k!='entries'}
            elif args.worker=='project':
                result=project(plan,args.catalog,destination)
            elif args.worker in {'benchmark','label'}:
                from .policy_family_landing import run_policy_family_evaluator_shard
                member=next(m for m in plan['members'] if m['policy']['name']==args.policy)
                result=run_policy_family_evaluator_shard(catalog_path=Path(plan['benchmark_catalog']) if args.worker=='benchmark' else args.catalog,
                    acquisition_frozen_policy=Path((plan['members'][0] if args.worker=='benchmark' else next(m for m in plan['members'] if m['policy']['name']==plan.get('proposer','pi_0')))['path']),evaluator_frozen_policy=Path(member['path']),
                    output_dir=destination/'result',shard_index=args.shard_index,shard_count=args.shard_count,max_ticks=400,
                    protocol_seed=9840201 if args.worker=='benchmark' else plan.get('label_seed',9841201),
                    execution_backend=args.backend,batch_size=8 if args.backend=='device' else 1)
                rows=read(destination/'result/labels.json')
                if result['environment_interactions']!=sum(r['environment_interactions'] for r in rows)+result.get('inactive_lane_interactions',0):
                    raise ValueError('device/serial cost telemetry mismatch')
                if args.worker=='benchmark':
                    result={**result,'labels':rows,'seconds':result['elapsed_seconds'],
                            'identities':{'policy':member['policy'],'catalog_sha256':file_sha(plan['benchmark_catalog']),'seed':9840201},
                            'scope':'execution comparison only; not snapshot replay or training data'}
            elif args.worker=='merge':
                from .policy_family_landing import merge_policy_family_evaluator_shards
                member=next(m for m in plan['members'] if m['policy']['name']==args.policy)
                merged=merge_policy_family_evaluator_shards(catalog_path=args.catalog,
                    shard_dirs=[Path(p) for p in read(output/f'{args.policy}_shards.json')],output_dir=destination/'result',
                    evaluator_name=args.policy,acquisition_frozen_policy=Path(next(m for m in plan['members'] if m['policy']['name']==plan.get('proposer','pi_0'))['path']),
                    evaluator_frozen_policy=Path(member['path']),max_ticks=400,protocol_seed=plan.get('label_seed',9841201))
                result={'status':'completed','environment_interactions':0,'merged_useful_label_interactions':merged['environment_interactions']}
            elif args.worker=='analyze':result=analyze(plan,output,destination)
            else:raise ValueError('unknown dense worker')
        result.update(worker_wall_seconds=time.perf_counter()-start)
        write(destination/'worker_report.json',result);return 0
    except BaseException as exc:
        print(traceback.format_exc(),flush=True)
        write(destination/'worker_failure.json',{'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()});return 2
