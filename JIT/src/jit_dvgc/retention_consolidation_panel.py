"""Read-only historical condition adapter; export a new locked pilot panel.

No reset, step, optimizer, teacher search or inference is performed here.
"""
from pathlib import Path
import copy
import json
import shutil
import numpy as np
from .generative_bridge.contracts import file_sha,digest
from .generative_bridge.protocol import atomic_json

BASE=Path('/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010')


def snapshot_time_provenance(path,lane,qpos,qvel,time):
    with np.load(path,allow_pickle=False) as tape:
        mask=np.asarray(tape['prefix_mask'][:,lane],bool)
        matches=[i for i in np.flatnonzero(mask) if float(tape['time_after'][i,lane])==time
                 and np.array_equal(tape['qpos'][i,lane],qpos) and np.array_equal(tape['qvel'][i,lane],qvel)]
    if len(matches)!=1:raise ValueError('snapshot time requires one exact active source physical frame')
    return dict(path=str(path),sha256=file_sha(path),lane=lane,tick=int(matches[0]),time_field='time_after')


def _read(path):return json.loads(Path(path).read_text())


def prepare_panel(output,*,protocol_sha256,teacher_evidence,source=BASE):
    from .retention_consolidation_data import collection_condition_identity
    from .unified_envelope_snapshot import load_unified_envelope_snapshot
    root=Path(output);root.mkdir(parents=True,exist_ok=True);source=Path(source)
    source_spec=_read(source/'B_002/DEV_pi0_spec.json');raw_cases=_read(source/'D2_002/report_004/new_dev_cases.json')
    for key,dest in [('initial_state_bank','DEV_initial.npz'),('frozen_request_table','DEV_requests.npz')]:
        src=Path(source_spec[key] if key=='initial_state_bank' else source_spec[key]['path'])
        if key=='frozen_request_table' and file_sha(src)!=source_spec[key]['sha256']:raise ValueError('historical fixed requests changed')
        out=root/dest
        if out.exists():
            if file_sha(out)!=file_sha(src):raise ValueError('existing pilot input differs')
        else:shutil.copyfile(src,out)
    spec=copy.deepcopy(source_spec);spec.update(initial_state_bank=str(root/'DEV_initial.npz'),budget=64*400,role='DEV',seed=1010258103)
    spec['frozen_request_table']=dict(path=str(root/'DEV_requests.npz'),sha256=file_sha(root/'DEV_requests.npz'))
    with np.load(root/'DEV_initial.npz',allow_pickle=False) as f:qpos=f['qpos'];qvel=f['qvel']
    with np.load(root/'DEV_requests.npz',allow_pickle=False) as f:requests=f['requested']
    cases=[];splits=[]
    for lane,raw in enumerate(raw_cases):
        # Nominal16 are one physical condition, with known batched numeric variation.
        if raw['group']=='A' and lane!=3:continue
        group={'A':'old_nominal','B':'old_random','C':'old_recovery','D':'accepted_parent_recovery'}[raw['group']]
        condition=collection_condition_identity(qpos[lane],qvel[lane],requests[:,lane],protocol_sha256=protocol_sha256)
        role='TRAIN_RETENTION' if raw['group']=='A' else 'DEV_RETENTION'
        case=dict(case_id=raw['case'],condition_identity=condition,group=group,role=role,mode='collect',
            spec=spec,capacity=64,horizon=400,lane=lane,repeat_seeds=[1010258103]*3,
            root_qpos_address=0,evaluation_batch_id='locked_DEV64',source_case=raw,
            source_descriptor=dict(path=str(source/'D2_002/report_004/new_dev_cases.json'),sha256=file_sha(source/'D2_002/report_004/new_dev_cases.json')))
        if role=='TRAIN_RETENTION':
            case['seen_condition_validation']=dict(schema='frontierbridge_seen_condition_validation_v1',
                training_condition_revisit=True,group='old_nominal',independent_dev=False,source_train_condition_identity=condition)
        else:
            splits.append(dict(root_id=raw['case'],ancestor_ids=[raw['ancestor']],condition_identity=condition,
                role=role,capability_group=group,phase='UNKNOWN'))
        cases.append(case)
    roots=_read(source/'D1_003/selected_roots.json');teachers={r['root_id']:r for r in _read(source/'D1_003/teacher_results.json')}
    for ordinal,raw in enumerate(r for r in roots if r['role']=='SOLVER_DEV'):
        row=copy.deepcopy(raw);snapshot=load_unified_envelope_snapshot(Path(row['snapshot']))
        xyz=np.asarray([row['coordinates']['root_'+axis+'_m'] for axis in ('x','y','z')],dtype=np.asarray(snapshot.qpos).dtype)
        if not np.array_equal(xyz,np.asarray(snapshot.qpos)[:3]):raise ValueError('native root coordinate address does not match source physical schema')
        row['time_provenance']=snapshot_time_provenance(Path(row['prefix_file']),row['collection_lane'],
            np.asarray(snapshot.qpos),np.asarray(snapshot.qvel),row['snapshot_time'])
        condition=digest(dict(snapshot_context=row['snapshot_context_sha256'],snapshot_time=row['snapshot_time']))
        g=teachers[row['root_id']]['methods']['G'];verified=g.get('repeat_verified') is True
        role='DEV_EXPANSION' if verified else 'DEV_RETENTION'
        row.update(index=0,candidate_id=0,role=role)
        path=root/f'solver_{ordinal:02d}_candidate.json';atomic_json(path,[row])
        solver_spec=dict(bank=source_spec['bank'],proposer=source_spec['proposer'],order=[source_spec['proposer']],
            candidates=str(path),horizon=400,budget=400,full_matrix=True,record_actor_preobservations=True,
            record_retention_diagnostics=True,reward_mode='original_all_phases',success_criterion='stable_forward_recovery',
            preserve_snapshot_episode_context=True,suffix_rng_count=17,suffix_rng_indices=[0])
        case=dict(case_id=row['root_id'],condition_identity=condition,group='teacher_verified_expansion' if verified else 'solver_unverified',
            role=role,mode='evaluate',spec=solver_spec,capacity=1,horizon=400,lane=0,repeat_seeds=[0]*3,
            root_qpos_address=0,parent_repeat_labels=raw['pi0_labels'],teacher_repeat_labels=g.get('winner_labels',[None]*3),
            teacher_status=g['status'],source_descriptor=dict(path=str(source/'D1_003/teacher_results.json'),sha256=file_sha(source/'D1_003/teacher_results.json')))
        if verified:
            if row['root_id'] not in teacher_evidence:raise ValueError('missing real DEV teacher evidence: '+row['root_id'])
            case['teacher_evidence']=teacher_evidence[row['root_id']]
        cases.append(case)
        splits.append(dict(root_id=raw['root_id'],ancestor_ids=[raw['root_episode_id']],condition_identity=condition,
            role=role,capability_group=case['group'],phase=raw['phase']))
    panel=dict(schema='frontierbridge_evaluation_panel_v1',cases=cases,split_records=splits,
        declared_counts=dict(nominal_seen_condition=1,DEV_random=16,DEV_nominal_pulse=16,DEV_random_pulse=16,
            solver_roots=7,teacher_verified_expansion=3),
        capability_notes={'accepted_parent_recovery':'Parent equals frozen pi0: D group protects inherited pi0 random+pulse recovery; no newer accepted parent skill is claimed.',
            'old_nominal':'One identical nominal condition revisits TRAIN; not an independent DEV condition. Original A16 physical lanes remain in batch; A03 counted once.',
            'solver_unverified':'Four solver roots retain unknown/unverified teacher labels and remain diagnostic; not declared teacher-solved expansion.'},
        cost=dict(unique_batches_per_repeat=8,capacity_sum=71,numerical_repeats=3,
            transitions_reserved_per_policy=71*3*400,new_environment_steps=0,teacher_search_steps=0))
    path=root/'evaluation_panel.json';atomic_json(path,panel)
    return dict(path=str(path),sha256=file_sha(path))
