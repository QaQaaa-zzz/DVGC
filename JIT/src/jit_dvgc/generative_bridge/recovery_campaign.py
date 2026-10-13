"""Explicit fresh teacher-recovery campaign, separate from legacy continuations."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from .contracts import file_sha, digest
from .protocol import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def campaign_parameters():
    from .student import recovery_training_contract
    return dict(rounds=100, physics_cap_per_round=3500000,
        teacher_query_count=128, teacher_generated_candidates=16, teacher_reviews=2,
        student=recovery_training_contract(),
        explorer=dict(num_envs=128, learning_rate=1e-4, max_learning_rate=1e-4,
            entropy_coefficient=.001, value_coefficient=.5, minibatch_size=256,
            epochs=4, clip=.2, target_kl=.01, max_epoch_kl=.03, max_grad_norm=1.))


def snapshot_entry(row, round_index):
    path=Path(row['snapshot'])/'identity.json'
    identity=read(path)
    step=identity['episode_step']
    if type(step) is not int or step<0:
        raise ValueError('original integer snapshot clock required')
    return dict(snapshot=str(Path(row['snapshot']).resolve()),
        snapshot_context_sha256=row['snapshot_context_sha256'],
        role='TRAIN', verified_success=True, round=round_index,
        snapshot_time=step*.02, snapshot_time_source=dict(path=str(path.resolve()),
            sha256=file_sha(path), control_step_field='episode_step', control_dt=.02))


def verified_teacher_roots(rows, teachers):
    return [row for row in rows if
        teachers.get(row['root_id'],{}).get('teacher_status')=='verified_solution'
        and teachers[row['root_id']].get('verification',{}).get('repeat_labels')==[1,1]
        and 'demo' in teachers[row['root_id']]]


def make_reset_inventory(runner, rows, teachers, seed_support):
    from ..retention_reset import validate_inventory
    current=verified_teacher_roots(rows,teachers)
    historical=read(seed_support)['entries']
    initial=runner.spec['recovery_complete_start']
    inventory=dict(schema='jit_teacher_recovery_reset_v1', role='TRAIN',
        round=runner.spec['round_index'],
        complete_start=dict(probability=.4, **initial),
        current_teacher_verified=dict(probability=.3,
            entries=[snapshot_entry(r,runner.spec['round_index']) for r in current]),
        historical_success=dict(probability=.3,
            entries=[snapshot_entry(r,0) for r in historical if r.get('witnessed')]))
    validate_inventory(inventory)
    path=runner.root/'student_reset_inventory.json'
    atomic_json(path,inventory)
    return str(path)


def complete_start_arrays(repository, runtime_config):
    import numpy as np
    import mujoco
    up=read(read(runtime_config)['inputs']['up_config_path'])
    xml=Path(repository)/up['model']['xml_path']
    if file_sha(xml)!=up['model']['xml_sha256']:
        raise ValueError('complete-start physical identity changed')
    model=mujoco.MjModel.from_xml_path(str(xml))
    key=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,up['reset']['keyframe'])
    free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
    if key<0 or len(free)!=1:
        raise ValueError('unique original root and keyframe required')
    qpos=model.key_qpos[key].copy();qvel=model.key_qvel[key].copy()
    qpos[model.jnt_qposadr[free[0]]]=2.5
    dof=model.jnt_dofadr[free[0]]
    qvel[dof]=up['reset']['initial_forward_velocity'];qvel[dof+2]=0.
    return dict(qpos=qpos.tolist(),qvel=qvel.tolist())


def reconcile_teacher_costs(parent_costs,worker_costs,completed_stages):
    """Replace a whole-worker reservation only with certified lower-level charges.

    Uncommitted stages keep their entire original reservation. This returns a
    new ledger; the failed attempt and its whole-worker reservation stay intact.
    """
    rows=deepcopy(parent_costs);matches=[i for i,c in enumerate(rows) if c['stage']=='teacher_worker']
    if len(matches)!=1 or not worker_costs:raise ValueError('one failed teacher worker ledger required')
    position=matches[0];original=rows[position]
    if original.get('phase')!='failed' or any(c.get('charged_updates',0) for c in parent_costs+worker_costs):
        raise ValueError('engineering recovery requires zero optimization and failed teacher worker')
    seen=set();completed=0;partial=0
    for i,c in enumerate(worker_costs):
        name=c['stage'];charge=c['charged_interactions']
        if name in seen or type(charge) is not int or charge<0:raise ValueError('invalid teacher cost ledger')
        seen.add(name)
        if c.get('phase')=='completed':
            proof=completed_stages.get(name)
            if proof is None or proof.get('charged_interactions')!=charge:
                raise ValueError('completed teacher cost lacks matching certified receipt')
            completed+=charge
        else:
            if i!=len(worker_costs)-1 or c.get('phase') not in ('running','failed'):
                raise ValueError('only the final uncommitted teacher stage may retain an upper bound')
            partial+=charge
    if set(completed_stages)!={c['stage'] for c in worker_costs if c.get('phase')=='completed'}:
        raise ValueError('completed teacher receipt set differs from ledger')
    total=completed+partial
    if total>original['charged_interactions']:raise ValueError('reconciled teacher exceeds original reservation')
    rows[position]={**original,'stage':'teacher_worker_prior_attempt','charged_interactions':total,
        'original_worker_reservation':original['charged_interactions'],
        'completed_interactions':completed,'partial_upper_interactions':partial,
        'accounting':'certified completed stage actuals plus full uncommitted stage reservation; original ledger immutable'}
    return rows


def reuse_completed_collection(reference,actual_config,new_round):
    """Require identical scientific/RNG configuration and immutable saved output."""
    for path,sha in reference['files'].items():
        if file_sha(path)!=sha:raise ValueError('saved collection changed: '+path)
    output=Path(reference['output']);status=read(output/'status.json')
    if status.get('phase')!='completed' or status.get('charged_interactions')!=reference['charged_interactions']:
        raise ValueError('saved collection completion/charge changed')
    old=Path(reference['previous_round']);new=Path(new_round)
    def normalize(value):
        if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
        if isinstance(value,list):return [normalize(v) for v in value]
        if isinstance(value,str) and value.startswith(str(new)+'/'):
            return str(old)+value[len(str(new)):]
        return value
    if normalize(actual_config)!=read(reference.get('spec_path',output/'hyperparameters.json')):
        raise ValueError('saved collection protocol/RNG differs from explicit recovery')
    return output


def dry_check_recovery_collection(original_spec,reference,directory):
    """Exercise the actual closed-loop configuration path, stopping before physics."""
    from .closed_loop import ClosedLoopRound
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    runner=ClosedLoopRound.__new__(ClosedLoopRound)
    runner.spec=deepcopy(original_spec);runner.spec['output']=str(directory)
    runner.root=directory;runner.runtime=runner.spec['source_runtime'];runner.source=runner.spec['source']
    class ConfigurationChecked(Exception):pass
    def stop(name,mode,config,maximum,**kwargs):
        if name!='collection' or mode!='collect':raise ValueError('unexpected stage before collection recovery check')
        reuse_completed_collection(reference,config,directory)
        atomic_json(directory/'collection_configuration_check.json',dict(phase='passed',
            original_spec_sha256=digest(original_spec),actual_configuration=config,new_physics_interactions=0))
        raise ConfigurationChecked()
    runner.measured=stop
    try:ClosedLoopRound._run(runner)
    except ConfigurationChecked:
        return read(directory/'collection_configuration_check.json')
    raise ValueError('collection configuration check did not stop before execution')


def _resume_stage_refs(stages):
    return {'eval_'+name:dict(path=row['journal_path'],sha256=row['journal_sha256'])
            for name,row in stages.items()}


def resume_round_spec(declaration):
    """Prepare the one explicitly recovered round; preserve every scientific field."""
    from .production import implementation_files
    d=declaration;info=d['engineering_recovery'];old=Path(info['previous_round'])
    if file_sha(old/'production.json')!=info['previous_spec_sha256']:
        raise ValueError('failed round declaration changed')
    spec=deepcopy(read(old/'production.json'));root=Path(d['campaign_output'])/'round_0001'
    root.mkdir(exist_ok=False)
    spec.update(output=str(root),repository=d['repository'],implementation_commit=d['implementation_commit'],
        implementation_files=implementation_files(d['repository']),campaign_output=d['campaign_output'],
        teacher_recovery_authorization=dict(path=str(Path(d['campaign_output'])/'declaration.json'),
            sha256=file_sha(Path(d['campaign_output'])/'declaration.json')),
        locks={**spec['locks'],**d['locks']},
        recovery=dict(previous=str(old),previous_contract_identity=info['previous_contract_identity'],stages=info['stages']),
        recovery_collection=info['collection'],
        recovery_started_unix=info.get('original_round_started_unix'))
    if info.get('completed_student') is not None:
        spec['recovery_completed_student']=info['completed_student']
        spec.pop('recovery_teacher_worker',None);spec.pop('recovery_completed_teacher_roots',None)
    else:
        spec.update(recovery_teacher_worker=info['teacher_worker'],
            recovery_completed_teacher_roots=info['completed_teacher_roots'])
    atomic_json(root/'production.json',spec);atomic_json(root/'panels.json',{})
    atomic_json(root/'costs.json',info['carried_round_costs'])
    atomic_json(root/'cost_reconciliation.json',dict(previous_round=str(old),
        previous_costs_sha256=info.get('previous_costs_sha256'),carried_costs=info['carried_round_costs'],
        audit=info.get('teacher_audit'),additional_physics_for_reuse=0,
        automatic_retry=False,partial_stage='UNKNOWN; retained full original stage reservation'))
    return spec


def _verify_resume_audit(path):
    from .verified_files import LOCK_CACHE
    audit=read(path)
    if (audit.get('schema')!='jit_teacher_worker_partial_audit_v1' or audit.get('phase')!='certified_partial'
        or audit.get('student_E_G_updates_started') is not False or audit.get('optimizer_updates')!=0):
        raise ValueError('certified pre-optimization teacher partial audit required')
    for filename,sha in audit['input_locks'].items():LOCK_CACHE.verify(filename,sha)
    return audit


def verify_evaluation_boundary(previous):
    """Certify the failed first round's completed teachers and32k student."""
    import pickle,jax,numpy as np
    from .worker import source_payload
    from .learner_continuation import validate_saved_learner
    from .actual_update import adam_update_count
    from ..handoff_bank import pytree_sha256
    root=Path(previous).resolve();rd=root/'round_0001';status=read(rd/'status.json')
    if status.get('phase')!='failed' or 'student_train_batch_0000' not in status.get('error',''):
        raise ValueError('diagnosed first post-learning student evaluation failure required')
    if any((rd/name).exists() for name in ('explorer_update','generator_incremental','current_source.json')):
        raise ValueError('post-learning recovery cannot discard later E/G or completed round work')
    contract=read(rd/'stages/round_contract.json');spec=read(rd/'production.json')
    if contract!={'sha256':digest(spec),'contract':spec}:raise ValueError('failed round contract changed')
    inherited=spec.get('recovery_completed_student')
    if inherited is not None:
        # Each explicitly requested evaluation-only successor keeps the same
        # original learning/teacher proof and adds its failed reservation.
        for path,sha in inherited['locks'].items():
            if file_sha(path)!=sha:raise ValueError('inherited completed boundary changed: '+path)
        original=verify_evaluation_boundary(Path(inherited['phase']['path']).parent.parent)
        for key in ('schema','phase','student','student_stage','inventory','checkpoint_probes','teachers',
            'teacher_roots','completed_training_transitions','optimizer_updates','learner_state_sha256',
            'optimizer_sha256','rng_sha256','extra_actor_updates','simulator_frame_continuity'):
            if inherited[key]!=original[key]:raise ValueError('inherited completed student identity changed: '+key)
        stage_path=rd/'stages/train_student.json';stage=read(stage_path);phase=read(inherited['phase']['path'])
        if (stage['input_sha256']!=digest({'round':contract['sha256'],'inputs':phase})
            or stage['output_sha256']!=digest(inherited['student']) or stage['result']!=inherited['student']):
            raise ValueError('inherited student journal changed')
        carried=read(root/'declaration.json')['engineering_recovery']['carried_round_costs']
        costs=read(rd/'costs.json');extra=costs[len(carried):]
        if (costs[:len(carried)]!=carried or not extra or any(c.get('stage')!='student_train_batch_0000'
            or c.get('phase')!='failed' or c.get('charged_updates',0)!=0
            or c.get('charged_interactions')!=12800 for c in extra)):
            raise ValueError('chained recovery permits only failed student evaluation reservations')
        charge=sum(c['charged_interactions'] for c in costs)
        if charge!=status['charged_interactions'] or sum(c['charged_interactions'] for c in carried)!=inherited['charged_interactions']:
            raise ValueError('chained failed round cost ledger changed')
        proof=deepcopy(inherited);proof.update(charged_interactions=charge,
            failed_evaluation_upper=inherited['failed_evaluation_upper']+sum(c['charged_interactions'] for c in extra))
        for name in ('production.json','status.json','costs.json','stages/round_contract.json','stages/train_student.json'):
            path=str(rd/name);proof['locks'][path]=file_sha(path)
        return proof
    phase_path=rd/'source_phase_result.json';phase=read(phase_path);stage_path=rd/'stages/train_student.json';stage=read(stage_path)
    if (stage['input_sha256']!=digest({'round':contract['sha256'],'inputs':phase})
        or stage['output_sha256']!=digest(stage['result'])):raise ValueError('completed student stage changed')
    student=stage['result'];training=Path(student['training']);report=read(training/'formal_report.json')
    if (read(training/'status.json').get('status')!='completed' or report.get('status')!='completed'
        or report.get('completed_training_transitions')!=32000):raise ValueError('completed32k student required')
    policy,_=source_payload(student['frozen_policy'])
    if policy!=student['policy']:raise ValueError('completed student policy changed')
    m=validate_saved_learner(student['learner'],policy,expected_local_transitions=32000)
    with Path(m['state_path']).open('rb') as stream:saved=pickle.load(stream)
    state=saved['training_state'];rng=saved['rng']
    for leaf in jax.tree.leaves((state,rng)):
        value=np.asarray(leaf)
        if value.dtype.kind in 'fciub' and not np.isfinite(value).all():raise ValueError('nonfinite completed student')
    for field,value in (('actor_sha256',state.params.policy),('critic_sha256',state.params.value),
        ('normalizer_sha256',state.normalizer_params),('optimizer_sha256',state.optimizer_state),('rng_sha256',rng)):
        if pytree_sha256(value)!=m[field]:raise ValueError('completed student '+field+' changed')
    updates=int(adam_update_count(state.optimizer_state))
    if int(state.env_steps)!=32000 or m['lifetime_transitions']!=32000 or updates<1:
        raise ValueError('completed student learner/optimizer clock mismatch')
    worker=rd/'teacher_worker';receipt=read(worker/'completed.json');teachers=read(worker/'results.json')
    panels=read(rd/'panels.json');roots={r['root_id'] for r in panels['new_roots']}
    if (receipt.get('phase')!='completed' or receipt.get('roots')!=128 or len(teachers)!=128
        or set(teachers)!=roots or receipt['results_sha256']!=file_sha(worker/'results.json')
        or receipt['request_sha256']!=file_sha(worker/'request.json')):raise ValueError('completed128 teacher receipt required')
    locks={}
    def pin(path,expected=None):
        path=str(path);sha=file_sha(path)
        if expected is not None and sha!=expected:raise ValueError('completed artifact changed: '+path)
        locks[path]=sha
    for name in ('production.json','status.json','costs.json','panels.json','source_phase_result.json',
                 'stages/round_contract.json','stages/train_student.json','teacher_worker/completed.json',
                 'teacher_worker/results.json','teacher_worker/request.json'):
        pin(rd/name)
    for row in teachers.values():
        if row['source_actor_sha256']!=spec['source']['actor_sha256'] or row['generator']!=spec['continuation']['generator']:
            raise ValueError('teacher source/G identity changed')
        pin(row['proposal_path'],row['proposal_sha256'])
        if row.get('demo'):pin(row['demo']['path'],row['demo']['sha256'])
    inventory=spec.get('student_reset_inventory')
    if inventory is None:
        config=policy['formal_config'];pin(config,policy['formal_config_file_sha256'])
        inventory=read(config)['student_reset_inventory']
    for path in (student['frozen_policy'],student['bank'],student['learner']['path'],m['state_path'],
        training/'status.json',training/'formal_report.json',phase['support_path'],phase['panel_support_path'],inventory):pin(path)
    probes=training/'checkpoint_action_probes.json';pin(probes)
    pin(phase['demo_manifest']['path'],phase['demo_manifest']['sha256'])
    demo=read(phase['demo_manifest']['path']);pin(demo['path'],demo['sha256'])
    costs=read(rd/'costs.json');charge=sum(c['charged_interactions'] for c in costs)
    if charge!=status['charged_interactions']:raise ValueError('failed round cost ledger changed')
    completed=[c for c in costs if c['stage']=='student_student' and c['phase']=='completed']
    teacher_cost=[c for c in costs if c['stage']=='teacher_worker' and c['phase']=='completed']
    failed=[c for c in costs if c['stage']=='student_train_batch_0000' and c['phase']=='failed']
    if (len(completed)!=1 or completed[0]['charged_interactions']!=32000+report['train_panel_interactions']
        or len(teacher_cost)!=1 or teacher_cost[0]['charged_interactions']!=receipt['charged_interactions']
        or teacher_cost[0]['worker_receipt_sha256']!=file_sha(worker/'completed.json') or len(failed)!=1):
        raise ValueError('completed learning/teacher cost proof required')
    return dict(schema='jit_post_learning_evaluation_boundary_v1',phase=dict(path=str(phase_path),sha256=file_sha(phase_path)),
        student=student,student_stage=dict(path=str(stage_path),sha256=file_sha(stage_path)),
        inventory=dict(path=inventory,sha256=file_sha(inventory)),
        checkpoint_probes=dict(path=str(probes),sha256=file_sha(probes)),
        teachers=dict(path=str(worker/'results.json'),sha256=file_sha(worker/'results.json')),
        teacher_roots=128,completed_training_transitions=32000,optimizer_updates=updates,
        learner_state_sha256=m['state_sha256'],optimizer_sha256=m['optimizer_sha256'],rng_sha256=m['rng_sha256'],
        charged_interactions=charge,failed_evaluation_upper=failed[0]['charged_interactions'],locks=locks,
        extra_actor_updates=0,new_physics_interactions=0,simulator_frame_continuity=False)


def completed_student_phase(runner):
    """Keep the original demo/support/reset paths and journal input identity."""
    proof=runner.spec['recovery_completed_student']
    for ref in (proof['phase'],proof['inventory'],proof['student_stage']):
        if file_sha(ref['path'])!=ref['sha256']:raise ValueError('completed student phase changed')
    phase=read(proof['phase']['path']);runner.spec['student_reset_inventory']=proof['inventory']['path']
    atomic_json(runner.root/'source_phase_result.json',phase)
    atomic_json(runner.root/'student_reuse.json',dict(source=proof['student_stage'],
        completed_training_transitions=32000,extra_actor_updates=0,charged_in_carried_costs=True,
        learner=proof['student']['learner'],simulator_frame_continuity=False))
    return phase


def prepare_evaluation_recovery(previous,output,repository,*,port=None):
    """Prepare an explicit sibling attempt after learning, with zero new updates."""
    from .production import implementation_identity
    parent=Path(previous).resolve();root=Path(output).resolve();repo=Path(repository).resolve();rd=parent/'round_0001'
    if root==parent or root.is_relative_to(parent):raise ValueError('fresh sibling attempt output required')
    old=read(parent/'declaration.json');status=read(parent/'status.json')
    if status.get('phase')!='failed' or status.get('completed_rounds')!=0:raise ValueError('failed first-round campaign required')
    started=read(parent/'started.json');pid=started.get('pid')
    if pid and (Path('/proc')/str(pid)/'cmdline').exists():
        command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
        aliases={str(parent),str(Path(previous)),old['campaign_output']}
        if any(alias.encode() in command for alias in aliases):raise ValueError('failed campaign coordinator is still alive')
    proof=verify_evaluation_boundary(parent);commit=implementation_identity(repo);spec=read(rd/'production.json')
    for path,sha in old['locks'].items():
        if file_sha(path)!=sha:raise ValueError('original initialization changed: '+path)
    stages={'train_student':proof['student_stage']};source_locks={}
    inherited_stages=spec.get('recovery',{}).get('stages',{}) if spec.get('recovery_completed_student') else {}
    source_refs={name:ref for name,ref in inherited_stages.items() if name.startswith('eval_source_suffix_batch_')}
    if not inherited_stages:
        source_refs={path.stem:dict(path=str(path),sha256=file_sha(path))
            for path in sorted((rd/'stages').glob('eval_source_suffix_batch_*.json')) if not path.name.endswith('.running.json')}
    for name,ref in source_refs.items():
        path=Path(ref['path'])
        if file_sha(path)!=ref['sha256']:raise ValueError('inherited source suffix journal changed')
        record=read(path)
        if record['output_sha256']!=digest(record['result']) or file_sha(record['result']['path'])!=record['result']['sha256']:
            raise ValueError('completed source suffix stage changed')
        stages[name]=ref
        source_locks[str(path)]=file_sha(path)
        source_locks[record['result']['path']]=record['result']['sha256']
        for row in read(record['result']['path']):
            for attempt in row.get('attempts',[]):source_locks[attempt['trace']]=attempt['trace_sha256']
    if len(stages)!=5:raise ValueError('four completed source suffix stages and completed student required')
    for path,sha in source_locks.items():
        if file_sha(path)!=sha:raise ValueError('completed source trace changed')
    # The existing collection reference may belong to the earlier OOM attempt.
    collection=spec.get('recovery_collection')
    if collection is None:raise ValueError('post-learning recovery requires the pinned completed collection reference')
    for path,sha in collection['files'].items():
        if file_sha(path)!=sha:raise ValueError('completed collection changed')
    carried=read(rd/'costs.json');additional=dict(student_suffix_evaluation=128*400,student_stress=9*32*400)
    cumulative=proof['charged_interactions']+sum(additional.values())
    if cumulative>spec['budgets']['max_physics']:raise ValueError('post-learning recovery exceeds original round ceiling')
    root.mkdir(parents=True,exist_ok=False);atomic_json(root/'initial_bundle.json',read(parent/'initial_bundle.json'))
    atomic_json(root/'post_learning_boundary.json',proof)
    d=deepcopy(old);d.pop('storage_continuation',None);d.pop('active_authorization_path',None)
    original_contract=(spec['recovery']['previous_contract_identity'] if spec.get('recovery_completed_student')
                       else read(rd/'stages/round_contract.json')['sha256'])
    info=dict(previous_campaign=str(parent),previous_round=str(rd),previous_spec_sha256=file_sha(rd/'production.json'),
        previous_contract_identity=original_contract,previous_costs_sha256=file_sha(rd/'costs.json'),
        original_round_started_unix=read(rd/'started.json')['started_unix'],carried_round_costs=carried,stages=stages,
        collection=collection,completed_student=proof,additional_round_budget=additional,cumulative_round_upper=cumulative,
        maximum_additional_round_interactions=sum(additional.values()),original_round_ceiling=spec['budgets']['max_physics'],
        reason='explicit inference-constructor repair after completed32k learning; preserve all states, outcomes and costs',automatic_retry=False)
    d.update(campaign_output=str(root),repository=str(repo),implementation_commit=commit,series_id=spec['series_id'],
        tensorboard_port=old['tensorboard_port'] if port is None else port,initial_bundle_sha256=file_sha(root/'initial_bundle.json'),
        engineering_recovery=info,original_scientific_campaign=old.get('original_scientific_campaign',str(parent)),
        locks={**old['locks'],**proof['locks'],**source_locks,str(parent/'declaration.json'):file_sha(parent/'declaration.json'),
            str(parent/'status.json'):file_sha(parent/'status.json'),str(root/'initial_bundle.json'):file_sha(root/'initial_bundle.json'),
            str(root/'post_learning_boundary.json'):file_sha(root/'post_learning_boundary.json')})
    atomic_json(root/'declaration.json',d);atomic_json(root/'status.json',dict(phase='prepared',completed_rounds=0,declared_rounds=100,
        completed_student_transitions=32000,recovered_round=1,extra_actor_updates=0,automatic_retry=False))
    return d


def prepare_resume(previous,output,repository,*,teacher_audit,engineering_receipts=(),engineering_reports=(),port=None):
    """Explicit fresh attempt after first-round engineering failure, zero GPU work."""
    from .production import implementation_identity
    parent=Path(previous).resolve();root=Path(output).resolve();repo=Path(repository).resolve()
    old=read(parent/'declaration.json');status=read(parent/'status.json');rd=parent/'round_0001'
    if root==parent or root.is_relative_to(parent):raise ValueError('fresh sibling attempt output required')
    if status.get('phase')!='failed' or status.get('completed_rounds')!=0:
        raise ValueError('resume supports only failed pre-optimization first round')
    if any((rd/name).exists() for name in ('students','explorer_update','generator_incremental')):
        raise ValueError('teacher engineering recovery cannot discard started optimization')
    if read(rd/'status.json').get('phase')!='failed':raise ValueError('failed original round required')
    # Preserve old execution even when its status is already failed.
    started=read(parent/'started.json');pid=started.get('pid')
    if pid and (Path('/proc')/str(pid)/'cmdline').exists():
        command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
        if str(parent).encode() in command:raise ValueError('failed campaign coordinator is still alive')
    audit_path=Path(teacher_audit).resolve();audit=_verify_resume_audit(audit_path)
    if (Path(audit['round_dir'])!=rd or audit.get('round_index')!=1
        or audit['round_manifest_sha256']!=file_sha(rd/'production.json')):
        raise ValueError('teacher audit belongs to a different failed round')
    original_spec=read(rd/'production.json');round_contract=read(rd/'stages/round_contract.json')
    if round_contract!={'sha256':digest(original_spec),'contract':original_spec}:
        raise ValueError('original round contract changed')
    parent_costs=read(rd/'costs.json');worker_costs=read(Path(audit['worker_dir'])/'costs.json')
    carried=reconcile_teacher_costs(parent_costs,worker_costs,audit['completed_stages'])
    if sum(c['charged_interactions'] for c in carried)!=audit['cost_reconciliation']['reconciled_round_interactions']:
        raise ValueError('audit and recomputed cost reconciliation disagree')
    for path,sha in old['locks'].items():
        if file_sha(path)!=sha:raise ValueError('original initialization lock changed: '+path)
    engineering=[];engineering_qualification=[];engineering_seen=set()
    for i,path in enumerate(engineering_receipts):
        path=Path(path).resolve()
        if str(path) in engineering_seen:raise ValueError('duplicate engineering cost receipt')
        engineering_seen.add(str(path));record=read(path);charge=record.get('charged_interactions')
        if record.get('phase')!='completed' or type(charge) is not int or charge<0:
            raise ValueError('completed engineering receipt with actual charged_interactions required')
        engineering.append(dict(path=str(path),sha256=file_sha(path),charged_interactions=charge))
        carried.append(dict(stage=f'engineering_recovery_check_{i:03d}',phase='completed',charged_updates=0,
            charged_interactions=charge,accounting='explicit bounded engineering receipt',receipt=str(path),receipt_sha256=file_sha(path)))
    for path in engineering_reports:
        path=Path(path).resolve();record=read(path)
        engineering_qualification.append(dict(path=str(path),sha256=file_sha(path),
            phase=record.get('phase'),cost_semantics='qualification evidence only; physical child charges separately retained'))
    completed=deepcopy(audit['completed_roots'])
    protocol=digest(dict(version=original_spec.get('protocol_version','1.1'),task='stable_forward_recovery',H=16,horizon=400,
        source_physics=original_spec['source']['xml_sha256'],source_reward=original_spec['source_runtime']['reward_mode']))
    for row in completed.values():row['protocol_sha256']=protocol
    remaining=audit['total_teacher_roots']-len(completed)
    additional_components=dict(teacher_remaining_upper=remaining*20800,student_training=32000,
        student_suffix_evaluation=128*400,student_stress=9*32*400,checkpoint_train_panels=2*2*2*400)
    maximum_new=sum(additional_components.values())
    cumulative=sum(c['charged_interactions'] for c in carried)+maximum_new
    if cumulative>original_spec['budgets']['max_physics']:
        raise ValueError('explicit recovery cumulative original round physical ceiling insufficient')
    collection=rd/'collection';collection_spec=rd/'collection_spec.json'
    files={str(p):file_sha(p) for p in collection.glob('*.json')}
    files[str(collection_spec)]=file_sha(collection_spec)
    for name in ('prefixes.npz','update_state.msgpack'):
        path=collection/name
        if not path.is_file():raise ValueError('completed collection payload missing: '+name)
        files[str(path)]=file_sha(path)
    collection_ref=dict(output=str(collection),previous_round=str(rd),spec_path=str(collection_spec),files=files,
        charged_interactions=read(collection/'status.json')['charged_interactions'])
    if read(collection/'status.json').get('phase')!='completed':raise ValueError('collection not completed')
    stages={};source_artifacts={}
    for path in sorted((rd/'stages').glob('eval_source_suffix_batch_*.json')):
        if path.name.endswith('.running.json'):continue
        record=read(path)
        if record['output_sha256']!=digest(record['result']) or file_sha(record['result']['path'])!=record['result']['sha256']:
            raise ValueError('completed source suffix receipt changed')
        results=read(record['result']['path']);status_path=Path(record['result']['path']).parent/'status.json'
        stage_status=read(status_path)
        if stage_status.get('phase')!='completed':raise ValueError('source suffix not completed')
        for row in results:
            for attempt in row.get('attempts',[]):
                trace=attempt['trace'];sha=attempt['trace_sha256']
                if trace in source_artifacts and source_artifacts[trace]!=sha:raise ValueError('source trace identity conflict')
                source_artifacts[trace]=sha
        source_artifacts[str(status_path)]=file_sha(status_path)
        source_artifacts[record['result']['path']]=record['result']['sha256']
        stages[path.stem]=dict(path=str(path),sha256=file_sha(path))
    if len(stages)!=4:raise ValueError('all original four source suffix batches required')
    from .verified_files import LOCK_CACHE
    for path,sha in source_artifacts.items():LOCK_CACHE.verify(path,sha)
    root.mkdir(parents=True,exist_ok=False)
    initial=read(parent/'initial_bundle.json');atomic_json(root/'initial_bundle.json',initial)
    d=deepcopy(old);d.update(campaign_output=str(root),repository=str(repo),implementation_commit=implementation_identity(repo),
        original_scientific_campaign=old.get('original_scientific_campaign',str(parent)),
        series_id=original_spec['series_id'],tensorboard_port=old['tensorboard_port'] if port is None else port,
        initial_bundle_sha256=file_sha(root/'initial_bundle.json'))
    info=dict(previous_campaign=str(parent),previous_round=str(rd),previous_spec_sha256=file_sha(rd/'production.json'),
        previous_costs_sha256=file_sha(rd/'costs.json'),previous_contract_identity=round_contract['sha256'],
        original_round_started_unix=read(rd/'started.json')['started_unix'],carried_round_costs=carried,stages=stages,
        collection=collection_ref,teacher_worker=dict(previous=audit['worker_dir'],
            previous_contract_identity=audit['worker_contract_sha256'],stages=_resume_stage_refs(audit['completed_stages']),
            partial_root=audit.get('partial_root'),artifact_locks=audit['input_locks']),
        completed_teacher_roots=completed,teacher_audit=dict(path=str(audit_path),sha256=file_sha(audit_path)),
        engineering_receipts=engineering,engineering_qualification_reports=engineering_qualification,
        source_artifacts=source_artifacts,additional_round_budget=additional_components,maximum_additional_round_interactions=maximum_new,
        cumulative_round_upper=cumulative,original_round_ceiling=original_spec['budgets']['max_physics'],
        reason='explicit engineering repair after kernel OOM; preserve all outcomes, seeds, protocol and failed costs',automatic_retry=False)
    d['engineering_recovery']=info
    d['locks']={**old['locks'],str(root/'initial_bundle.json'):file_sha(root/'initial_bundle.json'),
        str(parent/'declaration.json'):file_sha(parent/'declaration.json'),str(parent/'status.json'):file_sha(parent/'status.json'),
        str(rd/'production.json'):file_sha(rd/'production.json'),str(rd/'costs.json'):file_sha(rd/'costs.json'),
        str(audit_path):file_sha(audit_path),**{x['path']:x['sha256'] for x in engineering+engineering_qualification}}
    atomic_json(root/'declaration.json',d)
    atomic_json(root/'recovery_preflight.json',dict(phase='prepared',**info,new_training_transitions=0,new_physics_interactions=0))
    atomic_json(root/'status.json',dict(phase='prepared',completed_rounds=0,declared_rounds=100,
        recovered_round=1,original_campaign=str(parent),automatic_retry=False))
    return d


def prepare(parent, output, repository, *, monitor_panel, monitor_initial, seed=8113001, port=6038):
    from .production import implementation_identity, implementation_files
    from .worker import source_payload
    from .artifacts import validate_generator_receipt, load_corpus
    root=Path(output).resolve();repo=Path(repository).resolve();parent=Path(parent).resolve()
    old=read(parent/'plan.json');previous=read(parent/'current_source.json')
    if read(parent/'status.json')['phase']!='completed':
        raise ValueError('completed audited G73 parent required; paused R107 is not initialization')
    pi0=old['baseline_frozen_policy'];policy,_=source_payload(pi0)
    phase=read(Path(old['parent_campaign'])/'stages/fresh_source_phase.json')['result']
    validate_generator_receipt(previous['generator']);load_corpus(previous['corpus'])
    if previous['round']!=73:
        raise ValueError('audited G73 initialization required')
    fixed=read(monitor_panel)
    if fixed['pi0']['policy']['actor_sha256']!=policy['actor_sha256']:
        raise ValueError('fixed teacher monitor pi0 differs')
    initial_report=read(monitor_initial)
    initial_status=read(Path(monitor_initial).parent/'status.json')
    if (initial_report.get('round')!=0 or initial_report.get('phase')!='completed'
        or initial_status.get('phase')!='completed'
        or initial_status.get('report_sha256')!=file_sha(monitor_initial)
        or initial_report.get('panel_sha256')!=file_sha(monitor_panel)
        or initial_report.get('generator_manifest_sha256')!=previous['generator']['checkpoint_manifest_sha256']):
        raise ValueError('completed G0 must bind the exact fixed panel and initialization generator')
    root.mkdir(parents=True,exist_ok=False)
    baseline=deepcopy(previous)
    for key in ('evaluated_student','acceptance','stress','explorer_metrics','feedback_counts'):
        baseline.pop(key,None)
    baseline.update(round=0,actor=dict(frozen_policy=pi0,policy=policy,
        actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256']),
        explorer=None,learner=None,tail_lineage=[],arrival_ledgers={},
        demo=phase['demo_manifest'],support=phase['support_path'],
        seed_support=dict(path=phase['panel_support_path'],sha256=file_sha(phase['panel_support_path'])))
    atomic_json(root/'initial_bundle.json',baseline)
    settings=campaign_parameters()
    declaration=dict(schema='jit_teacher_recovery_campaign_authorization_v1',
        implementation_commit=implementation_identity(repo),campaign_output=str(root),
        repository=str(repo),rounds=100,user_authorized=True,formal_acceptance=False,
        max_physics_per_round=settings['physics_cap_per_round'],teacher_query_count=128,
        teacher_generated_candidates=16,teacher_reviews=2,
        known_limitations=['single independent outer-loop lineage; not formal15-methodmatrix',
            'fixed teacher DEV panel N7; no final TEST; no monotonic recovery guarantee'],
        parameters=settings,baseline_frozen_policy=pi0,parent_generator_source=str(parent),
        parent_campaign=old['parent_campaign'],source_runtime=read(Path(old['parent_campaign'])/'production.json')['source_runtime'],
        retention_ref=phase['retention_ref'],stress_reference=old['reference_panels']['baseline'],
        complete_start=complete_start_arrays(repo,policy['formal_config']),
        monitor_panel=str(Path(monitor_panel).resolve()),monitor_initial=str(Path(monitor_initial).resolve()),
        seed=seed,tensorboard_port=port,max_wall_seconds=604800,
        max_wall_seconds_scope='per_round_engineering_upper_bound',
        stop_conditions=['engineering/nonfinite/identity failure',
            'free disk below20GiB; preserve all prior artifacts',
            'no verified current teacher reset group or no fresh/historical demonstration group; no silent fallback'],
        initial_bundle_sha256=file_sha(root/'initial_bundle.json'),
        compiler=dict(cache='/home/qy/DVGC/JIT/runs/cache/jit_full_outer_shared',
            XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0'),
        final_test_open=False,automatic_retry=False,automatic_extension=False)
    declaration['locks']={str(p):file_sha(p) for p in [Path(pi0),root/'initial_bundle.json',
        Path(monitor_panel),Path(monitor_initial),Path(declaration['stress_reference']),
        parent/'current_source.json',parent/'plan.json',parent/'status.json']}
    atomic_json(root/'declaration.json',declaration)
    atomic_json(root/'status.json',dict(phase='prepared',completed_rounds=0,declared_rounds=100))
    return declaration


def round_spec(declaration, previous, round_index):
    from .production import implementation_files
    from ..probe_bank import lock_probe_bank
    d=declaration;root=Path(d['campaign_output'])/f'round_{round_index:04d}'
    root.mkdir(exist_ok=False)
    actor=previous['actor'];source=actor['policy'];settings=d['parameters']
    bank=read(d['source_runtime']['bank'])
    bank={k:bank[k] for k in ('task','max_ticks','label_interaction_budget','max_candidates_per_process')}
    bank.update(version=f'teacher_recovery_{round_index}',members=[dict(
        frozen_policy=actor['frozen_policy'],roles=['proposer','evaluator'])])
    lock_probe_bank(bank,root/'bank.json')
    runtime=dict(d['source_runtime'],bank=str(root/'bank.json'),proposer=source['name'],
        order=[source['name']],seed=d['seed']+round_index,round_index=round_index,
        preserve_snapshot_episode_context=True,
        gate=dict(kind='gpu_shared',minimum_free_mib=20000,authorization='start_without_gpu_idle_wait'))
    baseline=read(d['baseline_frozen_policy'])['policy']
    protocol=digest(dict(version='1.2',task='stable_forward_recovery',H=16,horizon=400,
        source_physics=baseline['xml_sha256'],source_reward=runtime['reward_mode']))
    identity=dict(actor_sha256=baseline['actor_sha256'],normalizer_sha256=baseline['normalizer_sha256'],
        model_sha256=baseline['xml_sha256'],protocol_sha256=protocol)
    declaration_path=Path(d.get('active_authorization_path',str(Path(d['campaign_output'])/'declaration.json')))
    from .teacher_monitor import should_evaluate_round
    monitor_reservation=145600 if should_evaluate_round(round_index) else 0
    spec=dict(schema='jit_bridge_experimental_round_v1',protocol_version='1.2',execute=True,
        output=str(root),repository=d['repository'],implementation_commit=d['implementation_commit'],
        implementation_files=implementation_files(d['repository']),
        source_frozen_policy=actor['frozen_policy'],source=source,source_runtime=runtime,
        seed=d['seed']+round_index,round_index=round_index,series_id=d.get('series_id',Path(d['campaign_output']).name),
        continuation=previous,baseline_identity=identity,
        generator_reference_frozen_policy=dict(path=d['baseline_frozen_policy'],sha256=file_sha(d['baseline_frozen_policy'])),
        retention_reference_frozen_policy=d['baseline_frozen_policy'],retention_ref=d['retention_ref'],
        teacher_layout='source_control_in_candidate_batch',teacher_execution='persistent_b1',
        teacher_recovery_authorization=dict(path=str(declaration_path),sha256=file_sha(declaration_path)),
        campaign_output=d['campaign_output'],
        teacher_replay_failure_policy='reject_finite_candidate',
        uniform_episode_fraction=0.,teacher_colored_noise_candidates=0,
        continuous_learning=True,allow_legacy_optimizer_bootstrap=round_index==1,
        nominal_evaluation=False,recovery_training_contract=settings['student'],
        recovery_explorer=settings['explorer'],recovery_complete_start=d['complete_start'],
        pulse_contract_version='scheduled_single_pulse_v1',pulse_start_schedule=[0,5,10,15],
        pulse_batch_mode='mixed',pulse_steps=3,delta_limit=[.25]*4,single_pulse_per_episode=True,
        generator_update_policy='last_valid',generator_charged_updates_before=previous['generator_lifetime_charged_updates'],
        generator_charged_updates_scope=previous.get('generator_charged_updates_scope','lifetime'),
        neighborhood=dict(neighbors=16,far_scale=2.,medium_halfwidths=[.2,.05,.1,.4,.2,.6,2.,5.,5.,10.,20.,20.],evidence_scope='train_history_v1'),
        locks={**d['locks'],previous['dev_fixture']:previous['dev_fixture_sha256'],
            actor['frozen_policy']:file_sha(actor['frozen_policy'])},
        budgets=dict(max_physics=settings['physics_cap_per_round']-monitor_reservation,
            max_supervised_updates=2000,max_wall_seconds=d['max_wall_seconds']),
        final_test_open=False,automatic_retry=False)
    if previous.get('explorer'):
        spec['locks'][previous['explorer']]=previous['explorer_sha256']
    atomic_json(root/'production.json',spec);atomic_json(root/'panels.json',{})
    return spec


def record_round_metrics(root, round_index, bundle):
    import csv
    import numpy as np
    rd=Path(root)/f'round_{round_index:04d}'
    f=read(rd/'explorer_feedback.json');e=read(rd/'explorer_update/metrics.json')
    eligible=sum(f['eligible'])
    outcomes=read(rd/'root_outcomes.json')
    teachers=[r for r in outcomes if r.get('teacher_found') is True]
    acquired=sum(r.get('student_label')==1 for r in teachers)
    selections=list(rd.glob('generator_incremental/attempt_*/generator_selection.json'))
    g=read(selections[-1]) if selections else {}
    values=[m['noise_mse'] for m in g.get('metrics',[]) if 'noise_mse' in m]
    metrics=dict(explorer_reward_mean=(sum(v for v,m in zip(f['rewards'],f['eligible']) if m)/eligible if eligible else None),
        explorer_eligible=eligible,explorer_quality_sum=f['component_sums']['quality'],
        explorer_novelty_sum=f['component_sums']['novelty'],explorer_kl=e['post_update_kl'],
        explorer_entropy=e['entropy'],explorer_learning_rate=e['learning_rate'],
        explorer_actual_epochs=e['completed_epochs'],explorer_optimizer_updates=e['optimizer_updates'],
        teacher_train_verified=len(teachers),student_train_independent_successes=acquired,
        student_fixed_dev_successes=sum(v['successes'] for v in bundle['stress'].values()),
        student_fixed_dev_total=sum(len(v['labels']) for v in bundle['stress'].values()),
        generator_training_loss_mean=float(np.mean(values)) if values else None,
        generator_new_updates=len(values),generator_parameter_age_updates=bundle['generator']['state_updates'])
    with (Path(root)/'monitoring/round_scalars.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(round=round_index,metrics=metrics))+'\n')
    plot=Path(root)/'plotting';plot.mkdir(exist_ok=True)
    path=plot/'round_metrics.csv';exists=path.exists()
    with path.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['round',*metrics],lineterminator='\n')
        if not exists:writer.writeheader()
        writer.writerow(dict(round=round_index,**metrics))
    atomic_json(plot/f'round_{round_index:04d}_artifacts.json',dict(round=round_index,
        generator=bundle['generator'],student=bundle['actor'],explorer=bundle['explorer'],
        source_rows=str(rd/'source_rows.json'),teacher_results=str(rd/'teacher_worker/results.json'),
        teacher_feedback=str(rd/'explorer_feedback.json'),teacher_student_outcomes=str(rd/'root_outcomes.json')))


def run_round(path):
    from .closed_loop import ClosedLoopRound
    spec=read(path)
    if spec.get('recovery_collection') is None:return ClosedLoopRound(spec).run()
    class ExplicitRecoveredRound(ClosedLoopRound):
        def __init__(self,manifest):
            super().__init__(manifest)
            original=manifest.get('recovery_started_unix')
            if original is not None:
                new_start=self.started_unix;self.started_unix=original
                atomic_json(self.root/'started.json',dict(started_unix=original,attempt_started_unix=new_start,
                    original_budget_clock_preserved=True,simulator_frame_continuity=False))
        def checkpoint_probes(self,student,source_phase,warmup):
            proof=self.spec.get('recovery_completed_student')
            if proof is None:return super().checkpoint_probes(student,source_phase,warmup)
            ref=proof['checkpoint_probes'];path=Path(ref['path'])
            if (path!=Path(student['training'])/'checkpoint_action_probes.json'
                or file_sha(path)!=ref['sha256']):raise ValueError('completed checkpoint probes changed')
            atomic_json(self.root/'checkpoint_probe_reuse.json',dict(source=ref,
                extra_actor_updates=0,new_physics_interactions=0,original_receipt_preserved=True))
            return path
        def teacher_search(self,incumbent):
            proof=self.spec.get('recovery_completed_student')
            if proof is None:return super().teacher_search(incumbent)
            ref=proof['teachers']
            if file_sha(ref['path'])!=ref['sha256']:raise ValueError('completed teacher results changed')
            teachers=read(ref['path'])
            if set(teachers)!={r['root_id'] for r in self.panels['new_roots']}:
                raise ValueError('completed teacher panel changed')
            if any(row['generator']!=incumbent for row in teachers.values()):raise ValueError('completed teacher generator changed')
            target=self.root/'teacher_worker';target.mkdir(exist_ok=False)
            atomic_json(target/'results.json',teachers)
            reuse=dict(source=ref,roots=len(teachers),new_physics_interactions=0,
                charged_in_carried_costs=True,all_negative_and_quarantine_results_preserved=True)
            atomic_json(target/'reuse.json',reuse)
            provenance=self.root/'teachers';provenance.mkdir(exist_ok=False)
            atomic_json(provenance/'reuse.json',reuse)
            return teachers
        def measured(self,name,mode,config,maximum,**kwargs):
            if name!='collection':return super().measured(name,mode,config,maximum,**kwargs)
            if mode!='collect' or getattr(self,'_collection_reused',False):raise ValueError('single explicit completed collection reuse required')
            output=reuse_completed_collection(self.spec['recovery_collection'],config,self.root)
            self._collection_reused=True
            atomic_json(self.root/'collection_reuse.json',dict(source=self.spec['recovery_collection'],
                additional_physics_interactions=0,charged_in_carried_costs=True))
            return output
    return ExplicitRecoveredRound(spec).run()


def verify_storage_completed_round(root):
    """Verify the completed first-round learner/E/G boundary without any update."""
    import io,pickle
    import jax,numpy as np,torch
    from .worker import source_payload
    from .learner_continuation import validate_saved_learner
    from .actual_update import adam_update_count
    from .artifacts import validate_generator_full_state
    from ..handoff_bank import pytree_sha256
    from ..rsl_pulse import restore
    root=Path(root).resolve();rd=root/'round_0001';status=read(rd/'status.json')
    if status.get('phase')!='completed' or status.get('training_transitions')!=32000:
        raise ValueError('storage handoff requires a fully completed32k first round')
    if (root/'round_0002').exists():raise ValueError('round2 already exists; storage handoff cannot replay it')
    costs=read(rd/'costs.json');charge=sum(c['charged_interactions'] for c in costs)
    if charge!=status['charged_interactions']:raise ValueError('completed round cost ledger mismatch')
    bundle=read(rd/'current_source.json')
    if bundle.get('round')!=1:raise ValueError('completed source round identity mismatch')
    policy,_=source_payload(bundle['actor']['frozen_policy'])
    if policy!=bundle['actor']['policy']:raise ValueError('completed source policy identity mismatch')
    m=validate_saved_learner(bundle['learner'],policy,expected_local_transitions=32000)
    with Path(m['state_path']).open('rb') as stream:saved=pickle.load(stream)
    state=saved['training_state'];rng=saved['rng']
    def finite(tree):
        for leaf in jax.tree.leaves(tree):
            value=np.asarray(leaf)
            if value.dtype.kind in 'fciub' and not np.isfinite(value).all():raise ValueError('nonfinite storage boundary state')
    finite((state,rng))
    for field,value in (('actor_sha256',state.params.policy),('critic_sha256',state.params.value),
        ('normalizer_sha256',state.normalizer_params),('optimizer_sha256',state.optimizer_state)):
        if pytree_sha256(value)!=m[field]:raise ValueError('serialized learner '+field+' mismatch')
    if pytree_sha256(rng)!=m['rng_sha256']:raise ValueError('serialized learner RNG mismatch')
    updates=int(adam_update_count(state.optimizer_state))
    if int(state.env_steps)!=32000 or m['lifetime_transitions']!=32000 or updates<1:
        raise ValueError('completed learner transition/optimizer clock mismatch')
    if file_sha(bundle['explorer'])!=bundle['explorer_sha256']:raise ValueError('completed explorer payload changed')
    e=restore(bundle['explorer']);esaved=torch.load(io.BytesIO(e['torch_state']),map_location='cpu',weights_only=False)
    if not esaved.get('optimizer') or 'rng' not in esaved or int(e['total_updates'])<1:
        raise ValueError('completed explorer requires optimizer and RNG')
    finite((e['params'],e['normalizer_mean'],e['normalizer_std'],e['rng'],esaved))
    g=validate_generator_full_state(bundle['generator'])
    gpath=Path(bundle['generator']['checkpoint_manifest']);gm=read(gpath)
    locks={str(rd/name):file_sha(rd/name) for name in ('status.json','current_source.json','costs.json')}
    for filename in (bundle['actor']['frozen_policy'],bundle['learner']['path'],m['state_path'],
        bundle['explorer'],str(gpath),str(gpath.parent/'state.msgpack')):locks[filename]=file_sha(filename)
    return dict(schema='jit_completed_round_storage_boundary_v1',phase='verified',round=1,next_round=2,
        completed_training_transitions=32000,charged_interactions=charge,locks=locks,
        source=dict(path=str(rd/'current_source.json'),sha256=file_sha(rd/'current_source.json')),
        learner=dict(manifest=bundle['learner'],state_sha256=m['state_sha256'],
            actor_sha256=m['actor_sha256'],critic_sha256=m['critic_sha256'],normalizer_sha256=m['normalizer_sha256'],
            optimizer_sha256=m['optimizer_sha256'],rng_sha256=m['rng_sha256'],optimizer_updates=updates),
        explorer=dict(path=bundle['explorer'],sha256=bundle['explorer_sha256'],updates=int(e['total_updates']),
            optimizer_sha256=pytree_sha256(esaved['optimizer']),rng_sha256=pytree_sha256((e['rng'],esaved['rng']))),
        generator=dict(manifest=bundle['generator'],state_sha256=gm['state_sha256'],updates=int(g['updates']),
            optimizer_sha256=pytree_sha256(g['optimizer']),rng_sha256=pytree_sha256(g['rng'])),
        simulator_frame_continuity=False,extra_actor_updates=0,new_physics_interactions=0)


def record_round_metrics_once(root,index,bundle):
    """Commit the completed round's missing coordinator metrics once."""
    root=Path(root);path=root/'monitoring/round_scalars.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    matching=[r for r in rows if r.get('round')==index]
    artifact=root/'plotting'/f'round_{index:04d}_artifacts.json'
    if matching:
        if len(matching)!=1 or not artifact.is_file():raise ValueError('partial/duplicate round metric publication requires reconciliation')
        old=read(artifact)
        if old['generator']!=bundle['generator'] or old['student']!=bundle['actor'] or old['explorer']!=bundle['explorer']:
            raise ValueError('completed round metric source changed')
        return
    (root/'monitoring').mkdir(exist_ok=True)
    record_round_metrics(root,index,bundle)


def prepare_storage_resume(existingrunroot_physical,repository,*,port=None):
    """Write a separate successor declaration after a completed storage boundary."""
    from .production import implementation_identity
    root=Path(existingrunroot_physical).resolve();repo=Path(repository).resolve()
    successor=root/'declaration_storage.json'
    if successor.exists():raise ValueError('storage successor declaration already exists')
    original=root/'declaration.json';old=read(original);original_sha=file_sha(original)
    proof=verify_storage_completed_round(root);bundle=read(proof['source']['path'])
    implementation_commit=implementation_identity(repo)
    for filename,sha in old['locks'].items():
        if file_sha(filename)!=sha:raise ValueError('original initialization lock changed: '+filename)
    transition=root/'storage_transition'
    receipt=transition/'completed_round1_verification.json'
    if receipt.exists():raise ValueError('storage boundary receipt already exists')
    done=[dict(round=1,learner=bundle['learner'],charged_interactions=proof['charged_interactions'])]
    current=root/'current_source.json';journal=root/'completed_rounds.json'
    if current.exists() and read(current)!=bundle:raise ValueError('coordinator source differs from completed round1')
    if journal.exists() and read(journal) not in ([],done):raise ValueError('completed coordinator journal differs')
    transition.mkdir(exist_ok=True)
    atomic_json(receipt,proof)
    if not current.exists():atomic_json(current,bundle)
    atomic_json(journal,done)
    record_round_metrics_once(root,1,bundle)
    d=deepcopy(old);d.update(repository=str(repo),implementation_commit=implementation_commit,
        campaign_output=str(root),active_authorization_path=str(successor),
        tensorboard_port=old['tensorboard_port'] if port is None else port,
        storage_continuation=dict(schema='jit_completed_round_storage_continuation_v1',
            completed_rounds=1,next_round=2,boundary=dict(path=str(receipt),sha256=file_sha(receipt)),
            original_declaration=dict(path=str(original),sha256=original_sha),
            source=proof['source'],completed_journal=done,simulator_frame_continuity=False,
            authorization='user explicitly approved storage-only migration; continue declared rounds2..100',
            automatic_retry=False,extra_actor_updates=0),
        locks={**old['locks'],**proof['locks'],str(original):original_sha,str(receipt):file_sha(receipt)})
    atomic_json(successor,d)
    return d


def campaign_start_state(declaration,root):
    """Fresh campaigns start1; an explicit completed storage boundary starts2."""
    storage=declaration.get('storage_continuation')
    if storage is None:return read(Path(root)/'initial_bundle.json'),[],1
    if (storage.get('schema')!='jit_completed_round_storage_continuation_v1'
        or storage.get('completed_rounds')!=1 or storage.get('next_round')!=2):
        raise ValueError('invalid completed storage cursor')
    for ref in (storage['boundary'],storage['source'],storage['original_declaration']):
        if file_sha(ref['path'])!=ref['sha256']:raise ValueError('storage continuation boundary changed')
    return read(storage['source']['path']),deepcopy(storage['completed_journal']),2


def monitoring_sources(root,*,start_round=1,existing_event_runs=False,completed_student_training=None):
    root=Path(root);sources=[]
    for i in range(start_round,101):
        training=root/f'round_{i:04d}/students/student/training/round_{i:04d}_student'
        if i==1 and completed_student_training is not None:training=Path(completed_student_training)
        for kind,filename in [('loss','metrics.jsonl'),('reward','episode_metrics.jsonl')]:
            run='student_'+kind
            if existing_event_runs:run=str(root/'monitoring/events'/run)
            sources.append(dict(name=f'round{i}_{kind}',run=run,path=str(training/filename),
                step='training_transitions',step_offset=(i-1)*32000,completion_status=str(training/'status.json')))
    run=str(root/'monitoring/events/outer_loop') if existing_event_runs else 'outer_loop'
    sources.append(dict(name='outer_loop_metrics',run=run,path=str(root/'monitoring/round_scalars.jsonl'),step='round'))
    return sources


def stop_owned_child(process, root):
    """Stop a verified active child's descendants; leave all monitoring services."""
    import signal
    root=Path(root)
    if process is None or process.poll() is not None:
        return []
    pid=process.pid
    command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(root) not in command:
        raise ValueError('refuse to terminate unowned active child')
    os.kill(pid,signal.SIGSTOP)
    parents={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            lines=(path/'status').read_text().splitlines()
            parents[int(path.name)]=int(next(s for s in lines if s.startswith('PPid:')).split(':')[1])
        except (OSError,StopIteration,ValueError):pass
    owned={pid}
    while True:
        new={p for p,parent in parents.items() if parent in owned}-owned
        if not new:break
        owned.update(new)
    receipt=[]
    for target in sorted(owned,reverse=True):
        try:
            cmd=(Path('/proc')/str(target)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            receipt.append(dict(pid=target,command=cmd))
            os.kill(target,signal.SIGTERM);os.kill(target,signal.SIGCONT)
        except ProcessLookupError:pass
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        for target in owned:
            try:os.kill(target,signal.SIGKILL)
            except ProcessLookupError:pass
        process.wait(timeout=10)
    atomic_json(root/'owned_child_stop_receipt.json',receipt)
    return receipt


def run_campaign(declaration_path):
    from .production import implementation_identity,start_notifications
    from .teacher_monitor import should_evaluate_round
    d=read(declaration_path);root=Path(d['campaign_output']);repo=Path(d['repository'])
    storage=d.get('storage_continuation') is not None
    started_path=(root/'storage_transition/started_storage.json') if storage else root/'started.json'
    if started_path.exists():
        raise ValueError('explicit recovery required; no automatic restart')
    if implementation_identity(repo)!=d['implementation_commit']:
        raise ValueError('implementation revision drift')
    for p,h in d['locks'].items():
        if file_sha(p)!=h:raise ValueError('locked initialization changed: '+p)
    completed_student=d.get('engineering_recovery',{}).get('completed_student')
    if d.get('engineering_recovery') is not None and not storage and completed_student is None:
        audit=d['engineering_recovery']['teacher_audit']
        if file_sha(audit['path'])!=audit['sha256']:raise ValueError('engineering recovery audit changed')
        _verify_resume_audit(audit['path'])
        from .verified_files import LOCK_CACHE
        for filename,sha in d['engineering_recovery'].get('source_artifacts',{}).items():LOCK_CACHE.verify(filename,sha)
    previous,done,first_round=campaign_start_state(d,root)
    atomic_json(started_path,dict(pid=os.getpid(),started_unix=time.time(),declaration=str(Path(declaration_path).resolve()),next_round=first_round))
    env=dict(os.environ,PYTHONPATH=str(repo/'JIT/src'),JAX_PLATFORMS='cpu',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',JIT_AUTO_PUBLISH='0',
        JAX_COMPILATION_CACHE_DIR=d['compiler']['cache'],XLA_FLAGS=d['compiler']['XLA_FLAGS'],
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def spawn(name,args,environment=env):
        with (root/(name+'.log')).open('ab') as stream:
            p=subprocess.Popen(args,cwd=repo,env=environment,stdout=stream,stderr=stream,start_new_session=True)
        atomic_json(root/(name+'_pid.json'),dict(pid=p.pid,argv=args));return p
    py=sys.executable;active=None
    initial=read(root/'initial_bundle.json');inherited_generator_cost=initial.get('generator_lifetime_charged_updates')
    initial_cost=read(Path(d['monitor_initial']).parent/'status.json')['charged_interactions']
    monitor_costs=[dict(round=0,charged_interactions=initial_cost,path=d['monitor_initial'])]
    def status(phase,**kwargs):
        owned_round_costs=[dict(c,round=p.parent.name) for p in root.glob('round_*/costs.json') for c in read(p)]
        monitor_records={x['round']:x for x in monitor_costs}
        for p in root.glob('teacher_monitor/G*/status.json'):
            record=read(p);monitor_records[record['round']]=dict(round=record['round'],
                charged_interactions=record['charged_interactions'],path=str(p.parent/'report.json'),phase=record['phase'])
        physics=sum(x['charged_interactions'] for x in owned_round_costs)+sum(x['charged_interactions'] for x in monitor_records.values())
        cap=d['parameters']['physics_cap_per_round']*100+142800
        if physics>cap:raise ValueError('new campaign physical budget exceeded')
        atomic_json(root/'cost_ledger.json',dict(rounds=done,round_stage_costs=owned_round_costs,
            teacher_monitors=list(monitor_records.values()),
            new_charged_interactions=physics,new_physical_cap=cap,
            inherited_generator_cost=inherited_generator_cost,
            optimization_units='G optimizer updates separate from physical control steps'))
        atomic_json(root/'status.json',dict(phase=phase,declared_rounds=100,completed_rounds=len(done),
            completed_student_transitions=max(len(done),int(completed_student is not None))*32000,student_transition_target=3200000,
            charged_interactions=physics,updated_unix=time.time(),**kwargs))
    monitoring=root/'monitoring';monitoring.mkdir(exist_ok=storage)
    manifest=monitoring/('tensorboard_manifest_storage.json' if storage else 'tensorboard_manifest.json')
    export_output=monitoring/'export_storage_receipt' if storage else monitoring/'events'
    atomic_json(manifest,dict(sources=monitoring_sources(root,start_round=first_round,existing_event_runs=storage,
        completed_student_training=completed_student['student']['training'] if completed_student else None)))
    spawn('exporter',[py,str(repo/'JIT/cli/export_tensorboard.py'),'--manifest',str(manifest),
        '--output',str(export_output),'--watch'])
    spawn('tensorboard',[py,'-m','tensorboard.main','--logdir',str(monitoring/'events'),'--host','0.0.0.0',
        '--port',str(d['tensorboard_port']),'--reload_interval','5'])
    atomic_json(root/'ACTIVE_RUN.json',dict(name=root.name,execution=str(root/'status.json'),lineage=str(root/'status.json')))
    status('starting')
    start_notifications(dict(output=str(root),repository=str(repo),protocol_version='teacher-recovery100'))
    try:
        for i in range(first_round,101):
            import shutil
            if shutil.disk_usage(root).free<20*1024**3:
                raise RuntimeError('free disk below20GiB margin; no artifact deletion')
            spec=(resume_round_spec(d) if i==1 and d.get('engineering_recovery') is not None else round_spec(d,previous,i));rd=Path(spec['output']);status('running',current_round=i,current_stage='full_loop')
            atomic_json(root/'ACTIVE_RUN.json',dict(name=root.name,execution=str(rd/'status.json'),lineage=str(root/'status.json')))
            p=spawn(f'round_{i:04d}',[py,str(repo/'JIT/cli/run_teacher_recovery_campaign.py'),'round','--spec',str(rd/'production.json')])
            active=p
            status('running',current_round=i,current_stage='full_loop',child_pid=p.pid)
            if p.wait()!=0:raise RuntimeError(f'round {i} failed; no automatic retry')
            active=None
            result=read(rd/'status.json');previous=read(rd/'current_source.json')
            if result['phase']!='completed' or result['training_transitions']!=32000:
                raise ValueError('incomplete full outer round')
            done.append(dict(round=i,learner=previous['learner'],charged_interactions=result['charged_interactions']))
            record_round_metrics(root,i,previous)
            atomic_json(root/'completed_rounds.json',done);atomic_json(root/'current_source.json',previous)
            if should_evaluate_round(i):
                status('running',current_round=i,current_stage='fixed_teacher_dev')
                out=root/'teacher_monitor'/f'G{i}';out.parent.mkdir(exist_ok=True)
                args=[py,str(repo/'JIT/cli/run_teacher_monitor.py'),'evaluate','--panel',d['monitor_panel'],
                    '--generator',previous['generator']['checkpoint_manifest'],'--output',str(out),'--round',str(i),
                    '--baseline',d['monitor_initial'],'--student-bank',previous['actor']['bank'],
                    '--student-proposer',previous['actor']['policy']['name']]
                p=spawn(f'G{i}_monitor',args,{**env,'JAX_PLATFORMS':'cuda,cpu'})
                active=p
                if p.wait()!=0:raise RuntimeError(f'fixed teacher G{i} validation failed; no automatic retry')
                active=None
                monitor_costs.append(dict(round=i,charged_interactions=read(out/'status.json')['charged_interactions'],
                    path=str(out/'report.json')))
                from .teacher_monitor import aggregate_reports
                aggregate_reports([x['path'] for x in monitor_costs],out.parent/f'aggregate_G{i}')
                done[-1]['teacher_monitor']=str(out/'report.json')
                atomic_json(root/'completed_rounds.json',done)
            status('running',current_round=i,current_stage='round_completed')
        status('completed',current_round=100)
    except BaseException as error:
        stop_owned_child(active,root)
        status('failed',error=repr(error));raise
