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
        gate=dict(kind='gpu_shared',minimum_free_mib=20000,authorization='user_authorized_new100'))
    baseline=read(d['baseline_frozen_policy'])['policy']
    protocol=digest(dict(version='1.2',task='stable_forward_recovery',H=16,horizon=400,
        source_physics=baseline['xml_sha256'],source_reward=runtime['reward_mode']))
    identity=dict(actor_sha256=baseline['actor_sha256'],normalizer_sha256=baseline['normalizer_sha256'],
        model_sha256=baseline['xml_sha256'],protocol_sha256=protocol)
    declaration_path=Path(d['campaign_output'])/'declaration.json'
    from .teacher_monitor import should_evaluate_round
    monitor_reservation=145600 if should_evaluate_round(round_index) else 0
    spec=dict(schema='jit_bridge_experimental_round_v1',protocol_version='1.2',execute=True,
        output=str(root),repository=d['repository'],implementation_commit=d['implementation_commit'],
        implementation_files=implementation_files(d['repository']),
        source_frozen_policy=actor['frozen_policy'],source=source,source_runtime=runtime,
        seed=d['seed']+round_index,round_index=round_index,series_id=Path(d['campaign_output']).name,
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
    return ClosedLoopRound(spec).run()


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
    if (root/'started.json').exists():
        raise ValueError('explicit recovery required; no automatic restart')
    if implementation_identity(repo)!=d['implementation_commit']:
        raise ValueError('implementation revision drift')
    for p,h in d['locks'].items():
        if file_sha(p)!=h:raise ValueError('locked initialization changed: '+p)
    atomic_json(root/'started.json',dict(pid=os.getpid(),started_unix=time.time()))
    env=dict(os.environ,PYTHONPATH=str(repo/'JIT/src'),JAX_PLATFORMS='cpu',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',JIT_AUTO_PUBLISH='0',
        JAX_COMPILATION_CACHE_DIR=d['compiler']['cache'],XLA_FLAGS=d['compiler']['XLA_FLAGS'],
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def spawn(name,args,environment=env):
        with (root/(name+'.log')).open('ab') as stream:
            p=subprocess.Popen(args,cwd=repo,env=environment,stdout=stream,stderr=stream,start_new_session=True)
        atomic_json(root/(name+'_pid.json'),dict(pid=p.pid,argv=args));return p
    py=sys.executable;done=[];previous=read(root/'initial_bundle.json');active=None
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
            inherited_generator_cost=previous.get('generator_lifetime_charged_updates'),
            optimization_units='G optimizer updates separate from physical control steps'))
        atomic_json(root/'status.json',dict(phase=phase,declared_rounds=100,completed_rounds=len(done),
            completed_student_transitions=len(done)*32000,student_transition_target=3200000,
            charged_interactions=physics,updated_unix=time.time(),**kwargs))
    monitoring=root/'monitoring';monitoring.mkdir()
    sources=[]
    for i in range(1,101):
        training=root/f'round_{i:04d}/students/student/training/round_{i:04d}_student'
        for kind,filename in [('loss','metrics.jsonl'),('reward','episode_metrics.jsonl')]:
            sources.append(dict(name=f'round{i}_{kind}',run='student_'+kind,path=str(training/filename),
                step='training_transitions',step_offset=(i-1)*32000,completion_status=str(training/'status.json')))
    sources.append(dict(name='outer_loop_metrics',run='outer_loop',path=str(monitoring/'round_scalars.jsonl'),step='round'))
    atomic_json(monitoring/'tensorboard_manifest.json',dict(sources=sources))
    spawn('exporter',[py,str(repo/'JIT/cli/export_tensorboard.py'),'--manifest',str(monitoring/'tensorboard_manifest.json'),
        '--output',str(monitoring/'events'),'--watch'])
    spawn('tensorboard',[py,'-m','tensorboard.main','--logdir',str(monitoring/'events'),'--host','0.0.0.0',
        '--port',str(d['tensorboard_port']),'--reload_interval','5'])
    atomic_json(root/'ACTIVE_RUN.json',dict(name=root.name,execution=str(root/'status.json'),lineage=str(root/'status.json')))
    status('starting')
    start_notifications(dict(output=str(root),repository=str(repo),protocol_version='teacher-recovery100'))
    try:
        for i in range(1,101):
            import shutil
            if shutil.disk_usage(root).free<20*1024**3:
                raise RuntimeError('free disk below20GiB margin; no artifact deletion')
            spec=round_spec(d,previous,i);rd=Path(spec['output']);status('running',current_round=i,current_stage='full_loop')
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
