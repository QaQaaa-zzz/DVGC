"""Authorized single-device BC execution and unchanged pulse replay backend."""
from pathlib import Path
import copy
import json
import os
import pickle
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import numpy as np
from .retention_consolidation import read, _descriptor_json, report
from .generative_bridge.contracts import file_sha
from .generative_bridge.protocol import atomic_json


def validate_collection(spec):
    if (spec.get('full_episode_rollout') is not True or spec.get('controller_mode')!='fixed_random'
            or (not spec.get('frozen_request_table') and np.any(np.asarray(spec.get('delta_limit',1))!=0)) or spec.get('explorer_checkpoint')
            or spec.get('episode_policy_mixture') or spec.get('reuse_prefix_collection')):
        raise ValueError('independent collection requires complete deterministic Actor and zero learned exploration residual')
    if spec.get('frozen_request_table'):
        from .retention_repair import load_requests
        load_requests(spec)


def validate_collection_requests(spec,tape):
    """External frozen requests may perturb the Actor; never accept learned pulses."""
    mask=np.asarray(tape['prefix_mask'],bool)
    actual=np.asarray(tape['requested_delta'])
    if actual.shape!=mask.shape+(4,) or not np.isfinite(actual).all():
        raise ValueError('invalid actual requested action tape')
    if spec.get('frozen_request_table'):
        from .retention_repair import load_requests
        expected,_=load_requests(spec)
        if mask.shape[1]!=expected.shape[1] or len(mask)>len(expected):raise ValueError('frozen request layout differs')
        if not np.array_equal(actual[mask],expected[:len(mask)][mask]):
            raise ValueError('actual requests differ from fixed frozen condition')
    elif np.any(actual[mask]!=0):raise ValueError('independent collection contains learned/random residual')


def evaluation_reservation(cases):
    """Full physical reserve for unique batched jobs; repeat indices never alias."""
    jobs={}
    for case in cases:
        for repeat,seed in enumerate(case['repeat_seeds']):
            spec=copy.deepcopy(case['spec']);spec['seed']=seed
            key=(case['mode'],repeat,json.dumps(spec,sort_keys=True))
            cost=case['capacity']*case['horizon']
            if key in jobs and jobs[key]!=cost:raise ValueError('shared job reservation layout differs')
            jobs[key]=cost
    return sum(jobs.values())


class Evaluator:
    def __init__(self,p):
        self.p=p;self.root=Path(p['output']);self.started=time.monotonic()
        self.deadline=self.started+p['config']['cost_limits']['wall_seconds']
        self.env=dict(os.environ,PYTHONPATH=str(Path(p['repository'])/'JIT/src'),
            JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',JIT_AUTO_PUBLISH='0',
            JAX_COMPILATION_CACHE_DIR=os.environ.get('JAX_COMPILATION_CACHE_DIR',str(Path.home()/'.cache/jax')))
        path=self.root/'cost_ledger.json'
        self.ledger=read(path) if path.exists() else dict(eval_reserved_transitions=0,entries=[])

    def check_clock(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('new-stage wall budget exhausted')

    def child(self,name,spec,mode,reservation):
        self.check_clock()
        if self.ledger['eval_reserved_transitions']+reservation>self.p['config']['cost_limits']['environment_transitions']:
            raise ValueError('evaluation budget exhausted before dispatch')
        entry=dict(stage=name,reserved_transitions=reservation,phase='running')
        self.ledger['entries'].append(entry);self.ledger['eval_reserved_transitions']+=reservation
        atomic_json(self.root/'cost_ledger.json',self.ledger)
        path=self.root/(name+'_spec.json');atomic_json(path,spec)
        from .retention_b import bounded_worker
        with (self.root/(name+'.log')).open('x') as log:
            try:
                bounded_worker([sys.executable,str(Path(self.p['repository'])/'JIT/cli/run_pulse_exploration.py'),
                    '--mode',mode,'--spec',str(path),'--output',str(self.root/name)],
                    cwd=self.p['repository'],env=self.env,stdout=log,timeout=max(1,self.deadline-time.monotonic()))
                receipt=read(self.root/name/'status.json')
                charged=receipt.get('charged_interactions')
                if receipt.get('phase')!='completed' or type(charged) is not int or not 0<=charged<=reservation:
                    raise ValueError('invalid evaluation cost/completion receipt')
                entry.update(phase='completed',charged_transitions=charged,active_transitions=receipt.get('active_interactions','UNKNOWN'),
                    padding_transitions=receipt.get('padding_interactions','UNKNOWN'))
            except BaseException as e:
                entry.update(phase='error',error=repr(e),charged_transitions=reservation)
                raise
            finally:atomic_json(self.root/'cost_ledger.json',self.ledger)
        return self.root/name

    def reference_spec(self,spec,policy,descriptor):
        from .probe_bank import load_probe_bank,lock_probe_bank
        source=load_probe_bank(Path(spec['bank']))
        from .generative_bridge.contracts import digest
        path=self.root/('reference_bank_'+file_sha(descriptor['path'])[:16]+'_'+digest(source)[:16]+'.json')
        if not path.exists():
            bank_spec={key:source[key] for key in ('task','max_ticks','label_interaction_budget','max_candidates_per_process')}
            bank_spec.update(version='retention_reference_'+file_sha(descriptor['path'])[:16],
                members=[dict(frozen_policy=descriptor['path'],roles=['proposer','evaluator'])])
            lock_probe_bank(bank_spec,path)
        result=copy.deepcopy(spec);result.update(bank=str(path),proposer=policy['name'],order=[policy['name']])
        return result

    def evaluate(self,tag,params_path,policy,reference_descriptor=None,*,case_ids=None):
        from .retention_consolidation_evaluation import (build_independent_replay_spec,
            independent_replay_label,independent_collection_label)
        panel=_descriptor_json(self.p['config']['evaluation_panel']);results={};jobs={}
        if case_ids is not None:
            if not case_ids or not set(case_ids)<=set(c['case_id'] for c in panel['cases']):
                raise ValueError('explicit existing nonempty evaluation subset required')
            panel=dict(panel,cases=[c for c in panel['cases'] if c['case_id'] in case_ids])
        initializer=dict(path=str(params_path),sha256=file_sha(params_path),
            source_actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256'])
        for ordinal,case in enumerate(panel['cases']):
            labels=[];receipts=[]
            for repeat,seed in enumerate(case['repeat_seeds']):
                spec=copy.deepcopy(case['spec']);spec['seed']=seed
                if reference_descriptor is None:raise ValueError('explicit parent/pi0 replay descriptor required')
                spec=self.reference_spec(spec,policy,reference_descriptor)
                if case['mode']=='evaluate':
                    spec=build_independent_replay_spec(spec,spec['candidates'],initializer)
                else:
                    validate_collection(spec);spec['warmup_initializer']=initializer
                name=f'{tag}_case{ordinal:03d}_repeat{repeat}'
                job=(case['mode'],repeat,json.dumps(spec,sort_keys=True))
                if job not in jobs:jobs[job]=self.child(name,spec,case['mode'],case['capacity']*case['horizon'])
                out=jobs[job]
                if case['mode']=='evaluate':
                    rows=read(out/'results.json');label=independent_replay_label(rows[case['lane']])
                else:
                    with np.load(out/'prefixes.npz',allow_pickle=False) as tape:
                        label=independent_collection_label(tape,case['lane'],case['horizon'])
                        validate_collection_requests(spec,tape)
                labels.append(label);receipts.append(str(out))
            results[case['case_id']]=dict(labels=labels,receipts=receipts)
        atomic_json(self.root/(tag+'_evaluation.json'),results)
        return results


def _rows(panel,parent,pi0,student,*,diagnostic=False):
    rows=[dict(case_id=c['case_id'],condition_identity=c['condition_identity'],group=c['group'],role=c['role'],
        parent_labels=parent[c['case_id']]['labels'],pi0_labels=pi0[c['case_id']]['labels'],
        student_labels=student[c['case_id']]['labels'],execution='independent_complete',
        teacher_repeat_labels=c.get('teacher_repeat_labels',[None]*3),seen_condition_validation=c.get('seen_condition_validation')) for c in panel['cases']]
    if diagnostic:
        for row in rows:
            if len(row['parent_labels'])!=len(row['pi0_labels']):raise ValueError('reference repeat provenance differs')
            row.update(reference_expected_repeats=len(row['parent_labels']),student_expected_repeats=len(row['student_labels']))
    return rows


def _scalar_http(p,arm,step):
    """Require current run reward, not merely a live server/event file."""
    port=p['config']['tensorboard']['port'];tag='BC/reward'
    url=f'http://127.0.0.1:{port}/data/plugin/scalars/scalars?'+urllib.parse.urlencode({'run':arm,'tag':tag})
    for _ in range(15):
        try:
            values=json.loads(urllib.request.urlopen(url,timeout=2).read())
            if any(int(v[1])==step for v in values):
                atomic_json(Path(p['output'])/(arm+'_tensorboard_verified.json'),dict(url=url,step=step,values=values));return
        except (OSError,ValueError):pass
        time.sleep(1)
    raise RuntimeError('current stage reward scalar not loaded in TensorBoard')


def train_stage(plan):
    import jax
    if jax.default_backend()!='gpu' or jax.device_count()!=1:raise ValueError('production stage requires exactly one GPU; multi-device guard unsupported')
    from brax.training.acme import running_statistics as rs
    from torch.utils.tensorboard import SummaryWriter
    from .ppo import make_network_factory
    from .generative_bridge.worker import source_payload
    from .generative_bridge.retention_bank import load_retention_bank
    from .generative_bridge.warmup import warmup_actor
    from .generative_bridge.learning_audit import action_probe
    from .handoff_bank import pytree_sha256
    from .retention_consolidation_evaluation import evaluate_capability_groups
    p=read(plan);c=p['config'];root=Path(p['output']);evaluator=Evaluator(p)
    from importlib import metadata
    versions={}
    for package in ('jax','jaxlib','brax','optax','numpy'):
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:versions[package]='NOT_INSTALLED'
    atomic_json(root/'runtime_environment.json',dict(python=sys.version,versions=versions,backend=jax.default_backend(),
        devices=[dict(id=d.id,platform=d.platform,device_kind=d.device_kind) for d in jax.devices()],
        jax_enable_x64=bool(jax.config.jax_enable_x64),default_matmul_precision=str(jax.config.jax_default_matmul_precision),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','UNSET'),JAX_PLATFORMS=os.environ.get('JAX_PLATFORMS','UNSET')))
    policy,payload=source_payload(c['parent']['path']);pi0_policy,pi0_payload=source_payload(c['pi0_reference']['path'])
    initializer=(payload.observation_normalizer,payload.actor_params,payload.critic_params)
    pi0=(pi0_payload.observation_normalizer,pi0_payload.actor_params,pi0_payload.critic_params)
    network=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
    _,fit,_=load_retention_bank(c['data_roles']['TRAIN_FIT']['path'],allowed_roles=('TRAIN_FIT',))
    bank_manifest,bank,_=load_retention_bank(c['data_roles']['TRAIN_GUARD']['path'],allowed_roles=('TRAIN_GUARD',))
    demo=_descriptor_json(c['data_roles']['TEACHER'])
    # Both arms access identical data; all frozen labels must use this parent.
    reference_norm,reference_actor=jax.tree.map(jax.lax.stop_gradient,initializer[:2])
    @jax.jit
    def prediction(obs):
        return network.parametric_action_distribution.mode(network.policy_network.apply(reference_norm,reference_actor,{'state':jax.numpy.asarray(obs)}))
    for b in (fit,bank):
        if not np.array_equal(np.asarray(prediction(b['observations'])),b['reference_actions']):raise ValueError('retention labels do not match fixed parent')
    panel=_descriptor_json(c['evaluation_panel'])
    def store(name,params):
        path=root/(name+'.pkl')
        with path.open('xb') as f:pickle.dump(jax.device_get(params),f)
        return path
    parent_path=store('parent',initializer);pi0_path=store('pi0',pi0)
    diagnostic=c['selection_config'].get('diagnostic_training') is True
    if diagnostic:
        from .retention_guard_calibration import load_baseline_reference_cache
        parent_eval,reuse=load_baseline_reference_cache(p,c,panel)
        atomic_json(root/'P0_reference_reuse.json',reuse)
        fresh_ids=[case['case_id'] for case in panel['cases'] if case['case_id'] not in parent_eval]
        if fresh_ids:parent_eval.update(evaluator.evaluate('P0',parent_path,policy,c['parent'],case_ids=fresh_ids))
        atomic_json(root/'P0_evaluation.json',parent_eval)
    else:parent_eval=evaluator.evaluate('P0',parent_path,policy,c['parent'])
    if (file_sha(c['parent']['path'])==file_sha(c['pi0_reference']['path']) and pytree_sha256(initializer)==pytree_sha256(pi0)):
        pi0_eval=copy.deepcopy(parent_eval)
        atomic_json(root/'pi0_reference_evaluation.json',pi0_eval)
        evaluator.ledger['pi0_reference_reuse']=dict(source='P0',reason='identical frozen descriptor and complete initializer trees',new_transitions=0)
        atomic_json(root/'cost_ledger.json',evaluator.ledger)
    else:pi0_eval=evaluator.evaluate('pi0_reference',pi0_path,pi0_policy,c['pi0_reference'])
    parent_rows=_rows(panel,parent_eval,pi0_eval,parent_eval,diagnostic=diagnostic)
    preflight=evaluate_capability_groups(parent_rows,c['selection_config'])
    atomic_json(root/'parent_preflight.json',preflight)
    if not preflight['continuation_eligible' if diagnostic else 'eligible']:raise ValueError('parent preflight has missing/unstable mandatory capability groups; zero optimizer updates')
    arms={};evaluator.ledger['supervised_environment_transitions']=0
    candidates={};atomic_json(root/'checkpoint_results.json',candidates)
    for arm in ('P1','P2'):
        candidates[arm]=[];writer=SummaryWriter(str(root/'tensorboard'/arm));count={}
        def metrics(row):
            evaluator.check_clock();count.update(attempts=row['attempt'],accepted=row['accepted_updates'])
            evaluator.ledger[arm]=dict(attempts=row['attempt'],accepted=row['accepted_updates'],rejected=row['rejected_updates'])
            evaluator.ledger['supervised_attempts']=sum(evaluator.ledger.get(a,{}).get('attempts',0) for a in ('P1','P2'))
            evaluator.ledger['accepted_updates']=sum(evaluator.ledger.get(a,{}).get('accepted',0) for a in ('P1','P2'))
            atomic_json(root/'cost_ledger.json',evaluator.ledger)
            # Reward here is the negative supervised objective, explicitly named.
            for key in ('loss','demo','keep','accepted_updates','rejected_updates','actor_grad_norm','accepted_delta_norm'):
                if key in row:writer.add_scalar('BC/'+key,row[key],row['attempt'])
            writer.add_scalar('BC/reward',-row['loss'],row['attempt']);writer.flush()
            if row['attempt']==1:_scalar_http(p,arm,1)
            atomic_json(root/'status.json',dict(phase='running',stage=arm,attempted_updates=row['attempt'],accepted_updates=row['accepted_updates'],
                declared_attempts=c['max_attempted_updates'],last_update_unix=time.time(),tensorboard_url=f"http://127.0.0.1:{c['tensorboard']['port']}",published_policy=None))
        def checkpoint(attempt,params):
            evaluator.check_clock()
            if pytree_sha256(params[0])!=pytree_sha256(initializer[0]) or pytree_sha256(params[2])!=pytree_sha256(initializer[2]):raise ValueError('frozen parent statistics/critic changed')
            path=root/arm/f'attempt_{attempt:04d}.pkl'
            ev=parent_eval if attempt==0 else evaluator.evaluate(f'{arm}_{attempt:04d}',path,policy,c['parent'])
            rows=_rows(panel,parent_eval,pi0_eval,ev,diagnostic=diagnostic)
            candidates[arm].append(dict(checkpoint=str(path),attempt=attempt,accepted_updates=count.get('accepted',0),
                identity_valid=True,data_valid=True,numerical_valid=True,rows=rows))
            atomic_json(root/'checkpoint_results.json',candidates)
            result=evaluate_capability_groups(rows,c['selection_config'])
            atomic_json(root/f'{arm}_{attempt:04d}_capabilities.json',result)
            writer.add_scalar('DEV/independent_success',result['expansion']['independent_success'],attempt)
            writer.add_scalar('DEV/old_lost',result['total_old_lost'],attempt);writer.flush()
            return not result['continuation_eligible' if diagnostic else 'eligible']
        try:
            warmup_actor(network,initializer,initializer[:2],demo,(fit['observations'],fit['weights']),root/arm,
                seed=c['seed'],updates=c['max_attempted_updates'],batch_size=256,demo_coefficient=1.,keep_coefficient=1.,
                update_guard=dict(bank=bank,config=c['guard_config'],identity=bank_manifest['identity'],enabled=arm=='P2'),
                checkpoint_steps=c['checkpoint_steps'],checkpoint_callback=checkpoint,metrics_callback=metrics,
                probe_callback=lambda attempt,params:action_probe(network,params[0],params[1],demo,reference=initializer[:2]),actual_gradient_audit=True)
        finally:
            writer.close()
            status_path=root/arm/'warmup_status.json'
            if status_path.exists():
                s=read(status_path);arms[arm]=dict(training_status=s['status'],attempts=s['attempts'],accepted_updates=s['accepted_updates'],rejected_updates=s['rejected_updates'])
                evaluator.ledger[arm]=dict(attempts=s['attempts'],accepted=s['accepted_updates'],wall_seconds=s.get('wall_seconds'))
                evaluator.ledger['supervised_attempts']=sum(evaluator.ledger.get(a,{}).get('attempts',0) for a in ('P1','P2'))
                evaluator.ledger['accepted_updates']=sum(evaluator.ledger.get(a,{}).get('accepted',0) for a in ('P1','P2'))
            atomic_json(root/'cost_ledger.json',evaluator.ledger)
    atomic_json(root/'status.json',dict(phase='completed',stage='P0_P1_P2',arms=arms,published_policy=None,automatic_next_stage=False,
        diagnostic_training=diagnostic,closed_loop_qualified=False if diagnostic else 'SEE_DEV_GATES',
        automatic_adoption=False,nominal_retention_status='UNVERIFIED_BASELINE_MIXED' if diagnostic else 'SEE_DEV_GATES'))
    return report(plan)



def verify_notification_service(root):
    """Authorized startup probe: command success never proves popup visibility."""
    root=Path(root)
    try:
        completed=subprocess.run(['notify-send','--app-name=JIT','--urgency=normal',
            'FrontierBridge monitoring active','Error and completion watcher is starting.'],
            check=True,capture_output=True,text=True,timeout=10)
        receipt=dict(command_returncode=completed.returncode,checked_unix=time.time(),user_saw_popup='UNKNOWN')
    except (OSError,subprocess.SubprocessError) as error:
        atomic_json(root/'notification_service_verified.json',dict(phase='error',error=repr(error),user_saw_popup='UNKNOWN'))
        raise RuntimeError('desktop notification service startup check failed') from error
    atomic_json(root/'notification_service_verified.json',receipt)
    return receipt


def launch(plan,authorization):
    p=read(plan);root=Path(p['output']);env=dict(os.environ,PYTHONPATH=str(Path(p['repository'])/'JIT/src'),
        JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    import socket
    port=p['config']['tensorboard']['port']
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
        try:probe.bind(('127.0.0.1',port))
        except OSError as error:raise ValueError('TensorBoard port already occupied; no services launched') from error
    atomic_json(root/'ACTIVE_RUN.json',dict(execution=str(root/'status.json'),lineage=str(root/'status.json'),name='FrontierBridge deterministic retention P0/P1/P2'))
    with (root/'watcher.log').open('x') as f:
        watcher=subprocess.Popen([sys.executable,str(Path(p['repository'])/'JIT/cli/watch_run_errors.py'),
            '--active-run',str(root/'ACTIVE_RUN.json'),'--state-dir',str(root/'notifications')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    port=p['config']['tensorboard']['port']
    with (root/'tensorboard.log').open('x') as f:
        tb=subprocess.Popen([sys.executable,'-m','tensorboard.main','--logdir',str(root/'tensorboard'),
            '--host','127.0.0.1','--port',str(port),'--reload_interval','1'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(root/'launch.json',dict(watcher_pid=watcher.pid,tensorboard_pid=tb.pid,plan=str(plan),
        tensorboard_url=f'http://127.0.0.1:{port}'))
    try:
        # A heartbeat alone is engineering health, not proof the user saw a popup.
        healthy=False
        for _ in range(20):
            if watcher.poll() is not None or tb.poll() is not None:raise RuntimeError('watcher/TensorBoard exited during launch')
            heartbeat=root/'notifications/notification_status.json'
            if heartbeat.exists():
                try:
                    state=read(heartbeat)
                    if state.get('pid')!=watcher.pid or not 0<=time.time()-state.get('checked_unix',0)<=10 or state.get('delivery_errors'):raise ValueError('stale watcher heartbeat')
                    urllib.request.urlopen(f'http://127.0.0.1:{port}',timeout=2).close();healthy=True;break
                except (OSError,ValueError):pass
            time.sleep(.5)
        if not healthy:raise RuntimeError('watcher heartbeat not verified')
        verify_notification_service(root)
        atomic_json(root/'status.json',dict(phase='running',stage='preflight',pid=os.getpid(),published_policy=None))
        from .retention_b import bounded_worker
        with (root/'worker.log').open('x') as f:
            bounded_worker([sys.executable,str(Path(p['repository'])/'JIT/cli/run_retention_consolidation.py'),
                'worker','--plan',str(plan),'--execute','--authorization',str(authorization)],cwd=p['repository'],
                env=env,stdout=f,timeout=p['config']['cost_limits']['wall_seconds'])
        return report(plan)
    except BaseException as e:
        atomic_json(root/'status.json',dict(phase='failed',error=repr(e),published_policy=None));raise
