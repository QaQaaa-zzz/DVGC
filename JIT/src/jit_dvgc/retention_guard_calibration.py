"""Bounded actual Adam calibration and GPU plumbing checks before a fresh pilot.

Only TRAIN data and locked DEV_RETENTION participate. The resulting thresholds
are observed finite-bank drift budgets; they do not establish global retention.
A distinct root-authorized reservation is required before any backend work.
"""
from pathlib import Path
import json
import math
import time
import numpy as np
from .generative_bridge.contracts import file_sha
from .generative_bridge.protocol import atomic_json

STEPS=[0,5,10,20]
PHYSICAL_STEPS=[0,20]
SELECTION_RULE='observed_update20_group_channel_drift_if_zero_old_losses_v1'
AUTH_SCHEMA='frontierbridge_calibration_authorization_v1'


def freeze_thresholds(candidates,groups):
    if [row.get('attempt') for row in candidates]!=STEPS:
        raise ValueError('fixed calibration schedule 0/5/10/20 required')
    usable=[]
    for row in candidates:
        metrics=row.get('guard',{}).get('groups',{})
        if set(metrics)!=set(groups):raise ValueError('all mandatory guard groups required')
        if row.get('finite') is not True:raise FloatingPointError('nonfinite calibration candidate')
        for group in groups:
            for key in ('mse','p99'):
                values=np.asarray(metrics[group][key],float)
                if values.shape!=(4,) or not np.isfinite(values).all() or np.any(values<0):
                    raise FloatingPointError('invalid finite calibration group/channel drift')
        if row['attempt']==20 and row.get('capabilities',{}).get('eligible') is True:usable.append(row)
    if not usable:raise ValueError('no qualifying nonzero calibration candidate; thresholds remain unresolved')
    return dict(mse_thresholds={g:np.max([r['guard']['groups'][g]['mse'] for r in usable],axis=0).tolist() for g in groups},
        p99_thresholds={g:np.max([r['guard']['groups'][g]['p99'] for r in usable],axis=0).tolist() for g in groups},
        qualified_attempts=[r['attempt'] for r in usable],threshold_rule=SELECTION_RULE,
        units={'mse':'normalized_action_squared_error','p99':'normalized_action_absolute_error'},
        zero_channel_budget='preserved_exactly_no_arbitrary_floor',
        evidence_boundary='finite TRAIN_GUARD behavior plus used DEV_RETENTION; no global retention or independent TEST claim')


def require_reservation(stage_plan,authorization):
    from .retention_consolidation import read,audit,_descriptor_json
    plan=read(stage_plan);grant=read(authorization)
    if (grant.get('schema')!=AUTH_SCHEMA or grant.get('stage_plan_sha256')!=file_sha(stage_plan)
            or grant.get('execute') is not True or not grant.get('user_authorization_record')
            or grant.get('selection_rule')!=SELECTION_RULE or grant.get('checkpoint_steps')!=STEPS
            or grant.get('physical_checkpoint_steps')!=PHYSICAL_STEPS
            or grant.get('calibration_attempts')!=20 or grant.get('max_supervised_attempts')!=22
            or not grant.get('output')):
        raise ValueError('actual authorization must bind calibration plan, fixed rule, attempts and fresh output')
    result=audit(stage_plan)
    forbidden=[x for x in result['unresolved'] if not (x=='guard_config' or x.startswith('guard_config.'))]
    if forbidden:raise ValueError('calibration inputs blocked: '+', '.join(forbidden))
    panel=_descriptor_json(plan['config']['evaluation_panel'])
    cases=[c for c in panel['cases'] if c['role'] in ('DEV_RETENTION','TRAIN_RETENTION') and c['mode']=='collect']
    from .retention_consolidation_runtime import evaluation_reservation
    expected=evaluation_reservation(cases)*len(PHYSICAL_STEPS)
    if type(grant.get('environment_transitions')) is not int or grant['environment_transitions']!=expected:
        raise ValueError('calibration physical reservation must equal fixed complete DEV schedule')
    if type(grant.get('wall_seconds')) not in (int,float) or not math.isfinite(grant['wall_seconds']) or grant['wall_seconds']<=0:
        raise ValueError('positive bounded calibration wall time required')
    if type(grant.get('tensorboard_port')) is not int or not 1024<=grant['tensorboard_port']<=65535:
        raise ValueError('explicit loopback calibration TensorBoard port required')
    return plan,grant,dict(cases=cases,split_records=[r for r in panel.get('split_records',[]) if r['role'] in ('DEV_RETENTION','TRAIN_RETENTION')])


def _gpu_micro(network,initializer,demo,fit,bank,thresholds,seed,output):
    import jax
    import jax.numpy as jp
    import optax
    import pickle
    from .generative_bridge.student import load_optional_demo
    from .generative_bridge.retention_guard import guarded_adam_step,require_finite
    from .policy_retention import action_mse
    from .handoff_bank import pytree_sha256
    norm,actor,critic=initializer
    def predict(p,x):return network.parametric_action_distribution.mode(network.policy_network.apply(norm,p,{'state':jp.asarray(x)}))
    predict=jax.jit(predict)
    obs,targets,weights=map(jp.asarray,load_optional_demo(demo));keep=jp.asarray(fit['observations'])
    key=jax.random.PRNGKey(seed);dk,kk=jax.random.split(jax.random.fold_in(key,1))
    ids=jax.random.choice(dk,len(obs),(256,),p=weights);kids=jax.random.choice(kk,len(keep),(256,),p=jp.asarray(fit['weights']))
    teacher=predict(actor,keep[kids])
    def loss(p):return action_mse(predict(p,obs[ids]),targets[ids])+action_mse(predict(p,keep[kids]),teacher)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-5));state=optimizer.init(actor)
    value,gradient=jax.jit(jax.value_and_grad(loss))(actor)
    delta,proposed=optimizer.update(gradient,state,actor);proposal=(value,gradient,delta,proposed)
    require_finite(proposal,'GPU micro proposal')
    candidate,new_state,accepted=guarded_adam_step(actor,state,None,predict,bank,optimizer,thresholds,proposal=proposal)
    zero=dict(mse_thresholds={g:[0.]*4 for g in thresholds['mse_thresholds']},p99_thresholds={g:[0.]*4 for g in thresholds['p99_thresholds']})
    rejected,old_state,denied=guarded_adam_step(actor,state,None,predict,bank,optimizer,zero,proposal=proposal)
    if not accepted['accepted'] or denied['accepted']:raise ValueError('actual GPU acceptance/rejection micro not demonstrated; stop before main pilot')
    if pytree_sha256(rejected)!=pytree_sha256(actor) or pytree_sha256(old_state)!=pytree_sha256(state):
        raise ValueError('GPU rejection mutated Actor or complete Adam state')
    payload=dict(actor=candidate,optimizer_state=new_state,normalizer=norm,critic=critic,rng=key,attempts=1,accepted_updates=1)
    path=Path(output)/'gpu_micro_learner.pkl'
    with path.open('xb') as f:pickle.dump(jax.device_get(payload),f)
    with path.open('rb') as f:loaded=pickle.load(f)
    if pytree_sha256(payload)!=pytree_sha256(loaded):raise ValueError('GPU micro learner roundtrip differs')
    before=np.asarray(predict(candidate,bank['observations']));after=np.asarray(predict(loaded['actor'],bank['observations']))
    if not np.array_equal(before,after):raise ValueError('deterministic GPU inference roundtrip differs')
    receipt=dict(phase='passed',backend=jax.default_backend(),physical_transitions=0,optimizer_proposals=1,
        guard_checks=2,accepted=accepted,rejected=denied,normalizer_sha256=pytree_sha256(norm),critic_sha256=pytree_sha256(critic),
        checkpoint=str(path),checkpoint_sha256=file_sha(path),state_sha256=pytree_sha256(loaded),
        closed_loop_validation='separate calibrated DEV receipts; micro itself is forward/update/serialization only')
    atomic_json(Path(output)/'gpu_micro.json',receipt)
    return receipt


def monitoring_pointer(root,grant):
    root=Path(root)
    lineage=Path(grant['overall_lineage_status']).resolve() if grant.get('overall_lineage_status') else root/'status.json'
    if grant.get('overall_lineage_status') and not lineage.is_file():
        raise ValueError('declared overall lineage status must exist before calibration launch')
    return dict(execution=str(root/'status.json'),lineage=str(lineage),name='FrontierBridge threshold calibration + GPU micro')


def run(stage_plan,authorization):
    """Foreground bounded run; caller must reserve global cost before dispatch."""
    plan,grant,panel=require_reservation(stage_plan,authorization)
    import jax
    if jax.default_backend()!='gpu' or jax.device_count()!=1:raise ValueError('actual calibration requires exactly one GPU')
    import pickle
    import copy
    import os
    import subprocess
    import urllib.request
    import socket
    from torch.utils.tensorboard import SummaryWriter
    from brax.training.acme import running_statistics as rs
    from .generative_bridge.worker import source_payload
    from .generative_bridge.retention_bank import load_retention_bank
    from .generative_bridge.retention_guard import guard_metrics
    from .generative_bridge.warmup import warmup_actor
    from .retention_consolidation import read,_descriptor_json
    from .retention_consolidation_runtime import Evaluator,_scalar_http,verify_notification_service
    from .retention_consolidation_evaluation import evaluate_capability_groups
    from .ppo import make_network_factory
    from .handoff_bank import pytree_sha256
    started=time.monotonic();root=Path(grant['output']).resolve();root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'authorization.json',dict(path=str(Path(authorization).resolve()),sha256=file_sha(authorization),grant=grant))
    atomic_json(root/'status.json',dict(phase='running',stage='calibration_preflight',attempts=0,environment_transitions=0))
    atomic_json(root/'cost_ledger.json',dict(eval_reserved_transitions=0,supervised_attempts=0,micro_optimizer_proposals=0,entries=[]))
    panel_path=root/'dev_retention_panel.json';atomic_json(panel_path,panel)
    p=copy.deepcopy(plan);p['output']=str(root);p['config']['evaluation_panel']=dict(path=str(panel_path),sha256=file_sha(panel_path))
    p['config']['cost_limits'].update(environment_transitions=grant['environment_transitions'],wall_seconds=grant['wall_seconds'])
    p['config']['tensorboard']=dict(host='127.0.0.1',port=grant['tensorboard_port'])
    atomic_json(root/'calibration_config.json',dict(plan=p,steps=STEPS,selection_rule=SELECTION_RULE,seed=p['config']['seed']))
    evaluator=Evaluator(p);writer=None;candidates=[]
    try:
        env=dict(os.environ,PYTHONPATH=str(Path(plan['repository'])/'JIT/src'),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        port=grant['tensorboard_port']
        with socket.socket() as sock:
            try:sock.bind(('127.0.0.1',port))
            except OSError as error:raise ValueError('calibration TensorBoard port already owned') from error
        atomic_json(root/'ACTIVE_RUN.json',monitoring_pointer(root,grant))
        verify_notification_service(root)
        with (root/'watcher.log').open('x') as log:
            watcher=subprocess.Popen([os.sys.executable,str(Path(plan['repository'])/'JIT/cli/watch_run_errors.py'),'--active-run',str(root/'ACTIVE_RUN.json'),'--state-dir',str(root/'notifications')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with (root/'tensorboard.log').open('x') as log:
            tb=subprocess.Popen([os.sys.executable,'-m','tensorboard.main','--host','127.0.0.1','--port',str(port),'--logdir',str(root/'tensorboard'),'--reload_interval','1'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(root/'monitoring.json',dict(watcher_pid=watcher.pid,tensorboard_pid=tb.pid,tensorboard_url=f'http://127.0.0.1:{port}'))
        healthy=False
        for _ in range(30):
            evaluator.check_clock()
            if watcher.poll() is not None or tb.poll() is not None:raise RuntimeError('calibration monitoring service exited')
            if (root/'notifications/notification_status.json').exists():
                try:urllib.request.urlopen(f'http://127.0.0.1:{port}',timeout=1).close();healthy=True;break
                except OSError:pass
            time.sleep(.5)
        if not healthy:raise RuntimeError('calibration watcher/TensorBoard health not verified')
        c=p['config'];policy,payload=source_payload(c['parent']['path']);initializer=(payload.observation_normalizer,payload.actor_params,payload.critic_params)
        network=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
        _,fit,_=load_retention_bank(c['data_roles']['TRAIN_FIT']['path'],allowed_roles=('TRAIN_FIT',))
        guard_manifest,bank,_=load_retention_bank(c['data_roles']['TRAIN_GUARD']['path'],allowed_roles=('TRAIN_GUARD',))
        demo=_descriptor_json(c['data_roles']['TEACHER']);groups=sorted(guard_manifest['group_weights'])
        ceiling=dict(mse_thresholds={g:[4.]*4 for g in groups},p99_thresholds={g:[2.]*4 for g in groups})
        def prediction(actor,x):return network.parametric_action_distribution.mode(network.policy_network.apply(initializer[0],actor,{'state':jax.numpy.asarray(x)}))
        prediction=jax.jit(prediction)
        if not np.array_equal(np.asarray(prediction(initializer[1],bank['observations'])),bank['reference_actions']):raise ValueError('actual parent guard labels drift')
        selection=copy.deepcopy(c['selection_config'])
        selection['required_groups']={g:dict(max_lost=0,max_rate_drop=0.) for g in groups}
        selection.update(mixed_policy='block',unknown_policy='block')
        writer=SummaryWriter(str(root/'tensorboard/calibration'));parent_results={}
        def checkpoint(attempt,params):
            evaluator.check_clock();path=root/'learner'/f'attempt_{attempt:04d}.pkl'
            metrics=guard_metrics(prediction,params[1],bank,ceiling)
            caps={};results={}
            if attempt in PHYSICAL_STEPS:
                results=evaluator.evaluate(f'calibration_{attempt:04d}',path,policy,c['parent'])
                if attempt==0:parent_results.update(results)
                rows=[dict(case_id=case['case_id'],condition_identity=case['condition_identity'],group=case['group'],role='DEV_RETENTION',declared_role=case['role'],
                    parent_labels=parent_results[case['case_id']]['labels'],pi0_labels=parent_results[case['case_id']]['labels'],student_labels=results[case['case_id']]['labels'],execution='independent_complete') for case in panel['cases']]
                caps=evaluate_capability_groups(rows,selection)
            candidates.append(dict(attempt=attempt,finite=True,closed_loop_evaluated=attempt in PHYSICAL_STEPS,capabilities=caps,guard=metrics,checkpoint=str(path),checkpoint_sha256=file_sha(path),results=results))
            atomic_json(root/'calibration_candidates.json',candidates)
            return attempt==0 and not caps['eligible']
        def metrics(row):
            evaluator.check_clock();evaluator.ledger['supervised_attempts']=row['attempt'];atomic_json(root/'cost_ledger.json',evaluator.ledger)
            for key in ('loss','demo','keep','actor_grad_norm'):writer.add_scalar('BC/'+key,row[key],row['attempt'])
            writer.add_scalar('BC/reward',-row['loss'],row['attempt']);writer.flush()
            atomic_json(root/'status.json',dict(phase='running',stage='calibration',attempts=row['attempt'],declared_attempts=20,eval_reserved_transitions=evaluator.ledger['eval_reserved_transitions']))
            if row['attempt']==1:_scalar_http(p,'calibration',1)
        warmup_actor(network,initializer,initializer[:2],demo,(fit['observations'],fit['weights']),root/'learner',seed=c['seed'],updates=20,batch_size=256,
            update_guard=dict(bank=bank,config=ceiling,identity=guard_manifest['identity'],enabled=False),checkpoint_steps=STEPS,checkpoint_callback=checkpoint,metrics_callback=metrics,actual_gradient_audit=True)
        if len(candidates)==1:raise ValueError('parent closed-loop preflight failed; zero calibration optimizer updates')
        frozen=freeze_thresholds(candidates,groups)
        receipt=dict(schema='frontierbridge_guard_calibration_v1',calibrated=True,data_role='TRAIN_GUARD',
            guard_bank_sha256=guard_manifest['sha256'],parent_actor_sha256=pytree_sha256(initializer[1]),
            cost_ledger=dict(path=str(root/'cost_ledger.json'),sha256=file_sha(root/'cost_ledger.json')),
            retention_gate_roles={case['case_id']:case['role'] for case in panel['cases']},
            seen_train_cases=[case['case_id'] for case in panel['cases'] if case['role']=='TRAIN_RETENTION'],
            candidate_thresholds=candidates,**frozen)
        atomic_json(root/'calibration_receipt.json',receipt)
        evaluator.check_clock()
        micro=_gpu_micro(network,initializer,demo,fit,bank,frozen,c['seed'],root)
        evaluator.ledger['micro_optimizer_proposals']=micro['optimizer_proposals'];atomic_json(root/'cost_ledger.json',evaluator.ledger)
        receipt['cost_ledger']['sha256']=file_sha(root/'cost_ledger.json');atomic_json(root/'calibration_receipt.json',receipt)
        atomic_json(root/'status.json',dict(phase='completed',stage='calibration_and_gpu_micro',attempts=20,eval_reserved_transitions=evaluator.ledger['eval_reserved_transitions'],wall_seconds=time.monotonic()-started))
        return receipt
    except BaseException as error:
        atomic_json(root/'status.json',dict(phase='failed',error=repr(error),attempts=evaluator.ledger.get('supervised_attempts',0),wall_seconds=time.monotonic()-started))
        raise
    finally:
        if writer is not None:writer.close()
