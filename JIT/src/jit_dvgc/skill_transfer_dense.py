"""Bounded dense query collection, one student update stage, independent checks."""
import json,pickle,time
from pathlib import Path
from .generative_bridge.protocol import atomic_json
from .generative_bridge.contracts import file_sha


def run_dense(config,output):
    import jax
    from tensorboardX import SummaryWriter
    from brax.training.acme import running_statistics as rs
    from .skill_transfer import audit
    from .skill_transfer_student_runtime import run_student_block
    from .skill_transfer_dual_teacher import merge_verified_labels
    from .recovery_acquisition_bc import warmup_recovery_actor
    from .ppo import make_network_factory
    from .skill_transfer_evaluation import evaluate_recovery
    from .retention_consolidation_runtime import Evaluator,evaluation_reservation
    c=json.loads(Path(config).read_text());audit(c)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    completed=c.get('completed_query_banks')
    if (root/'dense_status.json').exists() and not completed:raise FileExistsError('dense run already exists')
    if completed and len(completed)!=28:raise ValueError('all28 complete query blocks required')
    for key in ('student_initializer','recovery_inference','retention_panel'):
        if file_sha(c[key]['path'])!=c[key]['sha256']:raise ValueError('dense input identity drift: '+key)
    positions=(0,1,2,4,8,12,16);banks=[Path(p) for p in completed] if completed else [];started=time.time()
    query_writer=SummaryWriter(str(root/'tensorboard'))
    charged=0
    atomic_json(root/('finish_declaration.json' if completed else 'declaration.json'),dict(config=c,positions=list(positions),batches=4,
        teacher_transition_ceiling=940800,visitation_transition_ceiling=2408,BC_updates=2000,
        final_TEST_opened=False,automatic_adoption=False))
    def status(phase,**kw):atomic_json(root/'dense_status.json',dict(phase=phase,updated_unix=time.time(),**kw))
    try:
        for batch in (() if completed else range(4)):
            for position in positions:
                name=f'batch{batch:02d}_step{position:02d}';child=dict(c)
                child.update(student_initializer=c['student_initializer']['path'],student_old_offset=batch*7,
                    query_namespace=name+'_',strict_query_outcomes=True,allow_empty_queries=True,query_only=True)
                child.pop('previous_online_banks',None)
                cfg=root/(name+'_config.json');atomic_json(cfg,child)
                status('running',stage='queries',block=name,completed_blocks=len(banks),declared_blocks=28)
                run_student_block(cfg,root/name,c['recovery_inference']['path'],updates=0,visit_steps=position)
                banks.append(root/name/'labels.json')
                receipt=root/name/'query_receipts.json'
                q=json.loads(receipt.read_text()) if receipt.exists() else {}
                charged+=q.get('charged_interactions',0)
                query_writer.add_scalar('queries/completed_blocks',len(banks),len(banks))
                query_writer.add_scalar('queries/labeled_states',q.get('labeled_queries',0),len(banks))
                query_writer.add_scalar('queries/charged_interactions',charged,len(banks));query_writer.flush()
                # Drop per-env executable captures; disk compilation cache and
                # immutable inputs remain. Never retain28 full Warp allocators.
                jax.clear_caches()
        query_writer.close()
        manifest=merge_verified_labels(banks,root/'bank')
        with Path(c['student_initializer']['path']).open('rb') as f:initializer=pickle.load(f)
        network=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
        writer=SummaryWriter(str(root/'tensorboard'))
        def metrics(row):
            for key in ('loss','actor_grad_norm','actor_delta_norm'):writer.add_scalar('S_BC/'+key,row[key],row['update'])
            writer.add_scalar('S_BC/supervised_reward',-row['loss'],row['update']);writer.flush()
            status('running',stage='BC',completed_updates=row['update'],declared_updates=2000,verified_labels=manifest['count'])
        try:warmup_recovery_actor(network,initializer,manifest,root/'learner',updates=2000,seed=10101402,metrics_callback=metrics)
        finally:writer.close()
        checkpoint=root/'learner/update_2000.pkl'
        status('running',stage='independent_TRAIN',verified_labels=manifest['count'])
        train=evaluate_recovery(checkpoint,root/'TRAIN',c['pi0_descriptor'],c['teacher_manifest']['path'],c['source_rows']['path'],c['repository'])
        panel=json.loads(Path(c['retention_panel']['path']).read_text());target=root/'retention_DEV';target.mkdir()
        plan=dict(output=str(target),repository=c['repository'],config=dict(evaluation_panel=c['retention_panel'],
            cost_limits=dict(wall_seconds=86400,environment_transitions=evaluation_reservation(panel['cases']))))
        atomic_json(target/'plan.json',plan);status('running',stage='independent_retention_DEV')
        from .generative_bridge.worker import source_payload
        policy,_=source_payload(c['pi0_descriptor']['path'])
        Evaluator(plan).evaluate('panel',checkpoint,policy,c['pi0_descriptor'])
        status('completed',verified_labels=manifest['count'],TRAIN_stable_successes=train['stable_successes'],
            full_TRAIN_denominator=8,wall_seconds=time.time()-started,adopted=False,final_TEST_opened=False)
    except BaseException as exc:
        query_writer.close()
        status('error',error=repr(exc),wall_seconds=time.time()-started)
        raise
