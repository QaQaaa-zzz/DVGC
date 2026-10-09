"""Explicit metadata-only recovery of a completed final G worker shutdown abort.

Never executes a worker, repeats optimization, edits the failed series, or waives
normal continuation validation. Dry runs build a disposable publication view.
"""
from pathlib import Path
import tempfile
from .contracts import digest, file_sha
from .protocol import atomic_json
from .integration_audit import Evidence, require, audit_generator, audit_learner
from .continuation_boundary import resolve_completed_boundary, _resolve_completed_boundary, _assert_not_live


def validate_exit(status, log):
    stages=status.get('stages',[])
    require(status.get('phase')=='failed' and len(stages)==1 and
            stages[0].get('returncode')==-6,'not a single SIGABRT worker exit')
    require('Fatal Python error: PyInterpreterState_Delete: remaining subinterpreters' in log
            and 'Python runtime state: finalizing' in log,'not the recognized CPython finalization abort')


def _inspect(original):
    root=Path(original).resolve();e=Evidence();spec=e.read(root/'production.json')
    series=root.parent;plan=e.read(series/'plan.json');status=e.read(series/'status.json')
    require(status['phase']=='failed','original series is not failed')
    _assert_not_live(series,e.lock)
    completed=e.read(series/'completed_rounds.json')
    require(len(completed)==status['completed_rounds'] and root.name==f"round_{plan['round_offset']+len(completed)+1:04d}",
            'recovery must be the next unpublished round')
    require(not (root/'current_source.json').exists() and not (root/'stages/generator_selection.json').exists(),
            'round already published or generator stage already committed')
    boundary=resolve_completed_boundary(series,allow_stopped_parent=True)
    require(spec['continuation']==boundary['bundle'],'round continuation differs from published preceding round')
    e.locks.update(boundary['locks'])
    require(spec['round_index']==boundary['bundle']['round']+1 and Path(spec['output']).resolve()==root,
            'same-round production identity mismatch')
    require(spec.get('continuous_learning') is True and spec.get('generator_update_policy')=='last_valid',
            'recovery only supports continuous last_valid rounds')
    for path,sha in spec['locks'].items():e.lock(path,sha)
    execution=root/'execution/generator_incremental_0000'
    executed=e.read(execution/'status.json');log=e.lock(execution/'generator_incremental_0000.log').read_text()
    validate_exit(executed,log)
    cfg_path=root/'generator_incremental_0000_config.json';cfg=e.read(cfg_path)
    require(executed['stages'][0]['argv'][-2:]==['--spec',str(cfg_path)],'worker executed another config')
    execution_plan=e.read(executed['plan_path'],executed['plan_sha256'])
    require(execution_plan==e.read(execution/'plan.json'),'execution plan mismatch')
    contract=e.read(root/'generator_incremental/contract.json')
    require(contract['corpus_sha256']==cfg['corpus']['sha256'] and contract['max_updates']==cfg['updates']==2000
            and contract['generator_update_policy']==cfg['generator_update_policy']=='last_valid'
            and contract['charged_updates_before']==cfg['charged_updates_before'], 'generator contract mismatch')
    result=e.read(root/'incremental_result.json')
    require(result==e.read(root/'generator_incremental/completed.json'),'worker result/completion mismatch')
    require(cfg['mode']=='incremental' and cfg['seed']==spec['seed'] and cfg['incumbent']==spec['continuation']['generator']
            and cfg['generator_reference_frozen_policy']==spec['generator_reference_frozen_policy']
            and cfg['charged_updates_before']==spec['generator_charged_updates_before']
            and cfg['output']==str(root/'generator_incremental') and cfg['result']==str(root/'incremental_result.json'),
            'worker config does not belong to this round')
    require(Path(result['checkpoint_manifest']).resolve()==root/'generator_incremental/attempt_0000/update_2000/manifest.json',
            'generator checkpoint not final same-round attempt')
    cost=e.read(root/'generator_incremental/attempt_0000/cost_receipt.json')
    require(cost['status']=='completed' and cost['charged_updates']==result['updates']==result['charged_updates']==2000,
            'incomplete generator budget receipt')
    costs=e.read(root/'costs.json');g=[c for c in costs if c['stage'].startswith('generator_')]
    require(len(g)==1 and g[0]['stage']=='generator_incremental_0000' and g[0]['charged_updates']==2000
            and g[0]['phase']=='failed' and all(c['phase']=='completed' for c in costs if c not in g),
            'unresolved or inconsistent stage costs')
    require(e.read(root/'stages/round_contract.json')=={'contract':spec,'sha256':digest(spec)},'round contract mismatch')
    for path in (root/'stages').glob('*.json'):
        if path.name=='round_contract.json':continue
        require(not path.name.endswith('.running.json'),'unresolved running stage')
        stage=e.read(path);require(stage['output_sha256']==digest(stage['result']),'stage output changed')
    errors=list((root/'stages/errors').glob('generator_selection_*.json'))
    expected_inputs={k:cfg[k] for k in ('corpus','incumbent','generator_update_policy','generator_reference_frozen_policy')}
    require(len(errors)==1 and e.read(errors[0])['input_sha256']==digest({'round':digest(spec),'inputs':expected_inputs}),
            'uncommitted generator input identity mismatch')
    # Lock all same-round metadata used by reconstructed publication, including
    # original failed status and stage error receipts. Large physical files are
    # validated transitively by the existing continuation validators.
    for path in root.rglob('*.json'):e.lock(path)
    e.read(root/'status.json')
    return root,spec,plan,boundary,e,cfg,result


def _build(root,spec,parent_plan,boundary,e,cfg,result,output,original_started_unix):
    from .closed_loop import assemble_completed_bundle, continuous_decision
    from .artifacts import load_corpus
    from .rewards import feedback
    output=Path(output).resolve()
    require(not output.is_relative_to(root.parent),'recovery output must be outside immutable original series')
    output.mkdir(exist_ok=False)
    pending=output/'publication_pending.json'
    atomic_json(pending,dict(original_round=str(root),phase='validating'))
    target=output/root.name;target.mkdir();(target/'stages').mkdir()
    excluded={'stages','status.json','costs.json','cost_ledger.json','started.json','current_source.json','tail_adoption.json'}
    for path in root.iterdir():
        if path.name not in excluded:(target/path.name).symlink_to(path,target_is_directory=path.is_dir())
    for path in (root/'stages').glob('*.json'):(target/'stages'/path.name).symlink_to(path)
    def read(name):return e.read(root/name)
    phase=read('source_phase_result.json');student=read('stages/train_student.json')['result']
    decision=read('acceptance.json');training=e.read(Path(student['training'])/'status.json')
    require(decision==continuous_decision(student,training['status']=='completed'),'student adoption decision mismatch')
    rows=read('source_rows.json')['rows'];after=[]
    for path in sorted((root/'stages').glob('eval_student_train*.json')):
        receipt=e.read(path)['result'];after.extend(e.read(receipt['path'],receipt['sha256']))
    require(len(rows)==len(after) and [r['root_id'] for r in rows]==[r['root_id'] for r in after],
            'student evaluation root/order mismatch')
    teachers={r['root_id']:r for r in [e.read(p) for p in sorted((root/'teachers').glob('*_result.json'))]}
    recorded=read('explorer_feedback.json');source=spec['source'];previous=spec['continuation']
    from .closed_loop import feedback_rows
    require(recorded['rows']==feedback_rows(rows,after,teachers,decision),'explorer feedback row mismatch')
    reward,eligible,cells,parts=feedback(recorded['rows'],previous.get('arrival_ledgers',{}).get(source['actor_sha256'],[]))
    require(recorded['rewards']==reward.tolist() and recorded['eligible']==eligible.tolist(),'explorer feedback mismatch')
    corpus_receipt=cfg['corpus'];require(Path(corpus_receipt['path']).resolve()==root/'feedback_corpus/manifest.json','cross-round corpus')
    corpus=load_corpus(corpus_receipt)
    bundle=assemble_completed_bundle(target,spec,source,spec['baseline_identity']['protocol_sha256'],
        student=student,decision=decision,updated_g=result,updated_e=root/'explorer_update',corpus_receipt=corpus_receipt,
        corpus=corpus,phase=phase,support=root/'training_support.json',stress=read('student_stress.json'),
        e_metrics=read('explorer_update/metrics.json'),cells=cells,rows=rows,after=after,teachers=teachers,aggregate=root/'source_rows.json')
    generator_audit=audit_generator(root,previous,bundle,e)
    learner_audit=audit_learner(previous,bundle,False,False,e)
    inputs={k:cfg[k] for k in ('corpus','incumbent','generator_update_policy','generator_reference_frozen_policy')}
    atomic_json(target/'stages/generator_selection.json',dict(input_sha256=digest({'round':digest(spec),'inputs':inputs}),
        output_sha256=digest(result),result=result))
    # All original costs, including this physically completed round, already
    # enter the failed parent's lifetime total. This container adds zero cost.
    atomic_json(target/'costs.json',[])
    for path in (output/'current_source.json',target/'current_source.json'):atomic_json(path,bundle)
    plan={**parent_plan,'output':str(output),'rounds':1,'round_offset':spec['round_index']-1,
          'continuation_parent':str(root.parent),'prior_physics_charged':boundary['prior_physics_charged'],
          'carried_physics':0,'original_started_unix':original_started_unix,
          'publication_only':True,'new_training_authorized':False,'locks':dict(e.locks)}
    atomic_json(output/'plan.json',plan);atomic_json(output/'completed_rounds.json',[str(target)])
    receipt=dict(schema='jit_finalization_publication_recovery_v1',original_round=str(root),original_series=str(root.parent),
        original_execution_phase='failed',original_returncode=-6,new_physics=0,new_student_transitions=0,new_generator_updates=0,
        recovered_student_transitions=128000,recovered_generator_updates=2000,original_declared_budgets=parent_plan['budgets'],
        already_charged_physics=boundary['prior_physics_charged'],prior_optimization_costs=boundary['prior_optimization_costs'],
        generator_audit=generator_audit,learner_audit=learner_audit,original_started_unix=original_started_unix,locks=dict(e.locks))
    atomic_json(output/'recovery_receipt.json',receipt)
    atomic_json(target/'status.json',dict(phase='completed',publication_only=True,original_execution_phase='failed'))
    atomic_json(output/'status.json',dict(phase='completed',completed_rounds=1,declared_rounds=1,publication_only=True,
        lifetime_physics_charged=boundary['prior_physics_charged']))
    try:
        validated=_resolve_completed_boundary(output,pending_root=output)
        from .generator_costs import reconcile_generator_billing
        billing=reconcile_generator_billing(validated)
        require(billing['total_charged_updates']==result['total_charged_updates'], 'recovered G lifetime billing mismatch')
        receipt['generator_billing']=billing
        atomic_json(output/'recovery_receipt.json',receipt)
        for path,sha in e.locks.items():require(file_sha(path)==sha,'original artifact changed during recovery')
        pending.unlink()  # Atomic publication commit; public resolution was blocked until now.
        validated=resolve_completed_boundary(output)
    except BaseException:
        atomic_json(pending,dict(original_round=str(root),phase='validation_failed'))
        atomic_json(output/'status.json',dict(phase='failed',completed_rounds=0,publication_only=True))
        raise
    return dict(round=spec['round_index'],bundle=validated['bundle'],receipt=receipt,validated=True,
                prior_physics_charged=validated['prior_physics_charged'])


def recover_finalization(original_round, output=None, *, original_started_unix, dry_run=True):
    """Validate or publish exactly one completed-but-unpublished final G stage.

    Explicit ``dry_run=False`` and a new output path are required for publication.
    Never call this container's plan through the training runner.
    """
    inspected=_inspect(original_round)
    require(original_started_unix==inspected[2].get('original_started_unix'), 'original deadline identity mismatch')
    if dry_run:
        with tempfile.TemporaryDirectory(prefix='jit-publication-audit-') as temp:
            report=_build(*inspected,Path(temp)/'publication',original_started_unix)
            report.pop('bundle');report['dry_run']=True
            return report
    require(output is not None,'new output required')
    return _build(*inspected,output,original_started_unix)
