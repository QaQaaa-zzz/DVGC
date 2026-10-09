"""Read-only resolution of atomic completed bundles, including stopped series.

A published Actor is authoritative. A later evaluated/rejected student is never
substituted. Legacy dependency locks are explicit migration evidence, not a
claim that those files had historical hash protection.
"""
import json
import os
from pathlib import Path
from .contracts import digest, file_sha


def _read(path):
    return json.loads(Path(path).read_text())


def _validate_actor(actor):
    from .worker import source_payload
    policy, _ = source_payload(actor['frozen_policy'])
    if actor['policy'] != policy:
        raise ValueError('published Actor policy identity mismatch')
    for key in ('actor_sha256', 'normalizer_sha256'):
        if actor.get(key, policy[key]) != policy[key]:
            raise ValueError('published Actor identity mismatch: '+key)
    return policy


def _validate_semantics(bundle, spec):
    from .artifacts import load_corpus, validate_generator_receipt
    from .student import load_optional_demo
    from .student_demo_bank import _allowed_teacher_tails
    validate_generator_receipt(bundle['generator'])
    load_corpus(bundle['corpus'])
    load_optional_demo(_read(bundle['demo']['path']))
    policy=bundle['actor']['policy']; baseline=spec['baseline_identity']
    source={k:policy[k] for k in ('actor_sha256','normalizer_sha256')}
    source.update(model_sha256=policy['xml_sha256'],protocol_sha256=baseline['protocol_sha256'])
    _allowed_teacher_tails(source,baseline,bundle['tail_lineage'])


def _assert_not_live(root, lock):
    for name in ('launch.json','started.json'):
        path=root/name
        if not path.exists():continue
        lock(path)
        record=_read(path)
        for key in ('pid','supervisor_pid','process_pid'):
            pid=record.get(key)
            if type(pid) is not int or pid<=0 or pid==os.getpid():continue
            proc=Path('/proc')/str(pid)
            try:
                command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
            except FileNotFoundError:continue
            if state!='Z' and (str(root) in command or
                    ('run_bridge_closed_loop.py' in command and root.name in command)):
                raise ValueError('parent supervisor is still running: '+str(root))
    # Children may outlive a stopped supervisor. Match actual executable/script
    # arguments and active inputs, never shell source text or --continue-from.
    worker_scripts={'run_bridge_closed_loop.py','run_pulse_exploration.py',
                    'run_generative_bridge.py','train_iterative_probe.py','train_unified_from_pi0.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not argv or Path(argv[0]).name in ('bash','sh','dash','zsh'):continue
        if not any(Path(arg).name in worker_scripts for arg in argv[:3]):continue
        for i,arg in enumerate(argv[:-1]):
            if arg not in ('--plan','--spec','--config','--output'):continue
            active=Path(argv[i+1]).resolve()
            if active==root or root in active.parents:
                raise ValueError('parent worker is still running: '+proc.name)


def resolve_completed_boundary(previous, allow_stopped_parent=False):
    """Resolve only committed publications; pending recovery views are private."""
    return _resolve_completed_boundary(previous, allow_stopped_parent)


def _resolve_completed_boundary(previous, allow_stopped_parent=False, *, pending_root=None):
    """Return validated bundle, dependency locks and retained historical costs.

    Paths in the result are absolute strings. ``parent_series`` is the requested
    series; ``bundle_series`` is the explicit ancestor that published the bundle.
    This function does not write files, deserialize optimizer state, or run physics.
    """
    requested=Path(previous).resolve(); root=requested; locks={}; legacy=[]; ancestry=[]
    if pending_root is not None and pending_root!=requested:
        raise ValueError('pending publication audit is restricted to requested root')
    def lock(path, expected=None):
        path=Path(path).resolve(); actual=file_sha(path)
        if expected is not None and actual!=expected:
            raise ValueError('continuation artifact hash mismatch: '+str(path))
        if str(path) in locks and locks[str(path)]!=actual:
            raise ValueError('continuation artifact changed while resolving: '+str(path))
        locks[str(path)]=actual
        return actual
    def dependencies(value):
        if isinstance(value,list):
            for item in value:dependencies(item)
        elif isinstance(value,dict):
            if isinstance(value.get('path'),str) and value.get('sha256'):
                lock(value['path'],value['sha256'])
            for key,item in value.items():
                if key in ('inputs','input_files','locks') and isinstance(item,dict):
                    for path,sha in item.items():
                        if isinstance(path,str) and Path(path).is_absolute() and isinstance(sha,str):lock(path,sha)
                elif key.endswith('_sha256') and key not in ('formal_config_sha256','config_sha256'):
                    target=value.get(key[:-7])
                    if key=='formal_config_file_sha256':target=value.get('formal_config')
                    if isinstance(target,str) and Path(target).is_absolute():lock(target,item)
                dependencies(item)
    visited=set()
    while True:
        if root in visited:raise ValueError('continuation parent cycle')
        visited.add(root)
        if (root/'publication_pending.json').exists() and root!=pending_root:
            raise ValueError('publication recovery is not committed: '+str(root))
        lock(root/'status.json');lock(root/'plan.json')
        status=_read(root/'status.json');plan=_read(root/'plan.json')
        if plan.get('publication_only'):
            receipt_path=root/'recovery_receipt.json';lock(receipt_path);receipt=_read(receipt_path)
            if (receipt.get('schema')!='jit_finalization_publication_recovery_v1' or
                    receipt.get('original_execution_phase')!='failed' or receipt.get('original_returncode')!=-6 or
                    any(receipt.get(key)!=0 for key in ('new_physics','new_student_transitions','new_generator_updates')) or
                    receipt.get('original_series')!=plan.get('continuation_parent')):
                raise ValueError('invalid publication recovery receipt')
            dependencies(receipt)
            original_status=Path(receipt['original_series'])/'status.json';lock(original_status)
            if _read(original_status).get('phase')!='failed':
                raise ValueError('recovery original failed status changed')
        _assert_not_live(root,lock)
        phase=status.get('phase'); count=status.get('completed_rounds')
        if type(count) is not int or count<0:raise ValueError('invalid completed round count')
        if phase=='completed':
            if count!=plan['rounds']:raise ValueError('partial series labelled completed')
        elif not allow_stopped_parent or phase not in ('cancelled','stopped','failed','interrupted'):
            raise ValueError('completed parent required; stopped parent requires explicit opt-in')
        costs=[]
        for path in sorted(root.glob('round_*/costs.json')):
            lock(path);costs.extend(_read(path))
        carried=plan.get('carried_physics',0)
        local=carried+sum(c['charged_interactions'] for c in costs)
        lifetime=max(plan.get('prior_physics_charged',0)+local,status.get('lifetime_physics_charged',0))
        ledger=root/'cost_ledger.json'
        if ledger.exists():
            lock(ledger);record=_read(ledger)
            lifetime=max(lifetime,record.get('lifetime_physics_charged',0))
        ancestry.append(dict(series=str(root),status=status,plan=plan,costs=costs,
                             local_physics=local,lifetime_physics=lifetime))
        if count:break
        if (root/'current_source.json').exists():raise ValueError('uncommitted bundle with zero completed rounds')
        if not plan.get('continuation_parent'):raise ValueError('no completed boundary or explicit continuation_parent')
        root=Path(plan['continuation_parent']).resolve()
    lock(root/'completed_rounds.json');completed=_read(root/'completed_rounds.json')
    if len(completed)!=count:raise ValueError('completed-round journal mismatch')
    boundary=Path(completed[-1]).resolve()
    if boundary.parent!=root:raise ValueError('completed round outside publishing series')
    lock(boundary/'status.json');lock(boundary/'production.json')
    if _read(boundary/'status.json').get('phase')!='completed':raise ValueError('partial round is not a completed boundary')
    spec=_read(boundary/'production.json'); bundle_path=root/'current_source.json'
    lock(bundle_path);lock(boundary/'current_source.json')
    bundle=_read(bundle_path)
    if bundle!=_read(boundary/'current_source.json') or bundle['round']!=spec['round_index']:
        raise ValueError('published bundle and completed round disagree')
    if bundle['round']!=plan.get('round_offset',0)+count:raise ValueError('last completed round identity mismatch')
    contract=boundary/'stages/round_contract.json';lock(contract)
    if _read(contract)!={'contract':spec,'sha256':digest(spec)}:raise ValueError('round contract identity mismatch')
    if list((boundary/'stages').glob('*.running.json')):raise ValueError('incomplete required stage')
    records={}
    for path in (boundary/'stages').glob('*.json'):
        if path.name=='round_contract.json':continue
        lock(path);record=_read(path)
        if record.get('output_sha256')!=digest(record.get('result')):raise ValueError('stage output identity mismatch')
        dependencies(record['result']);records[path.stem]=record
    phase_path=boundary/'source_phase_result.json';lock(phase_path);phase_result=_read(phase_path)
    dependencies(phase_result)
    required={'train_student':(phase_result,bundle['evaluated_student']),
        'generator_selection':({'corpus':bundle['corpus'],'incumbent':spec['continuation']['generator']},bundle['generator'])}
    if 'generator_update_policy' in spec:
        required['generator_selection'][0]['generator_update_policy']=spec['generator_update_policy']
    if 'generator_reference_frozen_policy' in spec:
        required['generator_selection'][0]['generator_reference_frozen_policy']=spec['generator_reference_frozen_policy']
    for name,(inputs,result) in required.items():
        record=records.get(name,{})
        if record.get('result')!=result or record.get('input_sha256')!=digest({'round':digest(spec),'inputs':inputs}):
            raise ValueError('required stage identity mismatch: '+name)
    # Required feedback/evaluation children must exist even if their stage file
    # was removed; validating only the files present would accept partial work.
    source_rows_path=boundary/'source_rows.json';lock(source_rows_path)
    source_rows=_read(source_rows_path)['rows']
    if not source_rows:raise ValueError('completed round lacks source rows')
    batches=(len(source_rows)+31)//32
    for family in ('source_suffix','student_train'):
        names=([f'eval_{family}'] if len(source_rows)<=32 else
               [f'eval_{family}_batch_{i:04d}' for i in range(batches)])
        for i,name in enumerate(names):
            if name not in records:raise ValueError('missing required evaluation stage: '+name)
            result=records[name]['result'];rows=_read(result['path'])
            expected=min(32,len(source_rows)-32*i) if len(source_rows)>32 else len(source_rows)
            if len(rows)!=expected:raise ValueError('incomplete evaluation stage: '+name)
            evaluation=boundary/'evaluations'/name.removeprefix('eval_')
            lock(evaluation/'candidates.json');lock(evaluation/'spec.json')
            evaluation_spec=_read(evaluation/'spec.json')
            inputs={'rows':_read(evaluation/'candidates.json'),'bank':evaluation_spec['bank'],
                    'actor':evaluation_spec['proposer'],'prefixes_sha256':None}
            if records[name].get('input_sha256')!=digest({'round':digest(spec),'inputs':inputs}):
                raise ValueError('required evaluation input identity mismatch: '+name)
    for child in ('collection','explorer_update'):
        path=boundary/child/'status.json';lock(path)
        if _read(path).get('phase')!='completed':raise ValueError('incomplete required child: '+child)
    policy=_validate_actor(bundle['actor']);lock(bundle['actor']['frozen_policy'])
    checkpoint=policy.get('checkpoint')
    if checkpoint:
        for path in Path(checkpoint).iterdir():
            if path.is_file():lock(path)
    dependencies(bundle)
    for key in ('explorer','dev_fixture'):
        lock(bundle[key],bundle[key+'_sha256'])
    generator=bundle['generator'];manifest=Path(generator['checkpoint_manifest'])
    lock(manifest,generator['checkpoint_manifest_sha256'])
    lock(manifest.parent/'state.msgpack',_read(manifest)['state_sha256'])
    for key in ('support','neighborhood_map'):
        if bundle.get(key):
            path=bundle[key];expected=bundle.get(key+'_sha256')
            lock(path,expected);dependencies(_read(path))
            if expected is None:legacy.append(dict(field=key,path=str(Path(path).resolve()),sha256=file_sha(path)))
    for key in ('corpus','demo'):
        dependencies(_read(bundle[key]['path']))
    if bundle.get('seed_support'):dependencies(_read(bundle['seed_support']['path']))
    map_path=boundary/'neighborhood_map.json'
    if spec.get('neighborhood') is not None:
        collection_path=boundary/'collection_spec.json';lock(collection_path)
        collection=_read(collection_path)
        if Path(collection['neighborhood_map']).resolve()!=map_path.resolve():
            raise ValueError('completed round neighborhood map mismatch')
        lock(map_path,collection['neighborhood_map_sha256']);dependencies(_read(map_path))
    _validate_semantics(bundle,spec)
    learner=bundle.get('learner') or bundle['actor'].get('learner')
    if learner:
        from .learner_continuation import validate_saved_learner
        metadata=validate_saved_learner(learner,policy,expected_local_transitions=128000)
        lock(metadata['state_path'],metadata['state_sha256'])
    elif plan.get('continuous_learning'):
        raise ValueError('continuous parent cannot bootstrap a missing learner again')
    lifetime=ancestry[-1]['lifetime_physics']
    for entry in reversed(ancestry[:-1]):lifetime=max(entry['lifetime_physics'],lifetime+entry['local_physics'])
    # Reservations are charged work, not necessarily optimizer updates executed.
    # Do not invent totals for history preceding the explicitly traversed chain.
    generator_costs=[dict(c,series=a['series']) for a in ancestry for c in a['costs']
                     if c.get('stage','').startswith('generator_')]
    optimization={
        'supervised_updates':sum(c.get('charged_updates',0) for a in ancestry for c in a['costs']),
        'explorer_optimizer_updates':sum(c.get('explorer_optimizer_updates',0) for a in ancestry for c in a['costs'])}
    optimization.update(generator_charged_updates=sum(c.get('charged_updates',0) for c in generator_costs),
        generator_cost_entries=generator_costs,
        scope='exact charged reservations in resolved series chain; older optimization history unknown',
        older_history_unknown=bool(ancestry[-1]['plan'].get('prior_physics_charged',0)),
        published_generator_state_updates=_read(manifest).get('updates'),
        published_generator_receipt_charged_updates=generator.get('total_charged_updates'))
    return dict(parent_series=str(requested),bundle_series=str(root),bundle_path=str(bundle_path),
        bundle=bundle,status=ancestry[0]['status'],locks=locks,ancestry=ancestry,
        prior_physics_charged=lifetime,prior_optimization_costs=optimization,
        migration=dict(schema='jit_completed_boundary_migration_v1',legacy_optimizer_bootstrap=learner is None,
            published_actor_preserved=True,legacy_dependency_locks=legacy,
            source_bundle_sha256=locks[str(bundle_path)],stopped_parent=ancestry[0]['status']['phase']!='completed'))
