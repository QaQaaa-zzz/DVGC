"""Gated teacher lifetime with legacy replay and opt-in two-review recovery."""
from pathlib import Path
import json
import shutil
import time
from .production import ProductionRunner,read
from .contracts import file_sha,digest
from .protocol import atomic_json
from .teacher_runtime import TeacherEvaluationSession


class PersistentTeacherRunner(ProductionRunner):
    def __init__(self,manifest):
        super().__init__(manifest,_teacher_worker=True)
        self.persistent_teacher=True
        self.session=TeacherEvaluationSession()

    def child(self,name,argv,maximum,*,updates=0,extra_env=None):
        from ..pulse_exploration_runtime import evaluate
        if updates or argv[:3]!=['JIT/cli/run_pulse_exploration.py','--mode','evaluate']:
            raise ValueError('persistent teacher only executes physical evaluations')
        if sum(c['charged_interactions'] for c in self.costs)+maximum>self.spec['budgets']['max_physics']:
            raise ValueError('teacher worker reservation exhausted')
        config=Path(argv[argv.index('--spec')+1]);out=Path(argv[argv.index('--output')+1])
        cost=dict(stage=name,charged_interactions=maximum,charged_updates=0,phase='running',
            accounting='conservative reservation until actual receipt')
        self.costs.append(cost);atomic_json(self.root/'costs.json',self.costs)
        self.status('running',stage=name)
        start=time.monotonic()
        evaluate(read(config),out,session=self.session)
        cost.update(phase='completed',wall_seconds=time.monotonic()-start)
        self.session.memory_boundary(name,self.root,
            max_rss_bytes=self.spec.get('teacher_host_rss_limit_bytes',8*1024**3))
        return cost


def dispatch_teacher(parent,incumbent):
    if parent.spec.get('teacher_layout')!='source_control_in_candidate_batch' or parent.spec.get('teacher_colored_noise_candidates')!=0:
        raise ValueError('persistent B1 requires original 17-world candidate layout')
    validate_teacher_acceptance(parent.spec)
    root=parent.root/'teacher_worker';root.mkdir(exist_ok=False)
    roots={r['root_id'] for r in parent.panels['new_roots']}
    reused=set(parent.spec.get('recovery_completed_teacher_roots',{}))
    if not reused<=roots:raise ValueError('teacher reuse contains undeclared roots')
    maximum=teacher_reservation(parent.spec,len(roots-reused))
    spec={**parent.spec,'output':str(root),'teacher_execution':'worker_internal',
          'budgets':{**parent.spec['budgets'],'max_physics':maximum}}
    if parent.spec.get('recovery_teacher_worker') is not None:
        spec['recovery']=parent.spec['recovery_teacher_worker']
    atomic_json(root/'production.json',spec);atomic_json(root/'panels.json',parent.panels)
    request=root/'request.json';atomic_json(request,dict(production=str(root/'production.json'),incumbent=incumbent,
        input_files={str(root/name):file_sha(root/name) for name in ('production.json','panels.json')}))
    request_sha=file_sha(request)
    cost=parent.child('teacher_worker',['JIT/cli/run_bridge_teacher_worker.py','--request',request],maximum)
    receipt=read(root/'completed.json')
    if (receipt['phase']!='completed' or file_sha(root/'results.json')!=receipt['results_sha256']
        or file_sha(request)!=request_sha or receipt.get('request_sha256')!=request_sha):
        raise ValueError('incomplete persistent teacher receipt')
    if receipt['charged_interactions']>maximum:raise ValueError('teacher actuals exceed reservation')
    teachers=read(root/'results.json')
    if set(teachers)!={r['root_id'] for r in parent.panels['new_roots']}:
        raise ValueError('persistent teacher omitted roots')
    # Preserve canonical round-level evidence paths while all heavy arrays stay
    # immutable in the worker directory. No original evaluation is overwritten.
    target=parent.root/'teachers';target.mkdir(exist_ok=False)
    for p in (root/'teachers').glob('*.json'):
        shutil.copyfile(p,target/p.name)
    cost.update(charged_interactions=receipt['charged_interactions'],
        active_interactions=receipt['active_interactions'],accounting='measured',
        worker_receipt=str(root/'completed.json'),worker_receipt_sha256=file_sha(root/'completed.json'))
    atomic_json(parent.root/'costs.json',parent.costs)
    return teachers


def run(request):
    request_path=Path(request);request_sha=file_sha(request_path);request=read(request_path)
    from .verified_files import LOCK_CACHE
    for path,sha in request['input_files'].items():LOCK_CACHE.verify(path,sha)
    spec=read(request['production']);root=Path(spec['output'])
    runner=PersistentTeacherRunner(spec)
    try:
        result=runner.teacher_search(request['incumbent'])
        if file_sha(request_path)!=request_sha:raise ValueError('teacher request changed during execution')
        for path,sha in request['input_files'].items():LOCK_CACHE.verify(path,sha)
        for p in (root/'teachers').glob('*_result.json'):
            row=read(p);proposal=p.with_name(p.name.replace('_result.json','_proposals.npz'))
            if proposal.exists():row['proposal_path']=str(proposal);atomic_json(p,row);result[row['root_id']]=row
        atomic_json(root/'results.json',result)
        runner.status('completed',stage='teacher_complete')
        atomic_json(root/'completed.json',dict(phase='completed',results_sha256=file_sha(root/'results.json'),
            charged_interactions=sum(c['charged_interactions'] for c in runner.costs),
            active_interactions=sum(c.get('active_interactions',0) for c in runner.costs),
            kernel_compile_count=runner.session.compile_count,roots=len(result),layout='B1_17_worlds',
            request_sha256=request_sha))
    except BaseException as exc:
        runner.status('failed',error=repr(exc));raise
    finally:runner.session.close()


def validate_teacher_acceptance(spec):
    """Require physical acceptance or a user-authorized, single-round trial.

    A trial records unresolved evidence and never constitutes formal acceptance.
    Its exact output/round binding prevents reuse for a subsequent campaign.
    """
    if recovery_training_enabled(spec):
        validate_recovery_authorization(spec)
        return
    trial=spec.get('teacher_experimental_trial')
    if trial is not None:
        if set(trial)!={'path','sha256'} or file_sha(trial['path'])!=trial['sha256']:
            raise ValueError('experimental trial receipt changed')
        report=read(trial['path'])
        if (report.get('schema')!='jit_b1_experimental_trial_v1'
            or report.get('implementation_commit')!=spec['implementation_commit']
            or report.get('output')!=str(Path(spec['output']).resolve())
            or report.get('round_index')!=spec['round_index']
            or report.get('user_authorized') is not True
            or report.get('formal_acceptance') is not False
            or not report.get('known_limitations')
            or not 0<spec['budgets']['max_physics']<=report.get('max_physics',0)<=1500000):
            raise ValueError('experimental trial scope mismatch')
        return
    receipt=spec.get('teacher_physical_acceptance')
    if not receipt or set(receipt)!={'path','sha256'}:
        raise ValueError('persistent B1 requires a passed physical acceptance receipt')
    if file_sha(receipt['path'])!=receipt['sha256']:raise ValueError('teacher acceptance changed')
    report=read(receipt['path'])
    if (report.get('schema')!='jit_b1_physical_acceptance_v1'
        or report.get('implementation_commit')!=spec['implementation_commit']
        or any(report.get(k) is not True for k in ('aba_passed','baseline_comparison_passed',
                  'full_search_replay_passed','rng_mapping_passed'))):
        raise ValueError('persistent B1 physical acceptance missing or failed')


def recovery_training_enabled(spec):
    contract=spec.get('recovery_training_contract')
    if contract is None:return False
    if not isinstance(contract,dict) or contract.get('schema')!='jit_teacher_recovery_training_v1':
        raise ValueError('unknown recovery training contract')
    return True


def teacher_reservation(spec,query_count):
    recovery=recovery_training_enabled(spec)
    if type(query_count) is not int or query_count<0 or (recovery and query_count>128):
        raise ValueError('recovery teacher query count must be within 128')
    return query_count*400*(1+17*(3 if recovery else 2))


def validate_recovery_authorization(spec):
    """Bind the new authorized campaign; leave legacy trial/acceptance untouched."""
    receipt=spec.get('teacher_recovery_authorization')
    if (not receipt or set(receipt)!={'path','sha256'}
            or file_sha(receipt['path'])!=receipt['sha256']):
        raise ValueError('recovery campaign authorization missing or changed')
    report=read(receipt['path'])
    campaign=Path(spec.get('campaign_output','')).resolve()
    output=Path(spec['output']).resolve()
    round_index=spec.get('round_index')
    if (report.get('schema')!='jit_teacher_recovery_campaign_authorization_v1'
            or report.get('implementation_commit')!=spec['implementation_commit']
            or report.get('campaign_output')!=str(campaign)
            or not output.is_relative_to(campaign) or output==campaign
            or report.get('rounds')!=100 or type(round_index) is not int or not 1<=round_index<=100
            or report.get('user_authorized') is not True or report.get('formal_acceptance') is not False
            or not report.get('known_limitations')
            or report.get('max_physics_per_round')!=3500000
            or not 0<spec['budgets']['max_physics']<=3500000
            or report.get('teacher_query_count')!=128
            or report.get('teacher_generated_candidates')!=16 or report.get('teacher_reviews')!=2
            or spec.get('teacher_layout')!='source_control_in_candidate_batch'
            or spec.get('teacher_colored_noise_candidates')!=0):
        raise ValueError('recovery campaign authorization scope mismatch')


def _complete_recovery_result(result,base,source,horizon,lane):
    """Verify a completed real lane under the already frozen endpoint contract."""
    import numpy as np
    from .production import lane_arrays
    from ..pulse_exploration_runtime import suffix_label
    if (any(result.get(k)!=base[k] for k in ('root_id','snapshot_context_sha256'))
            or result.get('label') not in (0,1) or len(result.get('attempts',[]))!=1):
        raise ValueError('incomplete recovery teacher result or changed root identity')
    attempt=result['attempts'][0]
    expected=dict(trace_lane=lane,actor_sha256=source['actor_sha256'],
        normalizer_sha256=source['normalizer_sha256'],model_sha256=source['xml_sha256'],
        snapshot_context_sha256=base['snapshot_context_sha256'],label=result['label'])
    if any(attempt.get(k)!=v for k,v in expected.items()):
        raise ValueError('recovery teacher trace identity changed')
    arrays=lane_arrays(attempt)
    if any(not np.isfinite(a).all() for a in arrays.values() if a.dtype.kind in 'fc'):
        raise ValueError('nonfinite recovery teacher trace')
    n=len(arrays['done'])
    if n!=attempt.get('steps') or not n or not (arrays['done'][-1] or n>=horizon):
        raise ValueError('incomplete recovery teacher trace')
    label,_=suffix_label(bool(arrays['valid_contact'][-1]),bool(arrays['physical_failure'][-1]),
        bool(arrays['timeout'][-1]),bool(arrays['done'][-1]),n>=horizon)
    if label!=result['label']:raise ValueError('recovery teacher terminal label mismatch')
    return arrays


def _recovery_proposals(reference,root_id,seed):
    import numpy as np
    from .proposals import stable_seed
    path=Path(reference['proposal_path'])
    if file_sha(path)!=reference['proposal_sha256']:raise ValueError('teacher reuse proposal changed')
    with np.load(path,allow_pickle=False) as data:
        arrays={k:data[k] for k in ('actions','diffusion_noise','colored_raw_draws')}
    expected=np.stack([np.random.default_rng(stable_seed(seed,root_id,cid)).normal(size=(16,4))
        for cid in range(1,17)]).astype(np.float32)
    if (arrays['actions'].shape!=(17,16,4) or arrays['colored_raw_draws'].shape!=(0,16,4)
            or not np.array_equal(arrays['diffusion_noise'],expected)
            or not np.isfinite(arrays['actions']).all() or np.any(np.abs(arrays['actions'])>1)):
        raise ValueError('teacher reuse original candidate/noise layout changed')
    return arrays


def reusable_teacher_roots(runner,incumbent,originals):
    """Validate every original search/review trace before any resumed physics."""
    from copy import deepcopy
    from .production import lane_arrays
    from .teacher import select_teacher
    refs=runner.spec.get('recovery_completed_teacher_roots',{})
    if not isinstance(refs,dict) or not set(refs)<=set(originals):raise ValueError('teacher reuse roots differ')
    output={}
    for rid,ref in refs.items():
        base=originals[rid]
        context=ref.get('root_context_sha256',ref.get('snapshot_context_sha256'))
        if (context!=base['snapshot_context_sha256'] or ref.get('source_actor_sha256')!=runner.source['actor_sha256']
                or ref.get('generator_checkpoint_manifest_sha256')!=incumbent['checkpoint_manifest_sha256']
                or ref.get('protocol_sha256')!=runner.protocol or file_sha(ref['path'])!=ref['sha256']):
            raise ValueError('teacher reuse identity or result receipt changed')
        row=read(ref['path'])
        if (row.get('root_id')!=rid or row.get('source_actor_sha256')!=runner.source['actor_sha256']
                or row.get('generator')!=incumbent or row.get('generated_only_search') is not True
                or row.get('proposal_sha256')!=ref['proposal_sha256']
                or row.get('historical_source_label')!=base['label']):
            raise ValueError('teacher reuse recorded search binding changed')
        _recovery_proposals(ref,rid,runner.spec['seed'])
        candidates=row.get('candidates',[])
        if ([c.get('candidate_id') for c in candidates]!=list(range(17))
                or [c.get('kind') for c in candidates]!=['source_only']+['diffusion']*16):
            raise ValueError('teacher reuse candidate layout changed')
        for lane,candidate in enumerate(candidates):
            if type(candidate.get('full_success')) is not bool or candidate.get('invalid') or candidate.get('incomplete'):
                raise ValueError('teacher reuse incomplete candidate')
            result=dict(root_id=rid,snapshot_context_sha256=context,label=int(candidate['full_success']),
                attempts=[candidate['attempt']])
            _complete_recovery_result(result,base,runner.source,400,lane)
        selected=select_teacher(candidates[1:])
        if selected is not None:
            verification=row.get('verification',{});reviews=verification.get('reviews',[])
            cid=selected['candidate_id']
            if (row.get('selected_candidate_id')!=cid or verification.get('candidate_id')!=cid
                    or len(reviews)!=2 or verification.get('repeat_complete')!=[True,True]):
                raise ValueError('teacher reuse fixed winner/reviews incomplete')
            for review in reviews:
                if (review.get('candidate_id')!=cid or review.get('candidate_ids')!=list(range(17))
                        or len(review.get('labels',[]))!=17 or review.get('complete') is not True):
                    raise ValueError('teacher reuse review layout changed')
                for lane,label in enumerate(review['labels']):
                    attempt=deepcopy(review['selected_attempt']);attempt.update(trace_lane=lane,label=label)
                    attempt['steps']=len(lane_arrays(attempt)['done'])
                    _complete_recovery_result(dict(root_id=rid,snapshot_context_sha256=context,label=label,
                        attempts=[attempt]),base,runner.source,400,lane)
            labels=[r['labels'][cid] for r in reviews]
            if labels!=verification.get('repeat_labels') or (row.get('teacher_status')=='verified_solution' and labels!=[1,1]):
                raise ValueError('teacher reuse verified labels changed')
        if row.get('teacher_status')=='verified_solution':
            demo=row.get('demo',{});metadata=demo.get('metadata',{})
            if (file_sha(demo['path'])!=demo['sha256'] or metadata.get('protocol_sha256')!=runner.protocol
                    or metadata.get('root_context_sha256')!=context
                    or metadata.get('normalizer_sha256')!=runner.source['normalizer_sha256']
                    or metadata.get('model_sha256')!=runner.source['xml_sha256']
                    or metadata.get('verification_receipt_sha256')!=digest(row['verification'])):
                raise ValueError('teacher reuse demonstration identity changed')
        row.update(proposal_path=ref['proposal_path'],reused_from=deepcopy(ref))
        output[rid]=row
    return output


def recovery_teacher_search(runner,incumbent):
    """Search G independently, then review the fixed winner twice in its layout."""
    import numpy as np
    import jax
    import jax.numpy as jp
    from .worker import generator_template,generator_reference
    from .diffusion import restore_state,ddim_sample
    from .proposals import make_candidate_pool,stable_seed
    from .teacher import select_teacher
    from .production import lane_arrays,teacher_candidate,trace_metadata
    from .performance import measure
    from ..unified_envelope_snapshot import load_unified_envelope_snapshot
    validate_recovery_authorization(runner.spec)
    teacher_reservation(runner.spec,len(runner.panels['new_roots']))
    horizon=runner.spec['source_runtime']['horizon']
    if horizon!=400:raise ValueError('recovery teacher preserves the frozen 400-step horizon')
    originals={r['root_id']:r for r in runner.panels['new_roots']}
    if len(originals)!=len(runner.panels['new_roots']):raise ValueError('duplicate recovery teacher roots')
    reused=reusable_teacher_roots(runner,incumbent,originals)
    if reused:
        atomic_json(runner.root/'teacher_root_reuse.json',dict(schema='jit_teacher_root_reuse_v1',
            roots=runner.spec['recovery_completed_teacher_roots'],new_physics_interactions=0,optimizer_updates=0))
    with measure('model_actor_generator_load',backend=jax.default_backend(),component='generator'):
        net,template,identity=generator_template(generator_reference(runner.spec),runner.spec['seed'])
        state=restore_state(Path(incumbent['checkpoint_manifest']).parent,template,identity)
    generation=jax.jit(lambda obs,noise:ddim_sample(lambda x,o,k:net.apply(state['ema'],x,o,k),obs,noise)) if getattr(runner,'persistent_teacher',False) else None
    compiled=None
    runner.teacher_in_progress=True
    source_results=runner.evaluate('teacher_source',runner.panels['new_roots'])
    if [r['root_id'] for r in source_results]!=list(originals):raise ValueError('recovery source roots changed or incomplete')
    output={};directory=runner.root/'teachers';directory.mkdir(exist_ok=True)
    for ordinal,base in enumerate(source_results):
        rid=base['root_id'];runner.active_teacher_root=rid
        if rid in reused:
            if (base.get('snapshot_context_sha256')!=originals[rid]['snapshot_context_sha256']
                    or base.get('label')!=reused[rid]['source_control_label_history'][1]):
                raise ValueError('teacher reuse source control/context changed')
            atomic_json(directory/f'{ordinal:04d}_result.json',reused[rid]);output[rid]=reused[rid]
            continue
        if base.get('label') not in (0,1):raise ValueError('unknown recovery source recheck')
        if len(base.get('attempts',[]))!=1:raise ValueError('incomplete recovery source trace')
        _complete_recovery_result(base,originals[rid],runner.source,horizon,base['attempts'][0]['trace_lane'])
        snapshot=load_unified_envelope_snapshot(Path(base['snapshot']))
        actual=lane_arrays(base['attempts'][0]);obs=actual['actor_observation_before'][0]
        noise=np.stack([np.random.default_rng(stable_seed(runner.spec['seed'],rid,cid)).normal(size=(16,4)) for cid in range(1,17)]).astype(np.float32)
        normalized=(jp.asarray(np.repeat(obs[None],16,axis=0))-state['normalizer']['mean'])/state['normalizer']['std']
        partial=runner.spec.get('recovery_teacher_worker',{}).get('partial_root',{})
        inherited_proposal=partial.get('root_id')==rid
        if inherited_proposal and (partial.get('ordinal')!=ordinal
                or partial.get('snapshot_context_sha256')!=base['snapshot_context_sha256']):
            raise ValueError('teacher reuse partial root ordinal/context changed')
        if generation is not None and compiled is None and not inherited_proposal:
            with measure('ddim_compile',backend=jax.default_backend(),batch_size=16):
                compiled=generation.lower(normalized,jp.asarray(noise)).compile()
        if inherited_proposal:
            inherited_arrays=_recovery_proposals(partial,rid,runner.spec['seed'])
            pool=dict(actions=inherited_arrays['actions'],kinds=['source_only']+['diffusion']*16,
                colored_raw_draws=inherited_arrays['colored_raw_draws'])
        else:
            with measure('ddim_generation',backend=jax.default_backend(),root=rid,batch_size=16):
                generated=jax.device_get(compiled(normalized,jp.asarray(noise)) if compiled is not None else
                    ddim_sample(lambda x,o,k:net.apply(state['ema'],x,o,k),normalized,jp.asarray(noise)))
            pool=make_candidate_pool(rid,actual['normalized_action_executed'][:16],generated,
                seed=runner.spec['seed'],colored_noise_candidates=0)
        path=directory/f'{ordinal:04d}_proposals.npz'
        if path.exists():raise ValueError('recovery teacher proposals already exist; no silent retry')
        if inherited_proposal:shutil.copyfile(partial['proposal_path'],path)
        else:np.savez_compressed(path,actions=pool['actions'],diffusion_noise=noise,colored_raw_draws=pool['colored_raw_draws'])
        candidates=[{**base,'index':cid,'candidate_id':cid} for cid in range(17)]
        controls=np.asarray([True]+[False]*16)
        results=runner.evaluate(f'teacher_{ordinal:04d}',candidates,prefixes=pool['actions'],source_only=controls)
        if [r['candidate_id'] for r in results]!=list(range(17)):raise ValueError('incomplete recovery search candidate ordering')
        for lane,result in enumerate(results):_complete_recovery_result(result,base,runner.source,horizon,lane)
        scored=[teacher_candidate(r,pool['kinds'][r['candidate_id']],r['candidate_id'],snapshot.last_action) for r in results]
        selected=select_teacher([c for c in scored if c['kind']=='diffusion'])
        source_labels=[originals[rid]['label'],base['label'],results[0]['label']]
        row=dict(root_id=rid,teacher_status='searched_no_solution',source_actor_sha256=runner.source['actor_sha256'],
            generator=incumbent,candidates=scored,proposal_sha256=file_sha(path),generated_only_search=True,
            historical_source_label=originals[rid]['label'],training_eligible=True,teacher_query_completed=True)
        reviews=[];last_arrays=None
        if selected is not None:
            cid=selected['candidate_id']
            for review in (1,2):
                repeated=runner.evaluate(f'review_batch_{ordinal:04d}_{review}',candidates,
                    prefixes=pool['actions'],source_only=controls)
                if [r['candidate_id'] for r in repeated]!=list(range(17)):
                    raise ValueError('incomplete recovery review candidate ordering')
                for lane,result in enumerate(repeated):
                    arrays=_complete_recovery_result(result,base,runner.source,horizon,lane)
                    if lane==cid:last_arrays=arrays
                source_labels.append(repeated[0]['label'])
                reviews.append(dict(review=review,candidate_id=cid,full_success=repeated[cid]['label']==1,
                    complete=True,selected_label=repeated[cid]['label'],candidate_ids=list(range(17)),
                    labels=[r['label'] for r in repeated],selected_attempt=repeated[cid]['attempts'][0]))
            verification=dict(candidate_id=cid,full_success=all(r['full_success'] for r in reviews),
                batch_size=17,selected_lane=cid,layout='same_as_search',
                repeat_labels=[r['selected_label'] for r in reviews],repeat_complete=[r['complete'] for r in reviews],
                reviews=reviews,fixed_selection=True)
            atomic_json(directory/f'{ordinal:04d}_reviews.json',verification)
            row.update(selected_candidate_id=cid,verification=verification,
                teacher_status='verified_solution' if verification['full_success'] else 'replay_rejected')
            if not verification['full_success']:row['reason']='fixed_generated_candidate_failed_review'
        consistent=all(label in (0,1) and label==source_labels[0] for label in source_labels)
        row.update(source_control_label_history=source_labels,source_control_consistent=consistent,
            source_recheck_label=source_labels[0] if consistent else None,new_gain_eligible=consistent)
        if not consistent:
            row.update(teacher_status='invalid' if selected is not None else 'not_scheduled',
                reason='source_control_repeat_conflict',quarantined=True,
                source_control_labels=[source_labels[0],next(v for v in source_labels if v!=source_labels[0])],
                selected_replay_label=reviews[-1]['selected_label'] if reviews else None,complete_finite_replay=True)
        elif row['teacher_status']=='verified_solution':
            # Only the final successful review is exported, after both reviews.
            last_arrays['action_origin']=np.where(last_arrays['action_origin_code']==1,'bridge_prefix','source_tail')
            metadata=trace_metadata(base,runner.source,runner.protocol,runner.panels['splits'],'verified_teacher')
            metadata.update(teacher_status='verified_solution',verification_receipt_sha256=digest(row['verification']),
                generator_checkpoint_manifest_sha256=incumbent['checkpoint_manifest_sha256'])
            trace=directory/f'{ordinal:04d}_verified.npz';np.savez_compressed(trace,**last_arrays)
            row['demo']=dict(count=len(last_arrays['done']),path=str(trace),sha256=file_sha(trace),metadata=metadata)
        atomic_json(directory/f'{ordinal:04d}_result.json',row);output[rid]=row
    runner.teacher_in_progress=False;runner.active_teacher_root=None
    return output
