"""Fixed DEV generator diagnostics, isolated from all training admission paths.

A single C16 search supplies every C-prefix metric. A winner is locked before
exactly two extra complete-layout reviews; no replacement search is permitted.
Physics uses the existing gated GPU suffix evaluator and original17-lane layout.
"""
from pathlib import Path
import json
import time
import numpy as np
from .contracts import file_sha
from .protocol import atomic_json

BUDGETS = (1, 2, 4, 8, 16)
SCHEDULE = (0, 5, 10, 20, 40, 60, 80, 100)


def read(path):
    return json.loads(Path(path).read_text())


def should_evaluate_round(round_index):
    return round_index in SCHEDULE


def validate_roots(rows):
    if not rows:
        raise ValueError('nonempty fixed DEV roots required')
    ids = [r['root_id'] for r in rows]
    contexts = [r['snapshot_context_sha256'] for r in rows]
    if len(set(ids)) != len(ids) or len(set(contexts)) != len(contexts):
        raise ValueError('unique DEV roots and contexts required')
    for row in rows:
        if row.get('role') not in ('DEV', 'SOLVER_DEV') or row.get('data_role') not in ('dev', 'solver_dev'):
            raise ValueError('only DEV roots permitted; TRAIN and final TEST excluded')
        if row.get('pi0_labels') != [0, 0, 0]:
            raise ValueError('three independently recorded pi0 failure labels required')
        if row.get('onset') not in (0, 5, 10, 15):
            raise ValueError('fixed four-onset protocol required')
        if not row.get('state_sha256') or not row.get('root_episode_id'):
            raise ValueError('complete state/ancestor identity required')


def verified_label(row):
    """Three-valued generated-only qualification, never source rescue."""
    labels = row['candidate_labels']
    if len(labels) != 16 or any(x not in (0, 1, None) for x in labels):
        raise ValueError('sixteen explicit generated candidate labels required')
    source = row['source_labels']
    if len(source) != 3 or any(x != 0 for x in source):
        return None
    if any(x is None for x in labels):
        return None
    cid = row['selected_candidate_id']
    if cid is None:
        return False if all(x == 0 for x in labels) else None
    if not 1 <= cid <= 16 or labels[cid-1] != 1:
        raise ValueError('locked winner must be a successful generated candidate')
    reviews = row['winner_review_labels']
    if len(reviews) != 2 or any(x not in (0, 1) for x in reviews):
        return None
    return reviews == [1, 1]


def _rate(values):
    return sum(x is True for x in values)/len(values) if values and all(x is not None for x in values) else None


def summarize(rows, *, costs):
    ids = [r['root_id'] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError('nonempty unique monitor results required')
    verified = [verified_label(r) for r in rows]
    first = {}; rates = {}
    excluded = [r['root_id'] for r in rows if any(x != 0 for x in r['source_labels'])]
    for row in rows:
        labels = row['candidate_labels']; successful = [i+1 for i, x in enumerate(labels) if x == 1]
        earliest=min(successful) if successful else None
        first_unknown=earliest is not None and any(x is None for x in labels[:earliest])
        first[row['root_id']] = dict(index=None if first_unknown else earliest,
            censored=not successful and all(x == 0 for x in labels))
        if first_unknown:
            first[row['root_id']]['earliest_known_success_index']=earliest
    for c in BUDGETS:
        outcomes = []
        for row in rows:
            labels = row['candidate_labels'][:c]
            outcomes.append(None if row['root_id'] in excluded else
                True if 1 in labels else False if all(x == 0 for x in labels) else None)
        rates[str(c)] = _rate(outcomes)
    phase = {}
    for onset in (0, 5, 10, 15):
        values = [v for r, v in zip(rows, verified) if r['onset'] == onset]
        phase[str(onset)] = dict(n=len(values), verified=sum(v is True for v in values),
            unknown=sum(v is None for v in values), vtrr_at_16=_rate(values))
    charge = sum(c['charged_interactions'] for c in costs)
    active = sum(c.get('active_interactions', 0) for c in costs)
    padding = sum(c.get('padding_interactions', 0) for c in costs)
    if any(c['charged_interactions'] < 0 or c.get('active_interactions',0) < 0 or c.get('padding_interactions',0) < 0 for c in costs):
        raise ValueError('nonnegative control-step cost required')
    for cost in costs:
        if 'active_interactions' in cost and 'padding_interactions' in cost and cost['charged_interactions'] != cost['active_interactions'] + cost['padding_interactions']:
            raise ValueError('charge must include all active and padding control steps')
    count = sum(v is True for v in verified)
    return dict(n=len(rows), verified=count, unknown=sum(v is None for v in verified),
        trr_at_c=rates, vtrr_at_16=_rate(verified),
        fixed_denominator_verified_lower_bound=count/len(rows), first_success=first,
        phase_coverage=phase, excluded_source_roots=excluded,
        charged_control_steps=charge, active_control_steps=active, padding_control_steps=padding,
        rse_per_million_control_steps=count/charge*1e6 if charge else None,
        rse_includes='full17lane search, H16 prefixes, pi suffixes, two full reviews and padding',
        independent_training_repetitions=0)


def compare_to_g0(current, baseline):
    if current['comparison_identity'] != baseline['comparison_identity']:
        raise ValueError('fixed panel/protocol/Actor comparison identity differs')
    now = {r['root_id']:verified_label(r) for r in current['roots']}
    old = {r['root_id']:verified_label(r) for r in baseline['roots']}
    if set(now) != set(old):
        raise ValueError('root identity differs')
    known = [rid for rid in now if now[rid] is not None and old[rid] is not None]
    return dict(newly_recovered=sorted(r for r in known if now[r] and not old[r]),
        lost_recovery=sorted(r for r in known if old[r] and not now[r]),
        paired_known=len(known), unknown_pairs=len(now)-len(known))


def action_diagnostics(actions, last_action):
    a = np.asarray(actions); last = np.asarray(last_action)
    if a.shape != (16,16,4) or last.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(last).all():
        raise ValueError('finite C16/H16/four-channel generated actions required')
    differences = np.diff(np.concatenate((np.broadcast_to(last,(16,1,4)),a),axis=1),axis=1)
    return dict(candidate_variance_mean=float(np.var(a,axis=0).mean()),
        candidate_variance_per_channel=np.var(a,axis=0).mean(axis=0).tolist(),
        mean_absolute_slew=float(np.abs(differences).mean()),
        squared_delta_sum=float(np.square(differences).sum()),
        at_action_limit_fraction=float((np.abs(a)>=1.-1e-6).mean()),
        raw_preclip_fraction=None,
        clipping_note='DDIM clamps predicted clean action internally; final saturation measured, raw clip count unavailable')


def fixed_noise_mse(state, predict, fixture):
    import jax.numpy as jp
    from .diffusion import cosine_schedule, noise_mse
    obs, actions, k, eps = fixture
    if (np.shape(obs)!=(len(actions),76) or np.shape(actions)!=(len(actions),16,4)
        or np.shape(k)!=(len(actions),) or np.shape(eps)!=np.shape(actions)
        or np.asarray(k).dtype.kind not in 'iu' or np.any(k<0) or np.any(k>=100)
        or not all(np.isfinite(x).all() for x in fixture) or not len(actions)):
        raise ValueError('valid immutable fixed denoising fixture required')
    ab = jp.asarray(cosine_schedule()[1])[jp.asarray(k),None,None]
    normalized = (jp.asarray(obs)-state['normalizer']['mean'])/state['normalizer']['std']
    noisy = jp.sqrt(ab)*jp.asarray(actions)+jp.sqrt(1-ab)*jp.asarray(eps)
    loss = float(noise_mse(predict(state['ema'],noisy,normalized,jp.asarray(k)),jp.asarray(eps)))
    if not np.isfinite(loss):
        raise FloatingPointError('nonfinite fixed noise MSE')
    return loss


def student_acquisition(rows, student_labels):
    eligible = [r['root_id'] for r in rows if verified_label(r) is True]
    labels = [student_labels.get(rid) for rid in eligible]
    unknown = sum(x not in (0,1) for x in labels)
    count = sum(x == 1 for x in labels)
    return dict(teacher_verified=len(eligible), student_success=count, unknown=unknown,
        sar=count/len(eligible) if eligible and not unknown else None)


def prepare_panel(source_plan, output, *, dev_fixture, proposal_seed=1010246101):
    """Freeze already qualified SOLVER_DEV snapshots; zero physics/training."""
    source_plan = Path(source_plan).resolve(); source = read(source_plan)
    rootfile = Path(source['output'])/'selected_roots.json'
    rows = [r for r in read(rootfile) if r.get('role') == 'SOLVER_DEV']
    validate_roots(rows)
    model = source['models']['pi0']; policy = model['policy']
    frozen = Path(model['frozen_policy']); bank = Path(model['bank'])
    fixture = Path(dev_fixture).resolve()
    locks = {str(p.resolve()):file_sha(p) for p in (source_plan,rootfile,frozen,bank,fixture)}
    for name in ('payload.pkl','identity.json'):
        path = Path(policy['checkpoint'])/name; locks[str(path.resolve())] = file_sha(path)
    for row in rows:
        for name in ('snapshot.pkl','identity.json'):
            path = Path(row['snapshot'])/name; locks[str(path.resolve())] = file_sha(path)
        attempt = row['pi0_attempt']; path = Path(attempt['trace'])
        if file_sha(path) != attempt['trace_sha256']:
            raise ValueError('saved pi0 reference trace changed')
        locks[str(path.resolve())] = attempt['trace_sha256']
    rows = [{**r,'role':'DEV','data_role':'dev','training_allowed':False,
        'generator_training_allowed':False,'explorer_training_allowed':False,
        'inherited_role':'SOLVER_DEV'} for r in rows]
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'roots.json',rows)
    panel = dict(schema='jit_fixed_teacher_dev_panel_v1',role='DEV',training_allowed=False,
        roots=str(output/'roots.json'),roots_sha256=file_sha(output/'roots.json'),n=len(rows),
        source_plan=str(source_plan),pi0=model,locks=locks,
        fixture=dict(path=str(fixture),sha256=file_sha(fixture),role='generator_dev',training_allowed=False),
        protocol=dict(horizon=400,prefix_horizon=16,candidates=16,layout=17,extra_reviews=2,
            proposal_seed=proposal_seed,ddim_timesteps=list(range(99,-1,-5)),budgets=list(BUDGETS),
            suffix_rng_count=17,suffix_rng_indices=list(range(17)),
            preserve_snapshot_episode_context=True,success_criterion='stable_forward_recovery'),
        evaluation_rounds=list(SCHEDULE),
        teacher_max_control_steps=len(rows)*3*17*400,
        optional_student_max_control_steps=len(rows)*400,
        scope='fixed reused D1 SOLVER_DEV N; no 100-state claim and no final TEST',
        inherited_acquisition_cost='historical D1; not recharged in checkpoint RSE',
        optimizer_updates=dict(E=0,G=0,student=0))
    atomic_json(output/'panel.json',panel)
    return panel


def audit_panel(path):
    panel = read(path)
    if panel.get('schema') != 'jit_fixed_teacher_dev_panel_v1' or panel.get('role') != 'DEV' or panel.get('training_allowed') is not False:
        raise ValueError('fixed nontraining DEV panel required')
    for file, sha in panel['locks'].items():
        if file_sha(file) != sha:
            raise ValueError('DEV panel input hash drift: '+file)
    if file_sha(panel['roots']) != panel['roots_sha256']:
        raise ValueError('DEV roots changed')
    rows = read(panel['roots']); validate_roots(rows)
    if len(rows) != panel['n']:
        raise ValueError('DEV denominator changed')
    p = panel['protocol']
    expected = dict(horizon=400,prefix_horizon=16,candidates=16,layout=17,extra_reviews=2,
        ddim_timesteps=list(range(99,-1,-5)),budgets=list(BUDGETS),suffix_rng_count=17,
        suffix_rng_indices=list(range(17)),preserve_snapshot_episode_context=True,
        success_criterion='stable_forward_recovery')
    if any(p.get(k)!=v for k,v in expected.items()):
        raise ValueError('frozen teacher protocol changed')
    return panel, rows


class _Evaluator:
    def __init__(self, output, model, maximum, wall_seconds):
        from .teacher_runtime import TeacherEvaluationSession
        self.output=Path(output);self.model=model;self.maximum=maximum
        self.start=time.monotonic();self.wall_seconds=wall_seconds;self.costs=[]
        self.session=TeacherEvaluationSession()

    def evaluate(self, name, rows, actions=None):
        from ..pulse_exploration_runtime import evaluate
        reservation=len(rows)*400
        if sum(c['charged_interactions'] for c in self.costs)+reservation>self.maximum:
            raise ValueError('teacher DEV reservation exhausted')
        if time.monotonic()-self.start>=self.wall_seconds:
            raise TimeoutError('teacher DEV wall budget exhausted')
        rowfile=self.output/(name+'_rows.json');atomic_json(rowfile,rows)
        spec=dict(bank=self.model['bank'],proposer=self.model['proposer'],order=[self.model['proposer']],
            horizon=400,budget=reservation,candidates=str(rowfile),full_matrix=True,
            record_actor_preobservations=True,record_retention_diagnostics=True,
            reward_mode='original_all_phases',success_criterion='stable_forward_recovery',
            preserve_snapshot_episode_context=True,suffix_rng_count=len(rows),suffix_rng_indices=list(range(len(rows))))
        if actions is not None:
            ap=self.output/(name+'_actions.npz')
            np.savez_compressed(ap,actions=actions,source_only=np.array([r['candidate_id']==0 for r in rows],bool),
                root_contexts=np.array([r['snapshot_context_sha256'] for r in rows]))
            spec['bridge_action_plan']=dict(path=str(ap),sha256=file_sha(ap))
        atomic_json(self.output/(name+'_spec.json'),spec)
        cost=dict(stage=name,charged_interactions=reservation,reservation=reservation,phase='running')
        self.costs.append(cost);atomic_json(self.output/'costs.json',self.costs)
        started=time.monotonic();out=self.output/name;evaluate(spec,out,session=self.session)
        receipt=read(out/'status.json');result=read(out/'results.json')
        if receipt.get('phase')!='completed' or len(result)!=len(rows) or receipt['charged_interactions']>reservation:
            raise ValueError('incomplete/overbudget DEV evaluation')
        for row, actual in zip(rows,result):
            if any(actual.get(k)!=row[k] for k in ('root_id','snapshot_context_sha256','candidate_id')):
                raise ValueError('DEV root/candidate/context order changed')
        cost.update(phase='completed',wall_seconds=time.monotonic()-started,
            **{k:receipt[k] for k in ('charged_interactions','active_interactions','padding_interactions')},
            results=str(out/'results.json'),results_sha256=file_sha(out/'results.json'))
        atomic_json(self.output/'costs.json',self.costs)
        return result


def _frozen_model(bank, proposer, xml_sha):
    from .worker import source_payload
    value=read(bank);members=[m for m in value['members'] if m['name']==proposer and 'evaluator' in m['roles']]
    if len(members)!=1:
        raise ValueError('one frozen suffix bank member required')
    member=members[0]
    if file_sha(member['frozen_policy'])!=member['frozen_file_sha256']:
        raise ValueError('suffix frozen manifest changed')
    policy,_=source_payload(member['frozen_policy'])
    if policy['xml_sha256']!=xml_sha:
        raise ValueError('cross matrix requires identical physical model')
    return dict(bank=str(Path(bank).resolve()),proposer=proposer,policy=policy,
        frozen_policy=member['frozen_policy'],bank_sha256=file_sha(bank))


def evaluate_checkpoint(panel_path, generator_manifest, output, *, round_index=0,
        baseline=None, tail_model=None, student_model=None, max_wall_seconds=3600):
    """Execute one bounded frozen G/tail DEV cell; optionally independent SAR.

    tail_model/student_model are {bank,proposer} references to saved frozen banks.
    Caller supervises process timeout. No training, lesson export or admission.
    """
    import jax
    import jax.numpy as jp
    from .worker import generator_template
    from .diffusion import restore_state,ddim_sample
    from .proposals import stable_seed
    from .production import lane_arrays,teacher_candidate
    from .teacher import select_teacher
    from ..unified_envelope_snapshot import load_unified_envelope_snapshot,physical_state_sha256,snapshot_context_sha256
    if jax.default_backend()!='gpu':
        raise RuntimeError('fixed DEV physical evaluation requires existing GPU evaluator')
    panel,rows=audit_panel(panel_path);panel_sha=file_sha(panel_path)
    gpath=Path(generator_manifest).resolve();gsha=file_sha(gpath);gmanifest=read(gpath)
    reference=panel['pi0'];xml_sha=reference['policy']['xml_sha256']
    tail=_frozen_model((tail_model or reference)['bank'],(tail_model or reference)['proposer'],xml_sha)
    net,template,identity=generator_template(reference['frozen_policy'],panel['protocol']['proposal_seed'])
    state=restore_state(gpath.parent,template,identity)
    for key in ('mean','std'):
        if not np.array_equal(np.asarray(state['normalizer'][key]),np.asarray(template['normalizer'][key])):
            raise ValueError('generator normalizer differs from fixed pi0 reference')
    if tail_model is None and any(tail['policy'][k]!=reference['policy'][k] for k in ('actor_sha256','normalizer_sha256')):
        raise ValueError('fixed pi0 suffix identity changed')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    evaluator=_Evaluator(output,tail,panel['teacher_max_control_steps'],max_wall_seconds)
    generation=jax.jit(lambda obs,noise:ddim_sample(lambda x,o,k:net.apply(state['ema'],x,o,k),obs,noise))
    comparison=dict(panel_sha256=panel_sha,roots_sha256=panel['roots_sha256'],protocol=panel['protocol'],
        tail_actor_sha256=tail['policy']['actor_sha256'],tail_normalizer_sha256=tail['policy']['normalizer_sha256'],
        xml_sha256=xml_sha,generator_reference=identity)
    result=dict(schema='jit_fixed_teacher_dev_report_v1',role='DEV',training_allowed=False,
        round=round_index,n=panel['n'],panel=str(Path(panel_path).resolve()),panel_sha256=panel_sha,
        generator_manifest=str(gpath),generator_manifest_sha256=gsha,generator_state_sha256=gmanifest['state_sha256'],
        generator_updates=gmanifest['updates'],inference='ema',tail=tail,comparison_identity=comparison,
        fixture=panel['fixture'],roots=[],optimizer_updates=dict(E=0,G=0,student=0))
    atomic_json(output/'request.json',result)
    started=time.monotonic();independent=None
    try:
        with np.load(panel['fixture']['path'],allow_pickle=False) as f:
            fixture=tuple(f[k] for k in ('observations','actions','timesteps','noise'))
        result['fixed_dev_noise_mse']=fixed_noise_mse(state,lambda p,x,o,k:net.apply(p,x,o,k),fixture)
        for ordinal,base in enumerate(rows):
            snap=load_unified_envelope_snapshot(Path(base['snapshot']))
            if physical_state_sha256(snap)!=base['state_sha256'] or snapshot_context_sha256(snap)!=base['snapshot_context_sha256']:
                raise ValueError('DEV full state/context identity changed')
            source=lane_arrays(base['pi0_attempt'])['normalized_action_executed'][:16]
            if not len(source):raise ValueError('missing fixed pi0 control reference')
            if len(source)<16:source=np.concatenate((source,np.repeat(source[-1:],16-len(source),axis=0)))
            noise=np.stack([np.random.default_rng(stable_seed(panel['protocol']['proposal_seed'],base['root_id'],cid)).normal(size=(16,4)) for cid in range(1,17)]).astype(np.float32)
            obs=np.repeat(np.asarray(snap.observation)[None],16,axis=0)
            normalized=(jp.asarray(obs)-state['normalizer']['mean'])/state['normalizer']['std']
            gen_start=time.monotonic();generated=np.asarray(generation(normalized,jp.asarray(noise)),np.float32)
            generation_seconds=time.monotonic()-gen_start
            actions=np.concatenate((source[None],generated))
            proposal=output/f'root_{ordinal:03d}_proposals.npz'
            np.savez_compressed(proposal,actions=actions,diffusion_noise=noise,observation=obs[0])
            candidates=[{**base,'index':cid,'candidate_id':cid} for cid in range(17)]
            search=evaluator.evaluate(f'root_{ordinal:03d}_search',candidates,actions)
            scored=[teacher_candidate(r,'diffusion',r['candidate_id'],snap.last_action) for r in search[1:] if r['label'] in (0,1)]
            winner=select_teacher(scored);cid=winner['candidate_id'] if winner else None
            lock=dict(root_id=base['root_id'],candidate_id=cid,proposal_sha256=file_sha(proposal),
                layout=17,suffix_rng_indices=list(range(17)),fixed_before_reviews=True,
                selection='fullsuccess, action delta ascending, task return descending, candidate id ascending')
            atomic_json(output/f'root_{ordinal:03d}_winner_lock.json',lock)
            reviews=[evaluator.evaluate(f'root_{ordinal:03d}_review_{i}',candidates,actions) for i in (1,2)]
            all_results=[search,*reviews]
            row=dict(root_id=base['root_id'],root_episode_id=base['root_episode_id'],onset=base['onset'],
                state_sha256=base['state_sha256'],snapshot_context_sha256=base['snapshot_context_sha256'],
                role='DEV',training_allowed=False,candidate_labels=[r['label'] for r in search[1:]],
                candidate_review_labels=[[r['label'] for r in batch[1:]] for batch in reviews],
                source_labels=[batch[0]['label'] for batch in all_results],selected_candidate_id=cid,
                winner_review_labels=[batch[cid]['label'] for batch in reviews] if cid is not None else [],
                winner_lock=lock,proposal_path=str(proposal),proposals_sha256=file_sha(proposal),
                raw_results=all_results,generation_seconds=generation_seconds,
                review_semantics='two separate physics dispatches; identical original layout and paired RNG; not independent random-seed robustness',
                action_diagnostics=action_diagnostics(generated,snap.last_action),
                invalid_contact_control_frames=0,physical_failure_candidates=0)
            for candidate in search[1:]:
                if not candidate.get('attempts'):continue
                arrays=lane_arrays(candidate['attempts'][0])
                if 'prohibited_contact' in arrays:
                    row['invalid_contact_control_frames']+=int(np.asarray(arrays['prohibited_contact']).sum())
                if 'physical_failure' in arrays:
                    row['physical_failure_candidates']+=int(np.any(arrays['physical_failure']))
            row['verified_label']=verified_label(row)
            atomic_json(output/f'root_{ordinal:03d}_result.json',row);result['roots'].append(row)
            atomic_json(output/'progress.json',dict(phase='running',round=round_index,completed_roots=len(result['roots']),n=panel['n'],
                charged_control_steps=sum(c['charged_interactions'] for c in evaluator.costs)))
        result.update(summary=summarize(result['roots'],costs=evaluator.costs),costs=evaluator.costs,
            wall_seconds=time.monotonic()-started,kernel_compile_count=evaluator.session.compile_count)
        if baseline is not None:
            result['g0_comparison']=compare_to_g0(result,read(baseline))
            result['g0_paired_ci']=paired_confidence_interval(result,read(baseline))
            result['g0_report']=dict(path=str(Path(baseline).resolve()),sha256=file_sha(baseline))
        else:
            result['g0_paired_ci']=paired_confidence_interval(result,result)
        if student_model is not None:
            evaluator.session.close()
            student=_frozen_model(student_model['bank'],student_model['proposer'],xml_sha)
            student_output=output/'student';student_output.mkdir(exist_ok=False)
            independent=_Evaluator(student_output,student,panel['optional_student_max_control_steps'],max_wall_seconds)
            try:
                independent_rows=[{**r,'index':i,'candidate_id':0} for i,r in enumerate(rows)]
                actual=independent.evaluate('independent_student',independent_rows)
                labels={r['root_id']:r['label'] for r in actual}
                result.update(student=student,student_acquisition=student_acquisition(result['roots'],labels),
                    independent_student_results=actual,independent_student_costs=independent.costs)
            finally:independent.session.close()
        audit_panel(panel_path)
        if file_sha(panel_path)!=panel_sha or file_sha(gpath)!=gsha or file_sha(gpath.parent/'state.msgpack')!=gmanifest['state_sha256']:
            raise ValueError('monitor inputs changed during execution')
        result['phase']='completed';atomic_json(output/'report.json',result)
        atomic_json(output/'status.json',dict(phase='completed',round=round_index,n=panel['n'],
            report_sha256=file_sha(output/'report.json'),charged_interactions=sum(c['charged_interactions'] for c in evaluator.costs)+
                (sum(c['charged_interactions'] for c in independent.costs) if independent else 0),
            wall_seconds=time.monotonic()-started))
        return result
    except BaseException as error:
        atomic_json(output/'status.json',dict(phase='failed',round=round_index,error=repr(error),
            charged_interactions=sum(c['charged_interactions'] for c in evaluator.costs)+
                (sum(c['charged_interactions'] for c in independent.costs) if independent else 0),
            completed_roots=len(result['roots']),n=panel['n']))
        raise
    finally:evaluator.session.close()


def paired_confidence_interval(current, baseline, *, seed=20261013, bootstrap_replicates=10000):
    """Descriptive paired bootstrap; all roots of an ancestor stay together."""
    comparison=compare_to_g0(current,baseline)
    if type(bootstrap_replicates) is not int or bootstrap_replicates<1:
        raise ValueError('positive paired bootstrap replicate count required')
    old={r['root_id']:r for r in baseline['roots']};groups={}
    for row in current['roots']:
        before=old[row['root_id']];a=verified_label(row);b=verified_label(before)
        if a is None or b is None:continue
        ancestor=row.get('root_episode_id',row['root_id'])
        if ancestor!=before.get('root_episode_id',before['root_id']):
            raise ValueError('paired root ancestor identity differs')
        groups.setdefault(ancestor,[]).append(int(a)-int(b))
    result=dict(confidence=.95,bootstrap_replicates=bootstrap_replicates,seed=seed,
        sampling_unit='root_episode_id ancestor; paired whole groups with replacement',
        estimator='mean verified recovery difference on known paired roots',
        paired_known_roots=comparison['paired_known'],unknown_pairs=comparison['unknown_pairs'],
        ancestor_groups=len(groups),independent_training_seeds=0,
        interpretation='descriptive fixed DEV ancestor bootstrap from one training lineage; no independent training-seed inference',
        units='fraction of paired known roots',mean_difference=None,lower=None,upper=None)
    if not groups:return result
    values=[groups[key] for key in sorted(groups)]
    sums=np.array([sum(v) for v in values],dtype=np.float64)
    counts=np.array([len(v) for v in values],dtype=np.int64)
    draws=np.random.default_rng(seed).integers(0,len(groups),size=(bootstrap_replicates,len(groups)))
    estimates=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    lo,hi=np.quantile(estimates,[.025,.975])
    result.update(mean_difference=float(sums.sum()/counts.sum()),lower=float(lo),upper=float(hi))
    return result


def aggregate_reports(report_paths, output):
    """Save four diagnostic PNGs and a source-backed JSON/Markdown index, CPU only.

    Accept complete fixed pi0 reports beginning with G0. No rollout, optimization,
    checkpoint selection or training admission is performed here. Cross matrices
    with other frozen tails need a separate aggregate for each fixed tail.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    if not report_paths:raise ValueError('completed G0 report required')
    loaded=[(Path(p).resolve(),read(p)) for p in report_paths]
    loaded.sort(key=lambda item:item[1]['round'])
    if loaded[0][1]['round']!=0 or len(set(r['round'] for _,r in loaded))!=len(loaded):
        raise ValueError('one G0 and unique subsequent completed rounds required')
    baseline=loaded[0][1]
    series=[]
    for path,report in loaded:
        if (report.get('phase')!='completed' or report.get('role')!='DEV'
            or report.get('training_allowed') is not False):
            raise ValueError('only completed nontraining DEV reports permitted')
        recomputed=summarize(report['roots'],costs=report.get('costs',[]))
        fields=('n','verified','unknown','trr_at_c','vtrr_at_16','phase_coverage')
        if report['n']!=recomputed['n'] or any(report['summary'].get(k)!=recomputed[k] for k in fields):
            raise ValueError('saved summary disagrees with raw candidate/review outcomes')
        paired=compare_to_g0(report,baseline)
        ci=paired_confidence_interval(report,baseline)
        series.append(dict(round=report['round'],n=report['n'],source_report=str(path),source_report_sha256=file_sha(path),
            generator_manifest_sha256=report['generator_manifest_sha256'],summary=report['summary'],
            paired_changes_vs_g0=paired,paired_ci_vs_g0=ci,sar=report.get('student_acquisition')))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    rounds=[r['round'] for r in series];labels=[f'G{k}' for k in rounds]
    note=f"Fixed DEV N={baseline['n']}, H16/C16; one lineage; source-only excluded"
    def save(fig,name):
        fig.text(.01,.01,note,fontsize=8);fig.tight_layout(rect=(0,.035,1,1))
        fig.savefig(output/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
    def numeric(value):return np.nan if value is None else value
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    for label,row in zip(labels,series):
        rates=[numeric(row['summary']['trr_at_c'][str(c)]) for c in BUDGETS]
        ax.plot(BUDGETS,np.asarray(rates)*100,'o-',label=label)
    ax.set(xlabel='Generated candidate prefix budget C (one frozen search)',ylabel='Full recovery TRR (%)',
        xticks=BUDGETS,ylim=(-3,103),title='Recovery success versus generated proposal budget')
    ax.grid(alpha=.25);ax.legend();save(fig,'candidate_budget')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    rates=np.array([numeric(r['summary']['vtrr_at_16']) for r in series])*100
    axes[0].plot(rounds,rates,'o-')
    for x,y,row in zip(rounds,rates,series):
        if np.isfinite(y):axes[0].annotate(f"{row['summary']['verified']}/{row['n']}",(x,y),xytext=(0,6),textcoords='offset points',ha='center')
    axes[0].set(title='Verified teacher recovery after two extra reviews',xlabel='Generator round',ylabel='VTRR@16 (%)',ylim=(-3,103),xticks=rounds)
    ci=[r['paired_ci_vs_g0'] for r in series]
    means=np.array([numeric(r['mean_difference']) for r in ci])*100
    lower=np.array([numeric(r['lower']) for r in ci])*100
    upper=np.array([numeric(r['upper']) for r in ci])*100
    axes[1].plot(rounds,means,'o-',label='Paired difference vs G0')
    axes[1].vlines(rounds,lower,upper,color='C0',label='Descriptive 95% ancestor bootstrap')
    axes[1].axhline(0,color='gray',linewidth=.8)
    axes[1].set(title='10,000 paired ancestor bootstrap replicates',xlabel='Generator round',ylabel='VTRR difference (percentage points)',xticks=rounds)
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.25)
    save(fig,'vtrr_and_paired_ci')
    fig,ax=plt.subplots(figsize=(7,max(3,1+.45*len(series))))
    matrix=np.array([[numeric(row['summary']['phase_coverage'][str(o)]['vtrr_at_16']) for o in (0,5,10,15)] for row in series])
    im=ax.imshow(matrix,vmin=0,vmax=1,aspect='auto',cmap='Blues')
    ax.set(xticks=range(4),xticklabels=['0','5','10','15'],yticks=range(len(series)),yticklabels=labels,
        xlabel='Pulse onset (control step)',ylabel='Generator checkpoint',title='Phase coverage: verified states / fixed phase denominator')
    for i,row in enumerate(series):
        for j,onset in enumerate((0,5,10,15)):
            phase=row['summary']['phase_coverage'][str(onset)]
            text=f"{phase['verified']}/{phase['n']}" if phase['n'] else 'NO ROOTS'
            if phase['unknown']:text+=' UNKNOWN'
            ax.text(j,i,text,ha='center',va='center',color='white' if matrix[i,j]>.6 else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='VTRR@16 fraction');save(fig,'phase_coverage')
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    sar=np.array([numeric(r['sar']['sar']) if r['sar'] else np.nan for r in series])*100
    ax.plot(rounds,sar,'o-')
    for x,y,row in zip(rounds,sar,series):
        if np.isfinite(y):
            a=row['sar'];ax.annotate(f"{a['student_success']}/{a['teacher_verified']}",(x,y),xytext=(0,6),textcoords='offset points',ha='center')
    if not np.isfinite(sar).any():ax.text(.5,.5,'Independent student SAR: NOT_RUN / UNKNOWN',ha='center',va='center',transform=ax.transAxes)
    ax.set(title='Independent student acquisition on verified teacher roots',xlabel='Paired generator/student round',ylabel='Conditional SAR (%)',ylim=(-3,103),xticks=rounds)
    ax.grid(alpha=.25);save(fig,'student_acquisition')
    result=dict(schema='jit_fixed_teacher_dev_aggregate_v1',role='DEV',training_allowed=False,
        rounds=rounds,n=baseline['n'],comparison_identity=baseline['comparison_identity'],series=series,
        figures=[str(output/(name+'.png')) for name in ('candidate_budget','vtrr_and_paired_ci','phase_coverage','student_acquisition')],
        environment_interactions=0,optimizer_updates=0,
        interpretation='One fixed DEV panel and one training lineage. Paired 95% intervals resample whole root_episode_id ancestors. Missing outcomes remain UNKNOWN; final TEST unopened.')
    atomic_json(output/'report.json',result)
    lines=['# Fixed DEV teacher evolution', '',note+'.','',result['interpretation'],'',
        '| G round | Verified/N | VTRR@16 | Difference vs G0 [95% interval] | SAR | Charged teacher control steps |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in series:
        s=row['summary'];ci=row['paired_ci_vs_g0'];sar=row['sar']
        rate='UNKNOWN' if s['vtrr_at_16'] is None else f"{s['vtrr_at_16']:.3f}"
        interval='UNKNOWN' if ci['mean_difference'] is None else f"{ci['mean_difference']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]"
        sar_text='NOT_RUN' if sar is None else 'UNKNOWN' if sar['sar'] is None else f"{sar['student_success']}/{sar['teacher_verified']} ({sar['sar']:.3f})"
        lines.append(f"| G{row['round']} | {s['verified']}/{row['n']} | {rate} | {interval} | {sar_text} | {s['charged_control_steps']} |")
    lines.extend(['','Intervals are descriptive fractions on known pairs; these are not independent training-seed repetitions. SAR is conditional on each checkpoint’s verified teacher roots, whose denominator may change.',''])
    for figure in result['figures']:
        name=Path(figure).name;lines.extend([f'![{name}]({name})',''])
    lines.extend(['## Source reports',''])
    for row in series:
        lines.append(f"- [G{row['round']}]({row['source_report']}), report SHA-256 `{row['source_report_sha256']}`, generator manifest SHA-256 `{row['generator_manifest_sha256']}`")
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return result
