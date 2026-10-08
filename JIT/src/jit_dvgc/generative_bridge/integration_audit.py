"""Read-only two-round pilot audit. Engineering readiness is not a performance gate."""
import json
from pathlib import Path
import numpy as np
from .contracts import file_sha


def require(condition,message):
    if not condition:raise ValueError(message)


def tree_equal(a,b):
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) or isinstance(b,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return np.array_equal(a,b)


def explorer_state_equal(a,b):
    # Existing writer conventions differ only for neighborhood list serialization.
    from ..rsl_pulse import neighborhood_config
    a=dict(a);b=dict(b)
    if a.get('neighborhood') is not None:a['neighborhood']=neighborhood_config(a['neighborhood'])
    if b.get('neighborhood') is not None:b['neighborhood']=neighborhood_config(b['neighborhood'])
    return tree_equal(a,b)


class Evidence:
    def __init__(self):self.locks={}
    def lock(self,path,sha=None):
        path=Path(path).resolve();actual=file_sha(path)
        require(sha is None or actual==sha,'audit artifact hash changed: '+str(path))
        require(str(path) not in self.locks or self.locks[str(path)]==actual,'artifact changed during audit')
        self.locks[str(path)]=actual
        return path
    def read(self,path,sha=None):return json.loads(self.lock(path,sha).read_text())
    def unpack(self,path):
        from flax.serialization import msgpack_restore
        return msgpack_restore(self.lock(path).read_bytes())


def audit_timing_rows(spec,tape,rows):
    from ..pulse_schedule import lane_onsets,pulse_outcome
    delays=lane_onsets(spec,spec['round_index']);duration=spec['pulse_steps']
    require(len(rows)==128 and len(delays)==128,'collection requires all128 episode rows')
    counts={str(int(s)):int(np.sum(delays==s)) for s in np.unique(delays)}
    require(counts=={'0':32,'5':32,'10':32,'15':32},'collection onset strata are not32 each')
    prefix=np.asarray(tape['prefix_mask'],bool);mask=np.asarray(tape['mask'],bool)
    terminal=np.asarray(tape['terminal'],bool)
    require(prefix.shape==mask.shape==terminal.shape==(15+duration,128),'collection tape shape changed')
    outcomes={k:0 for k in ('pre_pulse_terminal','during_pulse_terminal','valid_post_pulse')}
    snapshot_steps={str(s):[] for s in (0,5,10,15)}
    for lane,row in enumerate(rows):
        real=np.flatnonzero(prefix[:,lane]);require(len(real)>0,'empty collection lane')
        require(np.array_equal(real,np.arange(real[-1]+1)),'noncontiguous physical prefix')
        end=int(real[-1]);onset=int(delays[lane]);applied=int(mask[:,lane].sum())
        expected=prefix[:,lane]&(np.arange(len(prefix))>=onset)
        require(np.array_equal(mask[:,lane],expected),'pulse mask outside single declared window')
        require(end<onset+duration,'lane advanced beyond its own post-pulse endpoint')
        require(not terminal[:end,lane].any(),'physical prefix continues after terminal')
        outcome=pulse_outcome(applied,duration,bool(terminal[end,lane]))
        require(row['index']==lane and row['pulse_scheduled_start_step']==onset,'candidate lane/onset drift')
        require(row['pulse_applied_steps']==applied and row['snapshot_control_step']==end+1,'candidate endpoint drift')
        require(row['pulse_outcome']==outcome and row['episode_performance_denominator'] is True,'terminal denominator drift')
        require(bool(row['prefix_terminal'])==(outcome!='valid_post_pulse'),'terminal marked as recoverable snapshot')
        if outcome=='valid_post_pulse':
            require(end+1==onset+duration and row['endpoint_kind']=='continuation_snapshot','post-pulse endpoint drift')
            snapshot_steps[str(onset)].append(end+1)
        outcomes[outcome]+=1
    return dict(episodes=len(rows),onset_counts=counts,outcomes=outcomes,snapshot_control_steps=snapshot_steps,
                active_interactions=int(prefix.sum()),pulse_actions=int(mask.sum()),
                charged_interactions=int(prefix.size),padding_interactions=int(prefix.size-prefix.sum()))


def audit_collection(root,previous,current,evidence):
    from .explorer_admission import validate_admission,behavior_identity
    cfg=evidence.read(root/'collection_spec.json');collection=root/'collection'
    receipt=evidence.read(collection/'explorer_admission.json')
    require(cfg.get('role')=='TRAIN','collection role is not TRAIN')
    require(cfg.get('explorer_checkpoint')==previous['explorer'],'collection did not inherit preceding E')
    evidence.lock(previous['explorer'],previous['explorer_sha256'])
    behavior=evidence.unpack(collection/'behavior.msgpack');parent=evidence.unpack(previous['explorer'])
    require(explorer_state_equal(behavior,parent),'collection E full state differs from preceding published E')
    for filename,sha in receipt['files'].items():evidence.lock(collection/filename,sha)
    with np.load(collection/'prefixes.npz',allow_pickle=False) as raw:tape={k:raw[k] for k in raw.files}
    identity=behavior_identity(behavior)
    bound={**cfg,'explorer_admission_v1_2':{**cfg['explorer_admission_v1_2'],**identity}}
    admitted=validate_admission(bound,receipt,tape,identity)
    update_cfg=evidence.read(root/'explorer_update_spec.json')
    require(Path(update_cfg['collection']).resolve()==collection and update_cfg['epochs']==4,'E update exposure/budget drift')
    feedback=evidence.read(update_cfg['feedback'])
    evidence.lock(current['explorer'],current['explorer_sha256'])
    require(Path(current['explorer']).resolve()==root/'explorer_update/state.msgpack','published E is not current update')
    learning_path=evidence.lock(root/'explorer_update/learning.npz')
    with np.load(learning_path,allow_pickle=False) as learning:
        expected_mask=admitted & np.asarray(feedback['eligible'],bool)[None,:]
        require(np.array_equal(learning['mask'],expected_mask),'E PPO mask includes padding/nonexecuted actions')
        terminal_reward=np.zeros(expected_mask.shape)
        for lane in range(expected_mask.shape[1]):
            ticks=np.flatnonzero(expected_mask[:,lane])
            if len(ticks):terminal_reward[ticks[-1],lane]=feedback['rewards'][lane]
        require(np.allclose(learning['reward'],terminal_reward),'E episode reward attribution changed with pulse duration')
    rows=evidence.read(collection/'candidates.json');result=audit_timing_rows(cfg,tape,rows)
    status=evidence.read(collection/'status.json')
    require(status['phase']=='completed','collection incomplete')
    for key in ('charged_interactions','active_interactions','padding_interactions'):
        require(status[key]==result[key],'collection cost mismatch: '+key)
    from ..unified_envelope_snapshot import load_unified_envelope_snapshot,snapshot_context_sha256
    for lane,row in enumerate(rows):
        require(row['prefix_sha256']==receipt['files']['prefixes.npz'],'candidate prefix hash drift')
        if row['pulse_outcome']!='valid_post_pulse':continue
        snapdir=Path(row['snapshot']);evidence.lock(snapdir/'identity.json');evidence.lock(snapdir/'snapshot.pkl')
        snap=load_unified_envelope_snapshot(snapdir);tick=row['snapshot_control_step']-1
        require(snapshot_context_sha256(snap)==row['snapshot_context_sha256'],'snapshot context drift')
        for attr,key in [('qpos','data/qpos'),('qvel','data/qvel'),('ctrl','data/ctrl'),
                         ('observation_fifo','history/frames'),('observation','obs/state'),
                         ('last_action','info/last_action'),('rng','info/rng')]:
            require(np.array_equal(getattr(snap,attr),tape['snap/'+key][tick,lane]),'snapshot is not lane-local final state: '+attr)
    result.update(behavior_identity=identity,requested_absolute_max=float(np.abs(tape['requested_delta']).max()),
                  effective_absolute_max=float(np.abs(tape['effective_delta'][tape['prefix_mask']]).max()),
                  clipping_channels=int(tape['action_clipped'][tape['prefix_mask']].sum()))
    return result


def audit_learner(previous,current,first,allow_bootstrap,evidence):
    from .learner_continuation import validate_saved_learner
    actor=current['actor'];receipt=current.get('learner') or actor.get('learner')
    require(receipt is not None,'published student has no full learner')
    saved=validate_saved_learner(receipt,actor['policy'],expected_local_transitions=128000)
    evidence.lock(receipt['path'],receipt['sha256']);evidence.lock(saved['state_path'],saved['state_sha256'])
    init=evidence.read(Path(actor['training'])/'learner/initialization.json')
    parent=previous.get('learner') or previous['actor'].get('learner')
    keys=('actor_sha256','critic_sha256','normalizer_sha256','optimizer_sha256','rng_sha256')
    if parent:
        prior=evidence.read(parent['path'],parent['sha256']);evidence.lock(prior['state_path'],prior['state_sha256'])
        require(init['mode']=='full_learner_continuation' and init['parent']==parent,'learner parent was reset')
        for key in keys:require(init[key]==prior[key],'learner initialization mismatch: '+key)
        require(init['lifetime_transition_offset']==prior['lifetime_transitions'],'learner counter reset')
    else:
        require(first and allow_bootstrap and init['mode']=='one_time_legacy_optimizer_bootstrap' and init['parent'] is None,
                'undeclared or repeated legacy learner bootstrap')
        for key in keys[:3]:require(init[key]==previous['actor']['policy'][key],'bootstrap source identity mismatch: '+key)
    require(saved['lifetime_transitions']==init['lifetime_transition_offset']+128000,'student lifetime count mismatch')
    return dict(mode=init['mode'],parent=parent,initialization={k:init[k] for k in keys},
                lifetime_transitions=saved['lifetime_transitions'])


def audit_generator(root,previous,current,evidence):
    from .artifacts import validate_generator_receipt
    old=previous['generator'];new=current['generator']
    validate_generator_receipt(old);validate_generator_receipt(new)
    def state(receipt):
        m=evidence.read(receipt['checkpoint_manifest'],receipt['checkpoint_manifest_sha256'])
        path=Path(receipt['checkpoint_manifest']).parent/'state.msgpack';evidence.lock(path,m['state_sha256'])
        raw=evidence.unpack(path)
        require(set(raw)=={'params','ema','optimizer','rng','normalizer','updates'},'incomplete generator training state')
        require(m['inference_parameters']=='ema' and int(raw['updates'])==m['updates'],'generator inference/count drift')
        def finite(x):
            if isinstance(x,dict):return all(finite(v) for v in x.values())
            return bool(np.isfinite(np.asarray(x)).all())
        require(finite(raw),'nonfinite generator state')
        return m,raw
    oldmeta,oldstate=state(old);newmeta,newstate=state(new)
    require(new.get('generator_update_policy')=='last_valid' and new.get('inference_parameters')=='ema','generator continuation rule drift')
    corpus=evidence.read(current['corpus']['path'],current['corpus']['sha256'])
    if new['status']=='skipped_no_new_data':
        require(not corpus['new_data'],'G skipped admitted new successes')
        require(tree_equal(oldstate,newstate) and new['updates']==0,'no-new-data changed generator state')
        require(new['total_charged_updates']==previous['generator_lifetime_charged_updates'],'no-new-data charged fabricated updates')
        return dict(status=new['status'],state_updates=newmeta['updates'],updates=0)
    require(new['status']=='completed' and corpus['new_data'],'generator stage not completed or fabricated training without new data')
    config=evidence.read(root/'generator_incremental_0000_config.json')
    require(config['incumbent']==old and config['generator_update_policy']=='last_valid','G worker restored wrong parent/policy')
    require(config['source_frozen_policy']==previous['actor']['frozen_policy'],'G teacher tail/source drift')
    require(config['dev_fixture']==previous['dev_fixture'] and config['dev_fixture_sha256']==previous['dev_fixture_sha256'],
            'G fixed old-dev monitoring fixture changed')
    evidence.lock(config['dev_fixture'],config['dev_fixture_sha256'])
    attempt=Path(new['checkpoint_manifest']).parent.parent
    initial=evidence.read(attempt/'incumbent/manifest.json')
    evidence.lock(attempt/'incumbent/state.msgpack',initial['state_sha256'])
    require(tree_equal(evidence.unpack(attempt/'incumbent/state.msgpack'),oldstate),'G params/EMA/optimizer/RNG/normalizer not inherited')
    report=evidence.read(attempt/'generator_selection.json')
    require(report['generator_update_policy']=='last_valid' and report['old_dev_metric']=='monitor_only','hidden old-dev selection')
    require(report['selected']==f"update_{report['updates']:04d}" and Path(new['checkpoint_manifest']).parent.name==report['selected'],
            'G selected checkpoint is not final complete state')
    require(report['state_updates']==newmeta['updates']==oldmeta['updates']+report['updates'],'G continued state count drift')
    require(tree_equal(oldstate['normalizer'],newstate['normalizer']),'G fixed normalizer changed')
    require(new['total_charged_updates']==previous['generator_lifetime_charged_updates']+new['charged_updates'],
            'G lifetime billing reset')
    require(newmeta['identity']==oldmeta['identity'],'G conditioning identity changed')
    return dict(status=new['status'],initial_state_updates=oldmeta['updates'],state_updates=newmeta['updates'],
        updates=report['updates'],total_charged_updates=new['total_charged_updates'],
        selected=new['checkpoint_manifest'],monitoring_best=new['monitoring_best'],inference_parameters='ema')


def audit_data(root,current,evidence):
    from .artifacts import load_corpus
    from .feedback_data import admit_trace,MIX
    corpus=load_corpus(current['corpus']);evidence.lock(current['corpus']['path'],current['corpus']['sha256'])
    require(corpus['requested_mix']==MIX,'generator sampler mix changed')
    splits={}
    for row in corpus['admission']:
        ancestor=row['root_episode_id'];split=row['inherited_split']
        require(ancestor not in splits or splits[ancestor]==split,'ancestor crosses TRAIN/DEV/TEST splits')
        splits[ancestor]=split
    counts={}
    for group,traces in corpus['groups'].items():
        counts[group]=len(traces)
        for trace in traces:
            m=trace['metadata'];require(m['role']=='train' and m['inherited_split']=='generator_train','DEV/TEST in G corpus')
            row=admit_trace(trace,adoption=trace.get('adoption',{}),expected={},splits=splits)
            require(row['eligible'],'G trace no longer admitted')
            if group=='actor_new':require(m['actor_sha256']==current['actor']['policy']['actor_sha256'],'student success belongs to another Actor')
    source=evidence.read(root/'source_rows.json')['rows'];feedback=evidence.read(root/'explorer_feedback.json')
    require(len(source)==len(feedback['rows'])==128,'TRAIN feedback denominator drift')
    roots={r['root_id'] for r in source}
    require(all(r.get('data_role')=='train' and r.get('panel')=='train' for r in source+feedback['rows']),
            'non-TRAIN E feedback')
    require({r['root_id'] for r in feedback['rows']}==roots,'E feedback uses unrelated contexts')
    return dict(corpus_counts=counts,requested_mix=corpus['requested_mix'],realized_mix=corpus['realized_mix'],
                train_feedback_episodes=128)


def audit_integration(plan_or_root):
    """Return JSON-ready evidence; failures never authorize a200-round continuation.

    Accept an in-memory plan, plan.json path, or run root. No files are written,
    no simulator is created, and no optimization or GPU worker is launched.
    """
    evidence=Evidence();report=dict(schema='jit_bridge_two_round_integration_audit_v1',ready_for_200=False,
        simulation_executed=False,optimization_executed=False,performance_gate=False,
        claim_scope='engineering state/timing/data/budget audit only; no capability improvement claim',rounds=[],errors=[])
    try:
        if isinstance(plan_or_root,dict):plan=plan_or_root;root=Path(plan['output']).resolve()
        else:
            path=Path(plan_or_root).resolve();root=path.parent if path.is_file() else path
            plan=evidence.read(root/'plan.json')
        status=evidence.read(root/'status.json')
        require(plan['rounds']==2 and status['phase']=='completed' and status['completed_rounds']==2,
                'pilot requires exactly two completed published rounds')
        from .continuation_profile import validate_profile,budget_dry_run
        profile=validate_profile(plan['profile']);budget_dry_run(plan)
        require(profile['pulse_steps']==3,'pilot audit requires declared L3')
        from .continuation_boundary import resolve_completed_boundary
        boundary=resolve_completed_boundary(root)
        evidence.locks.update(boundary['locks'])
        paths=evidence.read(root/'completed_rounds.json');require(len(paths)==2,'completed journal count mismatch')
        initial=plan['continuation_bundle'];previous=evidence.read(initial['path'],initial['sha256'])
        total_physics=0
        for local,path in enumerate(paths):
            roundroot=Path(path).resolve();require(roundroot.parent==root,'round outside pilot')
            spec=evidence.read(roundroot/'production.json');current=evidence.read(roundroot/'current_source.json')
            require(evidence.read(roundroot/'status.json')['phase']=='completed','partial round')
            require(spec['continuation']==previous,'round parent bundle differs from preceding publication')
            require(current['actor']==current['evaluated_student'],'published Actor is not this continuous round student')
            require(spec['generator_update_policy']=='last_valid','round G update policy drift')
            source=evidence.read(spec['source_frozen_policy'])['policy']
            for key in ('actor_sha256','critic_sha256','normalizer_sha256'):
                require(source[key]==previous['actor']['policy'][key],'source is not preceding published Actor: '+key)
            row=dict(round=current['round'],collection=audit_collection(roundroot,previous,current,evidence),
                learner=audit_learner(previous,current,local==0,plan.get('allow_legacy_optimizer_bootstrap',False),evidence),
                generator=audit_generator(roundroot,previous,current,evidence),data=audit_data(roundroot,current,evidence))
            costs=evidence.read(roundroot/'costs.json')
            require(all(type(c['charged_interactions']) is int and c['charged_interactions']>=0 for c in costs),
                    'invalid physical charged counts')
            physical=sum(c['charged_interactions'] for c in costs)
            require(physical<=plan['budgets']['per_round_physics'],'round physics budget exceeded')
            require(sum(c.get('charged_updates',0) for c in costs if c['stage'].startswith('generator_'))<=2000,
                    'round G update budget exceeded')
            require(all(c.get('phase')=='completed' for c in costs),'unreconciled stage reservation')
            row['charged_physics']=physical;total_physics+=physical;report['rounds'].append(row);previous=current
        require(total_physics<=plan['budgets']['max_physics'],'pilot physical budget exceeded')
        require(previous==boundary['bundle'],'final published bundle mismatch')
        report.update(ready_for_200=True,charged_physics=total_physics,
                      continuation_requires_explicit_200_round_plan=True)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,IndexError) as error:
        report['errors'].append(str(error))
    report['evidence_locks']=evidence.locks
    return report
