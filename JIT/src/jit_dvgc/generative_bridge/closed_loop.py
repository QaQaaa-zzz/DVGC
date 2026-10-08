"""Bounded experimental Actor/E/G continuation; formal adoption stays separate."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from .campaign import CampaignRunner
from .production import (read, implementation_identity, implementation_files,
                         actor_traces, read_teacher_traces, is_source_conflict_quarantine,
                         start_notifications, wall_deadline)
from .contracts import file_sha, digest
from .protocol import atomic_json


def continuation_decision(student, nominal, learner_completed):
    return {**{k:student[k] for k in ('actor_sha256','normalizer_sha256') if k in student},
        'adopted':bool(learner_completed and len(nominal)==4 and all(v==1 for v in nominal)),
        'scope':'experimental_continuation','formal_adopted':False,
        'nominal_labels':nominal,'forgetting_is_veto':False,
        'criterion':'completed finite learner and all declared nominal full-task successes',
        'authorization':'user 2026-10-08: allow forgetting; iterate and evaluate generalization'}


def continuous_decision(student,learner_completed):
    if not learner_completed:raise ValueError('incomplete learner cannot continue')
    return {**{k:student[k] for k in ('actor_sha256','normalizer_sha256') if k in student},
        'adopted':True,'scope':'experimental_continuous_training','formal_adopted':False,
        'nominal_evaluation':'disabled_by_user','performance_gate':False,
        'criterion':'continue completed finite learner regardless of performance',
        'authorization':'user 2026-10-08: cancel nominal checks; continue the student across 200 rounds'}


def feedback_rows(before, after, teachers, decision):
    after={r['root_id']:r for r in after};rows=[]
    for original in before:
        r=deepcopy(original);rid=r['root_id'];teacher=teachers.get(rid,{})
        r.update(initial_label=r['label'],learning_attempted=rid in after,
                 successor_adopted=decision['adopted'])
        if rid in after:r['label']=after[rid]['label']
        if teacher.get('teacher_status')=='invalid':
            r.update(initial_label=None,source_recheck_label=None)
        elif 'source_recheck_label' in teacher:
            r['source_recheck_label']=teacher['source_recheck_label']
        rows.append(r)
    return rows


def stress_conditions():
    return [dict(amplitude=a,onset=t,episode_ids=list(range(i*32,(i+1)*32)),
                 master_seed=8100801,role='student_dev',round=0)
            for i,(a,t) in enumerate((a,t) for a in (.25,.4,.6) for t in (0,10,20))]


def compare_stress(baseline,candidate):
    from ..policy_retention import paired_counts
    if set(baseline)!=set(candidate):raise ValueError('stress conditions differ')
    result={}
    for key,before in baseline.items():
        after=candidate[key]
        if before['condition']!=after['condition'] or len(before['labels'])!=len(after['labels']):
            raise ValueError('stress conditions or denominator differ')
        result[key]=dict(paired_counts(before['labels'],after['labels']),
            source_success=before['labels'].count(1),student_success=after['labels'].count(1),
            denominator=len(before['labels']),condition=before['condition'])
    return result


def tail_identity(policy, protocol):
    return dict(actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256'],
                model_sha256=policy['xml_sha256'],protocol_sha256=protocol)


def adopt_tail(path, baseline, previous, student, decision):
    if not decision['adopted']:raise ValueError('cannot publish unadopted experimental tail')
    receipt=dict(schema='jit_bridge_teacher_tail_adoption_v1_2',adopted=True,
        baseline_identity=baseline,previous_tail=previous,candidate_tail=student['policy'],
        frozen_policy={'path':student['frozen_policy'],'sha256':file_sha(student['frozen_policy'])},
        decision=decision,scope='experimental_continuation',formal_adopted=False)
    atomic_json(path,receipt)
    return {'path':str(path),'sha256':file_sha(path)}


def prepare_closed_loop(previous,output,repository,*,neighborhood=False):
    from .artifacts import validate_generator_receipt,load_corpus
    from .worker import source_payload
    previous=Path(previous).resolve();output=Path(output).resolve();repo=Path(repository).resolve()
    if read(previous/'status.json')['phase']!='completed':raise ValueError('completed parent required')
    audit=read(previous/'recovery_audit.json');campaign=Path(audit['previous'])
    for p,h in audit['input_sha256'].items():
        if file_sha(p)!=h:raise ValueError('parent evidence changed')
    parent=read(campaign/'production.json');bundle=read(previous/'current_source.json')
    candidate=read(campaign/'acceptance_report.json')['C']['student']
    source_payload(candidate['frozen_policy']);validate_generator_receipt(bundle['generator']);load_corpus(bundle['corpus'])
    if read(Path(candidate['training'])/'status.json')['status']!='completed':raise ValueError('C incomplete')
    spec=dict(schema='jit_bridge_experimental_closed_loop_v1',output=str(output),repository=str(repo),
        implementation_commit=implementation_identity(repo),implementation_files=implementation_files(repo),
        parent_reports=str(previous),parent_campaign=str(campaign),initial_student=candidate,
        prior_physics_charged=read(previous/'status.json')['charged_interactions'],
        protocol_version='1.2_experimental',initial_bundle=bundle,baseline_frozen_policy=parent['source_frozen_policy'],
        source_runtime=parent['source_runtime'],rounds=2,seed=8100802,
        uniform_episode_fraction=0.,teacher_colored_noise_candidates=0,
        budgets=dict(max_physics=3300000,per_round_physics=1500000,max_wall_seconds=86400,
                     student_transitions=256000,generator_updates=4000,explorer_epochs_per_round=4),
        stress_panel=stress_conditions(),experimental_adoption=True,formal_adoption=False,
        final_test_open=False,automatic_extension=False,automatic_retry=False,
        locks={str(previous/'current_source.json'):file_sha(previous/'current_source.json'),
               str(campaign/'acceptance_report.json'):file_sha(campaign/'acceptance_report.json')})
    output.mkdir(parents=True,exist_ok=False)
    if neighborhood:
        from .neighborhood_history import seed_history,neighborhood_config
        history=output/'initial_neighborhood_history.json'
        atomic_json(history,seed_history(campaign))
        spec.update(neighborhood=neighborhood_config(),neighborhood_history=str(history))
        spec['locks'][str(history)]=file_sha(history)
    atomic_json(output/'plan.json',spec)
    atomic_json(output/'status.json',{'phase':'prepared','completed_rounds':0})
    return spec


def continuation_budget(rounds):
    if type(rounds) is not int or not 1<=rounds<=200:
        raise ValueError('additional rounds must be an integer in [1,200]')
    return dict(max_physics=1500000*rounds,per_round_physics=1500000,max_wall_seconds=604800,
                student_transitions=128000*rounds,generator_updates=2000*rounds,explorer_epochs_per_round=4)


def prepare_continuation(previous,output,repository,*,rounds,continuous=False):
    """Continue only a completed published bundle, in a fresh bounded series."""
    from .artifacts import validate_generator_receipt,load_corpus
    from .worker import source_payload
    old=Path(previous).resolve();root=Path(output).resolve();repo=Path(repository).resolve()
    status=read(old/'status.json');plan=read(old/'plan.json')
    if status['phase']!='completed' or status['completed_rounds']!=plan['rounds']:
        raise ValueError('completed parent series required')
    bundle_path=old/'current_source.json';bundle=read(bundle_path)
    source_payload(bundle['actor']['frozen_policy'])
    validate_generator_receipt(bundle['generator']);load_corpus(bundle['corpus'])
    for path,h in [(bundle['explorer'],bundle['explorer_sha256']),
                   (bundle['demo']['path'],bundle['demo']['sha256']),
                   (bundle['dev_fixture'],bundle['dev_fixture_sha256'])]:
        if file_sha(path)!=h:raise ValueError('parent continuation artifact changed')
    panels=plan.get('reference_panels',dict(baseline=str(old/'round_0001/baseline_stress.json'),
                                          initial=str(old/'round_0001/initial_stress.json')))
    plan=deepcopy(plan)
    for key in ('original_started_unix','baseline_reuse','recovery'):
        plan.pop(key,None)
    plan.update(schema='jit_bridge_experimental_continuation_v1',output=str(root),repository=str(repo),
        implementation_commit=implementation_identity(repo),implementation_files=implementation_files(repo),
        rounds=rounds,round_offset=bundle['round'],budgets=continuation_budget(rounds),
        authorization='user_requested_additional_rounds',continuation_parent=str(old),
        uniform_episode_fraction=0.,teacher_colored_noise_candidates=0,
        continuation_bundle={'path':str(bundle_path),'sha256':file_sha(bundle_path)},
        reference_panels=panels,minimum_free_disk_bytes=20*1024**3,
        prior_physics_charged=status['lifetime_physics_charged'])
    for path in (bundle_path,old/'status.json',old/'plan.json',*map(Path,panels.values())):
        plan['locks'][str(path)]=file_sha(path)
    root.mkdir(parents=True,exist_ok=False)
    if continuous:
        # The legacy pilot retained C. User now explicitly chooses its latest student.
        bundle=deepcopy(bundle)
        candidate=bundle['evaluated_student'];source_payload(candidate['frozen_policy'])
        decision=continuous_decision(candidate,read(Path(candidate['training'])/'status.json')['status']=='completed')
        old_source=bundle['actor']['policy'];bundle['actor']=candidate
        spec=read(old/f"round_{bundle['round']:04d}/production.json")
        if candidate['actor_sha256']!=old_source['actor_sha256']:
            bundle['tail_lineage'].append(adopt_tail(root/'initial_continuous_tail.json',spec['baseline_identity'],
                old_source,candidate,decision))
        bundle['learner']=candidate.get('learner')
        if 'seed_support' not in bundle:
            seed=Path(read(old/f"round_{bundle['round']:04d}/source_phase_result.json")['panel_support_path'])
            bundle['seed_support']={'path':str(seed),'sha256':file_sha(seed)}
        initial=root/'initial_continuation.json';atomic_json(initial,bundle)
        plan.update(continuous_learning=True,nominal_evaluation=False,
            continuation_bundle={'path':str(initial),'sha256':file_sha(initial)},
            allow_legacy_optimizer_bootstrap=bundle['learner'] is None)
        plan['locks'][str(initial)]=file_sha(initial)
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',completed_rounds=0,declared_rounds=rounds,
                                      round_offset=bundle['round']))
    return plan


def baseline_reuse(name, mode, spec, maximum, entry):
    """Reuse only pinned completed P0 evaluation, never learning or partial work."""
    if name not in {f'baseline_stress_{i:02d}' for i in range(9)} or mode!='collect':
        raise ValueError('only baseline stress collection can be reused')
    output=Path(entry['output'])
    required={entry['spec'],entry['execution'],str(output/'status.json'),str(output/'candidates.json')}
    if not required.issubset(entry['locks']):raise ValueError('missing reuse evidence locks')
    for p,h in entry['locks'].items():
        if file_sha(p)!=h:raise ValueError('reuse evidence changed')
    before=read(entry['spec'])
    if {k:v for k,v in before.items() if k!='gate'}!={k:v for k,v in spec.items() if k!='gate'}:
        raise ValueError('reused baseline protocol differs')
    status=read(output/'status.json');execution=read(entry['execution']);cost=entry['cost']
    if (status['phase']!='completed' or execution['phase']!='completed'
            or not execution['stages'] or any(s['phase']!='completed' or s['returncode']!=0 for s in execution['stages'])
            or cost['stage']!=name or cost.get('accounting')!='measured' or cost.get('phase')!='completed'
            or cost.get('charged_updates')!=0 or type(status['charged_interactions']) is not int
            or not 0<=status['charged_interactions']<=maximum
            or cost['charged_interactions']!=status['charged_interactions']
            or len(read(output/'candidates.json'))!=32):
        raise ValueError('incomplete or inconsistent baseline reuse')
    return output


class ClosedLoopRound(CampaignRunner):
    """Reuse the validated teacher/student implementations, with fresh round data."""
    def measured(self,name,mode,spec,maximum,*,output=None):
        entry=self.spec.get('baseline_reuse',{}).get(name)
        if entry is not None:
            reused=baseline_reuse(name,mode,spec,maximum,entry)
            if any(c['stage']==name for c in self.costs):raise ValueError('duplicate reuse charge')
            self.costs.append({**entry['cost'],'reused_from':str(reused)})
            atomic_json(self.root/'costs.json',self.costs)
            atomic_json(self.root/(name+'_reuse.json'),entry)
            return reused
        path=self.root/(name+'_spec.json');atomic_json(path,spec)
        output=Path(output) if output else self.root/name
        cost=self.child(name,['JIT/cli/run_pulse_exploration.py','--mode',mode,
            '--spec',path,'--output',output],maximum)
        receipt=read(output/'status.json')
        if receipt['phase']!='completed':raise ValueError('incomplete child')
        charged=receipt['charged_interactions']
        if type(charged) is not int or not 0<=charged<=maximum:
            raise ValueError('child exceeds interaction reservation')
        cost.update(charged_interactions=receipt['charged_interactions'],accounting='measured')
        atomic_json(self.root/'costs.json',self.costs)
        return output

    def stress(self,name,bank,policy):
        panels={}
        for i,c in enumerate(stress_conditions()):
            cfg={**self.runtime,'bank':bank,'proposer':policy,'order':[policy],
                'controller_mode':'fixed_random','num_envs':32,'pulse_steps':3,'seed':8100801,'round_index':0,
                'full_episode_rollout':True,'pulse_start_schedule':[c['onset']],
                'delta_limit':[c['amplitude']]*4,'nominal_source_rollout':False,
                'pulse_protocol_v1_2':{k:c[k] for k in ('master_seed','role','round','episode_ids')}}
            path=self.measured(f'{name}_{i:02d}','collect',cfg,12800)
            rows=read(path/'candidates.json');labels=[r.get('prefix_label') for r in rows]
            if len(labels)!=32:raise ValueError('stress denominator changed')
            panels[f"a{c['amplitude']}_t{c['onset']}"]=dict(condition=c,labels=labels,
                evaluated_policy=policy,bank_sha256=file_sha(bank),
                successes=labels.count(1),failures=labels.count(0),unknown=labels.count(None),
                path=str(path/'candidates.json'))
        atomic_json(self.root/(name+'.json'),panels)
        return panels

    def _run(self):
        from .source_data import build_training_support
        from .source_phase import fresh_panels
        from .student_demo_bank import build_student_demo_bank
        from .artifacts import load_corpus,save_corpus
        from .feedback_data import build_corpus
        from .series import corpus_history
        from .rewards import feedback,WEIGHTS
        previous=self.spec['continuation'];index=self.spec['round_index']
        if self.spec.get('continuous_learning'):
            seed=previous['seed_support'];seed_support=Path(seed['path'])
            if file_sha(seed_support)!=seed['sha256']:raise ValueError('historical seed support changed')
            atomic_json(self.root/'seed_support_reuse.json',dict(**seed,physics=0,
                purpose='historical witnessed TRAIN reset states; not current Actor nominal qualification'))
        else:
            seed={**self.runtime,'controller_mode':'fixed_random','nominal_repeats':1}
            seed_path=self.measured('seed_support','seed_support',seed,80400)
            seed_support=seed_path/'support.json'
        cfg={**self.runtime,'controller_mode':'learned_residual','explorer_backend':'rsl_rl',
            'explorer_initialization':{'mode':'symmetric','latent_std':.6},
            'explorer_checkpoint':previous.get('explorer'),'num_envs':128,'round_index':index,
            'quality_mode':'discovery_conversion','minibatch_size':256,'clip':.2,'target_kl':.01,
            'value_coefficient':1.,'entropy_coefficient':.01,'epochs':4,
            'full_episode_rollout':False,'delta_limit':[.25]*4,'pulse_start_schedule':[0],
            'explorer_admission_v1_2':dict(run_id=self.spec['series_id'],collection_id=f'round_{index}',
                master_seed=self.spec['seed'],round=index,episode_ids=list(range(128)),uniform_episode_fraction=self.spec.get('uniform_episode_fraction',.2))}
        if self.spec.get('neighborhood') is not None:
            from .neighborhood_history import freeze_history
            path=self.root/'neighborhood_map.json'
            freeze_history(path,previous['neighborhood_history'],self.source['actor_sha256'],self.spec['neighborhood'],index)
            cfg.update(neighborhood=self.spec['neighborhood'],neighborhood_map=str(path),neighborhood_map_sha256=file_sha(path))
        collected=self.measured('collection','collect',cfg,384)
        original=read(collected/'candidates.json')
        for row in original:
            row.update(root_id=row['snapshot_context_sha256'],root_episode_id=row['prefix_file']+'::'+str(row['index']))
        # Batched evaluate preserves the original 128-row order, including terminal prefixes.
        rows=self.evaluate('source_suffix',original)
        for row in rows:
            row.update(root_id=row['snapshot_context_sha256'],root_episode_id=row['prefix_file']+'::'+str(row['index']),
                       data_role='train',panel='train',source_actor_sha256=self.source['actor_sha256'])
        aggregate=self.root/'source_rows.json'
        atomic_json(aggregate,dict(rows=rows,source_policy=self.source['name'],prior_labels_imported=False,
            inputs={str(collected/'candidates.json'):file_sha(collected/'candidates.json')}))
        support=self.root/'training_support.json'
        build_training_support(seed_support,aggregate,support,source_policy=self.source['name'],
            allow_historical_seed=bool(self.spec.get('continuous_learning')))
        retain_pending(support,previous['support'])
        self.panels=fresh_panels(rows,seed=self.spec['seed']+index)
        history=corpus_history(load_corpus(previous['corpus']))
        for trace in history:
            m=trace['metadata'];self.panels['splits'][m['root_episode_id']]=m['inherited_split']
        atomic_json(self.root/'panels.json',self.panels)
        teachers=self.teacher_search(previous['generator'])
        if any(t.get('teacher_status') in ('invalid','incomplete') and not is_source_conflict_quarantine(t)
               for t in teachers.values()):raise ValueError('teacher incomplete or engineering error')
        teaching=read_teacher_traces(teachers)
        demo=build_student_demo_bank(teaching,self.root/'student_demo_bank',
            source_identity=tail_identity(self.source,self.protocol),previous=previous['demo'],round_id=index,
            baseline_identity=self.spec['baseline_identity'],teacher_tail_lineage=previous['tail_lineage'])
        phase=dict(support_path=str(support),panel_support_path=str(seed_support),
            retention_ref=self.spec['retention_ref'],demo_manifest={'path':str(self.root/'student_demo_bank/manifest.json'),
                'sha256':file_sha(self.root/'student_demo_bank/manifest.json')})
        atomic_json(self.root/'source_phase_result.json',phase)
        student=self.journal.stage('train_student',phase,lambda:self.train_arm('student',phase,None))
        self.checkpoint_probes(student,phase,None)
        after=self.evaluate('student_train',rows,policy_bank=student['bank'],policy_name=student['policy']['name'])
        nominal_cfg={**self.runtime,'bank':student['bank'],'proposer':student['policy']['name'],
            'order':[student['policy']['name']],'nominal_source_rollout':True,'full_episode_rollout':True,
            'num_envs':4,'nominal_repeats':4,'pulse_steps':400,'pulse_start_schedule':[0],
            'delta_limit':[0.]*4,'explorer_checkpoint':None}
        if not self.spec.get('continuous_learning'):
            nominal_path=self.measured('student_nominal','collect',nominal_cfg,1600)
            nominal=[r['prefix_label'] for r in read(nominal_path/'candidates.json')]
        from .worker import source_payload
        _,payload=source_payload(student['frozen_policy'])
        import jax
        if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(
                (payload.actor_params,payload.observation_normalizer,payload.critic_params))):
            raise ValueError('nonfinite student checkpoint')
        learner_completed=read(Path(student['training'])/'status.json')['status']=='completed'
        decision=(continuous_decision(student,learner_completed) if self.spec.get('continuous_learning')
                  else continuation_decision(student,nominal,learner_completed))
        atomic_json(self.root/'acceptance.json',decision)
        from .outcomes import root_outcome,result_matrix
        usage=read(Path(student['training'])/'bridge_demo_usage.json')['demo_samples_by_root']
        evaluated={r['root_id']:r['label'] for r in after}
        outcomes=[root_outcome(dict(root_id=r['root_id'],training_eligible=True,
            teacher_status=teachers.get(r['root_id'],{}).get('teacher_status','not_scheduled'),
            source_recheck_label=teachers.get(r['root_id'],{}).get('source_recheck_label'),
            new_gain_eligible=teachers.get(r['root_id'],{}).get('new_gain_eligible'),
            quarantine_reason=teachers.get(r['root_id'],{}).get('reason'),
            student_label=evaluated.get(r['root_id']),student_adopted=decision['adopted'],
            n_direct_demo_examples_available=teachers.get(r['root_id'],{}).get('demo',{}).get('count',0),
            n_direct_demo_samples_used=usage.get(r['root_id'],0)),round_demo_samples_used=sum(usage.values()))
            for r in self.panels['original_pending']]
        atomic_json(self.root/'root_outcomes.json',outcomes)
        atomic_json(self.root/'result_matrix.json',result_matrix(outcomes))
        stress=self.stress('student_stress',student['bank'],student['policy']['name'])
        feedback_input=feedback_rows(rows,after,teachers,decision)
        # Novelty is relative to this frozen source's own arrival ledger.
        seen=previous.get('arrival_ledgers',{}).get(self.source['actor_sha256'],[])
        reward,eligible,cells,parts=feedback(feedback_input,seen)
        feedback_path=self.root/'explorer_feedback.json'
        atomic_json(feedback_path,dict(rewards=reward.tolist(),eligible=eligible.tolist(),components=parts,
            component_sums={k:float(np.sum(v)) for k,v in parts.items()},weights=WEIGHTS,
            teacher_success_bonus=0.,rows=feedback_input,adoption_scope='experimental_continuation'))
        receipt=read(collected/'explorer_admission.json')
        update_cfg={**cfg,'collection':str(collected),'feedback':str(feedback_path),'epochs':4,
            'explorer_admission_v1_2':{**cfg['explorer_admission_v1_2'],
                **{k:receipt[k] for k in ('behavior_actor_sha256','behavior_normalizer_sha256')}}}
        updated_e=self.measured('explorer_update','update',update_cfg,0)
        e_metrics=read(updated_e/'metrics.json')
        self.costs[-1]['explorer_optimizer_updates']=e_metrics['optimizer_updates']
        atomic_json(self.root/'costs.json',self.costs)
        actors=actor_traces([r for r in after if not r.get('prefix_terminal')],student['policy'],self.protocol,self.panels['splits'])
        corpus=build_corpus(history,teaching,actors,adoption=decision,
            expected={'model_sha256':self.source['xml_sha256'],'protocol_sha256':self.protocol},splits=self.panels['splits'])
        corpus_receipt=save_corpus(corpus,self.root/'feedback_corpus')
        updated_g=self.generator('incremental',corpus_receipt,previous['dev_fixture'],incumbent=previous['generator'])
        tail_lineage=list(previous['tail_lineage'])
        if decision['adopted']:
            tail_lineage.append(adopt_tail(self.root/'tail_adoption.json',self.spec['baseline_identity'],self.source,student,decision))
        ledgers={**previous.get('arrival_ledgers',{}),self.source['actor_sha256']:cells}
        bundle=dict(actor=student if decision['adopted'] else previous['actor'],generator=updated_g,
            explorer=str(updated_e/'state.msgpack'),explorer_sha256=file_sha(updated_e/'state.msgpack'),
            corpus=corpus_receipt,demo=phase['demo_manifest'],dev_fixture=previous['dev_fixture'],dev_fixture_sha256=previous['dev_fixture_sha256'],
            support=str(support),tail_lineage=tail_lineage,arrival_ledgers=ledgers,
            round=index,acceptance=decision,stress=stress,evaluated_student=student,formal_adopted=False,
            feedback_counts={k:len(v) for k,v in corpus['groups'].items()},explorer_metrics=e_metrics)
        if self.spec.get('continuous_learning'):
            bundle.update(learner=student['learner'],seed_support=previous['seed_support'])
        if self.spec.get('neighborhood') is not None:
            from .neighborhood_history import actor_evidence,teacher_evidence,source_recheck_evidence
            ancestors={r['root_episode_id'] for r in rows}
            bundle['neighborhood_history']=(previous['neighborhood_history']+
                actor_evidence(rows,ancestors,str(aggregate))+
                actor_evidence(after,ancestors,str(self.root/'evaluations/student_train'))+
                teacher_evidence(teachers,rows,str(self.root/'teachers'))+
                source_recheck_evidence(teachers,rows,str(self.root/'teachers')))
        atomic_json(self.root/'current_source.json',bundle)
        self.status('completed',stage='closed_loop_round',training_transitions=128000)
        return bundle


def retain_pending(current,previous):
    """Keep historical pending eligibility with original labels, never invent current labels."""
    from ..iterative_probe_training import candidate_support_view
    from ..evidence_integrity import canonical_sha256
    fresh=read(current);old=read(previous)
    pending={r['key']:r for r in old['entries'] if r.get('evidence_status')=='pending'}
    pending.update({r['key']:r for r in fresh['entries'] if r.get('evidence_status')=='pending'})
    if not pending:return
    witnessed=deepcopy(fresh)
    witnessed['entries']=[r for r in fresh['entries'] if r.get('witnessed')]
    # Recover the original witnessed schema from the current support adapter.
    from ..iterative_probe_training import SUPPORT_SCHEMA
    witnessed['schema']=SUPPORT_SCHEMA
    witnessed.pop('support_sha256',None);witnessed['support_sha256']=canonical_sha256(witnessed)
    result=candidate_support_view(witnessed,list(pending.values()),
        {**old['inputs'],str(previous):file_sha(previous)},pending_fraction=.5,max_pending_per_phase=len(pending))
    atomic_json(current,result)


def run_closed_loop(plan):
    root=Path(plan['output'])
    if (root/'started.json').exists():raise ValueError('explicit recovery required; never replay E updates')
    try:return _run_closed_loop(plan)
    except BaseException as error:
        if not (root/'status.json').exists() or read(root/'status.json').get('phase')!='failed':
            completed=read(root/'completed_rounds.json') if (root/'completed_rounds.json').exists() else []
            atomic_json(root/'status.json',dict(phase='failed',completed_rounds=len(completed),
                error=repr(error),**write_costs(plan)))
        raise


def _run_closed_loop(plan):
    from ..probe_bank import lock_probe_bank
    from .worker import source_payload
    root=Path(plan['output']);started=plan.get('original_started_unix',time.time());completed=[]
    if (root/'started.json').exists():raise ValueError('explicit recovery required; never replay E updates')
    atomic_json(root/'started.json',{'started_unix':started});start_notifications(plan)
    remaining_wall=plan['budgets']['max_wall_seconds']-(time.time()-started)
    if remaining_wall<=0:raise TimeoutError('original closed-loop deadline exhausted')
    continuation=plan.get('schema')=='jit_bridge_experimental_continuation_v1'
    if plan.get('experimental_adoption') is not True or plan.get('formal_adoption') is not False:
        raise ValueError('explicit experimental continuation required')
    if continuation:
        if plan.get('authorization')!='user_requested_additional_rounds':raise ValueError('missing continuation authorization')
        expected=continuation_budget(plan['rounds'])
        if type(plan.get('round_offset')) is not int or plan['round_offset']<1:
            raise ValueError('invalid completed-round offset')
        if plan.get('minimum_free_disk_bytes')!=20*1024**3:raise ValueError('disk guard changed')
    else:
        if plan.get('rounds')!=2:raise ValueError('explicit two-round experimental continuation required')
        expected=dict(max_physics=3300000,per_round_physics=1500000,max_wall_seconds=86400,
            student_transitions=256000,generator_updates=4000,explorer_epochs_per_round=4)
    if plan['budgets']!=expected:
        raise ValueError('declared closed-loop budget changed')
    for path,h in plan['locks'].items():
        if file_sha(path)!=h:raise ValueError('closed-loop input changed: '+path)
    if implementation_identity(plan['repository'])!=plan['implementation_commit']:
        raise ValueError('closed-loop code identity changed')
    parent=read(Path(plan['parent_campaign'])/'production.json')
    phase=read(Path(plan['parent_campaign'])/'stages/fresh_source_phase.json')['result']
    baseline,_=source_payload(plan['baseline_frozen_policy'])
    protocol=digest({'version':'1.2','task':'stable_forward_recovery','H':16,'horizon':400,
                     'source_physics':baseline['xml_sha256'],'source_reward':parent['source_runtime']['reward_mode']})
    identity=tail_identity(baseline,protocol)
    if continuation:
        item=plan['continuation_bundle']
        if file_sha(item['path'])!=item['sha256']:raise ValueError('published continuation changed')
        previous=read(item['path'])
        if previous['round']!=plan['round_offset']:raise ValueError('continuation round mismatch')
    else:
        initial=plan['initial_student']
        # Existing C passed all four nominal checks; this explicit new receipt does not alter old rejection.
        olddecision=read(Path(plan['parent_campaign'])/'acceptance_report.json')['C']['acceptance']
        n=olddecision['counts']['nominal']
        if n['retained']!=4 or n['lost'] or n['unknown']:raise ValueError('initial C nominal evidence invalid')
        decision=continuation_decision(initial,[1]*4,True)
        lineage=[adopt_tail(root/'initial_experimental_adoption.json',identity,baseline,initial,decision)]
        previous=dict(actor=initial,generator=plan['initial_bundle']['generator'],corpus=plan['initial_bundle']['corpus'],
            demo=plan['initial_bundle']['student_demo_bank'],dev_fixture=phase['bootstrap']['dev_fixture'],
            dev_fixture_sha256=phase['bootstrap']['dev_fixture_sha256'],
            support=phase['support_path'],tail_lineage=lineage,explorer=None)
        if plan.get('neighborhood') is not None:
            history=plan['neighborhood_history']
            if file_sha(history)!=plan['locks'][history]:raise ValueError('initial history changed')
            previous['neighborhood_history']=read(history)
    try:
        with wall_deadline(remaining_wall):
            for index in range(plan.get('round_offset',0)+1,plan.get('round_offset',0)+plan['rounds']+1):
                if continuation:
                    import shutil
                    if shutil.disk_usage(root).free<plan['minimum_free_disk_bytes']:
                        raise RuntimeError('free disk below declared 20GiB margin')
                remaining_physics=plan['budgets']['max_physics']-write_costs(plan)['charged_interactions']
                if remaining_physics<=0:raise ValueError('series physical budget exhausted')
                actor=previous['actor'];directory=root/f'round_{index:04d}';directory.mkdir()
                bank=read(parent['source_runtime']['bank'])
                bank={k:bank[k] for k in ('task','max_ticks','label_interaction_budget','max_candidates_per_process')}
                bank.update(version=f'closed_loop_{index}',members=[{'frozen_policy':actor['frozen_policy'],'roles':['proposer','evaluator']}])
                lock_probe_bank(bank,directory/'bank.json')
                runtime={**parent['source_runtime'],'bank':str(directory/'bank.json'),
                         'proposer':actor['policy']['name'],'order':[actor['policy']['name']],
                         'seed':plan['seed']+index,'round_index':index}
                if plan.get('execution_gate') is not None:runtime['gate']=plan['execution_gate']
                locks={**plan['locks'],previous['dev_fixture']:previous['dev_fixture_sha256'],
                    actor['frozen_policy']:file_sha(actor['frozen_policy'])}
                if previous.get('explorer'):locks[previous['explorer']]=previous['explorer_sha256']
                spec=dict(schema='jit_bridge_experimental_round_v1',protocol_version='1.2',execute=True,
                    output=str(directory),repository=plan['repository'],implementation_commit=plan['implementation_commit'],
                    implementation_files=plan['implementation_files'],source_frozen_policy=actor['frozen_policy'],
                    source=actor['policy'],source_runtime=runtime,seed=plan['seed']+index,round_index=index,
                    series_id=root.name,continuation=previous,baseline_identity=identity,
                    generator_reference_frozen_policy={'path':plan['baseline_frozen_policy'],'sha256':file_sha(plan['baseline_frozen_policy'])},
                    retention_reference_frozen_policy=plan['baseline_frozen_policy'],retention_ref=phase['retention_ref'],
                    teacher_layout=('source_control_in_candidate_batch' if plan.get('teacher_colored_noise_candidates')==0
                                    else 'source_control_in_32_world_batch'),locks=locks,
                    uniform_episode_fraction=plan.get('uniform_episode_fraction',.2),
                    teacher_colored_noise_candidates=plan.get('teacher_colored_noise_candidates',15),
                    budgets={'max_physics':min(remaining_physics,plan['budgets']['per_round_physics']+(230400 if index==1 else 0)),
                             'max_supervised_updates':2000,'max_wall_seconds':plan['budgets']['max_wall_seconds']-(time.time()-started)})
                if plan.get('neighborhood') is not None:spec['neighborhood']=plan['neighborhood']
                if plan.get('continuous_learning'):
                    if plan.get('nominal_evaluation') is not False:raise ValueError('continuous nominal check must be disabled')
                    spec.update(continuous_learning=True,allow_legacy_optimizer_bootstrap=
                        bool(index==plan['round_offset']+1 and plan.get('allow_legacy_optimizer_bootstrap')))
                if index==1 and plan.get('baseline_reuse'):spec['baseline_reuse']=plan['baseline_reuse']
                atomic_json(directory/'production.json',spec);atomic_json(directory/'panels.json',{})
                runner=ClosedLoopRound(spec)
                atomic_json(root/'status.json',dict(phase='running',round=index,completed_rounds=len(completed),declared_rounds=plan['rounds'],round_offset=plan.get('round_offset',0),current_round=str(directory)))
                atomic_json(root/'ACTIVE_RUN.json',dict(name=root.name,lineage=str(root/'status.json'),execution=str(directory/'status.json')))
                if index==1:
                    runner.stress('baseline_stress',parent['source_runtime']['bank'],baseline['name'])
                    runner.stress('initial_stress',runtime['bank'],actor['policy']['name'])
                previous=runner.run();completed.append(str(directory))
                baseline_panel=read(plan['reference_panels']['baseline'] if continuation else root/'round_0001/baseline_stress.json')
                initial_panel=read(plan['reference_panels']['initial'] if continuation else root/'round_0001/initial_stress.json')
                atomic_json(directory/'generalization_comparison.json',dict(
                    versus_P0=compare_stress(baseline_panel,previous['stress']),
                    versus_initial_C=compare_stress(initial_panel,previous['stress']),
                    evidence_role='fixed development stress panel; not final TEST',
                    equal_total_budget_control=False))
                atomic_json(root/'current_source.json',previous)
                atomic_json(root/'completed_rounds.json',completed)
                write_costs(plan)
        atomic_json(root/'status.json',dict(phase='completed',completed_rounds=len(completed),declared_rounds=plan['rounds'],round_offset=plan.get('round_offset',0),**write_costs(plan)))
    except BaseException as error:
        atomic_json(root/'status.json',dict(phase='failed',completed_rounds=len(completed),error=repr(error),**write_costs(plan)))
        raise
    return previous


def write_costs(plan):
    costs=[dict(c,round=p.parent.name) for p in Path(plan['output']).glob('round_*/costs.json') for c in read(p)]
    carried=plan.get('carried_physics',0)
    if type(carried) is not int or carried<0:raise ValueError('invalid carried physics cost')
    physical=carried+sum(c['charged_interactions'] for c in costs)
    if physical>plan['budgets']['max_physics']:raise ValueError('series physical budget exceeded')
    result=dict(charged_interactions=physical,lifetime_physics_charged=physical+plan['prior_physics_charged'],supervised_updates=sum(c.get('charged_updates',0) for c in costs),
                explorer_optimizer_updates=sum(c.get('explorer_optimizer_updates',0) for c in costs))
    atomic_json(Path(plan['output'])/'cost_ledger.json',dict(costs=costs,carried_physics=carried,limits=plan['budgets'],**result))
    return result
