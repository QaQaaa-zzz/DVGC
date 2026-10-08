"""Opt-in v1.2 staged campaign; legacy fixed-source series is unchanged."""
from copy import deepcopy
from pathlib import Path
import math
import time
import numpy as np
from .production import (ProductionRunner,read,implementation_identity,implementation_files,
                         read_teacher_traces,actor_traces)
from .protocol import atomic_json
from .contracts import file_sha,digest
from .stage_plan import require_stage


def nominal_evidence(status, rows):
    if status.get('phase')!='completed':raise ValueError('nominal execution not completed')
    return {'source_identity_valid':True,'action_parity':True,
            'nominal_success':bool(rows) and all(r.get('prefix_label')==1 for r in rows),
            'nominal_repeats':len(rows),'independent_conditions':1}


def acceptance_v12(before,after,student):
    from ..policy_retention import paired_counts
    counts={k:paired_counts(before[k],after[k]) for k in ('core','protected','new_roots','nominal')}
    n=counts['protected']['old_positive'];allowed=math.floor(n*.02+1e-10)
    adopted=(n>0 and counts['core']['old_positive']>0 and counts['core']['lost']==0
        and counts['protected']['lost']<=allowed and not any(c['unknown'] for c in counts.values())
        and counts['new_roots']['gained']-counts['new_roots']['lost']>=1
        and bool(before['nominal']) and all(v==1 for v in before['nominal']+after['nominal']))
    return {**student,'adopted':adopted,'counts':counts,'protected_allowed_losses':allowed,
            'thresholds':{'core_max_lost':0,'protected_min_retention':.98,'new_net_gain':1},
            'evidence_role':'development; final TEST unopened'}


def source_root_labels(rows,teachers):
    return [None if teachers[r['root_id']].get('reason')in ('same_layout_source_recheck','source_control_repeat_conflict')
        else teachers[r['root_id']]['source_recheck_label'] for r in rows]


def prepare_campaign(stage_plan, audit_dir, output, repository):
    audit_dir=Path(audit_dir).resolve();root=Path(output).resolve();repo=Path(repository).resolve()
    plan=read(stage_plan);audit=read(audit_dir/'source_audit.json')
    if plan['schema']!='jit_bridge_stage_plan_v1_2' or plan.get('execute') is not True:raise ValueError('explicit v1.2 execution declaration required')
    if (plan['source_checkpoint']!=audit['source_checkpoint'] or plan['actor_sha256']!=audit['parameter_hashes']['actor']
            or plan['normalizer_sha256']!=audit['parameter_hashes']['normalizer'] or audit['new_training_transitions']!=0
            or audit['action_parity']['exact'] is not True):raise ValueError('source audit/plan mismatch')
    base=Path(stage_plan).resolve().parent;runtime=read(base/'a0_nominal/spec.json')
    from ..pulse_exploration import pulse_delay
    pulse_delay(runtime,runtime['round_index'])  # CPU preflight before queueing GPU work.
    commit=implementation_identity(repo);root.mkdir(parents=True,exist_ok=False)
    runtime.update(bootstrap_config=audit['runtime_config'],nominal_source_rollout=False,
        full_episode_rollout=False,num_envs=32,pulse_steps=3,pulse_start_schedule=[0],
        pulse_batch_mode='single',delta_limit=[.25]*4,pending_fraction=.5)
    policy=read(audit['frozen_policy'])['policy']
    for filename,value in [('stage_plan.json',plan),('panels.json',{})]:atomic_json(root/filename,value)
    locks={str(p):file_sha(p) for p in [Path(stage_plan).resolve(),audit_dir/'source_audit.json',
        Path(audit['frozen_policy']),Path(audit['runtime_config']),Path(runtime['bank'])]}
    spec=dict(schema='jit_bridge_campaign_v1_2',protocol_version='1.2',execute=True,
        output=str(root),repository=str(repo),implementation_commit=commit,implementation_files=implementation_files(repo),
        source_frozen_policy=audit['frozen_policy'],source=policy,source_runtime=runtime,seed=plan['master_seed'],
        source_audit=str(audit_dir/'source_audit.json'),nominal_gate=str(base/'a0_nominal'),stage_plan=str(root/'stage_plan.json'),
        teacher_layout='source_control_in_32_world_batch',locks=locks,
        budgets={'max_physics':plan['budgets']['physics_cap'],'max_supervised_updates':24000,
                 'max_wall_seconds':plan['budgets']['wall_seconds']},final_test_open=False,automatic_retry=False)
    atomic_json(root/'production.json',spec);atomic_json(root/'status.json',{'phase':'prepared','training_transitions':0})
    atomic_json(root/'explorer_provenance.json',{'mode':'fixed_uniform_collection_only','optimizer_updated':False,
        'historical_explorer_imported':False,'stage_B_enabled':False,'reward_weights':{
        'adoption_bonus':2.,'conversion':0.,'failure':.1,'pulse_failure':2.,'novelty':.02,'repeat':.02,'teacher_success_bonus':0.}})
    return spec


class CampaignRunner(ProductionRunner):
    def __init__(self,manifest):
        if (Path(manifest['output'])/'started.json').exists():
            raise ValueError('campaign already started; reconcile receipts into a new explicit recovery attempt')
        super().__init__(manifest)

    def status(self,phase,**kwargs):
        super().status(phase,**kwargs)
        atomic_json(self.root/'cost_ledger.json',{'costs':self.costs,'limits':self.spec['budgets'],
            'physics_charged_or_reserved':sum(c['charged_interactions'] for c in self.costs),
            'supervised_updates_charged_or_reserved':sum(c.get('charged_updates',0) for c in self.costs),
            'comparison':'equal PPO only; all acquisition, padding, supervision and retries charged'})

    def collect_child(self,name,spec,maximum):
        directory=self.root/'development'/name;directory.mkdir(parents=True,exist_ok=False)
        atomic_json(directory/'spec.json',spec)
        cost=self.child(name,['JIT/cli/run_pulse_exploration.py','--mode','collect','--spec',directory/'spec.json',
            '--output',directory/'rollout'],maximum)
        status=read(directory/'rollout/status.json');cost.update(charged_interactions=status['charged_interactions'],accounting='measured',
            **{k:status[k] for k in ('active_interactions','padding_interactions','peak_rss_kib') if k in status})
        atomic_json(self.root/'costs.json',self.costs)
        return read(directory/'rollout/candidates.json')

    def evaluate_candidate(self,name,rows,bank,policy,*,warmup=None,prefix_policy=None,prefixes=None):
        directory=self.root/'development'/name;directory.mkdir(parents=True,exist_ok=False)
        atomic_json(directory/'candidates.json',rows)
        spec={**self.runtime,'bank':bank,'proposer':policy,'order':[policy],
            'candidates':str(directory/'candidates.json'),'budget':len(rows)*400,'full_matrix':True,
            'record_actor_preobservations':True}
        if warmup is not None:spec['warmup_initializer']=warmup
        if prefix_policy is not None:spec['closed_loop_prefix_policy']=prefix_policy
        if prefixes is not None:
            path=directory/'prefixes.npz';np.savez_compressed(path,actions=prefixes,
                source_only=np.zeros(len(rows),bool),root_contexts=np.array([r['snapshot_context_sha256'] for r in rows]))
            spec['bridge_action_plan']={'path':str(path),'sha256':file_sha(path)}
        atomic_json(directory/'spec.json',spec)
        cost=self.child(name,['JIT/cli/run_pulse_exploration.py','--mode','evaluate','--spec',directory/'spec.json',
            '--output',directory/'rollout'],len(rows)*400)
        receipt=read(directory/'rollout/status.json');cost.update(charged_interactions=receipt['charged_interactions'],accounting='measured',
            **{k:receipt[k] for k in ('active_interactions','padding_interactions','peak_rss_kib') if k in receipt})
        atomic_json(self.root/'costs.json',self.costs)
        return read(directory/'rollout/results.json')

    def full_development(self,name,source_phase,bank,policy):
        plan=read(source_phase['collection_plan'])
        panels={'student_dev_id':[],'student_dev_temporal':[]}
        for index,batch in enumerate(plan['batches']):
            if not batch['full_episode']:continue
            spec=read(batch['collection_spec']);spec.update(bank=bank,proposer=policy,order=[policy])
            panels[batch['panel']].extend(self.collect_child(f'{name}_dev_{index:03d}',spec,32*400))
        nominal=read(Path(self.spec['nominal_gate'])/'spec.json')
        nominal.update(bank=bank,proposer=policy,order=[policy])
        panels['nominal']=self.collect_child(name+'_nominal',nominal,1600)
        return panels

    def warmup(self,source_phase):
        directory=self.root/'warmup';directory.mkdir(exist_ok=False)
        config=dict(source_frozen_policy=self.spec['source_frozen_policy'],
            retention_reference_actor={'path':self.spec['source_frozen_policy'],'sha256':file_sha(self.spec['source_frozen_policy'])},
            demo_manifest=source_phase['demo_manifest'],retention_trace_observations=source_phase['retention_ref'],
            output=str(directory/'training'),result=str(directory/'result.json'),seed=self.spec['seed'],updates=2000,mode='warmup')
        atomic_json(directory/'config.json',config)
        warmup_cost=self.child('student_warmup',['JIT/cli/run_generative_bridge.py','worker','--spec',directory/'config.json'],0,updates=2000)
        result=read(directory/'result.json')
        warmup_cost.update(charged_updates=result['completed_updates'],accounting='measured supervised updates')
        atomic_json(self.root/'costs.json',self.costs)
        # Eight declared solver-dev conditions, collected once with P0; no TRAIN filtering.
        rows=[]
        for index,amplitude in enumerate((.1,.25)):
            spec={**self.runtime,'controller_mode':'fixed_random','num_envs':4,'full_episode_rollout':False,
                'pulse_batch_mode':'single','pulse_start_schedule':[0],'pulse_steps':3,'delta_limit':[amplitude]*4,
                'pulse_protocol_v1_2':{'master_seed':self.spec['seed'],'role':'solver_dev','round':0,
                    'episode_ids':list(range(index*4,index*4+4))}}
            rows.extend(self.collect_child('warmup_dev_roots_'+str(index),spec,12))
        scores={};evaluations={}
        for candidate in result['candidates']:
            evaluated=self.evaluate_candidate('warmup_select_'+str(candidate['update']),rows,
                self.runtime['bank'],self.source['name'],warmup=candidate)
            labels=[r['label'] for r in evaluated]
            if any(v is None for v in labels):raise ValueError('unknown warmup development result')
            scores[candidate['update']]=sum(v==1 for v in labels);evaluations[candidate['update']]=labels
        from .warmup import select_warmup_checkpoint
        selected=select_warmup_checkpoint(scores)
        selection={'selected_update':selected,'scores':scores,'labels':evaluations,
            'criterion':'8 fixed solver-dev post-pulse roots; full continuation success count, earliest tie',
            'not_full_task_from_student_start':True,'automatic_adoption':False}
        atomic_json(directory/'selection.json',selection)
        return next(c for c in result['candidates'] if c['update']==selected)

    def train_arm(self,name,source_phase,warmup=None):
        from ..iterative_probe_training import make_config
        from ..unified_policy_freeze import freeze_development_checkpoint
        from ..probe_bank import lock_probe_bank
        directory=self.root/'students'/name;directory.mkdir(parents=True,exist_ok=False)
        config=directory/'config.json';support=read(source_phase['support_path'])
        candidate=support['schema']=='jit_iterative_candidate_support_v1'
        raw=make_config(source_phase['support_path'],self.spec['source_frozen_policy'],self.runtime['bootstrap_config'],
            config,self.root.name+'_'+name,1,128000,self.spec['seed'],checkpoints=[32000,64000,96000,128000],
            panel_support_path=source_phase['panel_support_path'] if candidate else None,
            pending_fraction=.5 if candidate else None,reward_mode='original_all_phases')
        raw['training_action_pulse']={'onsets':[0],'steps':3,'delta_limit':[.25]*4,'probability':1.,
            'pulse_scope':'full_task_start_only','training_conditions':[{'probability':.2,'amplitude':0.},
                {'probability':.4,'amplitude':.1},{'probability':.4,'amplitude':.25}],
            'logical_episode_rng':{'master_seed':self.spec['seed'],'role':'student_ppo','round':0,'slot_ids':list(range(128))}}
        contract={'schema':'jit_bridge_student_v1_2','source_actor_sha256':self.source['actor_sha256'],
            'source_normalizer_sha256':self.source['normalizer_sha256'],
            'retention_reference_actor':{'path':self.spec['source_frozen_policy'],'sha256':file_sha(self.spec['source_frozen_policy'])},
            'retention_trace_observations':source_phase['retention_ref'],
            'demo_manifest':None if name=='A' else source_phase['demo_manifest'],
            'demo_coefficient_start':0. if name=='A' else .2,'demo_coefficient_end':0. if name=='A' else .05,
            'max_first_behavior_kl':.05,'retention_coefficient':.2,'demo_batch_size':256,'retention_batch_size':256}
        if warmup is not None:contract['warmup_initializer']=warmup
        raw['generative_bridge_student']=contract;atomic_json(config,raw)
        cost=self.child('student_'+name,['JIT/cli/train_unified_from_pi0.py','--config',config,'--run-id',raw['run_declaration']['run_id']],
            136000,extra_env={'JIT_RUN_ROOT':str(directory/'training')})
        training=directory/'training'/raw['run_declaration']['run_id'];report=read(training/'formal_report.json')
        cost.update(charged_interactions=report['completed_training_transitions']+report['train_panel_interactions'],accounting='measured')
        atomic_json(self.root/'costs.json',self.costs)
        freeze_development_checkpoint(directory/'frozen',config_path=config,checkpoint=training/'checkpoints/transition_128000',name=self.root.name+'_'+name)
        frozen=directory/'frozen/frozen_unified_policy.json';policy=read(frozen)['policy']
        bank=read(self.runtime['bank']);bank={k:bank[k] for k in ('task','max_ticks','label_interaction_budget','max_candidates_per_process')}
        bank.update(version=name,members=[{'frozen_policy':str(frozen),'roles':['proposer','evaluator']},
            {'frozen_policy':self.spec['source_frozen_policy'],'roles':['proposer','evaluator']}])
        lock_probe_bank(bank,directory/'bank.json')
        return {'frozen_policy':str(frozen),'policy':policy,'bank':str(directory/'bank.json'),
            'actor_sha256':policy['actor_sha256'],'normalizer_sha256':policy['normalizer_sha256'],'training':str(training)}

    def checkpoint_probes(self,student,source_phase,warmup):
        import pickle
        from brax.training.acme import running_statistics
        from ..ppo import make_network_factory
        from ..checkpoint import load_checkpoint
        from .worker import source_payload
        from .learning_audit import action_probe
        _,source=source_payload(self.spec['source_frozen_policy'])
        _,final=source_payload(student['frozen_policy'])
        net=make_network_factory()({'state':76,'privileged_state':106},4,
            preprocess_observations_fn=running_statistics.normalize)
        demo=read(source_phase['demo_manifest']['path']);reference=(source.observation_normalizer,source.actor_params)
        initial=reference
        if warmup is not None:
            if file_sha(warmup['path'])!=warmup['sha256']:raise ValueError('warmup selected checkpoint changed')
            with Path(warmup['path']).open('rb') as stream:initial=pickle.load(stream)[:2]
        probes={'0':action_probe(net,*initial,demo,reference=reference)}
        for step in (32000,64000,96000,128000):
            payload=load_checkpoint(Path(student['training'])/'checkpoints'/f'transition_{step}',expected=final.identity)
            probes[str(step)]=action_probe(net,payload.observation_normalizer,payload.actor_params,demo,reference=reference)
        for step,probe in probes.items():
            for group,values in probe['groups'].items():
                old=np.asarray(probes['0']['groups'][group]['mse_channels']);now=np.asarray(values['mse_channels'])
                values['relative_mse_reduction_from_initial']=[float((a-b)/a) if a>0 else None for a,b in zip(old,now)]
        path=Path(student['training'])/'checkpoint_action_probes.json'
        atomic_json(path,{'steps':probes,'fixed_demo_manifest':source_phase['demo_manifest'],
            'gradients_from':'learning_probe.json; actual guarded PPO loss','zero_is_actual_initializer':True})
        return str(path)

    def four_combinations(self,source_phase,student):
        traces=read_teacher_traces(source_phase['teachers'])[:8]
        root_map={r['root_id']:r for r in self.panels['new_roots']}
        if not traces:
            report={'status':'not_run_empty_verified_demo','physics':0,'rows':[]}
            atomic_json(self.root/'four_combinations.json',report);return report
        rows=[root_map[t['metadata']['root_id']] for t in traces]
        prefixes=np.stack([t['arrays']['normalized_action_executed'][:16] for t in traces])
        report={'status':'completed','rows':[],'same_layout_repeat_is_not_robust_probability':True}
        for repeat in range(2):
            for branch,tail,prefix_policy,actions in (
                ('teacher_prefix_old_tail',self.source['name'],None,prefixes),
                ('student_prefix_old_tail',self.source['name'],student['policy']['name'],None),
                ('teacher_prefix_student_tail',student['policy']['name'],None,prefixes),
                ('student_full',student['policy']['name'],None,None)):
                result=self.evaluate_candidate(f'four_{repeat}_{branch}',rows,student['bank'],tail,
                    prefix_policy=prefix_policy,prefixes=actions)
                report['rows'].append({'repeat':repeat,'branch':branch,'results':result})
        first={r['branch']:[v['label'] for v in r['results']] for r in report['rows'] if r['repeat']==0}
        second={r['branch']:[v['label'] for v in r['results']] for r in report['rows'] if r['repeat']==1}
        report['repeat_conflicts']={k:[rows[i]['root_id'] for i,(a,b) in enumerate(zip(first[k],second[k])) if a!=b] for k in first}
        report['unknown']=any(any(v['label'] is None for v in r['results']) for r in report['rows'])
        report['teacher_reference_invalid']=any(v!=1 for v in first['teacher_prefix_old_tail']+second['teacher_prefix_old_tail'])
        report['layout']='same declared diagnostic root batch across repeats; differs from 32-candidate teacher search'
        report['quarantined']=report['unknown'] or report['teacher_reference_invalid'] or any(report['repeat_conflicts'].values())
        atomic_json(self.root/'four_combinations.json',report);return report

    def export_lost_roots(self,name,before,after,source_phase,panels,roots,student):
        import csv,json
        fields=['run_id','panel','root_id','ancestor_id','initial_phase','pulse_onset','requested_amplitude',
            'source_label','student_label','source_failure_reason','student_failure_reason','source_trace_uri',
            'student_trace_uri','policy_hashes','layout_id','repeat_status']
        allrows=read(source_phase['aggregate_path'])['rows'];idrows=[r for r in allrows if r['panel']=='student_dev_id']
        original={'core':idrows[::2],'protected':idrows[1::2],'new_roots':self.panels['new_roots'],
            'nominal':read(Path(self.spec['nominal_gate'])/'rollout/candidates.json')}
        actual={'core':panels['student_dev_id'][::2],'protected':panels['student_dev_id'][1::2],
            'new_roots':roots,'nominal':panels['nominal']}
        path=self.root/'lost_roots.csv';exists=path.exists()
        with path.open('a',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            if not exists:writer.writeheader()
            for panel,labels in before.items():
                for i,(old,new) in enumerate(zip(labels,after[panel])):
                    if old!=1 or new==1:continue
                    a=original[panel][i];b=actual[panel][i]
                    row={k:None for k in fields};row.update(run_id=self.root.name+'_'+name,panel=panel,
                        root_id=a.get('root_id',a.get('snapshot_context_sha256')),ancestor_id=a.get('root_episode_id'),
                        initial_phase=a.get('phase'),pulse_onset=a.get('requested_onset',a.get('pulse_start_step')),
                        requested_amplitude=a.get('requested_amplitude'),source_label=old,student_label=new,
                        source_failure_reason=a.get('attempts',[{}])[0].get('outcome') if a.get('attempts') else a.get('prefix_outcome'),
                        student_failure_reason=b.get('attempts',[{}])[0].get('outcome') if b.get('attempts') else b.get('prefix_outcome'),
                        source_trace_uri=a.get('attempts',[{}])[0].get('trace') if a.get('attempts') else a.get('prefix_file'),
                        student_trace_uri=b.get('attempts',[{}])[0].get('trace') if b.get('attempts') else b.get('prefix_file'),
                        policy_hashes=json.dumps({'source':self.source['actor_sha256'],'student':student['actor_sha256']}),
                        layout_id='declared_32_lane_full_start' if panel in ('core','protected') else None,
                        repeat_status='not_repeated')
                    writer.writerow({k:'null' if v is None else v for k,v in row.items()})

    def finish_feedback(self,source_phase,results):
        from .artifacts import load_corpus,save_corpus
        from .feedback_data import build_corpus
        from .series import corpus_history
        from .outcomes import root_outcome,result_matrix
        matrices={}
        for arm,result in results.items():
            evaluated={r['root_id']:r['label'] for r in result['root_results']}
            usage=read(Path(result['student']['training'])/'bridge_demo_usage.json')['demo_samples_by_root']
            total=sum(usage.values());rows=[]
            for pending in self.panels['original_pending']:
                rid=pending['root_id'];teacher=source_phase['teachers'].get(rid,{'teacher_status':'not_scheduled'})
                rows.append(root_outcome({'root_id':rid,'teacher_status':teacher['teacher_status'],
                    'student_label':evaluated.get(rid),'student_adopted':result['acceptance']['adopted'],
                    'training_eligible':True,'n_direct_demo_examples_available':teacher.get('demo',{}).get('count',0),
                    'n_direct_demo_samples_used':usage.get(rid,0)},round_demo_samples_used=total))
            atomic_json(self.root/('root_outcomes_'+arm+'.json'),rows);matrices[arm]=result_matrix(rows)
        atomic_json(self.root/'result_matrix.json',matrices)
        primary=results['C'];student=primary['student'];decision=primary['acceptance']
        teachers=read_teacher_traces(source_phase['teachers'])
        actors=actor_traces(primary['root_results'],student['policy'],self.protocol,self.panels['splits'])
        corpus=build_corpus(corpus_history(load_corpus(source_phase['bootstrap']['corpus'])),teachers,actors,
            adoption=decision,expected={'model_sha256':self.source['xml_sha256'],'protocol_sha256':self.protocol},
            splits=self.panels['splits'])
        receipt=save_corpus(corpus,self.root/'feedback_corpus')
        updated=self.generator('incremental',receipt,source_phase['bootstrap']['dev_fixture'],incumbent=source_phase['incumbent'])
        bundle={'schema':'jit_bridge_A2_selection_v1_2','primary_arm':'C',
            'actor_frozen_policy':student['frozen_policy'] if decision['adopted'] else self.spec['source_frozen_policy'],
            'actor_acceptance':decision,'generator':updated,'corpus':receipt,
            'student_demo_bank':source_phase['demo_manifest'],'stage_B_enabled':False,
            'control_arms_do_not_supply_primary_generator':True}
        atomic_json(self.root/'current_source.json',bundle)
        learning={name:{'loss_audit':read(Path(r['student']['training'])/'learning_probe.json'),
            'checkpoints':read(Path(r['student']['training'])/'checkpoint_action_probes.json')} for name,r in results.items()}
        atomic_json(self.root/'learning_probe.json',{'arms':learning,'four_combinations':read(self.root/'four_combinations.json'),
            'warmup_selection':read(self.root/'warmup/selection.json')})
        summary={'source_checkpoint':read(self.spec['source_audit'])['source_checkpoint'],
            'source_actor_sha256':self.source['actor_sha256'],'stage':'A2_completed','final_test_open':False,
            'arms':{name:r['acceptance'] for name,r in results.items()},'primary_arm':'C',
            'cumulative_demo_transitions':source_phase['demo_count'],
            'cumulative_demo_roots':len(read(source_phase['demo_manifest']['path'])['entries']),
            'generator_feedback_counts':{k:len(v) for k,v in corpus['groups'].items()},
            'physics_charged':sum(c['charged_interactions'] for c in self.costs),
            'supervised_updates_charged':sum(c.get('charged_updates',0) for c in self.costs),
            'limitations':['single training seed','equal PPO is not equal total cost',
                'Stage B disabled; no learned explorer update','cumulative bank has only one new-source round']}
        atomic_json(self.root/'CORE_RESULTS.json',summary)
        lines=['# v1.2 核心结果','',f"源 Actor：{self.source['actor_sha256']}。PPO 每臂 128,000 步；主候选 C。",
            '', '| 臂 | 采用 | core lost | protected lost | 新根 gained/lost |',
            '|---|---:|---:|---:|---:|']
        for name,r in results.items():
            a=r['acceptance'];c=a['counts']
            lines.append(f"| {name} | {a['adopted']} | {c['core']['lost']} | {c['protected']['lost']} | {c['new_roots']['gained']}/{c['new_roots']['lost']} |")
        lines.extend(['','仅单种子开发结果；等 PPO 步数不代表等总预算。最终 TEST 未开启，主动探索未训练。',
            '完整逐失败根见 lost_roots.csv；监督和物理成本见 cost_ledger.json。'])
        (self.root/'CORE_RESULTS.md').write_text('\n'.join(lines)+'\n')

    def _run(self):
        from .source_phase import run_source_phase
        plan=read(self.spec['stage_plan']);gate=Path(self.spec['nominal_gate'])
        while True:
            status=read(gate/'execution/status.json')
            if status['phase'] not in ('waiting','running','prepared'):break
            self.status('waiting',stage='source_nominal_gate');time.sleep(15)
        if status['phase']!='completed':raise RuntimeError('nominal gate execution failed: '+str(status))
        evidence=nominal_evidence(status,read(gate/'rollout/candidates.json'))
        atomic_json(self.root/'source_swap_audit.json',{**read(self.spec['source_audit']),**evidence})
        if not evidence['nominal_success']:
            self.status('needs_input_resolution',stage='A0',reason='specified_source_failed_full_task_nominal',training_transitions=0)
            return {'phase':'needs_input_resolution','source_substitution_allowed':False}
        require_stage(plan,'A1',evidence)
        measured=read(gate/'rollout/status.json')['charged_interactions']
        if not any(c['stage']=='A0_nominal' for c in self.costs):
            self.costs.append({'stage':'A0_nominal','charged_interactions':measured,'accounting':'measured separate nominal receipt'})
        source_phase=self.journal.stage('fresh_source_phase',{},lambda:run_source_phase(self))
        evidence.update(data_ready=True,teacher_semantics_valid=True)
        require_stage(plan,'A2',evidence)
        warm=self.journal.stage('warmup',source_phase,lambda:self.warmup(source_phase))
        aggregate=read(source_phase['aggregate_path'])['rows']
        idrows=[r for r in aggregate if r['panel']=='student_dev_id']
        before={'core':[r['label'] for r in idrows[::2]],'protected':[r['label'] for r in idrows[1::2]],
            'new_roots':source_root_labels(self.panels['new_roots'],source_phase['teachers']),
            'nominal':[r['prefix_label'] for r in read(gate/'rollout/candidates.json')]}
        results={}
        for name in ('A','B','C'):
            student=self.journal.stage('train_'+name,{'source':source_phase,'warmup':warm if name=='C' else None},
                lambda:self.train_arm(name,source_phase,warm if name=='C' else None))
            self.checkpoint_probes(student,source_phase,warm if name=='C' else None)
            panels=self.full_development(name,source_phase,student['bank'],student['policy']['name'])
            roots=self.evaluate_candidate(name+'_roots',self.panels['new_roots'],student['bank'],student['policy']['name']) if self.panels['new_roots'] else []
            labels=[r.get('prefix_label') for r in panels['student_dev_id']]
            after={'core':labels[::2],'protected':labels[1::2],'new_roots':[r['label'] for r in roots],
                'nominal':[r.get('prefix_label') for r in panels['nominal']]}
            decision=acceptance_v12(before,after,{k:student[k] for k in ('actor_sha256','normalizer_sha256')})
            if name=='C':
                four=self.four_combinations(source_phase,student)
                if four.get('quarantined'):decision.update(adopted=False,quarantine_reason='four_combination_repeat_conflict_or_unknown')
            self.export_lost_roots(name,before,after,source_phase,panels,roots,student)
            results[name]={'student':student,'acceptance':decision,'temporal_labels':[r.get('prefix_label') for r in panels['student_dev_temporal']],
                'root_results':roots}
            atomic_json(self.root/'acceptance_report.json',results)
        self.finish_feedback(source_phase,results)
        atomic_json(self.root/'cost_ledger.json',{'costs':self.costs,'limits':plan['budgets'],'comparison':'equal PPO only; all other costs charged'})
        self.status('completed',stage='A2',training_transitions=384000)
        return results
