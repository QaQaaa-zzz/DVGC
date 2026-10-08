"""Bounded fresh-source A1 orchestration; A2 student optimization is separate."""
from pathlib import Path
import numpy as np

from .contracts import digest, file_sha
from .protocol import atomic_json
from .proposals import stable_seed
from .production import read, actor_traces, lane_arrays, read_teacher_traces, is_source_conflict_quarantine
from .source_data import prepare_collection_plan, aggregate_collection, build_training_support


def fresh_panels(rows, *, seed):
    splits = {r['root_episode_id']:('generator_train' if r['data_role']=='train' else 'generator_dev')
              for r in rows if r['data_role'] in ('train','generator_dev')}
    pending = [r for r in rows if r['data_role']=='train' and r['label']==0 and not r.get('prefix_terminal')]
    ordered = sorted(pending,key=lambda r:stable_seed(seed,r['root_id'],'teacher_panel'))
    selected=[]; seen=set()
    for row in ordered:
        if row['root_episode_id'] in seen:continue
        selected.append(row);seen.add(row['root_episode_id'])
        if len(selected)==32:break
    successful = [r for r in rows if r['data_role'] in ('train','generator_dev')
                  and r['label']==1 and not r.get('prefix_terminal')]
    return dict(original_pending=pending,new_roots=selected,bootstrap=successful,
                core=[],protected=[],splits=splits,
                selection='fixed seeded distinct TRAIN ancestors; all valid negatives retain support')


def bootstrap_from_results(runner, results):
    """Reuse source-only successful recording; never call evaluate here."""
    from .feedback_data import build_corpus
    from .artifacts import save_corpus
    from .data import build_action_windows
    traces=actor_traces(results,runner.source,runner.protocol,runner.panels['splits'],bootstrap=True)
    corpus=build_corpus(traces,[],[],adoption={},expected={'model_sha256':runner.source['xml_sha256'],
        'protocol_sha256':runner.protocol},splits=runner.panels['splits'])
    dev=[t for t in traces if t['metadata']['inherited_split']=='generator_dev']
    if not corpus['groups']['history'] or not dev:
        raise ValueError('fresh source real TRAIN and generator-dev successes required')
    receipt=save_corpus(corpus,runner.root/'bootstrap_corpus')
    observations=[];actions=[];ancestors=[]
    for trace in dev:
        windows=build_action_windows(trace)
        observations.extend(windows['observations']);actions.extend(windows['actions'])
        ancestors.extend([trace['metadata']['root_episode_id']]*len(windows['actions']))
    if not observations:raise ValueError('no actual H16 generator-dev windows')
    rng=np.random.default_rng(runner.spec['seed']);n=min(256,len(observations))
    ids=rng.choice(len(observations),n,replace=False)
    fixture=runner.root/'generator_dev_fixture.npz'
    np.savez_compressed(fixture,observations=np.asarray(observations)[ids],actions=np.asarray(actions)[ids],
        timesteps=rng.integers(0,100,n,dtype=np.int32),noise=rng.normal(size=(n,16,4)).astype(np.float32))
    atomic_json(runner.root/'generator_dev_manifest.json',dict(path=str(fixture),sha256=file_sha(fixture),
        ancestors=sorted(set(ancestors)),role='generator_dev',training_allowed=False,source='fresh_source_recordings'))
    return dict(corpus=receipt,dev_fixture=str(fixture),dev_fixture_sha256=file_sha(fixture))


def retention_from_successes(runner, rows):
    """Keep every real pre-action state of complete successful TRAIN episodes."""
    observations=[];weights=[];ancestors=[];inputs={}
    for row in rows:
        if row['data_role']!='train' or row['label']!=1 or row.get('prefix_terminal'):continue
        prefix=Path(row['prefix_file'])
        if file_sha(prefix)!=row['prefix_sha256']:raise ValueError('TRAIN prefix changed')
        with np.load(prefix,allow_pickle=False) as data:
            mask=np.asarray(data['prefix_mask'][:,row['index']],bool)
            before=np.asarray(data['actor_observation_before'][:,row['index']])[mask]
        suffix=lane_arrays(row['attempts'][0])['actor_observation_before']
        full=np.concatenate((before,suffix),axis=0)
        if not len(full):raise ValueError('successful TRAIN trajectory lacks observations')
        observations.extend(full);weights.extend(np.full(len(full),1/len(full)))
        ancestors.append(row['root_episode_id'])
        inputs[str(prefix)]=row['prefix_sha256']
        inputs[row['attempts'][0]['trace']]=row['attempts'][0]['trace_sha256']
    if not observations:raise ValueError('fresh successful TRAIN retention trajectories required')
    path=runner.root/'retention_observations.npz'
    np.savez_compressed(path,actor_observation_before=np.asarray(observations,np.float32),
                        weights=np.asarray(weights,np.float64))
    reference=dict(path=str(path),sha256=file_sha(path),role='train',full_success=True,
        actor_sha256=runner.source['actor_sha256'],normalizer_sha256=runner.source['normalizer_sha256'],
        source_frozen_policy=runner.spec['source_frozen_policy'],ancestors=ancestors,inputs=inputs,
        weighting='equal complete trajectory mass; all real prefix and source suffix preobservations')
    atomic_json(runner.root/'retention_observations.json',reference)
    return reference


def _measured_child(runner,name,mode,spec,output,maximum):
    cost=runner.child(name,['JIT/cli/run_pulse_exploration.py','--mode',mode,
        '--spec',spec,'--output',output],maximum)
    status=read(Path(output)/'status.json')
    if status['phase']!='completed':raise ValueError('fresh-source child incomplete')
    measured=status['charged_interactions']
    if measured>maximum:raise ValueError('fresh-source child exceeded declared reservation')
    cost.update(charged_interactions=measured,accounting='measured')
    for key in ('active_interactions','padding_interactions'):
        if key in status:cost[key]=status[key]
    atomic_json(runner.root/'costs.json',runner.costs)
    return status


def run_source_phase(runner):
    """Execute only A1 with caller-owned gating, budgets, notifications and journal."""
    from .student_demo_bank import build_student_demo_bank
    runner.status('running',stage='fresh_source_A1')
    reused=runner.spec.get('recovery_preparation')
    if reused:
        if file_sha(reused['path'])!=reused['sha256']:raise ValueError('recovery preparation drift')
        prepared=read(reused['path'])
        for path,sha in prepared['inputs'].items():
            if file_sha(path)!=sha:raise ValueError('recovery input drift: '+path)
        if prepared['source_actor_sha256']!=runner.source['actor_sha256']:
            raise ValueError('recovery source identity mismatch')
        aggregate_path=Path(prepared['aggregate_path']);support_path=Path(prepared['support_path'])
        seed_dir=Path(prepared['seed_dir']);collection_plan=prepared['collection_plan']
        runner.panels=prepared['panels'];retention=prepared['retention_ref']
        bootstrap=prepared['bootstrap'];incumbent=prepared['incumbent']
        from .artifacts import validate_generator_receipt,load_corpus
        validate_generator_receipt(incumbent);load_corpus(bootstrap['corpus'])
    else:
        seed_spec={**runner.runtime,'seed_stride':10,'record_actor_preobservations':True,
            'controller_mode':'fixed_random','full_episode_rollout':False,
            'order':[runner.source['name']], 'horizon':400}
        for key in ('pulse_protocol_v1_2','pulse_event_schedule','initial_velocity_randomization',
                    'reuse_results','reuse_collection','reuse_prefix_collection','bridge_action_plan','evaluation_batch_size'):
            seed_spec.pop(key,None)
        seed_path=runner.root/'seed_support_spec.json';atomic_json(seed_path,seed_spec)
        seed_dir=runner.root/'seed_support'
        _measured_child(runner,'fresh_seed_support','seed_support',seed_path,seed_dir,80400)
        seed_support=read(seed_dir/'support.json')
        if {r['phase'] for r in seed_support['entries']} != {'upstream','downstream'}:
            raise ValueError('fresh nominal seed support must witness both phases')
        plan=prepare_collection_plan(runner.runtime,runner.root/'source_collections',master_seed=runner.spec['seed'])
        for index,batch in enumerate(plan['batches']):
            _measured_child(runner,f'source_collect_{index:03d}','collect',batch['collection_spec'],
                batch['collection_output'],32*(400 if batch['full_episode'] else 3))
            if not batch['full_episode']:
                _measured_child(runner,f'source_suffix_{index:03d}','evaluate',batch['suffix_spec'],
                    batch['suffix_output'],32*400)
        aggregate_path=runner.root/'source_rows.json'
        aggregate=aggregate_collection(runner.root/'source_collections/plan.json',aggregate_path,
                                       source_actor_sha256=runner.source['actor_sha256'])
        support_path=runner.root/'training_support.json'
        build_training_support(seed_dir/'support.json',aggregate_path,support_path,source_policy=runner.source['name'])
        runner.panels=fresh_panels(aggregate['rows'],seed=runner.spec['seed'])
        atomic_json(runner.root/'panels.json',runner.panels)
        retention=retention_from_successes(runner,aggregate['rows'])
        bootstrap=bootstrap_from_results(runner,runner.panels['bootstrap'])
        incumbent=runner.generator('pretrain',bootstrap['corpus'],bootstrap['dev_fixture'])
        smoke=runner.smoke()
        if smoke.get('status')!='passed':raise ValueError('semantic smoke did not pass')
        collection_plan=str(runner.root/'source_collections/plan.json')
    if runner.spec.get('teacher_layout')!='source_control_in_32_world_batch':
        raise ValueError('A1 teacher requires source control in declared 32-world layout')
    teachers=runner.teacher_search(incumbent)
    # Exceptions and unknown source/teacher outcomes are never an empty-demo arm.
    if any(r.get('teacher_status') in ('invalid','incomplete') and not is_source_conflict_quarantine(r)
           for r in teachers.values()):
        raise ValueError('unknown or incomplete teacher prevents A1 completion')
    demo=build_student_demo_bank(read_teacher_traces(teachers),runner.root/'student_demo_bank',
        source_identity=dict(actor_sha256=runner.source['actor_sha256'],
            normalizer_sha256=runner.source['normalizer_sha256'],model_sha256=runner.source['xml_sha256'],
            protocol_sha256=runner.protocol),round_id=0)
    result=dict(bootstrap=bootstrap,incumbent=incumbent,teachers=teachers,
        demo_manifest=dict(path=str(runner.root/'student_demo_bank/manifest.json'),
                           sha256=file_sha(runner.root/'student_demo_bank/manifest.json')),
        retention_ref=retention,aggregate_path=str(aggregate_path),support_path=str(support_path),
        collection_plan=collection_plan,
        panel_support_path=str(seed_dir/'support.json'),panels=runner.panels,
        source_actor_sha256=runner.source['actor_sha256'],new_training_transitions=0,
        generator_updates=incumbent.get('updates'),demo_count=demo['count'])
    atomic_json(runner.root/'source_phase_result.json',result)
    return result
