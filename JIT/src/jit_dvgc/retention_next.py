"""Explicit finite retention stages; preparation never steps physics or optimizes."""
from pathlib import Path
import subprocess
import numpy as np
from .retention_repair import read,write,sha,MONITOR,CAMPAIGN

CONFIG=Path(__file__).resolve().parents[2]/'configs/retention_next.json'


def budget_layout(stage):
    if stage=='V': return dict(main=2*4*250*400,total=800000,hard_cap=1000000,wall_seconds=14400)
    if stage!='D1':raise ValueError('D2 needs a completed D1 receipt')
    # Both possible TRAIN blocks and independent solver-dev are budgeted before selection.
    result=dict(collection=(2*128+32)*18,all_root_pi0=(2*128+32)*400,
        selected_pi0_repeats=(32+8)*2*400,R5_control=(32+8)*400,
        teacher_search_and_replay=(32+8)*2*3*17*400)
    result.update(total=sum(result.values()),hard_cap=2000000,wall_seconds=21600)
    return result


def baseline_class(labels):
    if not labels or any(x not in (0,1) for x in labels):raise ValueError('incomplete baseline is not failure')
    return 'stable_failure' if all(x==0 for x in labels) else 'baseline_success' if all(x==1 for x in labels) else 'baseline_ambiguous'


def qualify_method(source_labels,winner_labels):
    if any(x not in (0,1) for x in source_labels):raise ValueError('unknown source label')
    if any(source_labels):return 'source_ambiguous'
    if winner_labels is None:return 'searched_no_solution'
    if len(winner_labels)!=3 or any(x not in (0,1) for x in winner_labels):raise ValueError('incomplete fixed winner replay')
    return 'verified_solution' if winner_labels==[1,1,1] else 'winner_ambiguous'


def choose_roots(rows,labels,target):
    pools={o:[r for r in rows if r['onset']==o and baseline_class(labels[r['root_id']])=='stable_failure'] for o in (0,5,10,15)}
    result=[]
    while len(result)<target and any(pools.values()):
        for pool in pools.values():
            if pool and len(result)<target:result.append(pool.pop(0))
    return result


def identities(models):
    from .checkpoint import load_checkpoint,CheckpointIdentity
    from .handoff_bank import pytree_sha256
    result={}
    for key,m in models.items():
        pol=m['policy'];side=read(Path(pol['checkpoint'])/'identity.json')
        fields={k:tuple(side[k]) if isinstance(side[k],list) else side[k] for k in ('config_sha256','xml_sha256','actor_frame_fields','actor_task_fields','action_order')}
        payload=load_checkpoint(Path(pol['checkpoint']),expected=CheckpointIdentity(**fields))
        result[key]={}
        for name,value in [('actor_sha256',payload.actor_params),('normalizer_sha256',payload.observation_normalizer),('critic_sha256',payload.critic_params)]:
            h=pytree_sha256(value)
            if h!=pol[name]:raise ValueError('actual checkpoint mismatch '+key+name)
            result[key][name]=h
        result[key]['checkpoint']=pol['checkpoint']
    return result


def random_initials(model,indices,n,seed):
    """Host forward geometry only, using original bounds/common height correction."""
    from scipy.spatial.transform import Rotation
    import mujoco,jax
    from .config import load_config
    from .model import load_host_model
    from .geometry import build_geometry_contract,collision_support_bounds
    cfg=read(model['policy']['formal_config']);up=load_config(Path(cfg['inputs']['up_config_path']),runtime_only=True)
    bundle=load_host_model(up);mj=bundle.mj_model;data=mujoco.MjData(mj)
    qi,vi=indices['root_qpos'],indices['root_dof'];baseq=np.array(indices['nominal_qpos']);basev=np.array(indices['nominal_qvel'])
    rng=np.random.default_rng(seed);offset=np.zeros((n,12));offset[:,:2]=rng.uniform(-1,1,(n,2))*[.1,.05]
    offset[:,3:6]=rng.uniform(-3,3,(n,3));offset[:,6:8]=rng.uniform(-.2,.2,(n,2));offset[:,9:12]=rng.uniform(-.1,.1,(n,3))
    q=np.tile(baseq,(n,1));v=np.tile(basev,(n,1));q[:,qi:qi+3]+=offset[:,:3];v[:,vi:vi+6]+=offset[:,6:12]
    rot=Rotation.from_euler('xyz',offset[:,3:6],degrees=True)*Rotation.from_quat(np.roll(baseq[qi+3:qi+7],-1))
    q[:,qi+3:qi+7]=np.roll(rot.as_quat(),1,axis=1)
    ids=np.asarray(build_geometry_contract(mj).robot_geom_ids)
    support=jax.jit(lambda pos,mat:collision_support_bounds(pos,mat,mj.geom_type[ids],mj.geom_size[ids]).min_z)
    precheck=[]
    for i in range(n):
        data.qpos[:]=q[i];data.qvel[:]=v[i];mujoco.mj_forward(mj,data)
        dz=max(0.,.001-float(np.min(support(data.geom_xpos[ids],data.geom_xmat[ids]))));q[i,qi+2]+=dz
        data.qpos[:]=q[i];mujoco.mj_forward(mj,data)
        precheck.append(dict(index=i,z_correction=dz,remaining_penetration=max([-data.contact[j].dist for j in range(data.ncon)],default=0.),excluded=False))
    return q,v,precheck


def prepare(stage,evidence_dir,output,repository):
    if stage=='D2':
        from .retention_next_report import prepare_d2
        return prepare_d2(evidence_dir,output,repository)
    code=Path(repository).resolve();evidence=Path(evidence_dir).resolve();root=Path(output).resolve()
    cfg=read(CONFIG);root.mkdir(parents=True,exist_ok=False)
    required=['README.md','summary.json','D0b_audit.json','normalizer_actor_factorial.json','historical_actor_gradients.json','repeat_first_differences.json','model_identity_audit.json','D1_preparation.json']
    locks={str(evidence/f):sha(evidence/f) for f in required};locks[str(CONFIG)]=sha(CONFIG)
    old=read(evidence/'D1_preparation.json');lineage=read(MONITOR/'continuous_key_metrics_20261010/lineage.json')
    template=read(MONITOR/'bridge_pi0_hard_initial1000_20261010/onset_00_bridge_pi0_spec.json')
    models={}
    for key,bank,name in [('pi0',template['bank'],template['proposer']),('R5',str(Path(lineage['5'])/'students/student/bank.json'),'round_0005_student')]:
        member=next(m for m in read(bank)['members'] if m['name']==name)
        models[key]=dict(bank=bank,proposer=name,policy=member['policy'],frozen_policy=member['frozen_policy'])
        for path in [bank,member['frozen_policy'],member['policy']['formal_config'],str(Path(member['policy']['checkpoint'])/'payload.pkl'),str(Path(member['policy']['checkpoint'])/'identity.json')]:locks[path]=sha(path)
    actual=identities(models)
    if any(actual[k][x]!=read(evidence/'model_identity_audit.json')['identities'][k][x] for k in models for x in ('actor_sha256','normalizer_sha256','critic_sha256')):raise ValueError('D0 identity mismatch')
    indices=read(MONITOR/'three_model_multidim1000_20261010/model_indices.json')
    batches=[]
    if stage=='V':
        q,v,precheck=random_initials(models['pi0'],indices,256,cfg['V_initial_seed']);write(root/'initial_precheck.json',precheck)
        requests=np.random.default_rng(cfg['V_request_seed']).uniform(-.25,.25,(256,3,4)).astype('f4')
        cases=[]
        for group,n in [('A',32),('B',256),('C',256),('D',256)]:
            for i in range(n):
                onset=cfg['onsets'][i//64] if group!='A' else 0
                cases.append(dict(case=f'{group}_{i:03d}',group=group,condition_index=i,onset=onset,
                    ancestor=f"DEV_CONFIRM-{cfg['V_initial_seed']}-"+(f'random-{i}' if group in 'BD' else 'nominal'),
                    request_id=f"DEV_CONFIRM-{cfg['V_request_seed']}-{i}" if group in 'CD' else 'zero',repeat=0,role='DEV_CONFIRM',
                    qpos=(q[i] if group in 'BD' else indices['nominal_qpos']).tolist() if group in 'BD' else indices['nominal_qpos'],
                    qvel=v[i].tolist() if group in 'BD' else indices['nominal_qvel'],request=requests[i].tolist() if group in 'CD' else np.zeros((3,4)).tolist()))
        for key,m in models.items():
            for ordinal in range(4):
                active=cases[ordinal*250:(ordinal+1)*250];padded=active+[dict(cases[0],case='padding')]*(250-len(active))
                name=f'{key}_main{ordinal}';bankfile=root/(name+'_initial.npz');reqfile=root/(name+'_requests.npz')
                np.savez_compressed(bankfile,qpos=np.array([c['qpos'] for c in padded],np.float32),qvel=np.array([c['qvel'] for c in padded],np.float32))
                table=np.zeros((400,250,4),np.float32)
                for lane,c in enumerate(padded):table[c['onset']:c['onset']+3,lane]=c['request']
                np.savez_compressed(reqfile,requested=table,onsets=np.array([c['onset'] for c in padded],np.int32))
                spec={k:v for k,v in template.items() if k not in ('neighborhood','neighborhood_map','neighborhood_map_sha256','neighborhood_reference_actor_sha256','explorer_checkpoint','explorer_backend','initial_state_bank')}
                spec.update(bank=m['bank'],proposer=m['proposer'],controller_mode='fixed_random',num_envs=250,
                    seed=cfg['V_execution_seed'],initial_state_bank=str(bankfile),frozen_request_table=dict(path=str(reqfile),sha256=sha(reqfile)),
                    record_actor_preobservations=True,record_retention_diagnostics=True,pulse_start_schedule=[0],role='DEV_CONFIRM')
                sp=root/(name+'_spec.json');write(sp,spec)
                for p in (bankfile,reqfile,sp):locks[str(p)]=sha(p)
                batches.append(dict(name=name,model=key,capacity=250,horizon=400,spec=str(sp),scored=len(active),cases=active,kind='main'))
    else:
        # Retain original policy identities; pi0 current flags only derive from pi0 evidence.
        e_cfg=read(old['E']['config']);historical=read(e_cfg['neighborhood_map'])
        if historical['config']!=e_cfg['neighborhood']:raise ValueError('E map config drift')
        atlas={**historical,'source_actor_sha256':models['pi0']['policy']['actor_sha256'],'frozen_before_collection':True,
               'source_scope':'Historical TRAIN evidence preserves evaluated Actor; unmeasured pi0 is UNKNOWN; no teacher leakage'}
        from .neighborhood import FrozenNeighborhood
        index=FrozenNeighborhood(atlas['rows'],atlas['source_actor_sha256'],atlas['config'])
        write(root/'pi0_neighborhood.json',atlas)
        write(root/'neighborhood_identity_audit.json',dict(current_actor=atlas['source_actor_sha256'],record_count=index.record_count,
            historical_labels_retagged=False,map_frozen_before_blocks=True,unmeasured_current_label='UNKNOWN'))
        locks[e_cfg['neighborhood_map']]=sha(e_cfg['neighborhood_map'])
        for path in [old['E']['checkpoint'],old['E']['config'],old['G']['manifest'],str(Path(old['G']['manifest']).parent/'state.msgpack'),str(root/'pi0_neighborhood.json')]:locks[path]=sha(path)
        for block,role,n,seed,offset in [('train0','TRAIN',128,cfg['TRAIN_seed'],0),('train1','TRAIN',128,cfg['TRAIN_seed'],128),('solver_dev','SOLVER_DEV',32,cfg['solver_dev_seed'],0)]:
            spec={k:v for k,v in e_cfg.items() if k not in ('collection','feedback','explorer_admission_v1_2','gate','initial_state_bank')}
            spec.update(bank=models['pi0']['bank'],proposer=models['pi0']['proposer'],order=[models['pi0']['proposer']],
                explorer_checkpoint=old['E']['checkpoint'],frozen_explorer_evaluation=True,neighborhood_map=str(root/'pi0_neighborhood.json'),
                neighborhood_map_sha256=sha(root/'pi0_neighborhood.json'),neighborhood_reference_actor_sha256=models['pi0']['policy']['actor_sha256'],
                num_envs=n,round_index=0,seed=seed,role=role,record_retention_diagnostics=True,
                explorer_admission_v1_2=dict(run_id=root.name,collection_id=block,master_seed=seed,round=0,
                    episode_ids=list(range(offset,offset+n)),uniform_episode_fraction=0.))
            sp=root/(block+'_spec.json');write(sp,spec);locks[str(sp)]=sha(sp)
            batches.append(dict(name=block,spec=str(sp),capacity=n,horizon=18,role=role,episode_offset=offset,optional=block=='train1'))
    for file in ['retention_next.py','retention_next_runtime.py','retention_next_report.py','pulse_exploration_runtime.py','generative_bridge/teacher_runtime.py']:
        path=code/'JIT/src/jit_dvgc'/file;locks[str(path)]=sha(path)
    layout=budget_layout(stage);write(root/'budget_dry_run.json',layout)
    plan=dict(schema='jit_retention_next_v1',stage=stage,output=str(root),code=str(code),
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),models=models,
        locks=locks,batches=batches,indices=indices,E=old['E'] if stage=='D1' else None,G=old['G'] if stage=='D1' else None,
        budget=layout,config=cfg,training_transitions=0,optimizer_updates=0,
        semantics='preserve complete snapshot episode/event/history context; canonical forward reconstruction; no reset at H16',
        original_campaign_status_path=str(CAMPAIGN/'series_0011_b1_resume_143/status.json'),
        original_campaign_status_sha256=sha(CAMPAIGN/'series_0011_b1_resume_143/status.json'),
        pointers=dict(learner_last=None,best_dev_candidate=models['R5'],published_policy=None),
        executable=True,execute_authorized=True,automatic_next_stage=False)
    write(root/'source_lock.json',dict(identities=actual,E=plan['E'],G=plan['G'],locks=locks))
    write(root/'plan.json',plan);write(root/'experiment_plan.json',plan)
    write(root/'status.json',dict(phase='prepared',stage=stage,charged_interactions=0,training_updates=0))
    audit(root/'plan.json');return root/'plan.json'


def audit(path):
    p=read(path)
    if p['schema']!='jit_retention_next_v1' or p['stage'] not in ('V','D1'):raise ValueError('unsupported executable plan')
    for file,h in p['locks'].items():
        if sha(file)!=h:raise ValueError('input changed '+file)
    if sha(p['original_campaign_status_path'])!=p['original_campaign_status_sha256']:raise ValueError('original stopped state changed')
    budget=(resume_budget(p['resume']['inherited_charge'],len(p['resume']['remaining_indices']),p['resume']['inherited_wall_seconds']) if p.get('resume') else budget_layout(p['stage']))
    if budget!=p['budget'] or budget['total']>budget['hard_cap']:raise ValueError('actual layout budget drift')
    if p['training_transitions'] or p['optimizer_updates'] or p['automatic_next_stage']:raise ValueError('stage scope drift')
    actual=identities(p['models'])
    frozen={}
    if p['stage']=='D1':
        from .rsl_pulse import restore
        from .generative_bridge.explorer_admission import behavior_identity
        estate=restore(p['E']['checkpoint']);frozen['E']=dict(file_sha256=sha(p['E']['checkpoint']),**behavior_identity(estate),updates=estate['total_updates'])
        g=read(p['G']['manifest']);statefile=Path(p['G']['manifest']).parent/'state.msgpack'
        if sha(statefile)!=g['state_sha256']:raise ValueError('G state differs from manifest')
        expected={k:p['models']['pi0']['policy'][k] for k in ('actor_sha256','normalizer_sha256','xml_sha256')}
        if g['identity']!=expected or g['inference_parameters']!='ema':raise ValueError('G reference or inference drift')
        frozen['G']=dict(manifest_sha256=sha(p['G']['manifest']),state_sha256=sha(statefile),reference=expected,updates=g['updates'],inference='ema',tail_validation='NOT_ESTABLISHED_BY_IDENTITY')
    result=dict(phase='passed',identities=actual,frozen_E_G=frozen,budget=budget,physics_steps=0,optimizer_updates=0)
    write(Path(p['output'])/'audit.json',result);return result


def resume_budget(inherited_charge,remaining_roots,elapsed):
    if inherited_charge<0 or not 0<=remaining_roots<=40 or not 0<=elapsed<21600:raise ValueError('invalid bounded resume')
    result=dict(inherited_charge=inherited_charge,remaining_teacher_roots=remaining_roots,
        remaining_teacher_search_and_replay=remaining_roots*2*3*17*400,
        total=inherited_charge+remaining_roots*2*3*17*400,hard_cap=2000000,wall_seconds=21600-elapsed)
    if result['total']>2000000:raise ValueError('resume exceeds original charged cap')
    return result


def prepare_resume(previous,output,repository):
    """Explicit new attempt; inherit completed roots only, retain unknown failed charge."""
    previous=Path(previous).resolve();old=read(previous/'plan.json');status=read(previous/'status.json')
    if old['stage']!='D1' or status['phase']!='failed':raise ValueError('only a failed finite D1 can be resumed')
    # Original code is verified against the actual committed tree, not current edited paths.
    for file,h in old['locks'].items():
        fp=Path(file)
        if fp.is_relative_to(Path(old['code'])/'JIT/src'):
            import hashlib
            rel=str(fp.relative_to(old['code']))
            raw=subprocess.check_output(['git','show',old['code_revision']+':'+rel],cwd=repository)
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('old code provenance changed '+file)
        elif sha(fp)!=h:raise ValueError('inherited locked input changed '+file)
    selected=read(previous/'selected_roots.json');completed=read(previous/'teacher_results.json')
    done={r['root_id'] for r in completed}
    if len(done)!=len(completed) or not done<=set(r['root_id'] for r in selected):raise ValueError('invalid partial roots')
    locks={k:v for k,v in old['locks'].items() if not Path(k).is_relative_to(Path(old['code'])/'JIT/src')}
    for file in ('plan.json','status.json','costs.json','selected_roots.json','teacher_results.json','collection_ledger.json','source_lock.json'):
        locks[str(previous/file)]=sha(previous/file)
    for r in completed:
        locks[r['proposals']]=sha(r['proposals'])
        if locks[r['proposals']]!=r['proposals_sha256']:raise ValueError('inherited actions changed')
        for m in r['methods'].values():
            for a in m['verified_attempts']:
                if sha(a['trace'])!=a['trace_sha256']:raise ValueError('inherited winner trace changed')
                locks[a['trace']]=a['trace_sha256']
    remaining=[(i,r) for i,r in enumerate(selected) if r['root_id'] not in done]
    # Keep whole search/replay batches immutable, including source lanes and losing candidates.
    for folder in previous.glob('root_*_*'):
        if not folder.is_dir() or not (folder/'results.json').exists():continue
        results=read(folder/'results.json')
        for file in ('results.json','status.json'):
            locks[str(folder/file)]=sha(folder/file)
        for r in results:
            for a in r['attempts']:
                if sha(a['trace'])!=a['trace_sha256']:raise ValueError('inherited batch trace changed')
                locks[a['trace']]=a['trace_sha256']
    for i,r in remaining:
        prop=previous/f'root_{i:03d}_proposals.npz'
        if prop.exists():locks[str(prop)]=sha(prop)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);code=Path(repository).resolve()
    for file in ['retention_next.py','retention_next_runtime.py','retention_next_report.py','pulse_exploration_runtime.py','generative_bridge/teacher_runtime.py']:
        fp=code/'JIT/src/jit_dvgc'/file;locks[str(fp)]=sha(fp)
    import time
    elapsed=time.time()-(status['updated_unix']-status['wall_seconds'])
    budget=resume_budget(status['charged_interactions'],len(remaining),elapsed)
    resumed={**old,'output':str(root),'code':str(code),'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
        'locks':locks,'budget':budget,'resume':dict(previous=str(previous),remaining_indices=[i for i,r in remaining],
            completed_indices=[i for i,r in enumerate(selected) if r['root_id'] in done],inherited_charge=status['charged_interactions'],
            inherited_wall_seconds=elapsed,previous_execution_wall_seconds=status['wall_seconds'],worker_lifetime='one root, same17world layout, search+2fixedreplays per method',
            reason='OOM during canonical snapshot mjx.forward after72 batched teacher dispatches; bounded root worker lifetimes prevent inter-root accumulation',
            completed_roots_recomputed=False,failed_search_label='UNKNOWN',automatic_retry=False)}
    write(root/'plan.json',resumed);write(root/'experiment_plan.json',resumed);write(root/'budget_dry_run.json',budget)
    write(root/'source_lock.json',dict(identities=identities(old['models']),E=old['E'],G=old['G'],locks=locks,
        parent=str(previous/'source_lock.json'),original_code=old['code_revision']))
    write(root/'selected_roots.json',selected);write(root/'collection_ledger.json',read(previous/'collection_ledger.json'))
    write(root/'status.json',dict(phase='prepared',stage='D1',charged_interactions=status['charged_interactions'],training_updates=0))
    audit(root/'plan.json');return root/'plan.json'
