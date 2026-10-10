"""Immutable D0 plans and fixed, policy-independent exogenous requests."""
import hashlib
import json
from pathlib import Path
import numpy as np

MONITOR=Path('/home/qy/DVGC/JIT/runs/monitoring')
CAMPAIGN=Path('/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)

def load_requests(spec):
    ref=spec['frozen_request_table']
    if sha(ref['path'])!=ref['sha256']:raise ValueError('request identity changed')
    if not spec.get('frozen_explorer_evaluation') or spec.get('controller_mode')!='fixed_random':
        raise ValueError('frozen requests require fixed-random evaluation only')
    with np.load(ref['path'],allow_pickle=False) as f:
        requests=np.asarray(f['requested'],np.float32);onsets=np.asarray(f['onsets'])
    n,h=spec['num_envs'],spec['horizon']
    if requests.shape!=(h,n,4) or onsets.shape!=(n,) or not np.isfinite(requests).all() or np.max(np.abs(requests))>.250001:
        raise ValueError('invalid fixed requests')
    if onsets.dtype.kind not in 'iu' or np.any(onsets<0) or np.any(onsets+spec['pulse_steps']>h):raise ValueError('invalid onsets')
    mask=(np.arange(h)[:,None]>=onsets)&(np.arange(h)[:,None]<onsets+spec['pulse_steps'])
    if np.any(requests[~mask]):raise ValueError('request outside single pulse window')
    return requests,onsets.astype(np.int32)

def relative_label(pi0,student,teacher):
    if pi0 is None or student is None:return 'unknown'
    if pi0 and not student:return 'retention_debt'
    if pi0:return 'baseline_retained'
    if teacher is None:return 'teacher_unknown'
    if teacher:return 'baseline_failure_teacher_recoverable_'+('learned' if student else 'pending')
    return 'baseline_failure_no_verified_teacher'

def validate_budget(plan):
    if plan['stage']!='D0' or plan['training_transitions']!=0:raise ValueError('only zero-training D0 authorized')
    charge=sum(b['capacity']*b['horizon'] for b in plan['batches'])
    if charge>plan['budget']['physical_transitions'] or plan['budget']['physical_transitions']>1500000:
        raise ValueError('physical budget exceeded including padding')
    if not 0<plan['budget']['wall_seconds']<=14400:raise ValueError('wall budget exceeded')
    return charge

def prepare(output):
    from scipy.spatial.transform import Rotation
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    lineage=read(MONITOR/'continuous_key_metrics_20261010/lineage.json')
    template=read(MONITOR/'bridge_pi0_hard_initial1000_20261010/onset_00_bridge_pi0_spec.json')
    banks={'pi0':(template['bank'],template['proposer'])}
    for r in (5,21,71,73):
        bank=Path(lineage[str(r)])/'students/student/bank.json'
        members=read(bank)['members'];name=f'round_{r:04d}_student'
        if not any(m['name']==name for m in members):raise ValueError('missing actual diagnostic student '+name)
        banks[f'R{r}']=(str(bank),name)
    models={};locks={}
    for key,(bank,name) in banks.items():
        member=next(m for m in read(bank)['members'] if m['name']==name);p=member['policy']
        paths=[bank,p['formal_config'],str(Path(p['checkpoint'])/'payload.pkl'),str(Path(p['checkpoint'])/'identity.json')]
        for path in paths:locks[path]=sha(path)
        models[key]={'bank':bank,'proposer':name,'policy':p,'missing':False}
    # Same published hard-initial bounds, independent DEV RNG and ancestry.
    indices=read(MONITOR/'three_model_multidim1000_20261010/model_indices.json')
    qi,vi=indices['root_qpos'],indices['root_dof'];baseq=np.array(indices['nominal_qpos']);basev=np.array(indices['nominal_qvel'])
    rng=np.random.default_rng(1010212026);offset=np.zeros((128,12))
    offset[:,:2]=rng.uniform(-1,1,(128,2))*[.1,.05];offset[:,3:6]=rng.uniform(-3,3,(128,3))
    offset[:,6:8]=rng.uniform(-.2,.2,(128,2));offset[:,9:12]=rng.uniform(-.1,.1,(128,3))
    q=np.tile(baseq,(128,1));v=np.tile(basev,(128,1));q[:,qi:qi+3]+=offset[:,:3]
    rot=Rotation.from_euler('xyz',offset[:,3:6],degrees=True)*Rotation.from_quat(np.roll(baseq[qi+3:qi+7],-1))
    q[:,qi+3:qi+7]=np.roll(rot.as_quat(),1,axis=1);v[:,vi:vi+6]+=offset[:,6:12]
    # Reproduce the documented common collision-height correction, without exclusion.
    import mujoco
    from .unified_formal import build_unified_formal_environment
    _,_,env=build_unified_formal_environment(Path(models['pi0']['policy']['formal_config']))
    model=env.mj_model;data=mujoco.MjData(model);zshift=[];legality=[]
    import jax
    from .geometry import collision_support_bounds
    ids=np.asarray(env._geometry.robot_geom_ids)
    support=jax.jit(lambda pos,mat:collision_support_bounds(pos,mat,model.geom_type[ids],model.geom_size[ids]).min_z)
    for i in range(128):
        data.qpos[:]=q[i];data.qvel[:]=v[i];mujoco.mj_forward(model,data)
        minz=float(np.min(np.asarray(support(data.geom_xpos[ids],data.geom_xmat[ids]))))
        dz=max(0.,.001-minz);q[i,qi+2]+=dz;zshift.append(dz)
        data.qpos[:]=q[i];mujoco.mj_forward(model,data)
        legality.append({'random_index':i,'z_correction':dz,'remaining_penetration':max([-data.contact[j].dist for j in range(data.ncon)],default=0.),'excluded':False})
    write(root/'initial_precheck.json',legality)
    np.savez_compressed(root/'random_initials.npz',qpos=q.astype('f4'),qvel=v.astype('f4'),offset=offset,z_correction=zshift)
    actions=np.random.default_rng(1010222026).uniform(-.25,.25,(128,3,4)).astype('f4')
    main=[]
    for group,n in [('A',64),('B',128),('C',128),('D',128)]:
        for i in range(n):
            onset=(0,5,10,15)[i//32] if group in ('B','C','D') else 0
            random=group in ('B','D');pulse=group in ('C','D')
            main.append({'case':f'{group}_{i:03d}','group':group,'condition_index':i,'onset':onset,
                'ancestor':f'DEV-retention-1010212026-'+(f'random-{i}' if random else 'nominal'),
                'request_id':f'DEV-request-1010222026-{i}' if pulse else 'zero',
                'qpos':(q[i] if random else baseq).tolist(),'qvel':(v[i] if random else basev).tolist(),
                'request':(actions[i] if pulse else np.zeros((3,4))).tolist(),'repeat':0,'role':'DEV'})
    reps=[]
    for group in ('A','B','C','D'):
        selected=[x for x in main if x['group']==group][::16][:8] if group!='A' else [x for x in main if x['group']==group][:8]
        for c in selected:
            for rep in range(1,4):reps.append({**c,'repeat':rep})
    batches=[]
    for key,m in models.items():
        sets=[('main0',main[:250]),('main1',main[250:])]
        if key in ('pi0','R71'):sets.append(('repeat',reps))
        for name,cases in sets:
            padded=cases+[dict(main[0],case='padding',group='padding')]*(250-len(cases))
            name=key+'_'+name;bankfile=root/(name+'_initial.npz');requestfile=root/(name+'_requests.npz')
            np.savez_compressed(bankfile,qpos=np.array([x['qpos'] for x in padded],np.float32),qvel=np.array([x['qvel'] for x in padded],np.float32))
            request=np.zeros((400,250,4),np.float32);onsets=np.array([x['onset'] for x in padded],np.int32)
            for lane,c in enumerate(padded):request[c['onset']:c['onset']+3,lane]=c['request']
            np.savez_compressed(requestfile,requested=request,onsets=onsets)
            spec={k:v for k,v in template.items() if k not in ('neighborhood','neighborhood_map','neighborhood_map_sha256','neighborhood_reference_actor_sha256','explorer_checkpoint','explorer_backend','initial_state_bank')}
            spec.update(bank=m['bank'],proposer=m['proposer'],controller_mode='fixed_random',num_envs=250,
                seed=1010232026,initial_state_bank=str(bankfile),delta_limit=[.25]*4,
                frozen_request_table={'path':str(requestfile),'sha256':sha(requestfile)},record_actor_preobservations=True,
                record_retention_diagnostics=True,pulse_start_schedule=[0],pulse_batch_mode='mixed',role='DEV')
            sp=root/(name+'_spec.json');write(sp,spec)
            for path in (bankfile,requestfile,sp):locks[str(path)]=sha(path)
            batches.append({'name':name,'model':key,'capacity':250,'horizon':400,'spec':str(sp),'scored':len(cases),'cases':cases,'kind':'repeat' if name.endswith('repeat') else 'main'})
    import subprocess
    code=Path(__file__).resolve().parents[3]
    plan={'schema':'jit_retention_repair_D0_v1','stage':'D0','output':str(root),'code':str(code),
        'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
        'models':models,'locks':locks,'batches':batches,'indices':indices,'training_transitions':0,
        'budget':{'physical_transitions':1500000,'wall_seconds':14400},'maximum_planned_transitions':1200000,
        'adaptive_E_historical':{'adaptive_disturbance':True,'source':str(MONITOR/'bridge_pi0_hard_initial1000_20261010/comparison.json')},
        'baseline_failure_teacher_student_matrix':'NOT_EXECUTED_D1','teacher':None,
        'G_E_training':False,'seed_role':'DEV; independent RNG from viewed hard1000; never TRAIN/TEST',
        'pointers':{'learner_last':None,'best_dev_candidate':None,'published_policy':None},
        'original_campaign_status':read(CAMPAIGN/'series_0011_b1_resume_143/status.json')}
    for file in ('random_initials.npz','initial_precheck.json'):plan['locks'][str(root/file)]=sha(root/file)
    validate_budget(plan);write(root/'plan.json',plan)
    write(root/'status.json',{'phase':'prepared','charged_interactions':0,'training_transitions':0})
    return root/'plan.json'

def audit(plan_path):
    p=read(plan_path);charge=validate_budget(p)
    for path,h in p['locks'].items():
        if sha(path)!=h:raise ValueError('locked input changed '+path)
    from .checkpoint import load_checkpoint,CheckpointIdentity
    from .handoff_bank import pytree_sha256
    identities={}
    for name,m in p['models'].items():
        pol=m['policy'];side=read(Path(pol['checkpoint'])/'identity.json')
        fields={k:side[k] for k in ('config_sha256','xml_sha256','actor_frame_fields','actor_task_fields','action_order')}
        for k in ('actor_frame_fields','actor_task_fields','action_order'):fields[k]=tuple(fields[k])
        payload=load_checkpoint(Path(pol['checkpoint']),expected=CheckpointIdentity(**fields))
        identities[name]={}
        for key,value in [('actor_sha256',payload.actor_params),('critic_sha256',payload.critic_params),('normalizer_sha256',payload.observation_normalizer)]:
            h=pytree_sha256(value)
            if h!=pol[key]:raise ValueError('policy identity mismatch '+name+key)
            identities[name][key]=h
        identities[name]['checkpoint']=pol['checkpoint']
    for b in p['batches']:
        spec=read(b['spec']);load_requests(spec)
        if any(c['role']!='DEV' or not c['ancestor'].startswith('DEV-') for c in b['cases']):raise ValueError('role leak')
    result={'phase':'passed','planned_charge_with_padding':charge,'identities':identities,'optimizer_updates':0}
    write(Path(p['output'])/'audit.json',result);return result
