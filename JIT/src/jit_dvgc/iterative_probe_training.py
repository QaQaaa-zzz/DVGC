"""Versioned Actor-only PPO on exact witnessed snapshots and fixed jump starts."""
from pathlib import Path
from types import SimpleNamespace
from .jump_evidence_validation import read,write,file_sha,verify_hash
from .evidence_integrity import canonical_sha256

SCHEMA='jit_iterative_probe_training_v1'
SUPPORT_SCHEMA='jit_iterative_witnessed_support_v1'



def checkpoint_evaluation_plan(steps, checkpoints=None, *, samples_per_phase=2, horizon=400):
    """Declare aligned milestones and worst-case cost before any training starts.

    Checkpoints are inference artifacts; they do not support optimizer restart.
    All milestones run inside one trainer invocation with its live learning state.
    """
    if type(steps) is not int or steps <= 0 or steps % 3200:
        raise ValueError('invalid aligned probe PPO budget')
    milestones = [steps] if checkpoints is None else list(checkpoints)
    if (not milestones or any(type(s) is not int or s <= 0 or s % 3200 or s > steps for s in milestones)
        or milestones != sorted(set(milestones)) or milestones[-1] != steps):
        raise ValueError('checkpoint schedule must be increasing, aligned and end at the training ceiling')
    if type(samples_per_phase) is not int or samples_per_phase <= 0:
        raise ValueError('positive samples_per_phase required')
    if type(horizon) is not int or not 0 < horizon <= 400:
        raise ValueError('panel horizon must be within the episode horizon')
    maximum_panel = len(milestones) * 2 * samples_per_phase * horizon
    return dict(schema='jit_probe_checkpoint_evaluation_v1', role='train',
        checkpoint_transitions=[0, *milestones], train_panel_transitions=milestones,
        samples_per_phase=samples_per_phase, horizon=horizon,
        maximum_training_transitions=steps, maximum_panel_interactions=maximum_panel,
        maximum_total_interactions=steps + maximum_panel, automatic_extension=False,
        optimizer_continuation='same_live_trainer_only', checkpoint_resume_supported=False,
        final_test_used=False)


def fixed_train_panel_identity(support, plan):
    """Lock the exact deterministic panel selection used by rollout_fixed_tube_panel."""
    from copy import deepcopy
    from .tube_rsi_smoke import fixed_indices
    entries = sorted(support['entries'], key=lambda r: (r['phase'] != 'upstream', r['key']))
    selected = []
    for phase in ('upstream', 'downstream'):
        rows = [r for r in entries if r['phase'] == phase]
        if not rows:
            raise ValueError('both TRAIN panel phases required')
        for index in fixed_indices(len(rows), plan['samples_per_phase']):
            selected.append(dict(phase=phase, entry_index=index, entry=deepcopy(rows[index])))
    identity = dict(schema='jit_probe_fixed_train_panel_v1', role='train', final_test_used=False,
        support_sha256=support['support_sha256'], selected_entries=selected,
        horizon=plan['horizon'], samples_per_phase=plan['samples_per_phase'],
        success_criterion='first_valid_landing', deterministic_policy=True,
        rng_contract='1100000 + phase_index*10000 + ordinal*horizon + tick',
        reset_contract='fresh episode and phase clocks; preserve saved event and controller context')
    return {**identity, 'panel_sha256': canonical_sha256(identity)}


def charged_train_panel_interactions(run_dir):
    """Completed panels charge actual ticks; interrupted panels keep their reservation."""
    total = 0
    for path in Path(run_dir).glob('train_panels/transition_*/reservation.json'):
        reservation = read(path)
        reserve = reservation['maximum_interactions']
        if type(reserve) is not int or reserve <= 0:
            raise ValueError('invalid checkpoint panel reservation')
        receipt = path.parent / 'completion.json'
        charge = reserve
        if receipt.exists():
            completion = read(receipt)
            report_path = path.parent / 'report.json'
            if not report_path.exists():
                raise ValueError('completed checkpoint panel report missing')
            report = read(report_path)
            charge = completion['charged_interactions']
            if (completion.get('panel_sha256') != reservation.get('panel_sha256')
                or report.get('fixed_train_panel', {}).get('panel_sha256') != reservation.get('panel_sha256')
                or report.get('training_checkpoint_transition') != reservation.get('training_checkpoint_transition')
                or type(report.get('environment_interactions')) is not int
                or report['environment_interactions'] != charge):
                raise ValueError('checkpoint panel receipt identity or accounting mismatch')
        if type(charge) is not int or not 0 < charge <= reserve:
            raise ValueError('invalid checkpoint panel charge')
        total += charge
    return total


def load_config(path):
    from .unified_formal import UnifiedFormalConfig,UnifiedFormalSchedule,UnifiedResetMixture
    from .unified_training import UnifiedPPOConfig
    raw=read(path)
    if raw.get('schema')!=SCHEMA:raise ValueError('unknown probe training schema')
    for p,sha in raw['input_files'].items():
        if file_sha(p)!=sha:raise ValueError('probe training input changed')
    support=read(raw['support']);verify_hash(support,'support_sha256')
    if support.get('schema')!=SUPPORT_SCHEMA or support.get('role')!='train' or support.get('final_test_used') is not False:
        raise ValueError('TRAIN witnessed support required')
    if not support['entries'] or any(not r['witnessed'] for r in support['entries']):raise ValueError('unwitnessed training reset')
    ppo=UnifiedPPOConfig(**raw['ppo'])
    if (ppo.num_parallel_envs!=128 or ppo.batch_size!=16 or ppo.num_minibatches!=8 or ppo.unroll_length!=25
        or ppo.episode_horizon!=400 or ppo.requested_transitions<=0 or ppo.requested_transitions%ppo.block_transitions):
        raise ValueError('invalid aligned probe PPO budget')
    if raw['jump_start_probability']!=.2 or raw['success_criterion']!='first_valid_landing':raise ValueError('probe reset/endpoint drift')
    init=raw['initialization']
    if init['actor']!='warm_start_frozen_unified' or init['critic']!='fresh' or init['optimizer']!='fresh':raise ValueError('probe initialization drift')
    for p,sha in support['inputs'].items():
        if file_sha(p)!=sha:raise ValueError('witnessed support input changed')
    plan=checkpoint_evaluation_plan(ppo.requested_transitions)
    if 'checkpoint_evaluation' in raw:
        declared=raw['checkpoint_evaluation']
        plan=checkpoint_evaluation_plan(ppo.requested_transitions, declared['train_panel_transitions'],
            samples_per_phase=declared['samples_per_phase'], horizon=declared['horizon'])
        if declared != plan:raise ValueError('checkpoint evaluation declaration drift')
        if raw.get('fixed_train_panel') != fixed_train_panel_identity(support, plan):
            raise ValueError('fixed TRAIN panel identity drift')
    return UnifiedFormalConfig(schema=SCHEMA,raw=raw,config_sha256=canonical_sha256(raw),
        runtime_naccdmax=1024,reset_mixture=UnifiedResetMixture('fixed_jump_start_20pct_exact_snapshot_80pct',.2,.8),
        ppo=ppo,formal=UnifiedFormalSchedule(tuple(plan['checkpoint_transitions']),tuple(plan['train_panel_transitions']),plan['samples_per_phase'],'fresh_only'),
        up_config_path=raw['inputs']['up_config_path'],up_config_sha256=raw['inputs']['up_config_sha256'],
        down_config_path=raw['inputs']['down_config_path'],down_config_sha256=raw['inputs']['down_config_sha256'],
        soft_tube_path=raw['support'],soft_tube_manifest_sha256=support['support_sha256'],
        tube_rsi_smoke_report=raw['support'],tube_rsi_smoke_report_sha256=file_sha(raw['support']))


def restore_params(path):
    from .unified_policy_freeze import load_frozen_unified_manifest,_checkpoint_identity,_load_policy_formal_config
    from .checkpoint import load_checkpoint
    from .handoff_bank import pytree_sha256
    config=load_config(path)
    policy=load_frozen_unified_manifest(Path(config.raw['initialization']['source_frozen_policy']))['policy']
    source=_load_policy_formal_config(Path(policy['formal_config']))
    checkpoint=load_checkpoint(Path(policy['checkpoint']),expected=_checkpoint_identity(source))
    values=(checkpoint.observation_normalizer,checkpoint.actor_params,checkpoint.critic_params)
    for name,value in zip(('normalizer_sha256','actor_sha256','critic_sha256'),values):
        if pytree_sha256(value)!=policy[name]:raise ValueError('initializer checkpoint identity drift')
    return values


def fresh_sample(sample):
    """Reset continuation clocks only; preserve controller and arrival context."""
    import jax.numpy as jp
    return {**sample, 'episode_step':jp.asarray(0,jp.int32),
            'phase_episode_step':jp.asarray(0,jp.int32),
            'episode_return':jp.asarray(0.,jp.float32),
            'events':{**sample['events'],'episode_step':jp.asarray(0,jp.int32)}}


def first_landing_state(state):
    import jax.numpy as jp
    from .constants import END_FIRST_VALID_LANDING
    success=state.info['down_events'].valid_contact_seen & ~state.info['physical_failure']
    truncated=state.info['truncated'] & ~success
    info={**state.info,'success':success,'timeout':state.info['timeout'] & ~success,
          'terminated':state.info['terminated'] | success,'truncated':truncated,
          'time_out':truncated.astype(jp.float32),
          'end_code':jp.where(success,END_FIRST_VALID_LANDING,state.info['end_code'])}
    metrics={**state.metrics,'terminal/success':success.astype(jp.float32),
             'terminal/descent_success':success.astype(jp.float32),
             'terminal/timeout':info['timeout'].astype(jp.float32),
             'terminal/descent_timeout':state.metrics['terminal/descent_timeout']*(~success)}
    return state.replace(done=jp.maximum(state.done,success.astype(state.done.dtype)),info=info,metrics=metrics)


def build_environment(config):
    import jax
    import jax.numpy as jp
    from .config import load_config as phase_config
    from .unified_env import UnifiedTubeRSIEnv
    from .snapshot_pool import SnapshotPool
    from .tube_rsi import TubeRSIPool
    from .handoff_snapshot import compatibility_identity
    from .semantics import initial_event_state
    from .descent_semantics import initial_descent_events
    from .unified_formal import load_unified_policy_formal_config
    from .unified_diagnostic import _load_runtime
    # One immutable bootstrap runtime supplies XML and the original task context.
    _,_,bootstrap_artifact,_=_load_runtime(load_unified_policy_formal_config(Path(config.raw['bootstrap_formal_config'])))
    support=read(config.raw['support']);entries=sorted(support['entries'],key=lambda r:(r['phase']!='upstream',r['key']))
    artifact=SimpleNamespace(root=Path(config.raw['support']).parent,entries=entries,
        manifest={'schema':SUPPORT_SCHEMA,'status':'completed','training_guidance_only':True,
                  'test_data_used':False,'validation_data_used':False,'manifest_sha256':support['support_sha256']})

    class ProbeEnv(UnifiedTubeRSIEnv):
        def reset(self,rng):
            decision,key=jax.random.split(rng)
            jump=jax.random.bernoulli(decision,.2)
            tube=self._tube_pool.sample(key)
            tube=fresh_sample(tube)
            start=self._natural_reset_sample(key,tube)
            root_x=jp.asarray(2.5,jp.float32);index=self._bundle.model_index.root_qpos_address
            up=initial_event_state(root_x,self._resolved_config);down=initial_descent_events(root_x)
            start={**start,'qpos':start['qpos'].at[index].set(root_x),
                'events':{n:jp.asarray(getattr(up,n)) for n in start['events']},
                'down_events':{n:jp.asarray(getattr(down,n)) for n in start['down_events']}}
            state=self._reset_from_tube_sample(self._select_reset_sample(jump,start,tube))
            return self._with_reset_source(state,soft_tube=~jump,jump_start=jump)

        def reset_tube_index(self,phase_index,entry_index):
            sample=fresh_sample(self._tube_pool.sample_at(phase_index,entry_index))
            return self._with_reset_source(self._reset_from_tube_sample(sample),soft_tube=True)

        def step(self,state,action):
            return first_landing_state(super().step(state,action))

    up=phase_config(Path(config.up_config_path));down=phase_config(Path(config.down_config_path))
    if up.config_sha256!=config.up_config_sha256 or down.config_sha256!=config.down_config_sha256:raise ValueError('phase config drift')
    env=ProbeEnv(up,down,bootstrap_artifact,runtime_naccdmax=1024)
    pool=SnapshotPool.from_paths([Path(r['snapshot']) for r in entries],compatibility=compatibility_identity(env))
    up_rows=[r for r in entries if r['phase']=='upstream'];down_rows=[r for r in entries if r['phase']=='downstream']
    if not up_rows or not down_rows:raise ValueError('both training phases required')
    weights=lambda rows:jp.asarray([r['sampling_weight'] for r in rows],jp.float32)
    uw,dw=weights(up_rows),weights(down_rows)
    env._tube_pool=TubeRSIPool(artifact,pool,uw,dw,jp.log(uw/uw.sum()),jp.log(dw/dw.sum()),
                              len(up_rows),len(down_rows),0,0,None)
    return artifact,env


def make_config(support_path,initializer_path,bootstrap_config,output,run_id,iteration,steps,seed,*,
                checkpoints=None,panel_samples_per_phase=2,panel_horizon=400):
    support_path,initializer_path,bootstrap_config=map(lambda p:Path(p).resolve(),(support_path,initializer_path,bootstrap_config))
    base=read(bootstrap_config)
    raw=dict(schema=SCHEMA,support=str(support_path),bootstrap_formal_config=str(bootstrap_config),
        inputs={k:base['inputs'][k] for k in ('up_config_path','up_config_sha256','down_config_path','down_config_sha256')},
        initialization={'actor':'warm_start_frozen_unified','critic':'fresh','optimizer':'fresh',
                        'source_frozen_policy':str(initializer_path)},
        jump_start_probability=.2,success_criterion='first_valid_landing',
        reward_contract='unchanged phase rewards; terminate at first valid landing',
        snapshot_reset_contract='fresh episode and phase clocks; preserve saved event and controller context',
        production_training_smoke_verified=False,
        ppo={**base['ppo'],'num_parallel_envs':128,'batch_size':16,'num_minibatches':8,
             'requested_transitions':steps,'seed':seed},
        run_declaration={'run_id':run_id},claim_boundary={'iteration':iteration,'test_data_used':False,'validation_data_used':False},
        input_files={str(p):file_sha(p) for p in (support_path,initializer_path,bootstrap_config)})
    if checkpoints is not None or panel_samples_per_phase != 2 or panel_horizon != 400:
        plan=checkpoint_evaluation_plan(steps,checkpoints,samples_per_phase=panel_samples_per_phase,horizon=panel_horizon)
        raw['checkpoint_evaluation']=plan
        raw['fixed_train_panel']=fixed_train_panel_identity(read(support_path),plan)
    write(output,raw);load_config(output)
    return raw
