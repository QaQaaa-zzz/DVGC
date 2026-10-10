"""Saved-data D2 micro audit and zero-physics formal preparation supplement."""
from pathlib import Path
import pickle
import numpy as np
from .retention_repair import read,write,sha


def finish(path):
    p=read(path);root=Path(p['output']);out=root/f'report_{len(list(root.glob("report_*")))+1:03d}';out.mkdir(exist_ok=False)
    verification=read(root/'gpu_micro/verification.json')
    if verification['phase']!='passed':raise ValueError('micro not verified')
    keep=read(root/'keep_receipt.json');demo=read(p['demo_manifest']);candidates=read(root/'keep_collection/candidates.json')
    successful={r['lane']:r for r in keep['episodes']}
    positive=[]
    from .config import load_config
    from .model import load_host_model
    from .constants import CTRL_DT,SIM_DT
    from types import SimpleNamespace
    from .exploration_continuation import snapshot_from_arrays
    from .unified_envelope_snapshot import save_unified_envelope_snapshot,physical_state_sha256,snapshot_context_sha256
    from .retention_repair_report import CachedArchive
    config=read(p['models']['pi0']['policy']['formal_config'])
    host=load_host_model(load_config(Path(config['inputs']['up_config_path']),runtime_only=True))
    env=SimpleNamespace(_bundle=host,dt=CTRL_DT,sim_dt=SIM_DT)
    tape=root/'keep_collection/prefixes.npz';tape_hash=sha(tape)
    with CachedArchive(tape) as f:
        for lane,episode in successful.items():
            tick=2
            arrays={k[5:]:f[k][tick,lane] for k in f.files if k.startswith('snap/')}
            if bool(arrays['info/done']):raise ValueError('nonterminal positive snapshot required')
            snap=snapshot_from_arrays(arrays,env=env,record=p['models']['pi0']['policy'],parent_trajectory=str(tape)+'::'+str(lane)+'::'+str(tick),parent_state_sha256=tape_hash)
            dest=out/'pi0_success_snapshots'/f'{lane:05d}';save_unified_envelope_snapshot(dest,snap)
            positive.append(dict(ancestor=episode['ancestor'],snapshot=str(dest),state_sha256=physical_state_sha256(snap),snapshot_context_sha256=snapshot_context_sha256(snap),snapshot_control_step=3,snapshot_time=float(f['time_after'][tick,lane]),episode_success=True,role='TRAIN',GPU_reset_validation='NOT_RUN'))
    teachers=read(p['teacher_manifest'])['TRAIN'];selected=read(Path(p['D1_receipt']).parent/'selected_roots.json');by={r['root_id']:r for r in selected}
    difficult=[dict(by[r['root_id']],teacher_lesson_sha256=r['sha256']) for r in teachers]
    pools=dict(nominal_complete=dict(probability=.25,qpos=p['indices']['nominal_qpos'],qvel=p['indices']['nominal_qvel']),
        random_complete=dict(probability=.25,path=str(root/'keep_initial.npz'),sha256=sha(root/'keep_initial.npz'),lanes=list(range(16,32))),
        pi0_success_snapshots=dict(probability=.25,entries=positive),teacher_recoverable=dict(probability=.25,entries=difficult),
        role='TRAIN',shared_by=['A','C'],empty_pool='raise; never renormalize',simulator_history='restore complete three-frame history and task context; no new H16 reset',
        actual_reset_counts='NOT_RUN',actual_transition_counts='NOT_RUN')
    write(out/'reset_pool_inventory.json',pools)
    from .retention_next import random_initials
    q,v,precheck=random_initials(p['models']['pi0'],p['indices'],16,1010258101)
    requests=np.random.default_rng(1010258102).uniform(-.25,.25,(16,3,4)).astype('f4')
    cases=[]
    for group in 'ABCD':
        for i in range(16):
            onset=[0,5,10,15][i//4]
            cases.append(dict(case=f'{group}_{i:02d}',group=group,onset=onset,role='DEV',ancestor=f'D2_DEV-1010258101-'+(str(i) if group in 'BD' else 'nominal'),request_id=f'D2_DEV-1010258102-{i}' if group in 'CD' else 'zero',
                qpos=q[i].tolist() if group in 'BD' else p['indices']['nominal_qpos'],qvel=v[i].tolist() if group in 'BD' else p['indices']['nominal_qvel'],request=requests[i].tolist() if group in 'CD' else np.zeros((3,4)).tolist()))
    write(out/'new_dev_cases.json',cases);write(out/'new_dev_precheck.json',precheck)
    write(out/'eval_contract.json',dict(role='DEV',capacity=64,horizon=400,charged_per_checkpoint=25600,paired='B/D same new starts, C/D same fixed requests; A one nominal condition16 repeats',
        baseline_models=['pi0','R5'],BC_checkpoints=[100,500,1000,2000],stop_checks='A/C B/D32episodes at32k checkpoints; decline>5pp relative each arm start requires repeat confirmation',
        four_combinations=dict(roots=p['solver_dev_roots'],layouts='original17-world layout including logical root lane, no handover reset',variants=['Teacher H16 + pi0','Student H16 + pi0','Teacher H16 + Student','Student independent']),
        evaluation_execution='NOT_RUN',TEST_used=False))
    # Same fixed TRAIN probes reconstructed from the actual saved before/after weights.
    import jax
    from brax.training.acme import running_statistics
    from .ppo import make_network_factory
    net=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=running_statistics.normalize)
    params={}
    for step in (0,50,100):
        with open(root/f'gpu_micro/inference_{step}.pkl','rb') as f:params[step]=pickle.load(f)
    with np.load(keep['path']) as f:keep_obs=f['actor_observation_before'][np.linspace(0,len(f['actor_observation_before'])-1,32,dtype=int)]
    with np.load(demo['path']) as f:
        ids=np.linspace(0,len(f['observations'])-1,32,dtype=int);demo_obs=f['observations'][ids];targets=f['actions'][ids]
    def action(param,obs):return np.asarray(net.parametric_action_distribution.mode(net.policy_network.apply(param[0],param[1],{'state':obs})))
    reference=action(params[0],keep_obs);probe=[]
    for before,after in ((0,50),(50,100)):
        b=action(params[before],keep_obs);a=action(params[after],keep_obs)
        probe.append(dict(before_transition=before,after_transition=after,keep_mse_before=float(np.mean((b-reference)**2)),keep_mse_after=float(np.mean((a-reference)**2)),
            demo_mse_before=float(np.mean((action(params[before],demo_obs)-targets)**2)),demo_mse_after=float(np.mean((action(params[after],demo_obs)-targets)**2)),action_change_mse=float(np.mean((a-b)**2))))
    write(out/'same_update_fixed_TRAIN_probes.json',dict(scope='posthoc deterministic inference on saved actual before/after checkpoints; separate from real sampled loss gradients',keep_source=keep['path'],demo_source=demo['path'],rows=probe))
    audit=read(root/'gpu_micro/actual_updates.json')
    scalar_coeff=dict(ppo_policy=1.,entropy=p['ppo_source']['entropy_cost'],entropy_gradient='already multiplied by entropy_cost',demo='effective_lambda_demo per row',keep=.2)
    from scipy.spatial import cKDTree
    with np.load(keep['path']) as f:all_keep=f['actor_observation_before']
    with np.load(demo['path']) as f:all_demo=f['observations'];all_targets=f['actions']
    mean=np.asarray(params[0][0].mean['state']);std=np.asarray(params[0][0].std['state'])
    standardized=lambda o:(o-mean)/np.maximum(std,1e-6)/np.sqrt(76)
    distances,nearest=cKDTree(standardized(all_keep)).query(standardized(all_demo))
    baseline_targets=action(params[0],all_keep[nearest]);action_rmse=np.sqrt(np.mean((all_targets-baseline_targets)**2,axis=-1))
    close=(distances<=.02)&(action_rmse>.05)
    write(out/'TRAIN_label_conflict_review.json',dict(scope='descriptive nearest neighbor diagnostic; no automatic relabel/filter',coordinate='pi0 normalized76D RMSE',near_threshold=.02,action_RMSE_threshold=.05,
        count=int(close.sum()),demo_examples=len(all_demo),keep_examples=len(all_keep),examples=[dict(demo_index=int(i),keep_index=int(nearest[i]),observation_RMSE=float(distances[i]),action_RMSE=float(action_rmse[i])) for i in np.flatnonzero(close)[:20]],DEV_used=False))
    budget=dict(p['budget'],BC_solver_dev_selection=4*len(p['solver_dev_roots'])*3*17*400,warning_repeat_confirm=8*32*400,formal_reset_GPU_check=8*25)
    budget['physics_total']=p['budget']['physics_total']+budget['BC_solver_dev_selection']+budget['warning_repeat_confirm']+budget['formal_reset_GPU_check']
    if budget['physics_total']>budget['hard_cap']:raise ValueError('supplement layout exceeds D2 budget')
    write(out/'budget_dry_run.json',budget)
    write(out/'actual_update_analysis.json',dict(coefficients=scalar_coeff,actual_updates=audit,fixed_probes=probe,global_clip='one combined Actor+critic norm, max0.75; Adam delta is actual parameter change, not per-term attribution',
        scope='2 engineering PPO+demo+keep updates; insufficient to infer long-run retention',keep_at_zero_update='pi0 copy has near-zero keep error/gradient by construction; does not justify multiplying keep600'))
    prep=dict(schema='jit_retention_D2_formal_supplement_v1',original_plan=str(path),original_plan_sha256=sha(path),reset_pool_inventory=str(out/'reset_pool_inventory.json'),new_DEV=str(out/'new_dev_cases.json'),
        BC_entry='generative_bridge.warmup.warmup_actor(full_learner=True,actual_gradient_audit=True,updates=2000); run B first after separate finite launch',
        PPO_fresh_once='A starts pi0, C exact selected B Actor/critic/normalizer with one explicitly fresh PPO optimizer; BC optimizer is separate and retained',
        same_stage_resume='LearnerHooks(preserve_stage_clock=True); restore full learner after each32k; do not rebootstrap critic/Adam',
        formal_execution_authorized=False,formal_training_started=False,automatic_next_stage=False,
        remaining_execution_gates=['GPU verification of four-pool training reset adapter and full context','new DEV physical baselines and finite B launch'],
        budget=budget,BC_selection=dict(candidates=[0,100,500,1000,2000],primary='independent student success count on locked7solver_dev roots',retention_guard='new full-task DEV B/D decline>5pp versus pi0 requires repeat confirmation; exclude confirmed regressed checkpoint',tie='earliest update',no_absorption='no PPO launch if no independent solver-dev gain',Teacher_no_solution='NA, never failure or student success'),actual_engineering_charged=12900,physics_charged_in_this_report=0,
        locks={str(out/name):sha(out/name) for name in ('reset_pool_inventory.json','new_dev_cases.json','eval_contract.json','budget_dry_run.json','TRAIN_label_conflict_review.json')})
    write(out/'formal_preparation.json',prep)
    text=f'''# D2 conditional preparation and real GPU micro

Source pi0 Actor/critic/normalizer; R5 frozen candidate. Formal A/B/C: NOT_RUN. D1 qualified8TRAIN lessons; new successful keep {len(successful)}/32 ({len(keep['groups']['nominal'])}/16 nominal, {len(keep['groups']['random'])}/16 random). Nominal samples are numerical repeats of one physical start. Keep weights balance nominal/random, then ancestor and trajectory. No DEV used in training inputs.

Actual engineering charged12900 =12800 keep acquisition +100 real GPU transitions. Two actual optimizer updates; whole normalizer frozen, source-to-checkpoint hash identical. Actor/critic/Adam changed; full state and RNG restoration exact, clock100, action difference0. lambda_demo={verification['checks']['lambda_values']}. Engineering invariants do not establish learning ability.

[TensorBoard](http://localhost:6025): current actual/reward_per_transition at50/100 verified HTTP; reward is logging only. [GPU receipt](../gpu_micro/verification.json), [actual weighted gradients and Adam deltas](actual_update_analysis.json), [fixed TRAIN before/after probes](same_update_fixed_TRAIN_probes.json).

D2 refined actual-layout budget{budget["physics_total"]}/2000000physics (includes all four BC solver-dev selection checkpoints, conditional warning repeats and GPU reset check), BC<=2000updates,12hours. Quarter whole-PPO LR7.5e-6, original remaining PPO/reward/network/action/physics/success settings retained; BC LR1e-5/demo1/keep1 inherited. Fresh_once PPO boundary allowed explicitly for A/C; later chunks preserve complete learner and cumulative128000 stage schedule.

[Formal preparation](formal_preparation.json), [four reset pools](reset_pool_inventory.json), [new DEV cases](new_dev_cases.json), [evaluation contract](eval_contract.json), [input locks](../source_lock.json), [refined budget](budget_dry_run.json), [label conflicts](TRAIN_label_conflict_review.json).

Decision: prepare B first; formal four-pool reset execution gates and new DEV baselines remain to verify; B selection score is locked in the supplement before a separate finite B launch. Student absorption, old-capability loss, independent final success: NOT_RUN. No automatic PPO arm or long training.
'''
    (out/'INDEX.md').write_text(text);(out/'analysis_update.md').write_text(text);return out/'INDEX.md'
