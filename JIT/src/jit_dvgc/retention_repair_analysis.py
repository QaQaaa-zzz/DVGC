"""Additional CPU evidence: historical first anomalies and coordinate factorial."""
import json
import pickle
from pathlib import Path
import numpy as np
from .retention_repair import read,write,sha,MONITOR,SOURCES
from .retention_repair_report import CachedArchive,liftoff_tick


def extend(plan_path,report_path):
    plan=read(plan_path);root=Path(report_path);summary=read(root/'summary.json');rawroot=Path(plan['output'])
    import jax
    from brax.training.acme import running_statistics as rs
    from .ppo import make_network_factory
    network=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
    predict=jax.jit(lambda n,a,x:network.parametric_action_distribution.mode(network.policy_network.apply(n,a,{'state':x})))
    with np.load(root/'D0b_fixed_observations.npz') as f:obs=f['raw_obs'];base=f['source_action']
    pol=plan['models']['pi0']['policy']
    with (Path(pol['checkpoint'])/'payload.pkl').open('rb') as f:pi0=pickle.load(f)
    factorial={}
    for m in ('R5','R21','R71','R73'):
        with (Path(plan['models'][m]['policy']['checkpoint'])/'payload.pkl').open('rb') as f:student=pickle.load(f)
        result={}
        for label,n,a in [('pi0_actor_student_normalizer',student.observation_normalizer,pi0.actor_params),
                ('student_actor_pi0_normalizer',pi0.observation_normalizer,student.actor_params),
                ('student_actor_student_normalizer',student.observation_normalizer,student.actor_params)]:
            delta=np.asarray(predict(n,a,obs))-base
            result[label]={'rmse_channels':np.sqrt(np.mean(delta**2,axis=0)).tolist(),'max_abs_difference':float(np.abs(delta).max())}
        factorial[m]=result
    write(root/'normalizer_actor_factorial.json',{'observations_role':'D0 DEV diagnostic only; never TRAIN anchors','observations':len(obs),
        'results':factorial,'limitation':'fixed-observation action sensitivity; swapping normalizer is not a physically tested repaired policy or causal effect on success'})
    lineage=read(MONITOR/'continuous_key_metrics_20261010/lineage.json');gradients={}
    for m in ('R5','R21','R71','R73'):
        r=Path(lineage[m[1:]]);p=r/'students/student/training'/f'round_{int(m[1:]):04d}_student'
        use=p/'bridge_demo_usage.json'
        if not use.exists():gradients[m]={'status':'MISSING'};continue
        u=read(use);contract=read(p/'bridge_student_contract.json')
        gradients[m]={'path':str(use),'sha256':sha(use),'loss_calls':u.get('learning_audit_loss_calls'),
            'first':u.get('learning_audit_first_losses',[None])[0],'latest':u.get('learning_audit_latest'),
            'keep_coefficient':contract.get('retention_coefficient'),'demo_coefficients':[contract.get('demo_coefficient_start'),contract.get('demo_coefficient_end')],
            'note':'unweighted fixed-probe auxiliary gradients vs actual PPO minibatch Actor gradient; distributions differ. Critic kept separate. Not a justification to increase keep blindly.'}
    write(root/'historical_actor_gradients.json',gradients)
    # Read original historical NPZ, not published envelope summaries.
    historical=[];histroot=MONITOR/'bridge_pi0_hard_initial1000_20261010'
    qi,vi=plan['indices']['root_qpos'],plan['indices']['root_dof']
    from scipy.spatial.transform import Rotation
    for model,folder in [('pi0','bridge_pi0'),('R71','bridge_latest')]:
        for onset in (0,5,10,15):
            path=histroot/f'onset_{onset:02d}_{folder}'/'prefixes.npz'
            with CachedArchive(path) as f:
                mask=f['prefix_mask'];q=f['qpos'];success=f['success'];failure=f['physical_failure'];contact=f['prohibited_contact']
                rolls=Rotation.from_quat(np.roll(q[...,qi+3:qi+7].reshape(-1,4),-1,axis=1)).as_euler('xyz')[:,0].reshape(mask.shape)
                for lane in range(mask.shape[1]):
                    n=int(mask[:,lane].sum());lift=liftoff_tick(f['front_wheel_clearance'][:n,lane],f['rear_wheel_clearance'][:n,lane])
                    def first(arr):
                        a=np.flatnonzero(arr[:n,lane]);return int(a[0]) if len(a) else None
                    def phase(t):
                        if t is None:return None
                        if f['valid_contact_seen_before'][t,lane]:return 'post_contact'
                        return 'airborne' if lift is not None and t>=lift else 'pre_liftoff'
                    rc=first(np.abs(rolls)>.6108652381980153);pc=first(contact);ff=first(failure)
                    historical.append({'model':model,'onset':onset,'lane':lane,'success':bool(success[:n,lane].any()),'steps':n,
                        'liftoff_control_tick':lift,'first_roll_above_35deg_tick':rc,'first_roll_above_35deg_phase':phase(rc),
                        'first_prohibited_contact_tick':pc,'first_prohibited_contact_phase':phase(pc),
                        'first_physical_failure_tick':ff,'first_physical_failure_phase':phase(ff),
                        'source':str(path),'adaptive_disturbance':True})
    counts={}
    for m in ('pi0','R71'):
        rr=[r for r in historical if r['model']==m];counts[m]={'n':len(rr),'successes':sum(r['success'] for r in rr)}
        for field in ('first_roll_above_35deg_phase','first_prohibited_contact_phase','first_physical_failure_phase'):
            counts[m][field]={phase:sum(r[field]==phase for r in rr) for phase in ('pre_liftoff','airborne','post_contact',None)}
    if counts['pi0']['successes']!=651 or counts['R71']['successes']!=309:raise ValueError('historical recount drift')
    write(root/'historical_first_anomalies.json',{'counts':counts,'episodes':historical,
        'limitations':'35deg roll is diagnostic source threshold; not retrospective relabeling.20ms frame resolution, no exact substep onset. Original task labels unchanged.'})
    # Qualify D1 as preparation only: frozen strong pi0 tail and pinned latest complete E/G.
    r73=Path(lineage['73']);g=read(r73/'incremental_result.json');gm=Path(g['checkpoint_manifest']);e=r73/'explorer_update/state.msgpack'
    write(root/'D1_preparation.json',{'schema':'jit_retention_teacher_preparation_v1','stage':'D1','execution_status':'not_started',
        'execute_authorized':False,'executable':False,'reason':'D0-only CLI; teacher/source sampling needs its own inspected executable plan',
        'pi0':pol,'G':{'manifest':str(gm),'manifest_sha256':sha(gm),'identity':read(gm),'inference':'ema','selection':'R73 last_valid complete; fixed for pilot'},
        'E':{'checkpoint':str(e),'sha256':sha(e),'config':str(r73/'explorer_update_spec.json'),'selection':'R73 last complete E; fixed for pilot'},
        'TRAIN_seed':SOURCES['teacher_train_seed'],'DEV_TEST_excluded':True,'inherit_hard1000_failures_for_training':False,
        'root_target':16,'root_cap':32,'onsets':[0,5,10,15],'pulse_steps':3,'delta_limit':[.25]*4,'H':16,
        'teacher_methods':['G16 candidates + separate source lane','correlated_noise16 candidates + separate source lane'],
        'pi0_independent_suffix_required_for_every_TRAIN_root':True,'same_snapshot_context_required':True,
        'relative_labels':['retention_debt','baseline_failure_teacher_recoverable_pending','baseline_failure_teacher_recoverable_learned','unknown'],
        'budget_upper_bound_requested':{'physical_transitions':2000000,'wall_seconds':21600,'student_updates':0},
        'actual_layout_dry_run':'PENDING; not authorized for execution by this preparation',
        'three_student_arms':'NOT_EXECUTABLE until repeatable pi0 failure -> teacher success witnesses exist',
        'student_initialization':{'all_arms':pol['checkpoint'],'actor_normalizer_critic_same_source':True,'optimizer':'fresh_once_if_pi0_optimizer_unavailable_then_complete_state',
            'freeze_actor_normalizer':True,'demo_clock':'completed_transitions','schedule_scope':'declared stage-local actual sampled transitions'},
        'student_limits':{'PPO_keep_transitions':128000,'BC_keep_updates':2000,'BC_PPO_keep_transitions':128000},
        'pointers':{'learner_last':None,'best_dev_candidate':None,'published_policy':None}})
    # Verify same-condition first trajectory differences in repetition data.
    batches=plan['batches']+(read(rawroot/'repeat_amendment.json')['batches'] if (rawroot/'repeat_amendment.json').exists() else []);difference=[];seen=set()
    for b in batches:
        if b['kind']!='repeat':continue
        with CachedArchive(rawroot/b['name']/'prefixes.npz') as f:
            for case in sorted(set(c['case'] for c in b['cases'])):
                cs=next(c for c in b['cases'] if c['case']==case)
                fingerprint=json.dumps([cs['qpos'],cs['qvel'],cs['request'],cs['onset']]);key=(b['model'],fingerprint)
                if key in seen:continue
                seen.add(key);lanes=[i for i,c in enumerate(b['cases']) if c['case']==case];end=[int(f['prefix_mask'][:,i].sum()) for i in lanes]
                overlap=min(end);anchor=lanes[0];values=[]
                for lane in lanes[1:]:
                    dif=np.max(np.abs(f['qpos'][:overlap,lane]-f['qpos'][:overlap,anchor]),axis=-1)
                    ix=np.flatnonzero(dif>1e-7);ad=np.max(np.abs(f['action'][:overlap,lane]-f['action'][:overlap,anchor]),axis=-1);ai=np.flatnonzero(ad>1e-7)
                    values.append({'first_qpos_difference_above_1e_7':int(ix[0]) if len(ix) else None,'first_action_difference_above_1e_7':int(ai[0]) if len(ai) else None})
                labels=[bool(f['success'][:n,i].any()) for i,n in zip(lanes,end)]
                difference.append({'model':b['model'],'case':case,'lanes':lanes,'labels':labels,'flipped':len(set(labels))>1,'differences':values})
    write(root/'repeat_first_differences.json',difference)
    import csv
    episodes=read(root/'episodes.json');parts=list(episodes[0]['reward_components'])
    fields=['model','group','case','repeat','success','steps','return','terminal_reward']+parts
    with (root/'reward_components.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({**{k:r[k] for k in fields if k not in parts},**r['reward_components']} for r in episodes)
    reward_checks=[]
    for b in batches:
        with CachedArchive(rawroot/b['name']/'prefixes.npz') as f:
            error=np.abs(f['reward']-f['metric/reward/pre_episode_return_override']-f['metric/reward/episode_return_override'])[f['prefix_mask']]
            maximum=float(error.max(initial=0.))
            if maximum>1e-4:raise ValueError('reward override reconstruction mismatch')
            reward_checks.append({'batch':b['name'],'maximum_per_step_error':maximum})
    write(root/'reward_reconstruction.json',{'checks':reward_checks,'equation':'reward = pre_episode_return_override + episode_return_override','physical_simulation':False})
    plot_details(plan,root)
    conclusions={}
    for model in ('R5','R21','R71','R73'):
        g=summary['four_cells'][model];conclusions[model]={'B_C_D_new':sum(g[x]['N01_new_vs_pi0'] for x in 'BCD'),'B_C_D_lost':sum(g[x]['N10_lost_vs_pi0'] for x in 'BCD')}
    text='\n\nAdditional audited diagnosis\n\n'
    text+='- Original adaptive-E NPZ recount confirms π0 651/1000 and R71 309/1000; first roll/prohibited-contact/failure times parsed independently of terminal codes. [Original trajectory anomalies](historical_first_anomalies.json).\n'
    text+='- [Normalizer/Actor factorial](normalizer_actor_factorial.json) isolates action-coordinate sensitivity on the same4096 saved observations. It is not a physically tested normalizer swap.\n'
    text+='- [R5/R21/R71/R73 historical Actor gradients and KL](historical_actor_gradients.json) are real training telemetry. R73 latest raw PPO/demo/keep norms14.0519/.1165/.1116; keep coefficient.2, PPO/demo cosine−.7909, PPO/keep cosine.1618, behavior KL.01266. Different probe distributions and critic gradients forbid treating these as a weight prescription.\n'
    text+='- [32 unique repeated conditions and first differences](repeat_first_differences.json); no strict numerical equivalence claim. Nominal A is one unique condition; new/lost counts there are execution observations, not distinct capabilities.\n'
    text+='- R71/R73 fail substantially already in B (random initial/no pulse); adaptive E cannot explain all loss. R73 also has31 lost C successes despite33 gained C successes, so nominal-pulse capability is redistributed. R5 is positive on this DEV panel but was not independently trained with a matched budget; no algorithm-superiority or adoption claim.\n'
    text+='- [Paired fixed-case reward and roll traces](paired_reward_roll.png) use the first declared case in each group, selected without outcomes. Per-episode rewards/terminal rewards/components remain in episodes.json.\n'
    text+='- [D1 fixed π0/E/G preparation](D1_preparation.json) freezes versions and defines independent TRAIN suffix labels; it is not executable or launched. No qualified baseline-failure teacher witnesses exist from D0, so three student arms remain preparation only.\n'
    text+='- Opt-in fixed-normalizer/explicit-transition schedule implementation is CPU-tested, including pinned Brax layout and full-learner hook composition; no student training or performance validation. Legacy runs and defaults unchanged. Actor coordinates freeze with all statistics leaves; critic weights can train independently.\n'
    text+='- [All episode signed reward components](reward_components.csv) and [per-step terminal override reconstruction](reward_reconstruction.json) checked on all14 batches; no additional simulation.\n'
    text+='- Analysis corrections: retain failed original report; normalized export now supplies both observation dictionary fields. Liftoff diagnostic waits for ground contact then3 airborne frames, excluding initial3cm suspension. It remains control-frame evidence, not exact substep liftoff.\n'
    for filename in ('INDEX.md','D0_diagnostic_report.md'):
        with (root/filename).open('a') as f:f.write(text)
    write(root/'status.json',{'phase':'completed','stage':'D0_and_D0b','physical_charged':summary['physical_charged_completed'],'training_updates':0,
        'original_execution_report_failure_preserved':True,'evaluation':'complete','adoption':'not_applicable','D1':'prepared_only_not_executable'})
    return counts,conclusions


def plot_details(plan,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(18,8))
    for b in plan['batches']:
        if b['kind']!='main':continue
        with CachedArchive(Path(plan['output'])/b['name']/'prefixes.npz') as f:
            for j,g in enumerate('ABCD'):
                found=[(i,c) for i,c in enumerate(b['cases']) if c['case']==g+'_000']
                if not found:continue
                i,c=found[0];n=int(f['prefix_mask'][:,i].sum());steps=np.arange(n)
                axes[0,j].plot(steps,f['reward'][:n,i],label=b['model'],lw=1)
                axes[1,j].plot(steps,np.rad2deg(f['roll'][:n,i]),label=b['model'],lw=1)
                if g in 'CD':
                    for ax in axes[:,j]:ax.axvspan(c['onset'],c['onset']+3,color='grey',alpha=.08)
    for j,g in enumerate('ABCD'):
        axes[0,j].set(title=g+' case000',ylabel='Reward per control step');axes[1,j].set(xlabel='Control step (20ms)',ylabel='Roll (deg)')
        for ax in axes[:,j]:ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('D0 first declared case per group; paired models; actual endpoints, fixed exogenous request window')
    fig.tight_layout();fig.savefig(root/'paired_reward_roll.png',dpi=160);plt.close(fig)
