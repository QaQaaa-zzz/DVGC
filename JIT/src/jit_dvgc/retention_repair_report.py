"""Read-only trajectory, paired capability and zero-update diagnostics."""
import csv
import pickle
from pathlib import Path
import numpy as np
from .retention_repair import read,write,sha


class CachedArchive:
    """Decompress each requested NPZ array once per analysis scope."""
    def __init__(self,path):self.raw=np.load(path,allow_pickle=False);self.files=self.raw.files;self.cache={}
    def __enter__(self):return self
    def __exit__(self,*args):self.raw.close()
    def __getitem__(self,key):
        if key not in self.cache:self.cache[key]=self.raw[key]
        return self.cache[key]

def normalized_actor_observations(stats,obs,privileged):
    from brax.training.acme import running_statistics
    return np.asarray(running_statistics.normalize({'state':obs,'privileged_state':privileged},stats)['state'])


def zero_update_audit(plan,output=None):
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from .ppo import make_network_factory
    from .checkpoint import save_checkpoint,load_checkpoint
    from .handoff_bank import pytree_sha256
    rawroot=Path(plan['output']);root=Path(output or rawroot);pol=plan['models']['pi0']['policy']
    with (Path(pol['checkpoint'])/'payload.pkl').open('rb') as f:source=pickle.load(f)
    copy=root/'D0b_pi0_zero_update_copy'
    if not copy.exists():save_checkpoint(copy,source)
    restored=load_checkpoint(copy,expected=source.identity)
    network=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=running_statistics.normalize)
    obs=[];priv=[];fifo_max=0.;continuity_max=0.;counter_mismatch=0;initial_counters=[]
    for b in plan['batches']:
        if b['model']!='pi0' or b['kind']!='main':continue
        with CachedArchive(rawroot/b['name']/'prefixes.npz') as f:
            m=f['prefix_mask'][:,:b['scored']];before=f['actor_observation_before'][:,:b['scored']];after=f['actor_observation_after'][:,:b['scored']]
            obs.append(before[m][::max(1,int(m.sum())//2048)][:2048]);priv.append(f['observation'][:,:b['scored']][m][::max(1,int(m.sum())//2048)][:2048])
            # Compare actual old->new FIFO observable coordinates excluding valid flags.
            error=np.abs(after[...,:75].reshape(*m.shape,3,25)[...,:2,:24]-before[...,:75].reshape(*m.shape,3,25)[...,1:,:24])
            fifo_max=max(fifo_max,float(error[m].max(initial=0.)))
            active=m[1:]&m[:-1]
            continuity_max=max(continuity_max,float(np.abs(before[1:]-after[:-1])[active].max(initial=0.)))
            counter_mismatch+=int(np.count_nonzero((f['success'][:,:b['scored']]&m)&(f['recovery_ticks'][:,:b['scored']]<25)))
            initial_counters.extend(f['recovery_ticks_before'][0,:b['scored']].tolist())
    obs=np.concatenate(obs).astype('f4');priv=np.concatenate(priv).astype('f4')
    mode=network.parametric_action_distribution.mode
    predict=jax.jit(lambda n,a,x:mode(network.policy_network.apply(n,a,{'state':x})))
    original=np.asarray(predict(source.observation_normalizer,source.actor_params,obs))
    copied=np.asarray(predict(restored.observation_normalizer,restored.actor_params,obs))
    exported=ppo_networks.make_inference_fn(network)((restored.observation_normalizer,restored.actor_params,restored.critic_params),deterministic=True)
    action_export=np.asarray(exported({'state':obs},jax.random.PRNGKey(0))[0])
    # Actor has no gradient and never changes: only installed normalization update.
    updated=running_statistics.update(source.observation_normalizer,{'state':obs,'privileged_state':priv})
    drifted=np.asarray(predict(updated,source.actor_params,obs))
    norm_raw=normalized_actor_observations(source.observation_normalizer,obs,priv)
    norm_changed=normalized_actor_observations(updated,obs,priv)
    np.savez_compressed(root/'D0b_fixed_observations.npz',raw_obs=obs,normalized_source=norm_raw,normalized_updated=norm_changed,
        source_action=original,copied_action=copied,exported_action=action_export,normalizer_only_action=drifted)
    hashes={name:{'source':pytree_sha256(getattr(source,field)),'copy':pytree_sha256(getattr(restored,field))} for name,field in [('actor','actor_params'),('normalizer','observation_normalizer'),('critic','critic_params')]}
    result={'schema':'jit_retention_zero_update_D0b_v1','observations':len(obs),'optimizer_updates':0,'hashes':hashes,
        'copy_action_max_abs_difference':float(np.abs(original-copied).max()),'export_action_max_abs_difference':float(np.abs(original-action_export).max()),
        'normalizer_only_actor_unchanged':True,'normalizer_before_sha256':pytree_sha256(source.observation_normalizer),
        'normalizer_after_sha256':pytree_sha256(updated),'normalizer_only_action_rmse_channels':np.sqrt(np.mean((drifted-original)**2,axis=0)).tolist(),
        'normalizer_only_action_max_abs_difference':float(np.abs(drifted-original).max()),
        'fifo_shift_max_abs_error':fifo_max,'observation_continuity_max_abs_error':continuity_max,
        'success_before_25_recovery_ticks':counter_mismatch,'nonzero_initial_recovery_counters':sum(x!=0 for x in initial_counters),
        'schedule_audit':'historical student.py lambda_demo uses normalizer.count; frozen-normalizer pilot requires explicit completed-transition clock',
        'actor_gradients':'NO_OPTIMIZER_IN_D0; not measured. Existing joint_loss_metrics separately records PPO/demo/keep norms and cosines during training',
        'normalizer_drift_interpretation':'fixed Actor observation-coordinate change only; not evidence it alone explains R71 retention loss'}
    if any(x['source']!=x['copy'] for x in hashes.values()) or result['copy_action_max_abs_difference']!=0 or result['export_action_max_abs_difference']!=0:raise ValueError('zero-update copy/export changed')
    if fifo_max>1e-6 or continuity_max>1e-6 or counter_mismatch:raise ValueError('history or success-counter audit failure')
    write(root/'D0b_audit.json',result);return result


def liftoff_tick(front,rear):
    ground=np.flatnonzero((front<=0)|(rear<=0))
    if not len(ground):return None
    airborne=(front>.01)&(rear>.01)
    for t in range(int(ground[0])+1,len(front)-2):
        if airborne[t:t+3].all():return t
    return None


def rows_for_batch(plan,b):
    root=Path(plan['output']);rows=[];qi=plan['indices']['root_qpos'];vi=plan['indices']['root_dof']
    with CachedArchive(root/b['name']/'prefixes.npz') as f:
        for lane,c in enumerate(b['cases']):
            mask=f['prefix_mask'][:,lane];n=int(mask.sum());end=n-1
            def first(field):
                ids=np.flatnonzero(f[field][:,lane]&mask);return int(ids[0]) if len(ids) else None
            liftoff=liftoff_tick(f['front_wheel_clearance'][:n,lane],f['rear_wheel_clearance'][:n,lane])
            def phase(t):
                if t is None:return None
                if bool(f['valid_contact_seen_before'][t,lane]):return 'post_contact'
                if liftoff is not None and t>=liftoff:return 'airborne_before_valid_contact'
                return 'pre_liftoff'
            roll=first('roll_limit');contact=first('prohibited_contact');fail=first('physical_failure');valid=first('first_valid_contact')
            # Plot diagnostic liftoff at control resolution; no endpoint change.
            # Initial3cm suspension is not called liftoff: ground seen then3 airborne frames.
            rewards={k[7:]:float(f[k][:n,lane].sum()) for k in f.files if k.startswith('metric/reward/')}
            row={'model':b['model'],'group':c['group'],'case':c['case'],'repeat':c['repeat'],'onset':c['onset'],'ancestor':c['ancestor'],
                'role':'DEV','request_sha256':sha(read(b['spec'])['frozen_request_table']['path']),
                'success':bool(np.any(f['success'][:,lane]&mask)),'invalid':False,'steps':n,'end_code':int(f['end_code'][end,lane]),
                'initial_x':c['qpos'][qi],'liftoff_control_tick':liftoff,'first_valid_contact_tick':valid,
                'first_roll_limit_tick':roll,'first_roll_limit_phase':phase(roll),'first_prohibited_contact_tick':contact,
                'first_prohibited_contact_phase':phase(contact),'first_physical_failure_tick':fail,'first_physical_failure_phase':phase(fail),
                'peak_abs_roll_rad':float(np.max(np.abs(f['roll'][:n,lane]))),'peak_abs_roll_rate_rad_s':float(np.max(np.abs(f['roll_rate'][:n,lane]))),
                'roll_at_first_valid_contact_rad':None if valid is None else float(f['roll'][valid,lane]),
                'roll_rate_at_first_valid_contact_rad_s':None if valid is None else float(f['roll_rate'][valid,lane]),
                'return':float(f['reward'][:n,lane].sum()),'terminal_reward':float(f['reward'][end,lane]),
                'clipped_channel_fraction':float(np.mean(f['action_clipped'][:n,lane])),
                'pulse_clipped_channel_fraction':float(np.mean(f['action_clipped'][:n,lane][f['mask'][:n,lane]])) if f['mask'][:n,lane].any() else None,
                'reward_components':rewards,'batch':b['name'],'lane':lane}
            rows.append(row)
    return rows


def report(plan_path,output=None):
    plan=read(plan_path);rawroot=Path(plan['output']);root=Path(output or rawroot);rows=[]
    root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():raise FileExistsError('preserve existing report')
    amendment=rawroot/'repeat_amendment.json'
    if amendment.exists():plan={**plan,'batches':plan['batches']+read(amendment)['batches']}
    for b in plan['batches']:
        if (rawroot/b['name']/'verification.json').exists():rows.extend(rows_for_batch(plan,b))
    if not rows:raise ValueError('no completed evidence; report cannot launch evaluation')
    write(root/'episodes.json',rows)
    scalar=[k for k in rows[0] if k!='reward_components']
    with (root/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=scalar);w.writeheader();w.writerows({k:r[k] for k in scalar} for r in rows)
    main=[r for r in rows if r['repeat']==0];baseline={r['case']:r for r in main if r['model']=='pi0'}
    four={};strata=[]
    for model in plan['models']:
        four[model]={}
        for group in 'ABCD':
            selected=[r for r in main if r['model']==model and r['group']==group]
            gains=sum(not baseline[r['case']]['success'] and r['success'] for r in selected)
            losses=sum(baseline[r['case']]['success'] and not r['success'] for r in selected)
            four[model][group]={'successes':sum(r['success'] for r in selected),'n':len(selected),'N01_new_vs_pi0':gains,'N10_lost_vs_pi0':losses,'net':gains-losses,
                'failed_before_liftoff':sum(not r['success'] and r['first_physical_failure_phase']=='pre_liftoff' for r in selected),
                'failed_airborne':sum(not r['success'] and r['first_physical_failure_phase']=='airborne_before_valid_contact' for r in selected),
                'failed_after_contact':sum(not r['success'] and r['first_physical_failure_phase']=='post_contact' for r in selected)}
            for onset in (0,5,10,15):
                ss=[r for r in selected if r['onset']==onset]
                if ss:strata.append({'model':model,'group':group,'onset':onset,'n':len(ss),'successes':sum(r['success'] for r in ss)})
    repeats=[]
    for model in ('pi0','R71'):
        cases=sorted(set(r['case'] for r in rows if r['model']==model and r['repeat']>0))
        for case in cases:
            rr=[r for r in rows if r['model']==model and r['case']==case and r['repeat']>0]
            repeats.append({'model':model,'case':case,'labels':[r['success'] for r in rr],'flipped':len(set(r['success'] for r in rr))>1,
                'first_failure_ticks':[r['first_physical_failure_tick'] for r in rr]})
    case_specs={c['case']:c for b in plan['batches'] if b['kind']=='main' for c in b['cases']}
    physical_repeat_keys={}
    import hashlib,json
    for r in repeats:
        c=case_specs[r['case']]
        key=hashlib.sha256(json.dumps([c['qpos'],c['qvel'],c['request'],c['onset']]).encode()).hexdigest()
        physical_repeat_keys.setdefault(r['model'],set()).add(key)
    summary={'four_cells':four,'onset_strata':strata,'numerical_repeats':repeats,
        'repeat_distinct_physical_conditions':{m:len(k) for m,k in physical_repeat_keys.items()},
        'repeat_case_id_count':{m:sum(r['model']==m for r in repeats) for m in ('pi0','R71')},
        'adaptive_E_historical':plan['adaptive_E_historical'],'teacher_student_same_root':'NOT_EXECUTED: D1 requires independent TRAIN roots and pi0 suffixes; D0 four-cell gains are full-task paired observations, not teacher conversion',
        'physical_charged_completed':sum(read(rawroot/b['name']/'status.json')['charged_interactions'] for b in plan['batches'] if (rawroot/b['name']/'verification.json').exists()),
        'scored_main':len(main),'repeat_episodes':sum(r['repeat']>0 for r in rows),'training_updates':0}
    complete=all((rawroot/b['name']/'verification.json').exists() for b in plan['batches'])
    summary['complete']=complete
    if complete:summary['D0b']=zero_update_audit(plan,root)
    write(root/'summary.json',summary)
    plot(plan,main,root)
    text='# D0 / D0b retention diagnostic\n\n![XY actual trajectories](xy_four_cells.png)\n\n'
    text+='Stage D0 only; zero policy updates. Baseline is bridge transition_0, not R73. All five same250-world layout; uniform exogenous requests, DEV only. A repeats one unique nominal state. C/D share request sequences; B/D share initial states. Effective action clipping and trajectories may differ. Historical adaptive E remains a separate negative stress test.\n\n'
    text+='|Model|A nominal/no pulse|B random/no pulse|C nominal/pulse|D random/pulse|\n|---|---|---|---|---|\n'
    for m,g in four.items():text+='|'+m+'|'+ '|'.join(f"{g[x]['successes']}/{g[x]['n']} (new {g[x]['N01_new_vs_pi0']}, lost {g[x]['N10_lost_vs_pi0']})" for x in 'ABCD')+'|\n'
    text+=f"\nRepeat physical conditions: {summary['repeat_distinct_physical_conditions']} (original repeated nominal IDs and7-condition supplement preserved in repeat_amendment.json).\n\nCharged physical transitions (padding included): {summary['physical_charged_completed']}; main episodes {len(main)}, repeats {summary['repeat_episodes']}; 4h /1.5M cap. Complete={complete}. No E/G/student optimization or policy publication.\n\n"
    if complete:
        a=summary['D0b'];text+=f"D0b copy and exported inference max action difference: {a['copy_action_max_abs_difference']}/{a['export_action_max_abs_difference']}. Actor/normalizer/critic copied byte-tree identities agree. FIFO shift error {a['fifo_shift_max_abs_error']}, continuity {a['observation_continuity_max_abs_error']}; success before25 recovery ticks {a['success_before_25_recovery_ticks']}. Frozen Actor, normalizer-only update action RMSE by channel: {a['normalizer_only_action_rmse_channels']}. This diagnoses coordinate sensitivity; it does not isolate the cause of historical forgetting.\n\n"
    text+='Same-root pi0/teacher/student conversion: **NOT EXECUTED**, no D1 TRAIN roots or teacher search in this stage. Training Actor gradient norms/KL: **NOT MEASURED**; no PPO batch or optimization in D0. Reward components and first roll/prohibited-contact/failure phases are in episodes.json/CSV. Missing teacher labels remain UNKNOWN, never new capability.\n\n'
    text+='Decision: keep pi0 fixed and R74 stopped. Do not launch D1/D2 automatically. Prepare an independent TRAIN-root qualification plan only after inspecting D0 and numerical repeat limits. A future frozen-normalizer student must use explicit completed-transition scheduling, same pi0 initialization and separate learner/best/published pointers. No reward or success-standard change is supported by loss/MSE alone.\n\n'
    text+='Evidence: [input plan](../plan.json), [identity audit](../audit.json), [full results](summary.json), [episodes](episodes.csv), [reward components and first anomalies](episodes.json), [D0b](D0b_audit.json), [raw fixed observations](D0b_fixed_observations.npz). Liftoff is a control-resolution plotting diagnostic; substep event mechanisms remain unobserved. DEV samples share paired ancestors, nominal repetitions are not independent starts; no TEST or training-seed claim.\n'
    (root/'D0_diagnostic_report.md').write_text(text);(root/'INDEX.md').write_text(text)
    if complete:
        from .retention_repair_analysis import extend
        extend(plan_path,root)
    return root/'D0_diagnostic_report.md'


def plot(plan,rows,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rawroot=Path(plan['output'])
    fig,axes=plt.subplots(1,4,figsize=(18,5));qi=plan['indices']['root_qpos']
    colors=dict(zip(plan['models'],('black','tab:blue','tab:orange','tab:green','tab:red')))
    for b in plan['batches']:
        if b['kind']!='main' or not (rawroot/b['name']/'verification.json').exists():continue
        with CachedArchive(rawroot/b['name']/'prefixes.npz') as f:
            for group,ax in zip('ABCD',axes):
                group_rows=[r for r in rows if r['batch']==b['name'] and r['group']==group]
                # deterministic representative subset; full rows and tapes retained
                for j,r in enumerate(group_rows[::8]):
                    q=f['qpos'][:r['steps'],r['lane']];ax.plot(q[:,qi],q[:,qi+1],c=colors[b['model']],alpha=.3,lw=.7,label=b['model'] if j==0 else None)
                    if not r['success']:ax.scatter(q[-1,qi],q[-1,qi+1],s=4,c=colors[b['model']],marker='x')
    for group,ax in zip('ABCD',axes):
        ax.set(title=group,xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        handles,labels=ax.get_legend_handles_labels();unique=dict(zip(labels,handles));ax.legend(unique.values(),unique.keys(),fontsize=7)
    fig.suptitle('D0 paired DEV actual XY; sampled every8th lane, failures stop at actual endpoint\nNo prescribed geometric path; original jump task unchanged')
    fig.tight_layout();fig.savefig(root/'xy_four_cells.png',dpi=180);plt.close(fig)
