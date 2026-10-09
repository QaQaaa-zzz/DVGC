"""Channel timelines and IMU-only calibration from frozen paired traces."""
import argparse
import json
from pathlib import Path
import numpy as np
from fast_path_probe import dump
from policy_runtime import Actor
from target_observation_adapter import TargetObservationAdapter


def load(path):
    with np.load(Path(path)/'trace.npz') as a:return {k:a[k] for k in a.files}


def first(error,threshold):
    mask=np.asarray(error)>threshold
    ticks=np.flatnonzero(np.any(mask,axis=1))
    return dict(tick=int(ticks[0]),channels=np.flatnonzero(mask[ticks[0]]).tolist()) if len(ticks) else dict(tick=None,channels=[])


def aligned_observations(trace,config):
    adapter=TargetObservationAdapter(config)
    return np.stack([adapter.transform(o,qpos=q,qvel=v) for o,q,v in zip(trace['observation'],trace['qpos'],trace['qvel'])])


def metrics(trace):
    events=trace['event'];apex=np.flatnonzero(events[:,5]);zone=np.flatnonzero(events[:,1])
    proxy=trace['wheel_kinematics'];contact=proxy[:,:,3]<.002
    slip=np.abs(proxy[:,:,5]);values=slip[contact]
    return dict(task_success=bool(len(apex)),apex_time_s=float(trace['time_s'][apex[0]]) if len(apex) else None,
        survival_time_s=float(trace['time_s'][-1]),max_abs_roll=float(np.max(np.abs(trace['rpy'][:,0]))),
        max_abs_yaw=float(np.max(np.abs(trace['rpy'][:,2]))),lateral_slip_mean_m_s=float(np.mean(values)) if len(values) else None,
        lateral_slip_measure='kinematic bottom point lateral velocity during geometric clearance<2mm; contact proxy, not actual solver friction force',
        forward_velocity_mean_m_s=float(np.mean(trace['qvel'][:,0])),jump_zone_arrival_time_s=float(trace['time_s'][zone[0]]) if len(zone) else None,
        apex_height_m=float(np.max(trace['qpos'][:,2])),end_code=int(trace['end_code'][-1]),post_apex_survival_s=float(trace['time_s'][-1]-trace['time_s'][apex[0]]) if len(apex) else None)


def timelines(sources,target_root,output,actor_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=False);summary={}
    fields=json.loads((Path(actor_path).parent/'identity.json').read_text())['actor_frame_fields']
    names=[f'frame{frame}_{field}' for frame in range(3) for field in fields]+['jump_signal']
    actor=Actor(actor_path);std=actor.p['std']
    for case,source in sources.items():
        tp=Path(target_root)/case
        if not (tp/'trace.npz').exists():continue
        s=load(source);t=load(tp);n=min(len(s['tick']),len(t['tick']))
        np.testing.assert_allclose(s['qpos'][0],t['qpos'][0],atol=2e-6,rtol=0)
        np.testing.assert_allclose(s['qvel'][0],t['qvel'][0],atol=2e-6,rtol=0)
        errors={key:np.abs(t[key][:n]-s[key][:n]) for key in ['observation','action','target','qpos','qvel','omega_body','omega_world','actor_acc_body']}
        normalized=errors['observation']/std
        obsfirst=first(normalized,.01);actfirst=first(errors['action'],.001)
        state_error=np.c_[errors['qpos']/.001,errors['qvel']/.01];statefirst=first(state_error,1.)
        numerical=first(errors['observation'],1e-5)
        finite_ticks=[v for v in [obsfirst['tick'],actfirst['tick'],statefirst['tick']] if v is not None]
        earliest=min(finite_ticks) if finite_ticks else None
        tied=[]
        for group,info,labels in [('observation',obsfirst,names),('action',actfirst,[f'action_{i}' for i in range(4)]),('state',statefirst,[f'qpos_{i}' for i in range(12)]+[f'qvel_{i}' for i in range(11)])]:
            if earliest is not None and info['tick']==earliest:tied.extend(f'{group}:{labels[i]}' for i in info['channels'])
        result=dict(FIRST_DIVERGENCE_TICK=earliest,
            FIRST_DIVERGENCE_CHANNEL=tied[0] if tied else None,simultaneously_divergent_channels=tied,
            FIRST_OBSERVATION_DIVERGENCE_TICK=obsfirst['tick'],FIRST_ACTION_DIVERGENCE_TICK=actfirst['tick'],FIRST_STATE_DIVERGENCE_TICK=statefirst['tick'],
            simultaneously_divergent_observation_channels=[names[i] for i in obsfirst['channels']],numerical_observation_first=numerical,
            thresholds=dict(observation_actor_std=.01,action=.001,qpos=.001,qvel=.01,numerical=1e-5),
            first_channel_rule='first observation index among simultaneous first-tick exceedances; not a unique causal attribution',
            observation_rmse=np.sqrt(np.mean(errors['observation']**2,axis=0)),action_rmse=np.sqrt(np.mean(errors['action']**2,axis=0)),
            source=metrics(s),target=metrics(t),paired_prefix_ticks=n-1)
        summary[case]=result;dump(output/f'{case}.json',result)
        arrays=[s['tick'][:n,None],s['time_s'][:n,None]];header=['tick','time_s']
        for engine,tr in [('source',s),('target',t)]:
            for key in ['observation','action','target','qpos','qvel','omega_body','omega_world','actor_acc_body']:
                arrays.append(tr[key][:n]);header.extend([f'{engine}_{name}' for name in names] if key=='observation' else [f'{engine}_{key}_{i}' for i in range(tr[key].shape[1])])
        arrays.append(normalized);header.extend([f'normalized_error_{name}' for name in names])
        np.savetxt(output/f'{case}_channels.csv',np.column_stack(arrays),delimiter=',',header=','.join(header),comments='')
        fig,ax=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
        im=ax[0,0].imshow(np.log10(np.maximum(normalized.T,1e-5)),origin='lower',aspect='auto',extent=[0,(n-1)*.02,0,76],vmin=-3,vmax=1)
        ax[0,0].set(title='76-channel log10 standardized absolute divergence',xlabel='Time (s)',ylabel='Observation channel');fig.colorbar(im,ax=ax[0,0])
        for k,label in enumerate(['steer','rear','hip','knee']):ax[0,1].plot(s['time_s'][:n],errors['action'][:,k],label=label)
        ax[0,1].set(title='Frozen Actor absolute action divergence',xlabel='Time (s)');ax[0,1].legend()
        for engine,tr,color in [('MJX-Warp',s,'black'),('PhysX',t,'#4477AA')]:
            ax[1,0].plot(tr['qpos'][:,0],tr['qpos'][:,2],label=engine,color=color)
            ax[1,1].plot(tr['time_s'],tr['rpy'][:,0],label=engine,color=color)
            ax[2,0].plot(tr['time_s'],tr['observation'][:,54],label=engine,color=color)
            ax[2,1].plot(tr['time_s'],tr['observation'][:,56],label=engine,color=color)
        for axis,title,ylabel in [(ax[1,0],'Actual X/Z trajectory','height (m)'),(ax[1,1],'Roll until real terminal','rad'),(ax[2,0],'Latest actor gyro Y','rad/s'),(ax[2,1],'Latest actor acceleration X','m/s2')]:
            axis.set(title=title,ylabel=ylabel,xlabel='X (m)' if axis is ax[1,0] else 'Time (s)');axis.legend();axis.grid(alpha=.2)
        fig.suptitle(f'{case} | frozen transition_4988928 | actual endpoints, no padded continuation')
        fig.savefig(output/f'{case}.png',dpi=150);fig.savefig(output/f'{case}.pdf');plt.close(fig)
    dump(output/'summary.json',summary)
    return summary


def calibrate(sources,target_root,output,actor_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);actor=Actor(actor_path);std=actor.p['std'][53:59]
    data={name:(load(path),load(Path(target_root)/name)) for name,path in sources.items()}
    configs={'identity':dict(a0=[1.]*6,a1=[0.]*6,b=[0.]*6)}
    for model in ['current_affine','short_history']:
        xs=[];ys=[]
        for case in ['natural','vx_minus','x_minus']:
            s,t=data[case];n=min(21,len(s['tick']),len(t['tick']));x=t['observation'][1:n,53:59]/std;prev=t['observation'][:n-1,53:59]/std
            xs.append(np.stack([x,prev if model=='short_history' else np.zeros_like(prev),np.ones_like(x)],axis=-1));ys.append(s['observation'][1:n,53:59]/std)
        x=np.concatenate(xs);y=np.concatenate(ys);coef=[]
        for channel in range(6):
            a=x[:,channel];b=y[:,channel];prior=np.array([1.,0.,0.]);coef.append(np.linalg.solve(a.T@a+.01*np.eye(3),a.T@b+.01*prior))
        coef=np.asarray(coef);configs[model]=dict(a0=coef[:,0].tolist(),a1=coef[:,1].tolist(),b=(coef[:,2]*std).tolist())
    scores={};channel_rows=[]
    for name,config in configs.items():
        errors=[]
        for case,(s,t) in data.items():
            obs=aligned_observations(t,config);n=min(len(obs),len(s['tick']),21)
            e=obs[1:n]-s['observation'][1:n];before=t['observation'][1:n]-s['observation'][1:n]
            action=actor(obs[1:n]);source_action=s['action'][1:n]
            dump(output/f'{name}__{case}.json',dict(observation_rmse_before=np.sqrt(np.mean(before**2,axis=0)),observation_rmse_after=np.sqrt(np.mean(e**2,axis=0)),
                action_rmse_before=np.sqrt(np.mean((t['action'][1:n]-source_action)**2,axis=0)),action_rmse_after=np.sqrt(np.mean((action-source_action)**2,axis=0)),
                ticks=list(range(1,n)),role='validation' if case in ['vx_plus','x_plus'] else 'calibration'))
            for channel in range(76):channel_rows.append([name,case,channel,float(np.sqrt(np.mean(before[:,channel]**2))),float(np.sqrt(np.mean(e[:,channel]**2)))])
            if case in ['vx_plus','x_plus']:errors.append(e[:,53:59]/std)
        scores[name]=float(np.sqrt(np.mean(np.concatenate(errors)**2)))
    selected='identity'
    for name in ['current_affine','short_history']:
        if scores[name]<.95*scores[selected]:selected=name
    dump(output/'candidates.json',configs);dump(output/'selection.json',dict(selected=selected,validation_imu_standardized_rmse=scores,
        selection_uses_reward=False,selection_uses_task_success=False,calibration_prefix_s=.4,untouched_fields='all except six IMU values in each valid frame'))
    dump(output/'selected_adapter.json',configs[selected])
    import csv
    with (output/'all_observation_channels.csv').open('w') as file:
        w=csv.writer(file);w.writerow(['model','case','channel','rmse_before','rmse_after']);w.writerows(channel_rows)
    print(json.dumps(dict(selected=selected,scores=scores)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',required=True);p.add_argument('--target',required=True);p.add_argument('--output',required=True)
    p.add_argument('--actor',default='policy/actor.npz');p.add_argument('--calibrate',action='store_true');a=p.parse_args()
    fn=calibrate if a.calibrate else timelines;fn(json.loads(Path(a.sources).read_text()),a.target,a.output,a.actor)
