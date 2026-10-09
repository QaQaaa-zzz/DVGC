"""Rebuild finite fast-path evidence tables/figures from recorded engine traces."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_response import read_trace, paired_samples, validate_trace
from fast_path_analysis import load, metrics, aligned_observations
from fast_path_probe import dump
from policy_runtime import Actor

ROOT=Path(__file__).resolve().parent
P=ROOT/'results/sim2sim_fast_path_20260928'
A=P/'analysis';A.mkdir(exist_ok=True)
BASE='rear0.005_steer2.5';CAND='rear0.005_steer10.0'


def table(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def actuator():
    rows=[];endpoints=[];paths={}
    for panel in ['suspended','loaded']:
        protocol=json.loads((P/f'actuator_{panel}.json').read_text())
        case_map={x['name']:x for x in protocol['cases']}
        for group in ['', '_refine', '_validation', '_exact']:
            protocol=json.loads((P/f'actuator{"_exact" if group=="_exact" else ""}_{panel}.json').read_text())
            case_map={x['name']:x for x in protocol['cases']}
            folder=P/f'physx_actuator{group}_{panel}'
            if not folder.exists():continue
            for candidate in sorted(x for x in folder.iterdir() if x.is_dir()):
                for case in sorted(x for x in candidate.iterdir() if x.is_dir()):
                    if not (case/'traces.npz').exists():continue
                    source_folder=P/f'mjx_actuator{"_exact" if group=="_exact" else ""}_{panel}'/case.name
                    s=read_trace(source_folder);t=read_trace(case)
                    validate_trace(s,protocol,case_map[case.name]);validate_trace(t,protocol,case_map[case.name])
                    si,ti=paired_samples(s['time_s'],t['time_s'],.02)
                    np.testing.assert_allclose(s['ctrl'][si],t['ctrl'][ti],atol=2e-6)
                    joint='rearwheel_joint' if case.name.startswith('rear') else 'steering_joint'
                    ji=s['joint_names'].tolist().index(joint)
                    actuator='cmd_rearwheel_f' if joint=='rearwheel_joint' else 'cmd_steering_v'
                    ci=s['actuator_names'].tolist().index(actuator)
                    qe=t['joint_q'][ti,ji]-s['joint_q'][si,ji];ve=t['joint_qd'][ti,ji]-s['joint_qd'][si,ji]
                    qs,vs=(2.4,12.) if ji==0 else (.1,2.)
                    role='exact_duration_validation' if group=='_exact' else ('validation' if group=='_validation' else 'calibration')
                    rows.append(dict(panel=panel,candidate=candidate.name,case=case.name,role=role,joint=joint,
                        loss=float(np.mean((qe/qs)**2+(ve/vs)**2)),q_rmse_rad=float(np.sqrt(np.mean(qe**2))),qd_rmse_rad_s=float(np.sqrt(np.mean(ve**2))),source=str(source_folder),target=str(case)))
                    paths[(panel,candidate.name,case.name)]=(s,t,ji,ci)
                    for k,(i,j) in enumerate(zip(si,ti)):
                        endpoints.append(dict(panel=panel,candidate=candidate.name,case=case.name,time_s=round(float(s['time_s'][i]),6),source_q=float(s['joint_q'][i,ji]),target_q=float(t['joint_q'][j,ji]),source_qd=float(s['joint_qd'][i,ji]),target_qd=float(t['joint_qd'][j,ji]),physical_target=float(t['ctrl'][j,ci]),source_effort_native_stage=float(s['actuator_force_native'][i,ci]),target_actual_motor_effort_available=False,target_projected_joint_effort=float(t['projected_joint_effort'][j,ji])))
    for r in rows:
        b=next(x for x in rows if x['candidate']==BASE and x['panel']==r['panel'] and x['case']==r['case'])
        r['q_guard']=r['q_rmse_rad']<=1.1*b['q_rmse_rad']+1e-4
        r['qd_guard']=r['qd_rmse_rad_s']<=1.1*b['qd_rmse_rad_s']+1e-3
    table(A/'actuator_metrics.csv',rows);table(A/'actuator_20ms_endpoints.csv',endpoints);dump(A/'actuator_metrics.json',rows)
    for panel in ['suspended','loaded']:
        for case in ['rear_12','steer_0p1','rear_6','rear_24','steer_neg0p08','steer_0p2','rear_12_t0','steer_0p1_t0']:
            if (panel,BASE,case) not in paths:continue
            s,t,ji,ci=paths[(panel,BASE,case)]
            shown=[('MJX-Warp',s,'black')]
            for cand,color in [(BASE,'#4477AA'),(CAND,'#CC6677'),('rear5.0_steer2.5','#228833')]:
                if (panel,cand,case) in paths:shown.append((cand,paths[(panel,cand,case)][1],color))
            fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
            for label,tr,color in shown:
                time=tr['time_s'];ax[0,0].step(time,tr['ctrl'][:,ci],where='pre',label=label,color=color)
                ax[0,1].plot(time,tr['joint_q'][:,ji],label=label,color=color)
                ax[1,0].plot(time,tr['joint_qd'][:,ji],label=label,color=color)
                if label=='MJX-Warp':ax[1,1].plot(time,tr['actuator_force_native'][:,ci],label='MJX native RK4-stage motor effort',color=color)
                else:ax[1,1].plot(time,tr['projected_joint_effort'][:,ji],label=label+' projected joint effort (NOT motor)',color=color,ls=':')
            for axis,title,unit in zip(ax.flat,['Applied target','Joint position','Endpoint joint velocity','Non-equivalent effort readbacks'],['rad/s' if ji==0 else 'rad','rad','rad/s','Nm']):
                axis.set(title=title,xlabel='Time (s)',ylabel=unit);axis.grid(alpha=.2)
                for tick in [.02,.1,.2]:axis.axvline(tick,color='.7',lw=.7,ls='--')
            ax[0,0].legend(fontsize=7);ax[1,1].legend(fontsize=6)
            fig.suptitle(f'{panel} | {case} | real-engine same input; PhysX native motor effort unavailable')
            fig.savefig(A/f'{panel}_{case}.png',dpi=140);fig.savefig(A/f'{panel}_{case}.pdf');plt.close(fig)
    return rows


def observation():
    actor=Actor(ROOT/'policy/actor.npz');configs=json.loads((P/'observation_alignment/candidates.json').read_text())
    configs['deterministic_20ms']=json.loads((P/'observation_alignment/deterministic_adapter.json').read_text())
    sources=json.loads((P/'sources.json').read_text());rows=[];action_rows=[]
    for name,cfg in configs.items():
        for case,path in sources.items():
            s=load(path);t=load(P/'target_baseline_retry'/case);obs=aligned_observations(t,cfg);n=min(21,len(obs),len(s['tick']))
            before=t['observation'][1:n]-s['observation'][1:n];after=obs[1:n]-s['observation'][1:n]
            for k in range(76):rows.append(dict(adapter=name,case=case,channel=k,rmse_before=float(np.sqrt(np.mean(before[:,k]**2))),rmse_after=float(np.sqrt(np.mean(after[:,k]**2)))))
            ae=actor(obs[1:n])-s['action'][1:n];ab=t['action'][1:n]-s['action'][1:n]
            action_rows.append(dict(adapter=name,case=case,action_rmse_before=float(np.sqrt(np.mean(ab**2))),action_rmse_after=float(np.sqrt(np.mean(ae**2))),imu_std_rmse=float(np.sqrt(np.mean((after[:,53:59]/actor.p['std'][53:59])**2)))))
    table(A/'observation_channels.csv',rows);table(A/'observation_actions.csv',action_rows)
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    names=list(configs);colors=['#888888','#CC6677','#AA4499','#4477AA']
    for name,color in zip(names,colors):
        ch=[np.mean([r['rmse_after'] for r in rows if r['adapter']==name and r['channel']==k and r['case'] in ['vx_plus','x_plus']]) for k in range(76)]
        ax[0].plot(ch,label=name,color=color)
    ax[0].set(title='Validation prefix: raw per-channel RMSE (mixed units)',xlabel='Observation index',ylabel='RMSE');ax[0].legend(fontsize=7)
    vals=[np.mean([r['action_rmse_after'] for r in action_rows if r['adapter']==name and r['case'] in ['vx_plus','x_plus']]) for name in names]
    ax[1].bar(names,vals,color=colors);ax[1].set(title='Offline frozen Actor action RMSE',ylabel='Normalized action');ax[1].tick_params(axis='x',rotation=20)
    fig.suptitle('Paired ticks 1..20 (0.4 s); offline fit is not closed-loop success')
    fig.savefig(A/'observation.png',dpi=150);fig.savefig(A/'observation.pdf');plt.close(fig)
    return action_rows


def closed_loop():
    sources=json.loads((P/'sources.json').read_text());rows=[];audit={}
    groups={'Source':sources,'Baseline':P/'target_baseline_retry','Obs affine (rejected)':P/'target_obs_only','Obs only':P/'target_obs_physical'}
    for label,folder in groups.items():
        for case in sources:
            path=Path(folder[case]) if isinstance(folder,dict) else folder/case
            rows.append(dict(group=label,case=case,**metrics(load(path)),trace=str(path/'trace.npz')))
    contacts=[]
    for mu in [.5,.75,1.,1.5,2.,3.,5.]:
        for case in ['natural','vx_minus']:
            folder=P/f'contact_mu_{mu}'/case
            runtime=json.loads((folder/'runtime.json').read_text());backend=np.asarray(runtime['wheel_material_properties_physx_backend'])
            np.testing.assert_allclose(backend[:,:,:2],mu,atol=1e-6)
            for pair,values in runtime['effective_contact_pairs'].items():
                assert values['combine']=='min' and values['static_friction']==mu and values['dynamic_friction']==mu
            audit[f'{mu}/{case}']={'wheel_backend':runtime['wheel_material_properties_physx_backend'],'pairs':runtime['effective_contact_pairs'],'runtime':str(folder/'runtime.json')}
            contacts.append(dict(mu=mu,case=case,**metrics(load(folder)),trace=str(folder/'trace.npz')))
    table(A/'contact_screening.csv',contacts);dump(A/'contact_runtime_audit.json',audit)
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for case,style in [('natural','o-'),('vx_minus','s--')]:
        rr=[r for r in contacts if r['case']==case];x=[r['mu'] for r in rr]
        for axis,key,title in zip(ax,['task_success','apex_height_m','survival_time_s'],['Apex success (0/1)','Maximum root height (m)','True terminal time (s)']):
            axis.plot(x,[r[key] for r in rr],style,label=case);axis.set(xlabel='Scalar friction mu',title=title);axis.grid(alpha=.2)
    ax[0].legend();fig.suptitle('Frozen Actor; actual wheel/floor/step material audit; two engineering ground states')
    fig.savefig(A/'contact.png',dpi=150);fig.savefig(A/'contact.pdf');plt.close(fig)
    ablations={'Baseline':'target_baseline_retry','Obs only':'target_obs_physical','Actuator only':'ablation_actuator','Contact only':'contact_mu_2.0','Obs + Actuator':'ablation_obs_actuator','Obs + Contact':'ablation_obs_contact','Actuator + Contact':'ablation_actuator_contact','Full alignment':'ablation_full'}
    ab=[]
    for label,folder in ablations.items():
        for case in ['natural','vx_minus']:
            path=P/folder/case
            if not (path/'summary.json').exists():continue
            ab.append(dict(group=label,case=case,**metrics(load(path)),trace=str(path/'trace.npz')))
    table(A/'ablation.csv',ab);dump(A/'ablation.json',ab)
    if (P/'final_validation/status.json').exists():
        for case in sources:
            folder=P/'final_validation'/case
            if (folder/'summary.json').exists():rows.append(dict(group='Final frozen repeat',case=case,**metrics(load(folder)),trace=str(folder/'trace.npz')))
    table(A/'closed_loop_metrics.csv',rows);dump(A/'closed_loop_metrics.json',rows)
    for case in sources:
        fig,ax=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for label,folder,color in [('MJX-Warp',Path(sources[case]),'black'),('PhysX baseline',P/'target_baseline_retry'/case,'#CC6677'),('PhysX 20ms acc',P/'target_obs_physical'/case,'#4477AA')]:
            t=load(folder);time=t['time_s'];ax[0,0].plot(t['qpos'][:,0],t['qpos'][:,1],color=color,label=label)
            for axis,val in [(ax[0,1],t['qpos'][:,2]),(ax[0,2],t['rpy'][:,0]),(ax[1,0],t['rpy'][:,2]),(ax[1,1],t['qvel'][:,0]),(ax[1,2],t['wheel_kinematics'][:,0,5])]:axis.plot(time,val,color=color,label=label)
            apex=np.flatnonzero(t['event'][:,5])
            if len(apex):ax[0,1].scatter(time[apex[0]],t['qpos'][apex[0],2],color=color,s=30)
        for axis,title,ylabel in zip(ax.flat,['XY trajectory','Height (dots: source event apex)','Roll','Yaw','World forward velocity','Front-wheel bottom lateral velocity'],['Y (m)','Z (m)','rad','rad','m/s','m/s (geometric proxy)']):
            axis.set(title=title,ylabel=ylabel,xlabel='X (m)' if axis is ax[0,0] else 'Time (s)');axis.grid(alpha=.2)
        ax[0,0].legend(fontsize=7);fig.suptitle(f'{case} | ground initial state | frozen transition_4988928 | ends at actual failure')
        fig.savefig(A/f'closed_loop_{case}.png',dpi=150);fig.savefig(A/f'closed_loop_{case}.pdf');plt.close(fig)
    return rows,ab,contacts


def csv_traces():
    count=0
    for f in P.rglob('trace.npz'):
        with np.load(f) as data:
            if 'observation' not in data:continue
            arrays=[];names=[];n=len(data['tick'])
            for key in data.files:
                v=data[key]
                if v.shape[0]!=n or v.dtype.kind not in 'bifu':continue
                v=v.reshape(n,-1);arrays.append(v);names.extend([key] if v.shape[1]==1 else [f'{key}_{i}' for i in range(v.shape[1])])
            np.savetxt(f.with_suffix('.csv'),np.column_stack(arrays),delimiter=',',header=','.join(names),comments='');count+=1
    return count


if __name__=='__main__':
    ar=actuator();ob=observation();cl,ab,co=closed_loop();count=csv_traces()
    dump(A/'build_status.json',dict(actuator_cases=len(ar),observation_evaluations=len(ob),closed_loop_rows=len(cl),ablation_rows=len(ab),contact_rows=len(co),full_trace_csv_count=count))
    print(json.dumps(dict(actuator_cases=len(ar),ablation_rows=len(ab),csv_traces=count)))
