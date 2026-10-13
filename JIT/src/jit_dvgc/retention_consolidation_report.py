"""Idempotent figures derived only from saved receipts; never runs simulation."""
from pathlib import Path
import json
import numpy as np
from .generative_bridge.protocol import atomic_json


def generate_figures(plan, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(plan['output']);dest=root/'figures';dest.mkdir(exist_ok=True)
    files=[];gaps=[]
    def save(fig,name):
        path=dest/(name+'.png');fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)
        files.append(str(path.relative_to(root)))
    # Trajectory comes first in the index when its coordinate identity is known.
    panel=plan['config'].get('evaluation_panel',{})
    if isinstance(panel,dict) and 'path' in panel:
        try:panel=json.loads(Path(panel['path']).read_text())
        except (OSError,ValueError):panel={};gaps.append('locked evaluation panel unavailable')
    def saved_json(path):
        try:return json.loads(Path(path).read_text())
        except (OSError,ValueError):gaps.append('saved receipt unavailable: '+str(path));return {}
    trajectory_count=0
    for case in panel.get('cases',[]):
        address=case.get('root_qpos_address')
        if type(address) is not int or address<0:
            gaps.append('trajectory coordinate schema unavailable: '+case['case_id']);continue
        fig,ax=plt.subplots();drawn=False
        for arm in ('P0','P1','P2'):
            tag='P0'
            if arm!='P0':
                selection=report.get('selection',{}).get(arm,{})
                selected=selection.get('stage_candidate')
                if not isinstance(selected,str):continue
                matching=[a for a in selection.get('audits',[]) if a['checkpoint']==selected]
                if not matching:continue
                tag=f"{arm}_{matching[0]['attempt']:04d}"
            receipt=root/(tag+'_evaluation.json')
            if not receipt.is_file():continue
            evidence=saved_json(receipt).get(case['case_id'],{})
            for repeat,out in enumerate(evidence.get('receipts',[])):
                out=Path(out);lane=case['lane'];trace=None
                if case['mode']=='evaluate':
                    rows=saved_json(out/'results.json')
                    if not isinstance(rows,list) or not 0<=lane<len(rows):continue
                    attempts=rows[lane].get('attempts',[])
                    if len(attempts)!=1:continue
                    if not attempts[0].get('trace') or type(attempts[0].get('trace_lane')) is not int:
                        gaps.append('trace identity unavailable: '+str(out));continue
                    trace=Path(attempts[0]['trace']);lane=attempts[0]['trace_lane'];mask_key='mask'
                else:trace=out/'prefixes.npz';mask_key='prefix_mask'
                try:
                    with np.load(trace,allow_pickle=False) as tape:
                        q=tape['qpos'];mask=np.asarray(tape[mask_key][:,lane],bool)
                        if q.ndim!=3 or q.shape[-1]<=address+2:raise ValueError('incompatible root coordinates')
                        x=q[mask,lane,address];z=q[mask,lane,address+2]
                        if not len(x) or not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('invalid trajectory')
                    ax.plot(x,z,label=f'{arm} repeat {repeat+1}',alpha=.65)
                    ax.scatter(x[-1],z[-1],s=15);drawn=True
                except (OSError,KeyError,ValueError,IndexError):gaps.append('trajectory array unavailable: '+str(trace))
        if drawn:
            ax.set(xlabel='world x (m)',ylabel='root height z (m)',title=case['case_id']+' — independent saved trajectories')
            ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=7)
            save(fig,'trajectory_'+str(trajectory_count));trajectory_count+=1
        else:plt.close(fig);gaps.append('trajectory evidence missing: '+case['case_id'])
    if trajectory_count==0:gaps.append('trajectory figures unavailable; result delivery incomplete')
    selections=report.get('selection',{})
    if any(s.get('audits') for s in selections.values()):
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        if (root/'P0_evaluation.json').is_file():axes[0].scatter([0],[0],marker='s',label='P0 fixed parent')
        for arm,s in selections.items():
            rows=[a for a in s.get('audits',[]) if 'evaluation' in a]
            if not rows:continue
            attempts=[a['attempt'] for a in rows];lost=[a['evaluation']['total_old_lost'] for a in rows]
            gained=[a['evaluation']['expansion']['gained'] for a in rows]
            axes[0].plot(lost,gained,'o-',label=arm)
            axes[1].plot(attempts,gained,'o-',label=arm+' gained')
            axes[1].plot(attempts,lost,'x--',label=arm+' old lost')
        axes[0].set(xlabel='old stable successes lost (count)',ylabel='independent DEV gains (count)')
        axes[1].set(xlabel='attempted updates',ylabel='condition count')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        save(fig,'retention_expansion')
        groups=sorted({g for s in selections.values() for a in s.get('audits',[]) for g in a.get('evaluation',{}).get('parent',{})})
        if groups:
            fig,ax=plt.subplots()
            for arm,s in selections.items():
                for group in groups:
                    rows=[a for a in s.get('audits',[]) if group in a.get('evaluation',{}).get('parent',{})]
                    if rows:ax.plot([a['attempt'] for a in rows],[a['evaluation']['parent'][group]['lost'] for a in rows],'o-',label=arm+'/'+group)
            ax.set(xlabel='attempted updates',ylabel='old stable successes lost (count)');ax.legend(fontsize=7);save(fig,'group_retention')
    arms=report.get('arms',{})
    if arms:
        fig,axes=plt.subplots(1,2,figsize=(9,4));names=sorted(arms)
        for i,arm in enumerate(names):
            a=arms[arm];attempts=a.get('attempts',0);accepted=a.get('accepted_updates',0)
            axes[0].bar(i-.15,attempts,.3,label='attempts' if i==0 else None)
            axes[0].bar(i+.15,accepted,.3,label='accepted' if i==0 else None)
            cost=report.get('costs',{}).get(arm,{}).get('wall_seconds')
            if isinstance(cost,(float,int)):axes[1].bar(i,cost)
        axes[0].set(xticks=range(len(names)),xticklabels=names,ylabel='optimizer count');axes[0].legend()
        axes[1].set(xticks=range(len(names)),xticklabels=names,ylabel='BC wall seconds (guard/evaluation included)')
        save(fig,'attempts_accepted_cost')
    update_costs={}
    for arm in ('P1','P2'):
        audit_path=root/arm/'update_audit.jsonl'
        if not audit_path.is_file():continue
        try:updates=[json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        except (OSError,ValueError):gaps.append('update telemetry unavailable: '+arm);continue
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        valid=[row for row in updates if type(row.get('attempt')) is int and type(row.get('accepted_updates')) is int]
        if valid:
            attempted=[row['attempt'] for row in valid];accepted=[row['accepted_updates'] for row in valid]
            axes[0,0].plot(attempted,accepted,'o-');axes[1,1].plot(attempted,
                [(a-b)/a if a else 0. for a,b in zip(attempted,accepted)],'o-')
        capability=[a for a in selections.get(arm,{}).get('audits',[]) if 'evaluation' in a]
        if capability:
            axes[0,1].plot([a['attempt'] for a in capability],
                          [a['evaluation']['expansion']['gained'] for a in capability],'o-',label='DEV gained')
            axes[0,1].plot([a['attempt'] for a in capability],
                          [a['evaluation']['expansion']['teacher_verified_absorption'] for a in capability],'s--',label='verified absorption')
            with_clock=[a for a in capability if type(a.get('accepted_updates')) is int]
            if with_clock:
                axes[1,0].plot([a['accepted_updates'] for a in with_clock],
                              [a['evaluation']['expansion']['gained'] for a in with_clock],'o-',label='DEV gained')
            else:gaps.append('accepted-update checkpoint clock unavailable: '+arm)
            axes[0,1].legend(fontsize=7)
        axes[0,0].set(xlabel='attempted updates',ylabel='accepted updates')
        axes[0,1].set(xlabel='attempted updates',ylabel='independent condition count')
        axes[1,0].set(xlabel='accepted updates',ylabel='independent DEV gains')
        axes[1,1].set(xlabel='attempted updates',ylabel='cumulative rejection fraction',ylim=(0,1))
        fig.suptitle(arm+' — attempted and accepted clocks')
        save(fig,arm+'_dual_clocks')
        costs={}
        for key in ('guard_seconds','proposal_seconds'):
            values=[row.get(key) for row in updates]
            if values and all(type(value) in (int,float) and np.isfinite(value) and value>=0 for value in values):
                costs[key]=float(sum(values))
            else:costs[key]='UNKNOWN'
        values=[row.get('wall_seconds') for row in updates]
        costs['aggregate_update_seconds']=(float(sum(values)) if values and all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in values) else 'UNKNOWN')
        if 'UNKNOWN' in (costs['guard_seconds'],costs['proposal_seconds']):
            gaps.append('separate guard/proposal timing unavailable: '+arm+'; aggregate time is not a decomposition')
        update_costs[arm]=costs
    result=dict(phase='partial_saved_evidence' if gaps else 'complete_saved_evidence',files=files,gaps=sorted(set(gaps)),
        environment_steps=0,update_costs=update_costs,source='saved checkpoint capability/cost receipts and original trajectory NPZ')
    atomic_json(dest/'index.json',result)
    text=['# Saved evidence figures','', 'Trajectory figures are listed first. Missing source evidence remains explicit.','']
    text.extend(f'- [{Path(path).stem}]({Path(path).name})' for path in files)
    text.extend(['','Gaps:']+[f'- {gap}' for gap in result['gaps']])
    (dest/'INDEX.md').write_text('\n'.join(text)+'\n')
    return result
