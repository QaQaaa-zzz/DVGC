"""Stream verified rollout batches into an all-trajectory comparison artifact."""
from collections import Counter
import csv
import os
from pathlib import Path

import numpy as np

from ..rsi_comparison import read, write, file_sha
from ..batched_pulse_comparison import load_tape
from .liftoff_alignment import liftoff_index, align_xz


def report(output, destination=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    output=Path(output);spec=read(output/'spec.json');receipts=read(output/'receipts.json')
    if len(receipts)!=len(spec['batches']):raise ValueError('incomplete comparison batches')
    destination=Path(destination) if destination else output/'comparison'
    destination.mkdir(exist_ok=False)
    font=Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    methods=spec['methods'];q0=spec['root_qpos_address'];colors=['#2369a1','#d15a28','#32845d','#8556a8']
    world_x_edges=np.linspace(2.4,5.5,156);aligned_x_edges=np.linspace(-.35,5.5,196)
    z_edges=np.linspace(.05,1.30,151)
    density={m['key']:{'world':np.zeros((len(world_x_edges)-1,len(z_edges)-1)),
                       'aligned':np.zeros((len(aligned_x_edges)-1,len(z_edges)-1))} for m in methods}
    ends={m['key']:[] for m in methods};peaks={m['key']:[] for m in methods}
    beyond={m['key']:0 for m in methods};excluded=Counter();counts={m['key']:Counter() for m in methods}
    seen={m['key']:set() for m in methods};outcomes={m['key']:{} for m in methods};hashes={};charged=active=0
    with (destination/'episodes.csv').open('x') as stream:
        writer=None
        for expected,receipt in zip(spec['batches'],receipts):
            if receipt['batch']!=expected:raise ValueError('batch plan drift')
            path=Path(receipt['verification'])
            if file_sha(path)!=receipt['verification_sha256']:raise ValueError('batch verification drift')
            verified=read(path)
            charged+=verified['charged_interactions'];active+=verified['active_interactions']
            hashes.update(verified['trace_sha256'])
            for row in verified['rows']:
                key=row['method'];episode=row['episode']
                if episode in seen[key]:raise ValueError('duplicate episode')
                seen[key].add(episode);outcomes[key][episode]=row['success']
                counts[key]['episodes']+=1;counts[key]['successes']+=int(row['success'])
                counts[key][row['terminal_reason']]+=1
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
            directory=path.parent
            for method in methods:
                key=method['key'];trace=directory/key/'prefixes.npz'
                if file_sha(trace)!=hashes[str(trace)]:raise ValueError('trajectory drift')
                tape=load_tape(trace)
                for lane in range(expected['count']):
                    ticks=np.flatnonzero(tape['prefix_mask'][:,lane]);points=tape['qpos'][ticks,lane][:,[q0,q0+2]]
                    good=bool(tape['success'][ticks[-1],lane]) and not bool(tape['physical_failure'][ticks[-1],lane])
                    peaks[key].append(float(points[:,1].max()))
                    beyond[key]+=int(np.any(points[:,0]>5.5))
                    density[key]['world']+=np.histogram2d(points[:,0],points[:,1],bins=(world_x_edges,z_edges))[0]/len(points)
                    if not good:ends[key].append(points[-1])
                    k=liftoff_index(tape['front_wheel_clearance'][ticks,lane],tape['rear_wheel_clearance'][ticks,lane])
                    if k is None:excluded[key]+=1
                    else:
                        aligned=align_xz(points,points[k,0])
                        density[key]['aligned']+=np.histogram2d(aligned[:,0],aligned[:,1],bins=(aligned_x_edges,z_edges))[0]/len(points)
    for method in methods:
        key=method['key']
        if seen[key]!=set(range(spec['episodes'])):raise ValueError('missing method episodes')
        hashes[method['nominal_trace']]=file_sha(method['nominal_trace'])
    _render_readable(destination,methods,colors,counts,density,ends,peaks,beyond,excluded,
                     world_x_edges,aligned_x_edges,z_edges,spec,q0)
    pairs={}
    for i,a in enumerate(methods):
        for b in methods[i+1:]:
            pairs[a['key']+'__'+b['key']]=dict(Counter(
                f'{int(outcomes[a["key"]][e])}{int(outcomes[b["key"]][e])}' for e in range(spec['episodes'])))
    display_contract=dict(world_x_min_m=2.4,world_x_max_m=5.5,
        aligned_x_min_m=-.35,aligned_x_max_m=5.5,z_min_m=.05,z_max_m=1.30,
        density='equal episode weight; each episode contributes 1/control_steps across occupied bins',
        trajectories_sampled=False,episodes_extending_beyond_world_x_max=beyond)
    summary=dict(methods={k:dict(v) for k,v in counts.items()},aligned_exclusions=dict(excluded),
        charged_interactions=charged,active_interactions=active,paired_outcomes=pairs,
        paired_random_draws_identical=True,paired_requests_identical_before_termination=True,
        episodes_per_method=spec['episodes'],role=spec['role'],trace_sha256=hashes,
        new_training_transitions=0,display_contract=display_contract,report_source_sha256=file_sha(__file__))
    if charged>spec['maximum_interactions']:raise ValueError('total comparison budget exceeded')
    write(destination/'summary.json',summary)
    link=os.path.relpath(destination,output)
    with (output/'INDEX.md').open('a') as stream:
        stream.write(f'\n## 可读版图件（x≤5.5m）\n\n![轨迹密度与差异]({link}/comparison.png)\n\n'
            f'[PDF]({link}/comparison.pdf) · [SVG]({link}/comparison.svg) · [逐回合CSV]({link}/episodes.csv) · [汇总及轨迹哈希]({link}/summary.json)\n\n')
        for m in methods:stream.write(f'- {m["label"]}：{counts[m["key"]]["successes"]}/{spec["episodes"]}。\n')
    return summary


def _wilson(successes,total,z=1.959963984540054):
    p=successes/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return center-radius,center+radius


def _render_readable(destination,methods,colors,counts,density,ends,peaks,beyond,excluded,
                     world_x_edges,aligned_x_edges,z_edges,spec,q0):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,LogNorm,Normalize
    labels=[m['label'] for m in methods]
    short_labels=['基线','RSI 800万步','Phase U 499万步'] if len(methods)==3 else labels
    vmax=max(float(density[m['key']]['world'].max()) for m in methods)
    positive=np.concatenate([density[m['key']]['world'][density[m['key']]['world']>0] for m in methods])
    vmin=max(float(np.quantile(positive,.08)),vmax/500) if len(positive) else 0
    world_norm=LogNorm(vmin=vmin,vmax=vmax) if vmax>vmin>0 else Normalize(vmin=0,vmax=1)
    fig=plt.figure(figsize=(16,10),layout='constrained')
    grid=fig.add_gridspec(2,3,height_ratios=(1.15,.85))
    for i,method in enumerate(methods):
        key=method['key'];color=colors[i];ax=fig.add_subplot(grid[0,i])
        cmap=LinearSegmentedColormap.from_list(key,['#ffffff',color])
        image=ax.pcolormesh(world_x_edges,z_edges,density[key]['world'].T,
            cmap=cmap,norm=world_norm,shading='auto',rasterized=True)
        nominal=load_tape(method['nominal_trace']);ticks=np.flatnonzero(nominal['prefix_mask'][:,0])
        pts=nominal['qpos'][ticks,0][:,[q0,q0+2]]
        ax.plot(pts[:,0],pts[:,1],color='#222222',lw=1.8,label='无扰动轨迹')
        endpoint=np.asarray(ends[key]);visible=endpoint[(endpoint[:,0]>=2.4)&(endpoint[:,0]<=5.5)&
            (endpoint[:,1]>=.05)&(endpoint[:,1]<=1.30)] if len(endpoint) else np.empty((0,2))
        if len(visible):ax.scatter(visible[:,0],visible[:,1],s=5,marker='x',color='#7a1f1f',alpha=.25,rasterized=True)
        ax.set(xlim=(2.4,5.5),ylim=(.05,1.30),title=method['label'],xlabel='世界位置 x（m）')
        if i==0:ax.set_ylabel('根部高度 z（m）')
        ax.grid(alpha=.12);ax.legend(loc='upper right',fontsize=8,frameon=False)
        ax.text(.02,.97,f'成功 {counts[key]["successes"]:,}/{spec["episodes"]:,}\n越过5.5m：{beyond[key]:,}回合',
            transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.82,edgecolor='none'))
        fig.colorbar(image,ax=ax,shrink=.68,pad=.01,label='回合等权轨迹密度（对数）')
    ax=fig.add_subplot(grid[1,0]);positions=np.arange(len(methods));rates=[];low=[];high=[]
    for method in methods:
        key=method['key'];rate=counts[key]['successes']/spec['episodes'];lo,hi=_wilson(counts[key]['successes'],spec['episodes'])
        rates.append(rate*100);low.append(max(0.,(rate-lo)*100));high.append(max(0.,(hi-rate)*100))
    rates=np.asarray(rates,float);low=np.asarray(low,float);high=np.asarray(high,float)
    ax.errorbar(positions,rates,yerr=np.vstack((low,high)),fmt='none',ecolor='#333333',capsize=5,lw=1.5)
    ax.scatter(positions,rates,c=colors[:len(methods)],s=90,zorder=3)
    for x,value,method in zip(positions,rates,methods):ax.text(x,value+1.25,f'{value:.2f}%\n{counts[method["key"]]["successes"]:,}/{spec["episodes"]:,}',ha='center',fontsize=9)
    ylo=max(0.,float(rates.min()-5));yhi=min(105.,float(rates.max()+5))
    ax.set(xticks=positions,xticklabels=short_labels,ylim=(ylo,yhi),ylabel='稳定恢复成功率（%）',title='成功率与95% Wilson区间')
    ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[1,1]);failure_groups=['roll_limit','pitch_limit','yaw_limit','stuck','other']
    failure_labels=['侧倾','俯仰','偏航','卡住','其他'];failure_colors=['#9c755f','#e15759','#b07aa1','#f28e2b','#bab0ac']
    left=np.zeros(len(methods))
    for group,label,color in zip(failure_groups,failure_labels,failure_colors):
        values=[]
        for method in methods:
            c=counts[method['key']]
            if group=='other':value=sum(v for k,v in c.items() if k not in {'episodes','successes','recovery_success','roll_limit','pitch_limit','yaw_limit','stuck'})
            else:value=c[group]
            values.append(100*value/spec['episodes'])
        ax.barh(short_labels,values,left=left,label=label,color=color,height=.58);left+=values
    ax.set(xlabel='占全部回合比例（%）',title='未成功终止原因（统一分母）',xlim=(0,max(left)*1.12));ax.grid(axis='x',alpha=.2)
    ax.legend(ncol=3,fontsize=8,frameon=False,loc='upper left',bbox_to_anchor=(0,1.01))
    ax=fig.add_subplot(grid[1,2]);quantiles=[]
    for method in methods:quantiles.append(np.quantile(peaks[method['key']],[.1,.25,.5,.75,.9]))
    for i,(q,color) in enumerate(zip(quantiles,colors)):
        ax.vlines(i,q[0],q[4],color=color,lw=2);ax.vlines(i,q[1],q[3],color=color,lw=10,alpha=.55)
        ax.scatter(i,q[2],s=55,color=color,zorder=3);ax.text(i,q[4]+.025,f'中位 {q[2]:.3f}m',ha='center',fontsize=9)
    ax.set(xticks=positions,xticklabels=short_labels,ylim=(.25,1.30),ylabel='单回合根部峰值高度（m）',title='峰值高度分布：10–90% / 25–75% / 中位数')
    ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'三策略配对随机扰动（每策略 {spec["episodes"]:,} 回合，显示范围 x≤5.5m）',fontsize=17)
    fig.supxlabel('上排为全部回合的等权轨迹密度；红叉为显示范围内的未成功终点。超过x=5.5m的轨迹仅裁剪显示，仍完整计入统计。',fontsize=10)
    for ext in ('png','pdf','svg'):fig.savefig(destination/f'comparison.{ext}',dpi=190)
    plt.close(fig)

    fig,axes=plt.subplots(1,len(methods),figsize=(16,4.6),sharex=True,sharey=True,layout='constrained')
    aligned_positive=np.concatenate([density[m['key']]['aligned'][density[m['key']]['aligned']>0] for m in methods])
    aligned_vmax=max(float(density[m['key']]['aligned'].max()) for m in methods)
    aligned_vmin=max(float(np.quantile(aligned_positive,.08)),aligned_vmax/500) if len(aligned_positive) else 0
    aligned_norm=LogNorm(vmin=aligned_vmin,vmax=aligned_vmax) if aligned_vmax>aligned_vmin>0 else Normalize(vmin=0,vmax=1)
    for i,(ax,method) in enumerate(zip(axes,methods)):
        key=method['key'];cmap=LinearSegmentedColormap.from_list(key+'aligned',['#ffffff',colors[i]])
        ax.pcolormesh(aligned_x_edges,z_edges,density[key]['aligned'].T,cmap=cmap,
            norm=aligned_norm,shading='auto',rasterized=True)
        ax.axvline(0,color='#555555',ls='--',lw=1);ax.set(xlim=(-.35,5.5),ylim=(.05,1.30),title=method['label'],xlabel='x − x_LO（m）')
        if i==0:ax.set_ylabel('根部高度 z（m）')
        ax.grid(alpha=.12)
    fig.suptitle('诊断离地点对齐后的轨迹密度（仅平移x，真实高度不变）',fontsize=15)
    fig.supxlabel('每个回合等权；显示截止到相对离地点 x=5.5m。未检出离地的回合不进入本图，但仍计入成功率。',fontsize=10)
    for ext in ('png','pdf','svg'):fig.savefig(destination/f'aligned_density.{ext}',dpi=190)
    plt.close(fig)
    arrays={}
    for method in methods:
        key=method['key'];arrays[key+'_world_density']=density[key]['world'];arrays[key+'_aligned_density']=density[key]['aligned'];arrays[key+'_peak_z']=np.asarray(peaks[key])
    np.savez_compressed(destination/'plot_data.npz',world_x_edges=world_x_edges,
        aligned_x_edges=aligned_x_edges,z_edges=z_edges,**arrays)
