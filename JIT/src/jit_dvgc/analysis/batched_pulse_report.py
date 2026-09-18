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
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
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
    segments={m['key']:{(view,good):[] for view in ('world','aligned') for good in (False,True)} for m in methods}
    ends={m['key']:[] for m in methods};excluded=Counter();counts={m['key']:Counter() for m in methods}
    seen={m['key']:set() for m in methods};outcomes={m['key']:{} for m in methods};hashes={};charged=active=0
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
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
                    segments[key]['world',good].append(points)
                    if not good:ends[key].append(points[-1])
                    k=liftoff_index(tape['front_wheel_clearance'][ticks,lane],tape['rear_wheel_clearance'][ticks,lane])
                    if k is None:excluded[key]+=1
                    else:segments[key]['aligned',good].append(align_xz(points,points[k,0]))
    for i,method in enumerate(methods):
        key=method['key'];color=colors[i%len(colors)]
        if seen[key]!=set(range(spec['episodes'])):raise ValueError('missing method episodes')
        tape=load_tape(method['nominal_trace']);hashes[method['nominal_trace']]=file_sha(method['nominal_trace'])
        ticks=np.flatnonzero(tape['prefix_mask'][:,0]);pts=tape['qpos'][ticks,0][:,[q0,q0+2]]
        axes[0,0].plot(pts[:,0],pts[:,1],color=color,label=method['label'],lw=2)
        for view,ax in [('world',axes[0,1]),('aligned',axes[1,0])]:
            for good in (True,False):
                ax.add_collection(LineCollection(segments[key][view,good],colors=color,
                    alpha=.04,linewidths=.5,linestyles='solid' if good else 'dashed',rasterized=True))
            ax.autoscale_view()
        if ends[key]:
            points=np.asarray(ends[key]);axes[0,1].scatter(points[:,0],points[:,1],color=color,marker='x',s=6,alpha=.2,rasterized=True)
    axes[0,0].legend(fontsize=9)
    axes[0,1].legend(handles=[Line2D([0],[0],color=colors[i%len(colors)],label=m['label']) for i,m in enumerate(methods)],fontsize=9)
    titles=['无扰动参考轨迹（复用）',f'每策略{spec["episodes"]:,}回合：全部世界坐标轨迹','全部可检出离地轨迹：仅平移x对齐']
    for ax,title in zip(axes.flat[:3],titles):
        ax.set(title=title,xlabel='x − x_LO（m）' if ax is axes[1,0] else '世界位置 x（m）',ylabel='根部高度 z（m）');ax.grid(alpha=.2)
    axes[1,0].axvline(0,color='gray',ls='--',lw=.8)
    values=[100*counts[m['key']]['successes']/spec['episodes'] for m in methods]
    bars=axes[1,1].bar([m['label'] for m in methods],values,color=colors[:len(methods)],width=.6)
    axes[1,1].set(ylim=(0,112),ylabel='稳定恢复成功率（%）',title='全部回合为分母，包含起扰前失败')
    for bar,m,value in zip(bars,methods,values):
        axes[1,1].text(bar.get_x()+bar.get_width()/2,value+2,f'{counts[m["key"]]["successes"]}/{spec["episodes"]}\n{value:.2f}%',ha='center')
    fig.suptitle('固定策略、配对随机扰动：三策略大样本比较',fontsize=17)
    fig.supxlabel('四通道±0.25，3步脉冲；实线成功、虚线未成功、叉号真实失败终点。轨迹层栅格化，未抽样。\n'
        +'未检出诊断离地：'+', '.join(f'{m["key"]}={excluded[m["key"]]}' for m in methods)
        +'；全部仍计入成功率。单策略checkpoint的开发评估，不是独立训练种子。',fontsize=10)
    for ext in ('png','pdf','svg'):fig.savefig(destination/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    pairs={}
    for i,a in enumerate(methods):
        for b in methods[i+1:]:
            pairs[a['key']+'__'+b['key']]=dict(Counter(
                f'{int(outcomes[a["key"]][e])}{int(outcomes[b["key"]][e])}' for e in range(spec['episodes'])))
    summary=dict(methods={k:dict(v) for k,v in counts.items()},aligned_exclusions=dict(excluded),
        charged_interactions=charged,active_interactions=active,paired_outcomes=pairs,
        paired_random_draws_identical=True,paired_requests_identical_before_termination=True,
        episodes_per_method=spec['episodes'],role=spec['role'],trace_sha256=hashes,
        new_training_transitions=0,report_source_sha256=file_sha(__file__))
    if charged>spec['maximum_interactions']:raise ValueError('total comparison budget exceeded')
    write(destination/'summary.json',summary)
    link=os.path.relpath(destination,output)
    with (output/'INDEX.md').open('a') as stream:
        stream.write(f'\n## 完成结果\n\n![全部轨迹]({link}/comparison.png)\n\n'
            f'[PDF]({link}/comparison.pdf) · [SVG]({link}/comparison.svg) · [逐回合CSV]({link}/episodes.csv) · [汇总及轨迹哈希]({link}/summary.json)\n\n')
        for m in methods:stream.write(f'- {m["label"]}：{counts[m["key"]]["successes"]}/{spec["episodes"]}。\n')
    return summary
