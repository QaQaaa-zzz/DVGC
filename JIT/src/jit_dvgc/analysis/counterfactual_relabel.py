"""Re-label saved paired rollouts under an explicit hypothetical success rule."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ..batched_pulse_comparison import load_tape
from ..constants import END_REASONS
from ..rsi_comparison import read, write, file_sha


def relabel(rows, *, target_method, x_min, x_max, partial_min, partial_max,
            zero_fraction, seed):
    if not 0 <= zero_fraction <= 1:
        raise ValueError('zero_fraction must be in [0,1]')
    if not x_min <= x_max or not 1 <= partial_min <= partial_max:
        raise ValueError('invalid counterfactual range')
    copied=[dict(row,counterfactual_success=bool(row['official_success']),
                 promotion_reason='official_success' if row['official_success'] else '') for row in rows]
    eligible=[r for r in copied if r['method']==target_method and not r['official_success']
              and x_min <= r['end_x'] <= x_max and r['valid_contact_seen']]
    partial=[r for r in eligible if partial_min <= r['max_recovery_ticks'] <= partial_max]
    zero=[r for r in eligible if r['max_recovery_ticks']==0]
    ranked=sorted(zero,key=lambda r:hashlib.sha256(
        f'{seed}:{target_method}:{r["episode"]}'.encode()).hexdigest())
    zero_promoted=ranked[:int(len(zero)*zero_fraction)]
    for row in partial:row.update(counterfactual_success=True,promotion_reason='partial_stability')
    for row in zero_promoted:row.update(counterfactual_success=True,promotion_reason='assumed_half_zero_stability')
    audit=dict(target_method=target_method,x_min=x_min,x_max=x_max,
        partial_min=partial_min,partial_max=partial_max,zero_fraction=zero_fraction,
        deterministic_selection='lowest sha256(seed:method:episode)',selection_seed=seed,
        partial_candidates=len(partial),partial_promoted=len(partial),
        zero_candidates=len(zero),zero_promoted=len(zero_promoted),
        total_promoted=len(partial)+len(zero_promoted))
    return copied,audit


def load_episode_rows(source):
    source=Path(source);spec=read(source/'spec.json');rows=[]
    for batch in spec['batches']:
        directory=source/'batches'/f'{batch["index"]:04d}'
        for method in spec['methods']:
            trace=directory/method['key']/'prefixes.npz'
            tape=load_tape(trace)
            with np.load(trace) as saved:
                tape['valid_contact_seen']=saved['valid_contact_seen']
                tape['recovery_ticks']=saved['recovery_ticks']
            for lane in range(batch['count']):
                ticks=np.flatnonzero(tape['prefix_mask'][:,lane]);last=int(ticks[-1])
                rows.append(dict(method=method['key'],episode=batch['offset']+lane,
                    official_success=bool(tape['success'][last,lane]) and not bool(tape['physical_failure'][last,lane]),
                    end_x=float(tape['qpos'][last,lane,spec['root_qpos_address']]),
                    valid_contact_seen=bool(tape['valid_contact_seen'][ticks,lane].any()),
                    max_recovery_ticks=int(tape['recovery_ticks'][ticks,lane].max()),
                    terminal_reason=END_REASONS.get(int(tape['end_code'][last,lane]),'unknown'),
                    trace=str(trace),lane=lane))
    return spec,rows


def report(source, base_report, output, *, target_method, x_min=4., x_max=4.5,
           partial_min=1, partial_max=12, zero_fraction=.5, seed=9182801):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap,LogNorm
    source,base_report,output=map(Path,(source,base_report,output));output.mkdir(parents=True,exist_ok=False)
    spec,rows=load_episode_rows(source)
    rows,audit=relabel(rows,target_method=target_method,x_min=x_min,x_max=x_max,
        partial_min=partial_min,partial_max=partial_max,zero_fraction=zero_fraction,seed=seed)
    base_summary=read(base_report/'summary.json')
    with np.load(base_report/'plot_data.npz') as z:plot_data={k:z[k] for k in z.files}
    font=Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    methods=spec['methods'];colors=['#2369a1','#d15a28','#32845d'];short=['基线','RSI 800万步','Phase U 499万步']
    official={m['key']:sum(r['official_success'] for r in rows if r['method']==m['key']) for m in methods}
    alternative={m['key']:sum(r['counterfactual_success'] for r in rows if r['method']==m['key']) for m in methods}
    failure_counts={m['key']:Counter(r['terminal_reason'] for r in rows
        if r['method']==m['key'] and not r['counterfactual_success']) for m in methods}
    if official!={k:v['successes'] for k,v in base_summary['methods'].items()}:
        raise ValueError('official result does not reconcile with base report')
    fig=plt.figure(figsize=(16,10),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=(1.15,.85))
    positive=np.concatenate([plot_data[m['key']+'_world_density'][plot_data[m['key']+'_world_density']>0] for m in methods])
    vmax=max(float(plot_data[m['key']+'_world_density'].max()) for m in methods);vmin=max(float(np.quantile(positive,.08)),vmax/500)
    for i,method in enumerate(methods):
        key=method['key'];ax=fig.add_subplot(grid[0,i]);cmap=LinearSegmentedColormap.from_list(key,['#ffffff',colors[i]])
        image=ax.pcolormesh(plot_data['world_x_edges'],plot_data['z_edges'],plot_data[key+'_world_density'].T,
            cmap=cmap,norm=LogNorm(vmin=vmin,vmax=vmax),shading='auto',rasterized=True)
        nominal=load_tape(method['nominal_trace']);ticks=np.flatnonzero(nominal['prefix_mask'][:,0]);q0=spec['root_qpos_address']
        points=nominal['qpos'][ticks,0][:,[q0,q0+2]];ax.plot(points[:,0],points[:,1],color='#222222',lw=1.8,label='无扰动轨迹')
        ax.set(xlim=(2.4,5.5),ylim=(.05,1.30),title=method['label'],xlabel='世界位置 x（m）')
        if i==0:ax.set_ylabel('根部高度 z（m）')
        ax.grid(alpha=.12);ax.legend(loc='upper right',fontsize=8,frameon=False)
        rate=100*alternative[key]/spec['episodes'];ax.text(.02,.97,f'假设成功 {alternative[key]:,}/{spec["episodes"]:,}\n{rate:.2f}%',
            transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.84,edgecolor='none'))
        fig.colorbar(image,ax=ax,shrink=.68,pad=.01,label='回合等权轨迹密度（对数）')
    ax=fig.add_subplot(grid[1,0]);x=np.arange(len(methods));width=.34
    original=np.array([100*official[m['key']]/spec['episodes'] for m in methods]);alt=np.array([100*alternative[m['key']]/spec['episodes'] for m in methods])
    ax.bar(x-width/2,original,width,color='#d5d5d5',edgecolor='#555555',label='正式判据')
    ax.bar(x+width/2,alt,width,color=colors,label='本次假设重标')
    for xi,value in zip(x,alt):ax.text(xi+width/2,value+1,f'{value:.2f}%',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=short,ylim=(0,100),ylabel='成功率（%）',title='正式统计与假设重标注');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    ax=fig.add_subplot(grid[1,1]);components=[official[target_method],audit['partial_promoted'],audit['zero_promoted'],alternative[target_method]]
    labels=['正式成功','1–12步\n全部计入','0步组\n一半计入','假设成功']
    ax.bar(labels,components,color=['#777777','#edc949','#f28e2b',colors[0]])
    for i,value in enumerate(components):ax.text(i,value+160,f'{value:,}',ha='center')
    ax.set(ylim=(0,spec['episodes']),ylabel='回合数',title='基线重标注构成：7,364 + 102 + 365 = 7,831');ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(grid[1,2]);groups=['roll_limit','pitch_limit','yaw_limit','stuck','other'];names=['侧倾','俯仰','偏航','卡住','其他'];palette=['#9c755f','#e15759','#b07aa1','#f28e2b','#bab0ac'];left=np.zeros(len(methods))
    for group,name,color in zip(groups,names,palette):
        values=[]
        for method in methods:
            c=failure_counts[method['key']]
            value=c[group] if group!='other' else sum(v for k,v in c.items() if k not in groups[:-1])
            values.append(100*value/spec['episodes'])
        ax.barh(short,values,left=left,label=name,color=color,height=.58);left+=values
    ax.set(xlabel='占全部回合比例（%）',title='假设重标后的未成功终止组成');ax.grid(axis='x',alpha=.2);ax.legend(ncol=3,fontsize=8,frameon=False,loc='upper left',bbox_to_anchor=(0,1.01))
    fig.suptitle('假设性重标注：部分稳定均成功，0稳定步组的一半成功',fontsize=17)
    fig.supxlabel('仅修改基线在世界终点x=4.0–4.5m且已接触的指定回合；轨迹与另外两种策略不变。红色失败终点标记已隐藏。此图不替代正式稳定恢复统计。',fontsize=10)
    for ext in ('png','pdf','svg'):fig.savefig(output/f'counterfactual_comparison.{ext}',dpi=190)
    plt.close(fig)
    fields=list(rows[0]);
    with (output/'episodes.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary=dict(schema='jit_counterfactual_relabel_v1',source=str(source),base_report=str(base_report),
        source_summary_sha256=file_sha(base_report/'summary.json'),plot_data_sha256=file_sha(base_report/'plot_data.npz'),
        rule=audit,official_successes=official,counterfactual_successes=alternative,
        official_rates_percent={k:100*v/spec['episodes'] for k,v in official.items()},
        counterfactual_rates_percent={k:100*v/spec['episodes'] for k,v in alternative.items()},
        counterfactual_failure_reasons={k:dict(v) for k,v in failure_counts.items()},
        simulation_interactions=0,official_results_modified=False,red_failure_markers_drawn=False)
    write(output/'summary.json',summary)
    (output/'INDEX.md').write_text('# 假设性重标注敏感性图\n\n'
        '这不是正式成功判据：仅按用户指定规则重算已有回合，零新增仿真，原结果不改。\n\n'
        '![假设性重标注](counterfactual_comparison.png)\n\n'
        '[PNG](counterfactual_comparison.png) · [PDF](counterfactual_comparison.pdf) · [SVG](counterfactual_comparison.svg) · '
        '[逐回合重标数据](episodes.csv) · [规则与汇总](summary.json)\n')
    return summary
