"""Evidence-backed x-z projections of pulse campaign trajectory witnesses."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def round_directories(root):
    root=Path(root).resolve()
    count=len(read(root/'training_metrics.json'))
    result=[]
    for i in range(count):
        current=root;seen=set()
        while not (current/f'round_{i:04d}'/'outcomes.json').exists():
            if current in seen:raise ValueError('cycle in run ancestry')
            seen.add(current)
            receipt=next((current/n for n in ('recovery.json','resume.json') if (current/n).exists()),None)
            if receipt is None:raise FileNotFoundError(f'missing round {i} in {current}')
            current=Path(read(receipt)['previous']).resolve()
        result.append(current/f'round_{i:04d}')
    return result


def active_xz(tape,lane,mask_key,q0):
    ticks=np.flatnonzero(tape[mask_key][:,lane])
    points=tape['qpos'][ticks,lane][:,[q0,q0+2]]
    if not len(ticks) or not np.isfinite(points).all():raise ValueError('empty/nonfinite active trajectory')
    return ticks,points


def extract(root,output):
    from jit_dvgc.config import load_config
    from jit_dvgc.model import load_host_model
    root=Path(root).resolve();output.mkdir(parents=True,exist_ok=False)
    spec=read(root/'declaration.json')['spec'];bank=read(spec['bank'])
    member=next(m for m in bank['members'] if m['name']==spec['proposer'])
    cfg=read(member['policy']['formal_config'])
    phase=load_config(Path(cfg['inputs']['up_config_path']),runtime_only=True)
    q0=load_host_model(phase).model_index.root_qpos_address
    hashes={};groups=defaultdict(list);candidates=[];seen=set();segments=[];metadata=[]
    def check(path,expected=None):
        path=str(Path(path).resolve())
        if path not in hashes:hashes[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if expected is not None and hashes[path]!=expected:raise ValueError('trace hash mismatch: '+path)
    for p in (root/'declaration.json',Path(spec['bank']),Path(member['policy']['formal_config']),Path(cfg['inputs']['up_config_path'])):check(p)
    for d in round_directories(root):
        p=d/'outcomes.json';check(p)
        for row in read(p):
            c=row.get('coordinates')
            if c:candidates.append(dict(round=int(row['round']),candidate=int(row['index']),context=row['snapshot_context_sha256'],label='unknown' if row['label'] is None else ('success' if row['label']==1 else 'unresolved'),x=c['root_x_m'],z=c['root_z_m']))
            if row['label']!=1:continue
            if row.get('prefix_terminal'):
                prefix=row['prefix_file'];check(prefix,row['prefix_sha256'])
                with np.load(prefix) as z:
                    ticks,pts=active_xz(z,row['index'],'prefix_mask',q0)
                    if 'physical_failure' not in z or bool(z['physical_failure'][ticks[-1],row['index']]):
                        raise ValueError('terminal-prefix success lacks conflict-free evidence')
                key=('terminal_prefix',row['snapshot_context_sha256'])
                if key not in seen:
                    seen.add(key);metadata.append(dict(segment=len(segments),kind='terminal_prefix',policy=row.get('witness'),context=row['snapshot_context_sha256'],trace=prefix,lane=row['index'],points=len(pts)))
                    segments.append(pts)
                continue
            for a in row['attempts']:
                if a['label']!=1:continue
                if a['snapshot_context_sha256']!=row['snapshot_context_sha256']:raise ValueError('context mismatch')
                key=(a['actor_sha256'],row['snapshot_context_sha256'])
                if key in seen:continue
                seen.add(key);groups[a['trace']].append((row,a))
    prefix_cache={}
    for path,items in groups.items():
        check(path)
        with np.load(path) as tape:
            for row,a in items:
                check(path,a['trace_sha256']);ticks,points=active_xz(tape,a['trace_lane'],'mask',q0)
                if len(ticks)!=a['steps'] or not bool(tape['success'][ticks[-1],a['trace_lane']]):raise ValueError('success/length mismatch')
                if bool(tape['physical_failure'][ticks[-1],a['trace_lane']]):raise ValueError('conflicting success')
                success_at=np.flatnonzero(tape['success'][ticks,a['trace_lane']])
                points=points[:int(success_at[0])+1]
                prefix=row['prefix_file'];check(prefix,row['prefix_sha256'])
                if prefix not in prefix_cache:
                    with np.load(prefix) as z:prefix_cache[prefix]={k:z[k] for k in ('qpos','prefix_mask')}
                _,pre=active_xz(prefix_cache[prefix],row['index'],'prefix_mask',q0)
                # Keep prefix/suffix distinct: do not draw an artificial replay-join segment.
                for kind,pts in (('prefix',pre),('suffix',points)):
                    metadata.append(dict(segment=len(segments),kind=kind,policy=a['policy'],context=row['snapshot_context_sha256'],trace=prefix if kind=='prefix' else path,lane=row['index'] if kind=='prefix' else a['trace_lane'],points=len(pts)))
                    segments.append(pts)
        prefix_cache.clear()
    if not segments:raise ValueError('no successful trajectory witnesses')
    offsets=np.r_[0,np.cumsum([len(s) for s in segments])]
    np.savez_compressed(output/'trajectory_points.npz',xz=np.concatenate(segments),offsets=offsets)
    (output/'trajectory_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with (output/'candidate_points.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['round','candidate','context','label','x','z']);w.writeheader();w.writerows(candidates)
    manifest=dict(lineage=str(root),success_criterion=spec.get('success_criterion','first_valid_landing'),rounds=len(round_directories(root)),successful_context_policy_pairs=len(seen),candidate_count=len(candidates),root_qpos_address=q0,input_sha256=hashes,new_simulation_interactions=0,claim='Observed root-position projection of successful prefix/suffix witnesses across policies; not one policy envelope or a filled reachable set. Unresolved candidates are not proven dead zones. Recovery segment is retained.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return segments,candidates,manifest


def render(data,output,labels,focus_width=3.3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import LineCollection
    font=Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11})
    allpts=np.concatenate([s for seg,_,_ in data for s in seg]);lo=allpts.min(0);hi=allpts.max(0)
    allnames=sorted({m['policy'] for i in range(len(data)) for m in read(output/f'scheme_{i+1}'/'trajectory_metadata.json') if m['policy']})
    ordered=sorted(allnames,key=lambda n:('repair_' in n,n))
    palette=plt.get_cmap('turbo');colors={n:palette(i/max(1,len(ordered)-1)) for i,n in enumerate(ordered)}
    for i,((segments,rows,manifest),label) in enumerate(zip(data,labels)):
        dest=output/f'scheme_{i+1}';meta=read(dest/'trajectory_metadata.json');by=defaultdict(list)
        for m,seg in zip(meta,segments):by[m['policy']].append(seg)
        for full in (False,True):
            right=hi[0]+.08 if full else min(hi[0]+.08,lo[0]+focus_width)
            fig,axes=plt.subplots(2,1,figsize=(15,8.5),gridspec_kw={'height_ratios':[2.3,1]},layout='constrained')
            for name in ordered:
                if name not in by:continue
                axes[0].add_collection(LineCollection(by[name],colors=[colors[name]],linewidths=.5,alpha=.15,rasterized=True))
                legend=('初始策略' if 'repair_' not in name else '后继策略 '+name.rsplit('repair_',1)[1])
                axes[0].plot([],[],color=colors[name],label=legend)
            axes[0].set_title('当前策略族的 tube：成功轨迹的 x–z 投影'+('（完整恢复）' if full else '（跳跃段局部放大）'),loc='left',weight='bold')
            axes[0].legend(ncol=2,fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1),title='成功接续策略（不等于全部已接纳）',title_fontsize=8)
            pts=np.array([[r['x'],r['z']] for r in rows if r['label']=='success'])
            if len(pts):axes[1].scatter(pts[:,0],pts[:,1],s=7,color='#176b87',alpha=.35,rasterized=True)
            axes[1].set_title('已保存的成功候选状态位置（实际采样点）',loc='left')
            for ax in axes:
                ax.set(xlim=(lo[0]-.05,right),ylim=(max(0,lo[1]-.03),hi[1]+.05),xlabel='前向位置 x（m）',ylabel='车体根部高度 z（m）')
                ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            fig.suptitle(label+'｜按策略分色，真实记录合并，不插值填充空隙',fontsize=14,weight='bold')
            fig.text(.01,.005,'轨迹数据在首次稳定恢复成功处结束。'+('本图包含完整恢复距离。' if full else '此图仅缩放显示范围；完整的落地后稳定恢复证据见完整图。'),fontsize=9)
            stem='tube_xz_full' if full else 'tube_xz'
            for ext in ('png','pdf','svg'):fig.savefig(dest/f'{stem}.{ext}',dpi=180)
            plt.close(fig)
        manifest.update(display_focus_width_m=focus_width,cutoff='first recorded recovery success; spatial focus is display only',policy_colors={n:list(colors[n]) for n in ordered if n in by})
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# 两种奖励方案的 x-z 投影','','主图按策略分色并放大跳跃段；完整图保留首次满足稳定恢复条件前的全部距离。两组采用相同坐标范围。下图只显示成功候选；全部成功/未解决/未知点仍保存在CSV。这里是策略族的经验投影，不是最终单策略的完整可控集。','']
    for i,label in enumerate(labels):lines.append(f'- {label}：[主图](scheme_{i+1}/tube_xz.png) · [PDF](scheme_{i+1}/tube_xz.pdf) · [完整恢复图](scheme_{i+1}/tube_xz_full.png) · [证据](scheme_{i+1}/manifest.json)')
    (output/'INDEX.md').write_text('\n'.join(lines)+'\n')


def build(lineages,output,labels=None,focus_width=3.3):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    labels=labels or [Path(p).parent.name for p in lineages]
    if len(labels)!=len(lineages):raise ValueError('one label per lineage required')
    if not np.isfinite(focus_width) or focus_width<=0:raise ValueError('focus width must be positive')
    data=[extract(p,output/f'scheme_{i+1}') for i,p in enumerate(lineages)]
    render(data,output,labels,focus_width)
    return [d[2] for d in data]
