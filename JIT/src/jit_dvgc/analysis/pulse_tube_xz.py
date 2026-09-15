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


def build(lineages,output,labels=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    labels=labels or [Path(p).parent.name for p in lineages]
    if len(labels)!=len(lineages):raise ValueError('one label per lineage required')
    data=[extract(p,output/f'scheme_{i+1}') for i,p in enumerate(lineages)]
    allpts=np.concatenate([s for seg,_,_ in data for s in seg]);cp=np.array([[r['x'],r['z']] for _,rows,_ in data for r in rows]);lo=allpts.min(0);hi=allpts.max(0)
    for i,((segments,rows,manifest),label) in enumerate(zip(data,labels)):
        dest=output/f'scheme_{i+1}';fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
        axes[0].add_collection(LineCollection(segments,colors='#176b87',linewidths=.35,alpha=.06,rasterized=True))
        axes[0].set(xlim=(lo[0]-.1,hi[0]+.1),ylim=(lo[1]-.03,hi[1]+.05),title=f'{label}: observed successful trajectories ({manifest["successful_context_policy_pairs"]} witnesses)')
        for status,color,name in [('success','#16855b','Successful handoff'),('unresolved','#d45d43','Unresolved after attempts'),('unknown','#888888','Unknown')]:
            pts=np.array([[r['x'],r['z']] for r in rows if r['label']==status])
            if len(pts):axes[1].scatter(pts[:,0],pts[:,1],s=4,c=color,alpha=.35,label=name,rasterized=True)
        axes[1].set(xlim=(cp[:,0].min()-.03,cp[:,0].max()+.03),ylim=(cp[:,1].min()-.03,cp[:,1].max()+.03),title='Explored handoff positions (not continuous dead/safe regions)');axes[1].legend()
        for ax in axes:ax.set_xlabel('Root forward position x (m)');ax.set_ylabel('Root height z (m)');ax.grid(alpha=.2)
        fig.suptitle('Empirical x-z projection; full recovery retained; no hull/interpolation')
        for ext in ('png','pdf','svg'):fig.savefig(dest/f'tube_xz.{ext}',dpi=180)
        plt.close(fig)
    lines=['# Pulse campaign x-z projections','','All schemes use identical axis ranges. Positions are robot root coordinates, not wheel clearance. Successful trajectories include recovery and may include subsequent bounces. Prefixes and suffixes are drawn separately; no artificial bridge or hull. Failed handoff points are not proven physical dead zones.','']
    for i,label in enumerate(labels):lines.append(f'- {label}: [PNG](scheme_{i+1}/tube_xz.png) · [PDF](scheme_{i+1}/tube_xz.pdf) · [evidence/data](scheme_{i+1}/manifest.json)')
    (output/'INDEX.md').write_text('\n'.join(lines)+'\n')
    return [d[2] for d in data]
