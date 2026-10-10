"""Read-only BC retention report; never dispatches simulation or training."""
from pathlib import Path
import numpy as np
from .retention_repair import read,write


def build_report(path):
    p=read(path);root=Path(p['output']);status=read(root/'status.json')
    out=root/'report_001';out.mkdir(exist_ok=False)
    if status['phase']!='completed':
        write(out/'summary.json',dict(status=status,claim='incomplete B; no ability conclusion'))
        return out
    rows=read(root/'B_checkpoint_results.json');finish=read(root/'B_completed.json');costs=read(root/'costs.json')
    from .retention_b import full_results
    baselines={name:full_results(p,root/('DEV_'+name)) for name in ('pi0','R5')}
    write(out/'summary.json',dict(source_revision=p['code_revision'],source_lock=str(root/'source_lock.json'),budget=p['budget'],actual_charged=sum(x['charged'] for x in costs),D2_prior_charged=12900,baselines=baselines,checkpoints=rows,finish=finish))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .retention_repair_report import CachedArchive
    for group in 'ABCD':
        fig,axes=plt.subplots(4,4,figsize=(12,12))
        lanes=[i for i,c in enumerate(p['batches'][0]['cases']) if c['group']==group]
        sources=[('pi0',root/'DEV_pi0'),('R5',root/'DEV_R5')]+[(f"BC{k}",Path(v['full']['output'])) for k,v in rows.items() if k!='0']
        for label,folder in sources:
            with CachedArchive(folder/'prefixes.npz') as z:
                for ax,lane in zip(axes.flat,lanes):
                    mask=z['prefix_mask'][:,lane];xy=z['qpos'][:,lane,:2][mask]
                    ax.plot(xy[:,0],xy[:,1],label=label,lw=.8);ax.scatter(xy[-1,0],xy[-1,1],s=6);ax.set_aspect('equal',adjustable='datalim')
                    ax.set_title(p['batches'][0]['cases'][lane]['case'],fontsize=7);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        axes.flat[0].legend(fontsize=6);fig.suptitle(f'New DEV {group}: complete task, actual endpoints; 16 paired cases')
        fig.tight_layout();fig.savefig(out/f'xy_DEV_{group}.png',dpi=130);plt.close(fig)
    text=['# D2 B-only results','',*[f'![New DEV {g}](xy_DEV_{g}.png)' for g in 'ABCD'],'',f"Code `{p['code_revision']}`; BC updates {finish['completed_updates']}/2000. PPO/E/G updates 0.",'','|update|TRAIN absorption /8|SOLVER_DEV /7|A /16|B /16|C /16|D /16|eligible|','|---|---|---|---|---|---|---|---|']
    for k,v in rows.items():text.append('|'+ '|'.join(map(str,[k,v['train_success'],v['solver_success'],*[v['cells'][g]['success'] for g in 'ABCD'],v['eligible']]))+'|')
    text+=['',f"Selected update {finish['selected_update']} by all-seven independent SOLVER_DEV, earliest tie, excluding confirmed B/D regressions.",'','TRAIN is seen-root absorption. SOLVER_DEV is small-sample development transfer. Full DEV measures paired old losses and new gains; no TEST or policy publication. R5 remains the global best_dev_candidate.','',f"Charged new physics {sum(x['charged'] for x in costs)}; prior D2 12900; maximum {p['budget']['total']}. Full immutable state/labels/inputs in summary.json and parent run."]
    (out/'INDEX.md').write_text('\n'.join(text)+'\n');return out
