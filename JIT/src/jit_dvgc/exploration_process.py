"""Inspectable optimization history and replot data for actor and critic."""
import csv
import json
from pathlib import Path


def append_update(root,row):
    with (Path(root)/'optimizer_updates.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')


def export_process(root,history,*,plots=False):
    root=Path(root)
    rows=[]
    for row in history:
        flat={k:v for k,v in row.items() if isinstance(v,(str,int,float,bool))}
        flat.update({'reward_'+k:v for k,v in row.get('reward_components',{}).items()})
        flat.update(row.get('loss_components',{}));rows.append(flat)
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (root/'training_process.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    if not plots or not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7));x=[r['batch'] for r in rows]
    for key in ('actor_loss','critic_loss','total_loss'):
        axes[0,0].plot(x,[r.get(key,float('nan')) for r in rows],label=key)
    for key in ('novelty','physical_failure','residual_energy'):
        axes[0,1].plot(x,[r.get('reward_'+key,0) for r in rows],label=key)
    axes[1,0].plot(x,[r.get('total_reward',0) for r in rows],label='total reward')
    axes[1,1].plot(x,[r['new_cells'] for r in rows],label='new provisional cells')
    for ax in axes.flat:ax.set_xlabel('Batch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png','pdf','svg'):fig.savefig(root/('training_process.'+suffix))
    plt.close(fig)
