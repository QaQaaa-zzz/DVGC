"""Separate extended-domain discovery, common-domain gain and unwitnessed states."""
from collections import Counter, defaultdict
from pathlib import Path
from ..jump_evidence_validation import read, write
from .policy_envelopes import write_csv


def classify_labels(values):
    if len(values)!=4 or any(type(v) is not int or v not in (0,1) for v in values):
        raise ValueError('four completed binary labels required; missing is not failure')
    return 'all_succeed' if sum(values)==4 else 'no_success_witness' if not any(values) else 'policy_disagreement'


def summarize_frontier(output, panels, labels, old_root, destination):
    names=list(panels)
    rows,receipts=[],[]
    grouped=defaultdict(Counter)
    old_domain,new_domain=set(),set()
    for name in names:
        plan=read(output/name/'plan.json')
        limit=float(read(plan['centerline'])['effective_centerline_max_x_m'])
        manifest=read(output/name/'analysis_inputs.json')
        catalog_path=Path(manifest['catalog']);catalog=read(catalog_path)
        source=catalog['entries']
        if len(source)!=len(panels[name]):raise ValueError('frontier catalog/projected count mismatch')
        trails=catalog.get('trajectory_receipts')
        if trails is None or len(trails)!=catalog['attempted_candidate_count']:
            raise ValueError('missing acquisition attempt receipts')
        if sum(r['environment_interactions'] for r in trails)!=catalog['environment_interactions']:
            raise ValueError('acquisition receipt cost drift')
        receipts += trails
        for i,(point,entry) in enumerate(zip(panels[name],source,strict=True)):
            if point['candidate_id']!=entry['candidate_id'] or point['state_sha256']!=entry['state_sha256']:
                raise ValueError('boundary source identity drift')
            values=[labels[name][n][i]['label'] for n in names]
            status=classify_labels(values)
            perturb=entry['perturbation']
            record={'proposer':name,'candidate_id':point['candidate_id'],'state_sha256':point['state_sha256'],
                'trajectory_id':point['trajectory_id'],'trajectory_step':point['trajectory_step'],
                'phase':point['phase'],'x_m':point['coordinates']['root_x_m'],
                'z_m':point['coordinates']['root_z_m'],'vz_mps':point['coordinates']['root_vz_mps'],
                'action':perturb['action_name'],'sign':perturb['sign'],'strength':perturb['strength'],
                'status':status,'successful_evaluators':sum(values),
                'within_previous_corridor':point['coordinates']['root_x_m']<=limit,
                'catalog':str(catalog_path),'snapshot':str((catalog_path.parent/entry['source_bank']/entry['snapshot']).resolve())}
            rows.append(record)
            key=(name,record['action'],record['sign'],record['strength'],record['phase'])
            grouped[key][status]+=1
            if any(values):
                (old_domain if record['within_previous_corridor'] else new_domain).add(point['root_cell'])
    write_csv(destination/'boundary_candidates.csv',rows)
    write_csv(destination/'trajectory_endings.csv',[{k:v for k,v in r.items() if k!='direction'} for r in receipts])
    write_csv(destination/'boundary_groups.csv',[dict(zip(('proposer','action','sign','strength','phase'),key),**{s:c[s] for s in ('all_succeed','policy_disagreement','no_success_witness')}) for key,c in grouped.items()])
    statuses=Counter(r['status'] for r in rows)
    stops=Counter(r['stop_reason'] for r in receipts)
    report={'candidate_status_counts':dict(statuses),'trajectory_stop_counts':dict(stops),
            'old_corridor_novel_root_cells':len(old_domain-old_root),
            'extended_corridor_only_novel_root_cells':len(new_domain-old_root-old_domain),
            'total_novel_root_cells':len((old_domain|new_domain)-old_root),
            'extended_domain_is_not_same_domain_widening':True,
            'truncated_trajectories':sum(r['truncated'] for r in receipts),
            'all_attempts_reached_terminal_outcome':not any(r['truncated'] for r in receipts),
            'no_success_witness_means_physical_impossibility':False,
            'training_admission_authorized':False,'correlated_frames_are_independent':False}
    write(destination/'frontier_summary.json',report)
    render_boundary(rows,destination)
    return report


def render_boundary(rows,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['pi_0','pi_1','pi_2','pi_3']
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for name,ax in zip(names,axes.flat):
        for status,color,marker in [('all_succeed','#aaaaaa','.'),('policy_disagreement','#E69F00','o'),('no_success_witness','#D55E00','x')]:
            selected=[r for r in rows if r['proposer']==name and r['status']==status]
            ax.scatter([r['x_m'] for r in selected],[r['z_m'] for r in selected],
                       c=color,marker=marker,s=12,label=status,rasterized=True)
        ax.set(title=name,xlabel='x (m)',ylabel='z (m)');ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Real arrivals: agreement and missing landing witnesses')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(destination/f'boundary_states.{ext}',dpi=300)
    plt.close(fig)
