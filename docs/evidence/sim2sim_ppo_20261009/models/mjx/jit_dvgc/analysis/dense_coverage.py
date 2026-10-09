"""Old versus dense observed support under unchanged physical-cell resolution."""
from pathlib import Path
from .policy_envelopes import write_csv
from ..jump_evidence_validation import write


def coverage_rows(old_points,old_report,new_points,new_report):
    if old_report['resolution'] != new_report['resolution']:
        raise ValueError('physical resolution differs; recompute both views before comparison')
    for name in [*old_report.get('policy_names',[]),'union']:
        if name not in old_report['masks']:raise ValueError('incomplete old policy masks')
    rows=[]
    for name in [*new_report['policy_names'],'union']:
        known=name in old_report['masks']
        row={'policy':name,'old_panel_evaluated':known}
        for field in ('root_cell','full_cell'):
            before=({p[field] for p,ok in zip(old_points,old_report['masks'][name],strict=True) if ok} if known else None)
            after={p[field] for p,ok in zip(new_points,new_report['masks'][name],strict=True) if ok}
            row.update({field+'_old':len(before) if known else None,field+'_new_panel':len(after),
                        field+'_novel':len(after-before) if known else None,field+'_cumulative':len(before|after) if known else None})
        rows.append(row)
    return rows


def compare_coverage(old_points,old_report,new_points,new_report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=coverage_rows(old_points,old_report,new_points,new_report)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'old_new_coverage.csv',rows)
    names=[*new_report['policy_names'],'union']
    fig,axes=plt.subplots(len(names),2,figsize=(10,2.4*len(names)),squeeze=False)
    for name,axs in zip(names,axes):
        known=name in old_report['masks']
        old=[p for p,ok in zip(old_points,old_report['masks'].get(name,[]),strict=known) if ok]
        before={p['root_cell'] for p in old}
        new=[p for p,ok in zip(new_points,new_report['masks'][name]) if ok and p['root_cell'] not in before]
        for ax,y,label in zip(axs,('root_z_m','root_vz_mps'),('z (m)','vz (m/s)')):
            for points,color,title in ((old,'#aaaaaa','Previous witnessed support'),(new,'#009E73','New root-state cells' if known else 'Current support; old panel untested')):
                ax.scatter([p['coordinates']['root_x_m'] for p in points],[p['coordinates'][y] for p in points],s=8,c=color,label=title,rasterized=True)
            # All rows have the same limits, including empty policy additions.
            allpoints=old_points+new_points
            for setter,key in ((ax.set_xlim,'root_x_m'),(ax.set_ylim,y)):
                values=[p['coordinates'][key] for p in allpoints];pad=max((max(values)-min(values))*.05,.01)
                setter(min(values)-pad,max(values)+pad)
            ax.set(xlabel='x (m)',ylabel=label,title=name.replace('pi_','π')+('' if known else ' — old panel untested'));ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('TRAIN | Previous + dense pilot; unchanged physical grid; observed points only')
    fig.tight_layout(rect=(0,0,1,.97))
    for ext in ('png','pdf','svg'):fig.savefig(output/f'old_new_support.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    report={'rows':rows,'baseline_scope':'previous shared-panel TRAIN support only; not all historical Tube versions',
            'physical_grid_unchanged':True,'continuous_volume_claim':False,
            'denser_sampling_is_not_by_itself_policy_improvement':True}
    write(output/'old_new_coverage.json',report)
    return report
