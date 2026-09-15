"""Offline fixed-schedule discovery comparisons with explicit unknowns and costs.

Inputs must already have verified catalog/protocol/source receipts. This module
checks aligned complete row identities, never supplies missing negative labels,
and charges acquisition, retries and padding upfront. Candidate label costs are
atomic in declared catalog order, not a reconstruction of adaptive GPU timing.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

BASELINE_NAME = 'historical_legacy_label_scope_conservative_exclusion'


def _cost(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _validate(arm):
    points, witnesses = arm['points'], arm['witnesses']
    costs = arm['per_candidate_label_cost']
    if len(points) != len(witnesses) or len(points) != len(costs):
        raise ValueError('incomplete aligned witnesses/costs')
    identities = set()
    previous_index = -1
    for index, (point, witness, cost) in enumerate(zip(points, witnesses, costs, strict=True)):
        identity = tuple(point.get(k) for k in ('candidate_id','state_sha256','phase'))
        if any(not isinstance(v, str) or not v for v in identity) or identity in identities:
            raise ValueError('missing or duplicate complete candidate identity')
        identities.add(identity)
        if identity != tuple(witness.get(k) for k in ('candidate_id','state_sha256','phase')):
            raise ValueError('witness identity drift')
        candidate_index = witness.get('candidate_index', index)
        expected_index = point.get('candidate_index', index)
        if type(candidate_index) is not int or candidate_index != expected_index or candidate_index <= previous_index:
            raise ValueError('candidate index identity/order drift')
        previous_index = candidate_index
        label = witness.get('label')
        if 'label' not in witness or (label is not None and (type(label) is not int or label not in (0, 1))):
            raise ValueError('explicit 0/1/null witness label required')
        outcomes = witness.get('per_policy_outcomes')
        if outcomes is not None:
            if not isinstance(outcomes, dict) or not outcomes:
                raise ValueError('nonempty declared evaluator outcomes required')
            statuses = [v.get('status') for v in outcomes.values()]
            if any(v not in {'success','failure','conflict','unknown','untested'} for v in statuses):
                raise ValueError('invalid evaluator outcome')
            aggregate = 1 if 'success' in statuses else 0 if all(v=='failure' for v in statuses) else None
            if label != aggregate:
                raise ValueError('aggregate witness disagrees with evaluator outcomes')
        if not isinstance(point.get('root_cell'), str) or not point['root_cell']:
            raise ValueError('root cell identity required')
        for field in ('root_x_m', 'root_z_m', 'root_vz_mps'):
            value = point['coordinates'][field]
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
                raise ValueError('finite geometry required')
        _cost(cost, 'candidate label cost')
    acquired = _cost(arm['acquisition_interactions'], 'acquisition_interactions')
    charged = _cost(arm['charged_interactions'], 'charged_interactions')
    training = _cost(arm['training_surcharge'], 'training_surcharge')
    overhead = charged-sum(costs)
    if overhead < acquired:
        raise ValueError('charged cost omits acquisition or candidate label work')
    return overhead, charged, training


def _count(points, witnesses, baseline, count):
    selected = list(zip(points[:count], witnesses[:count], strict=True))
    cells = {p['root_cell'] for p,w in selected if w['label']==1}
    return dict(completed_candidates=count,
                positive_candidates=sum(w['label']==1 for _,w in selected),
                negative_candidates=sum(w['label']==0 for _,w in selected),
                unknown_candidates=sum(w['label'] is None for _,w in selected),
                witnessed_root_cells=len(cells), novel_root_cells=len(cells-baseline))


def summarize_arms(arms, baseline_cells, *, baseline_name=BASELINE_NAME):
    """Return complete full-schedule metrics and conservative equal-cost views.

    Arm fields: points (project_row dictionaries), aligned witnesses (existence
    entries), aligned per_candidate_label_cost (useful steps), acquisition_interactions,
    charged_interactions (all exploration), training_surcharge, trajectory_receipts.
    Global candidate_index, when noncontiguous, must occur in both aligned rows.
    Caller establishes identical declared acquisition schedules and evaluator banks.
    """
    if not arms or not baseline_name:
        raise ValueError('named baseline and at least one arm required')
    baseline = set(baseline_cells)
    validated = {name:_validate(arm) for name,arm in arms.items()}
    budgets = {scenario:min(charged+(training if scenario=='training_inclusive' else 0)
                           for _,charged,training in validated.values())
               for scenario in ('exploration_only','training_inclusive')}
    result = dict(schema='jit_checkpoint_discovery_comparison_v1', baseline_name=baseline_name,
                  baseline_root_cells=len(baseline), arm_metrics=[], curves=[], equal_budget=[], points=[],
                  complete_evaluator_matrix=False, final_test_used=False,
                  interpretation='TRAIN development; fixed declared candidate-order offline cost truncation, all non-candidate overhead upfront; no interpolated reachable region or training repayment claim',
                  common_budgets=budgets)
    novel_sets={name:{p['root_cell'] for p,w in zip(a['points'],a['witnesses']) if w['label']==1}-baseline
                for name,a in arms.items()}
    for name, arm in arms.items():
        overhead, charged, training = validated[name]
        points, witnesses = arm['points'], arm['witnesses']
        clean=[r for r in arm['trajectory_receipts'] if r.get('valid_landing') is True
               and r.get('physical_failure') is False and r.get('truncated') is False]
        peaks=[float(r['peak_root_z_m']) for r in clean if r.get('peak_root_z_m') is not None
               and math.isfinite(float(r['peak_root_z_m']))]
        others=set().union(*(cells for other,cells in novel_sets.items() if other!=name))
        result['arm_metrics'].append(dict(arm=name, **_count(points,witnesses,baseline,len(points)),
            scope='full_actual_fixed_schedule', acquisition_interactions=arm['acquisition_interactions'],
            charged_interactions=charged, training_surcharge=training, training_inclusive_interactions=charged+training,
            upfront_exploration_overhead=overhead, candidate_label_interactions=sum(arm['per_candidate_label_cost']),
            exclusive_novel_root_cells=len(novel_sets[name]-others), trajectory_count=len(arm['trajectory_receipts']),
            clean_forward_landings=len(clean), minimum_clean_forward_peak_root_z_m=min(peaks) if peaks else None))
        for index,(point,witness) in enumerate(zip(points,witnesses,strict=True)):
            result['points'].append(dict(arm=name,candidate_index=witness.get('candidate_index',index),
                **{k:point[k] for k in ('candidate_id','state_sha256','phase','root_cell')},
                **point['coordinates'],label=witness['label'],
                novel_positive=witness['label']==1 and point['root_cell'] not in baseline))
        for scenario,budget in budgets.items():
            surcharge=training if scenario=='training_inclusive' else 0
            spent=overhead+surcharge
            curve=[dict(arm=name,scenario=scenario,interactions=0,**_count(points,witnesses,baseline,0))]
            curve.append(dict(arm=name,scenario=scenario,interactions=spent,**_count(points,witnesses,baseline,0)))
            for index,cost in enumerate(arm['per_candidate_label_cost'],1):
                spent+=cost
                curve.append(dict(arm=name,scenario=scenario,interactions=spent,**_count(points,witnesses,baseline,index)))
            result['curves'].extend(curve)
            affordable=[row for row in curve if row['interactions']<=budget]
            matched=affordable[-1]
            result['equal_budget'].append(dict(matched,common_budget=budget,
                budget_below_training_surcharge=budget<surcharge,
                training_surcharge=surcharge, full_schedule_interactions=charged+surcharge,
                scope='conservative_offline_declared_order'))
    return result


def _csv(path, rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def render(summary, output_dir):
    """Export every row and two complete PNG/PDF/SVG comparison figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    for filename,key in [('all_points.csv','points'),('all_cost_curves.csv','curves'),
                         ('metrics.csv','arm_metrics'),('equal_budget.csv','equal_budget')]:
        _csv(output/filename,summary[key])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    names=[r['arm'] for r in summary['arm_metrics']]
    phases=sorted({r['phase'] for r in summary['points']}) or ['no observed points']
    fig,axes=plt.subplots(len(phases),2,figsize=(12,4*len(phases)),squeeze=False)
    for row,phase in enumerate(phases):
        for col,y in enumerate(('root_z_m','root_vz_mps')):
            ax=axes[row,col]
            for index,name in enumerate(names):
                for label,marker in ((1,'o'),(0,'x'),(None,'+')):
                    points=[p for p in summary['points'] if p['arm']==name and p['phase']==phase and p['label']==label]
                    if points:
                        ax.scatter([p['root_x_m'] for p in points],[p[y] for p in points],s=14,
                            color=f'C{index%10}',marker=marker,alpha=.65,
                            label=f'{name}: '+{1:'witness',0:'no bank witness',None:'unknown'}[label])
            ax.set(xlabel='root x (m)',ylabel='root z (m)' if col==0 else 'root vz (m/s)',title=phase)
            if ax.collections:ax.legend(fontsize=7)
    fig.suptitle('Observed candidate geometry; discrete evidence, no hull')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(output/f'geometry.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,scenario in zip(axes,('exploration_only','training_inclusive')):
        for name in names:
            rows=[r for r in summary['curves'] if r['arm']==name and r['scenario']==scenario]
            ax.step([r['interactions'] for r in rows],[r['novel_root_cells'] for r in rows],where='post',label=name)
        ax.axvline(summary['common_budgets'][scenario],linestyle='--',color='black',label='common budget')
        ax.set(xlabel='Charged interactions (offline declared order)',ylabel='Novel witnessed root cells',title=scenario.replace('_',' '))
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(output/f'coverage_cost.{ext}',dpi=160)
    plt.close(fig)
    manifest=dict(schema='jit_checkpoint_discovery_figures_v1',point_count=len(summary['points']),
                  files=sorted(p.name for p in output.iterdir() if p.is_file()),baseline_name=summary['baseline_name'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
