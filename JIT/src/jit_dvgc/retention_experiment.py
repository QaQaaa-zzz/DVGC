"""Development retention/gain evaluation and inspectable training plots."""
from pathlib import Path
import csv
from .jump_evidence_validation import read, write


def export_training(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import json
    run_dir = Path(run_dir)
    for stem in ('metrics', 'episode_metrics'):
        path = run_dir / (stem + '.jsonl')
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        keys = sorted({k for row in rows for k in row['metrics']})
        plot_dir = run_dir / 'process' / stem
        plot_dir.mkdir(parents=True, exist_ok=True)
        with (plot_dir / 'data.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['training_transitions'] + keys)
            writer.writeheader()
            writer.writerows({'training_transitions': r['training_transitions'], **r['metrics']} for r in rows)
        index = []
        for i, key in enumerate(keys):
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot([r['training_transitions'] for r in rows], [r['metrics'].get(key, float('nan')) for r in rows])
            ax.set(xlabel='PPO transitions', ylabel=key)
            ax.grid(alpha=.2)
            fig.tight_layout()
            for ext in ('png', 'pdf', 'svg'):
                fig.savefig(plot_dir / f'{i:03d}.{ext}')
            plt.close(fig)
            index.append(f'- {key}: [{i:03d}.png]({i:03d}.png)')
        (plot_dir / 'INDEX.md').write_text('\n'.join(index) + '\n')


def evaluate(spec_path, output):
    from .unified_policy_freeze import freeze_development_checkpoint
    from .probe_bank import load_probe_bank, lock_probe_bank
    from .pulse_exploration_runtime import evaluate as suffix_evaluate
    from .policy_retention import paired_counts
    spec, output = read(spec_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bank = load_probe_bank(Path(spec['bank']))
    source = next(m for m in bank['members'] if m['name'] == spec['source'])
    members = [dict(frozen_policy=source['frozen_policy'], roles=['evaluator'])]
    order = [spec['source']]
    for arm in spec['arms']:
        config = read(arm['config'])
        run = Path(arm['run_dir'])
        export_training(run)
        for step in spec['checkpoints']:
            name = arm['name'] + '_' + str(step)
            frozen = output / 'frozen' / name
            freeze_development_checkpoint(frozen, config_path=Path(arm['config']),
                checkpoint=run / 'checkpoints' / f'transition_{step}', name=name)
            members.append(dict(frozen_policy=str(frozen / 'frozen_unified_policy.json'), roles=['evaluator']))
            order.append(name)
    bank_path = output / 'bank.json'
    lock_probe_bank(dict(version='retention_development', task=bank['task'],
        max_ticks=spec['horizon'], label_interaction_budget=spec['budget'],
        max_candidates_per_process=bank['max_candidates_per_process'], members=members), bank_path)
    suffix_evaluate(dict(bank=str(bank_path), candidates=spec['candidates'], horizon=spec['horizon'],
        budget=spec['budget'], order=order, proposer=spec['source'], full_matrix=True), output / 'suffixes')
    rows = read(output / 'suffixes/results.json')
    training_cost = {}
    for arm in spec['arms']:
        report = read(Path(arm['run_dir']) / 'formal_report.json')
        training_cost[arm['name']] = dict(
            ppo=report['completed_training_transitions'],
            checkpoint_panels=report['train_panel_interactions'])
    suffix_cost = read(output / 'suffixes/status.json')
    write(output / 'cost.json', dict(training=training_cost, suffix=suffix_cost,
        total_new_interactions=sum(sum(c.values()) for c in training_cost.values()) + suffix_cost['charged_interactions'],
        inherited_data_and_pi0_cost='not included; reused frozen historical artifacts'))
    summary = {}
    for name in order[1:]:
        summary[name] = {}
        for role in ('old_support', 'pending'):
            selected = [r for r in rows if r['evaluation_group'] == role]
            before = [next(a['label'] for a in r['attempts'] if a['policy'] == spec['source']) for r in selected]
            after = [next(a['label'] for a in r['attempts'] if a['policy'] == name) for r in selected]
            summary[name][role] = paired_counts(before, after)
    write(output / 'comparison.json', {'role': 'TRAIN development mechanism evaluation',
        'final_test_used': False, 'summary': summary, 'cost': read(output / 'suffixes/status.json')})
    with (output / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['policy', 'group', 'retained', 'lost', 'gained',
            'unchanged_negative', 'unknown', 'old_positive'])
        writer.writeheader()
        for name, groups in summary.items():
            for group, counts in groups.items():
                writer.writerow(dict(policy=name, group=group, **counts))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = list(summary)
    for key in ('retained', 'lost', 'unknown'):
        axes[0].plot(names, [summary[n]['old_support'][key] for n in names], marker='o', label=key)
    axes[1].bar(names, [summary[n]['pending']['gained'] for n in names])
    axes[0].set_ylabel('Old-support paired states')
    axes[0].legend()
    axes[1].set_ylabel('Pending states: source failure to checkpoint success')
    for ax in axes:
        ax.tick_params(axis='x', rotation=35)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / ('comparison.' + ext))
    plt.close(fig)
    (output / 'INDEX.md').write_text('# Retention and gain\n\n'
        'TRAIN development states; not independent holdout or continuous safety evidence.\n\n'
        '[Paired counts](comparison.csv), [full context-bound attempts](suffixes/results.json), '
        '[cost](suffixes/status.json). Training process figures are in each arm/process/.\n')
    return summary
