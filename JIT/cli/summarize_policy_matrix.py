#!/usr/bin/env python3
"""Summarize a complete frozen development panel, keeping unknowns separate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    declaration = json.loads((root/'declaration.json').read_text())
    names = declaration['policies']
    matrix, costs, hashes = [], [], {}
    for shard in declaration['shards']:
        result = root/shard/'results.json'
        rows = json.loads(result.read_text())
        hashes[str(result)] = hashlib.sha256(result.read_bytes()).hexdigest()
        costs.append(json.loads((root/shard/'status.json').read_text()))
        for row in rows:
            attempts = {a['policy']: a for a in row['attempts']}
            if set(attempts) != set(names):
                raise ValueError('Incomplete policy matrix')
            matrix.append({'context': row['snapshot_context_sha256'],
                           **{name: attempts[name]['label'] for name in names}})
    if len(matrix) != declaration['candidate_count'] or len({r['context'] for r in matrix}) != len(matrix):
        raise ValueError('Panel missing rows or contains duplicate contexts')
    output = root/'analysis'
    output.mkdir(exist_ok=False)
    with (output/'matrix.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['context', *names])
        writer.writeheader(); writer.writerows(matrix)
    summaries = []
    for i, name in enumerate(names):
        old = names[max(0, i-1)]
        summaries.append(dict(policy=name, success=sum(r[name] == 1 for r in matrix),
            failure=sum(r[name] == 0 for r in matrix), unknown=sum(r[name] is None for r in matrix),
            gained_from_previous=sum(r[old] == 0 and r[name] == 1 for r in matrix),
            lost_from_previous=sum(r[old] == 1 and r[name] == 0 for r in matrix)))
    report = dict(role='development_reused_training_states', final_test_used=False,
        candidate_count=len(matrix), policies=summaries,
        charged_interactions=sum(c['charged_interactions'] for c in costs),
        active_interactions=sum(c['active_interactions'] for c in costs), source_hashes=hashes)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(names)), [s['success'] for s in summaries])
    ax.set_xticks(range(len(names)), names, rotation=20, ha='right')
    ax.set_ylabel(f'Successful contexts / {len(matrix)}')
    ax.set_title('Fixed development panel — not held-out performance')
    fig.tight_layout()
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'policy_success.{extension}')
    lines = ['# Fixed development policy matrix', '',
        'Reused training states; unknown is not failure. No final TEST used.', '',
        '| Policy | Success | Failure | Unknown | Gained vs previous | Lost vs previous |',
        '|---|---:|---:|---:|---:|---:|']
    lines += ['| '+ ' | '.join(str(s[k]) for k in ('policy','success','failure','unknown','gained_from_previous','lost_from_previous'))+' |' for s in summaries]
    lines += ['', f"Charged interactions: {report['charged_interactions']}; active: {report['active_interactions']}.",
              '', '[Plot](policy_success.png) · [Data](matrix.csv) · [Evidence and costs](summary.json)']
    (output/'INDEX.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
