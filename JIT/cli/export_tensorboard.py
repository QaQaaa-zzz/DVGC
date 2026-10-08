#!/usr/bin/env python3
"""Export declared JIT CSV/JSON metrics to TensorBoard without simulation imports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time


def metric_rows(source):
    path = Path(source['path'])
    if path.suffix == '.jsonl':
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append({**{k:v for k,v in row.items() if k!='metrics'}, **row.get('metrics',{})})
        return rows
    if path.suffix == '.csv':
        with path.open() as stream:
            return list(csv.DictReader(stream))
    value = json.loads(path.read_text())
    return value[source['rows_key']] if source.get('rows_key') else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    from tensorboardX import SummaryWriter
    declaration = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    writers, seen, source_hashes, finished = {}, {}, {}, set()
    try:
        while True:
            for source in declaration['sources']:
                name = source['name']
                path = Path(source['path'])
                if name in finished or (args.watch and not path.is_file()):
                    continue
                if name not in writers:
                    writers[name] = SummaryWriter(str(args.output/name))
                    seen[name] = set()
                    writers[name].add_text('provenance/source', json.dumps(source, indent=2))
                    for config in source.get('configs', []):
                        writers[name].add_text('configuration/'+Path(config).name,
                            '```json\n'+Path(config).read_text()+'\n```')
                path = Path(source['path'])
                # Observe completion before reading metrics: final rows must already be flushed.
                completed = False
                if source.get('completion_status'):
                    try:
                        completed = json.loads(Path(source['completion_status']).read_text()).get('status') == 'completed'
                    except (OSError, ValueError):
                        pass
                try:
                    rows = metric_rows(source)
                except (OSError, ValueError):
                    if args.watch:
                        continue
                    raise
                source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                for ordinal, row in enumerate(rows):
                    step = int(float(row[source['step']])) if source.get('step') else ordinal
                    for tag, value in row.items():
                        if tag == source.get('step'):
                            continue
                        try:
                            number = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(number):
                            continue
                        identity = (step, tag)
                        if identity not in seen[name]:
                            writers[name].add_scalar(tag, number, step)
                            seen[name].add(identity)
                writers[name].flush()
                if completed:
                    writers.pop(name).close()
                    finished.add(name)
            receipt = dict(phase='watching' if args.watch else 'completed',
                updated_unix=time.time(), scalar_count=sum(map(len, seen.values())),
                sources=source_hashes, manifest=str(args.manifest.resolve()),
                note='Derived logs. Original step units retained; no synthetic reward or unknown labels.')
            temporary = args.output/'receipt.tmp'
            temporary.write_text(json.dumps(receipt, indent=2)+'\n')
            temporary.replace(args.output/'receipt.json')
            if not args.watch:
                break
            time.sleep(15)
    finally:
        for writer in writers.values():
            writer.close()


if __name__ == '__main__':
    main()
