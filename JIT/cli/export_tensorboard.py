#!/usr/bin/env python3
"""Export declared JIT CSV/JSON metrics to TensorBoard without simulation imports.

Each source has a unique ``name``. Optional ``run`` groups sources into one
TensorBoard run (default: name); ``step_offset`` adds a nonnegative integer to
the original step (default: zero). Optional ``step_offset_receipt`` contains
``path`` and ``field`` naming a JSON integer that must equal the declared offset;
watch mode defers a source until this receipt exists. Declare future sources
before watch starts.
"""
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
    run_sources = {}
    names = set()
    for source in declaration['sources']:
        name = source['name']
        if name in names:
            raise ValueError(f'Duplicate source name: {name}')
        names.add(name)
        offset = source.get('step_offset', 0)
        if type(offset) is not int or offset < 0:
            raise ValueError(f'step_offset must be a nonnegative integer: {name}')
        run_sources.setdefault(source.get('run', name), set()).add(name)
    args.output.mkdir(parents=True, exist_ok=False)
    writers, seen, source_hashes, finished = {}, {}, {}, set()
    run_seen = {}
    try:
        while True:
            for source in declaration['sources']:
                name = source['name']
                run = source.get('run', name)
                path = Path(source['path'])
                if name in finished or (args.watch and not path.is_file()):
                    continue
                if source.get('step_offset_receipt'):
                    offset_receipt = source['step_offset_receipt']
                    try:
                        receipt_data = json.loads(Path(offset_receipt['path']).read_text())
                    except FileNotFoundError:
                        if args.watch:
                            continue
                        raise
                    actual_offset = receipt_data.get(offset_receipt['field'])
                    if (type(actual_offset) is not int or
                            actual_offset != source.get('step_offset', 0)):
                        raise ValueError(f'step_offset receipt mismatch or noninteger for {name}: '
                                         f'{actual_offset!r} != {source.get("step_offset", 0)}')
                if run not in writers:
                    writers[run] = SummaryWriter(str(args.output/run))
                    run_seen[run] = {}
                if name not in seen:
                    seen[name] = set()
                    provenance_tag = ('provenance/source' if len(run_sources[run]) == 1
                                      else 'provenance/source/' + name)
                    writers[run].add_text(provenance_tag, json.dumps(source, indent=2))
                    for config in source.get('configs', []):
                        config_tag = ('configuration/' if len(run_sources[run]) == 1
                                      else 'configuration/' + name + '/')
                        writers[run].add_text(config_tag+Path(config).name,
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
                    step += source.get('step_offset', 0)
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
                            owner = run_seen[run].get(identity)
                            if owner is not None and owner != name:
                                raise ValueError(f'Scalar overlap in run {run}: {identity}, '
                                                 f'sources {owner} and {name}')
                            writers[run].add_scalar(tag, number, step)
                            seen[name].add(identity)
                            run_seen[run][identity] = name
                writers[run].flush()
                if completed:
                    finished.add(name)
                    if run_sources[run] <= finished:
                        writers.pop(run).close()
            receipt = dict(phase='watching' if args.watch else 'completed',
                updated_unix=time.time(), scalar_count=sum(map(len, seen.values())),
                sources=source_hashes, manifest=str(args.manifest.resolve()),
                note='Derived logs. Step = original step + declared step_offset (default 0); '
                     'original step units retained. No synthetic reward or unknown labels.')
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
