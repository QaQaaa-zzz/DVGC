"""Bound peak memory with sequential, provenance-locked evaluation processes.

A shard is only an execution partition: candidate order, complete snapshots,
policy identity and original nonterminal RNG lanes are retained. Each child
uses canonical restoration and its own active/padded interaction accounting.
"""
from pathlib import Path
import signal
import time
from .exploration_loop import read, write
from .probe_bank import _file_sha
from .gated_execution import run_gated_plan


def evaluation_shards(rows, size):
    if type(size) is not int or size < 1:
        raise ValueError('evaluation_batch_size must be a positive integer')
    original_lanes = []
    count = 0
    for row in rows:
        original_lanes.append(None if row.get('prefix_terminal', False) else count)
        count += not row.get('prefix_terminal', False)
    return [(rows[start:start+size],
             [i for i in original_lanes[start:start+size] if i is not None], count)
            for start in range(0, len(rows), size)]


def _identity(row):
    return {k:v for k,v in row.items() if k not in ('label','witness','attempts')}


def evaluate_batched(spec, output):
    if len(spec['order']) != 1 or spec.get('reuse_results'):
        raise ValueError('process-batched evaluation requires one policy and fresh labels')
    rows = read(spec['candidates'])
    batches = evaluation_shards(rows, spec['evaluation_batch_size'])
    if len(rows)*spec['horizon'] > spec['budget']:
        raise ValueError('insufficient total evaluation reservation')
    output = Path(output).resolve();output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); merged = []; receipts = []
    charged = active = padding = peak = 0
    reused_interactions = 0
    incomplete_reservation = 0
    active_shard = None
    def status(phase, **extra):
        write(output/'status.json', dict(phase=phase, charged_interactions=charged,
            reused_interactions=reused_interactions,
            active_interactions=active, padding_interactions=padding,
            evaluation_batch_size=spec['evaluation_batch_size'], completed_shards=len(receipts),
            total_shards=len(batches), peak_child_rss_kib=peak,
            wall_seconds=time.monotonic()-start, **extra))
    def stop(signum, frame):
        raise SystemExit(f'evaluation supervisor received signal {signum}')
    previous_handler = signal.signal(signal.SIGTERM, stop)
    try:
        status('running')
        for ordinal, (batch, lanes, full_count) in enumerate(batches):
            directory=output/f'shard_{ordinal:04d}';directory.mkdir()
            candidates=directory/'candidates.json';write(candidates,batch)
            child={k:v for k,v in spec.items() if k not in ('evaluation_batch_size','resume_evaluation_root')}
            child.update(candidates=str(candidates),budget=len(batch)*spec['horizon'],
                         suffix_rng_count=full_count,suffix_rng_indices=lanes)
            child_spec=directory/'spec.json';write(child_spec,child)
            locks={**spec['input_files'],str(candidates):_file_sha(candidates),str(child_spec):_file_sha(child_spec)}
            plan=dict(schema='jit_gated_plan_v1',gate=spec['gate'],input_files=locks,
                source_locks=spec['source_locks'],max_interactions=max(1,child['budget']),
                wait_timeout_seconds=spec['wait_timeout_seconds'],stages=[dict(name='evaluate',
                argv=[spec['python'],'JIT/cli/run_pulse_exploration.py','--mode','evaluate',
                      '--spec',str(child_spec),'--output',str(directory/'evaluation')],
                cwd=spec['repo'],env=dict(JAX_PLATFORMS='cuda,cpu',CUDA_VISIBLE_DEVICES='0',
                    PYTHONPATH=str(Path(spec['repo'])/'JIT/src'),
                    XLA_PYTHON_CLIENT_PREALLOCATE='false',JIT_AUTO_PUBLISH='0'),
                timeout_seconds=spec['stage_timeout_seconds'],max_interactions=child['budget'])])
            path=directory/'plan.json';write(path,plan)
            status('running',active_shard=ordinal,reserved_attempt_interactions=child['budget'])
            active_shard=ordinal
            incomplete_reservation=0
            prior = (Path(spec['resume_evaluation_root'])/f'shard_{ordinal:04d}'
                     if spec.get('resume_evaluation_root') else None)
            reused = bool(prior is not None and (prior/'evaluation/status.json').exists()
                          and read(prior/'evaluation/status.json')['phase']=='completed')
            if reused:
                from .current_policy_iteration import verify_stage_reuse
                for filename in ('spec.json','evaluation/status.json','evaluation/results.json'):
                    artifact=(prior/filename).resolve()
                    if spec['input_files'].get(str(artifact)) != _file_sha(artifact):
                        raise ValueError('completed shard reuse requires matching artifact locks')
                verify_stage_reuse(read(prior/'spec.json'),child)
                receipt=read(prior/'evaluation/status.json');evaluated=read(prior/'evaluation/results.json')
                write(directory/'reuse.json',dict(previous=str(prior),new_interactions=0,
                    receipt_sha256=_file_sha(prior/'evaluation/status.json')))
                # Keep canonical artifacts discoverable across repeated resumes.
                (directory/'evaluation').symlink_to((prior/'evaluation').resolve(),target_is_directory=True)
            else:
                incomplete_reservation=child['budget']
                result=run_gated_plan(path,directory/'execution',wait=True,poll_seconds=5)
                incomplete_reservation=result.get('reserved_interactions',incomplete_reservation)
                if result['phase']!='completed':
                    raise RuntimeError(f'evaluation shard {ordinal} stopped: {result["phase"]}')
                receipt=read(directory/'evaluation/status.json');evaluated=read(directory/'evaluation/results.json')
            if (receipt['phase']!='completed' or len(evaluated)!=len(batch)
                    or any(_identity(a)!=_identity(b) for a,b in zip(batch,evaluated))):
                raise ValueError('evaluation shard candidate identity/order drift')
            c=receipt['charged_interactions'];a=receipt['active_interactions'];p=receipt['padding_interactions']
            if not (0<=a<=c<=child['budget'] and p==c-a):
                raise ValueError('evaluation shard cost mismatch')
            if reused:
                reused_interactions+=c
            else:
                charged+=c;active+=a;padding+=p
            peak=max(peak,receipt.get('peak_rss_kib',0))
            incomplete_reservation=0
            active_shard=None
            merged.extend(evaluated);receipts.append(dict(shard=ordinal,receipt=receipt))
            write(output/'shards.json',receipts);status('running')
        write(output/'results.json',merged)
        status('completed',successes=sum(r['label']==1 for r in merged))
    except BaseException as exc:
        status('error',error=f'{type(exc).__name__}: {exc}',no_automatic_retry=True,
               active_shard=active_shard,incomplete_reserved_interactions=incomplete_reservation,
               accounting='completed_shards_only_with_incomplete_reservation')
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
