"""Optional wall-time events; no device synchronization or inferred GPU timing."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


def record_event(path, event, started_unix, elapsed_seconds, **fields):
    if not path:
        return
    row = dict(schema='jit_performance_event_v1', event=event,
        round=None, root=None, stage=None, pid=os.getpid(), backend=None,
        batch_size=None, started_unix=started_unix,
        finished_unix=started_unix+elapsed_seconds, elapsed_seconds=elapsed_seconds,
        input_bytes=None, output_bytes=None, hash_bytes=None, cache_hit=None,
        gpu_peak_memory_bytes=None, status='completed')
    row.update(fields)
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    # One O_APPEND write per small event; inter-process writers do not share buffers.
    payload=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    try:
        if os.write(fd,payload)!=len(payload):raise OSError('short performance event write')
    finally:os.close(fd)


@contextmanager
def measure(event, *, path=None, **fields):
    path=path or os.environ.get('JIT_PERFORMANCE_FILE')
    if not path:
        yield fields
        return
    started=time.time();clock=time.perf_counter()
    try:
        yield fields
    except BaseException:
        fields['status']='failed'
        raise
    finally:
        record_event(path,event,started,time.perf_counter()-clock,**fields)


def interval_union(intervals):
    ordered=sorted((float(a),float(b)) for a,b in intervals)
    if any(b<a for a,b in ordered):raise ValueError('negative time interval')
    merged=[]
    for a,b in ordered:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return merged


def audit_round(root):
    """Use saved wall timestamps, not a sum of potentially parallel child times."""
    root=Path(root);read=lambda p:json.loads(p.read_text())
    status=read(root/'status.json');production=read(root/'production.json')
    start=read(root/'started.json')['started_unix'];end=start+status['wall_seconds']
    stages=[]
    for p in sorted((root/'execution').glob('*/status.json')):
        for s in read(p).get('stages',[]):
            if 'finished_unix' in s:
                stages.append(dict(stage=s['name'],pid=s.get('pid'),
                    start=s['started_unix'],end=s['finished_unix'],status=s['phase']))
    union=interval_union((max(start,s['start']),min(end,s['end'])) for s in stages
                         if s['end']>=start and s['start']<=end)
    gaps=[];cursor=start
    for a,b in union:
        if a>cursor:gaps.append([cursor,a])
        cursor=b
    if cursor<end:gaps.append([cursor,end])
    return dict(schema='jit_round_wall_audit_v1',round=root.name,
        implementation_commit=production['implementation_commit'],
        repository=production['repository'],wall_seconds=end-start,
        child_interval_union_seconds=sum(b-a for a,b in union),
        outside_child_seconds=sum(b-a for a,b in gaps),child_intervals=union,
        outside_intervals=gaps,stages=stages,
        attribution='Uninstrumented historical gaps have unknown internal causes.')


def timed(event):
    """Measure a host operation without adding device completion barriers."""
    from functools import wraps
    def decorate(function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            with measure(event):return function(*args,**kwargs)
        return wrapped
    return decorate
