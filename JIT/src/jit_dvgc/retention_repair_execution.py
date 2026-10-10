"""Finite sequential D0 execution, charged padding, no training or promotion."""
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from .retention_repair import audit,read,write,load_requests,validate_budget

PY='/home/qy/mujoco_playground/.venv/bin/python'

def verify(batch,root):
    b=batch;out=root/b['name'];status=read(out/'status.json')
    if status['phase']!='completed':raise ValueError('incomplete batch')
    spec=read(b['spec']);requests,_=load_requests(spec)
    with np.load(out/'prefixes.npz') as f,np.load(spec['initial_state_bank']) as initial:
        mask=f['prefix_mask'];np.testing.assert_allclose(f['initial_qpos'],initial['qpos'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(f['initial_qvel'],initial['qvel'],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(f['requested_delta'][mask],requests[mask])
        if np.any(f['requested_delta'][~mask]):raise ValueError('inactive lane requests')
        if not np.all(f['finite'][mask]):raise ValueError('nonfinite active trajectory')
        if np.any(mask[1:]&~mask[:-1]):raise ValueError('revived terminal lane')
        end=mask.sum(0)-1
        success=np.any(f['success']&mask,axis=0)
        np.testing.assert_array_equal(success,f['end_code'][end,np.arange(b['capacity'])]==12)
        result={'phase':'passed','successes':int(success[:b['scored']].sum()),'scored':b['scored'],
            'charged_interactions':status['charged_interactions'],'active_interactions':status['active_interactions'],
            'padding_interactions':status['padding_interactions'],'request_elementwise_match':True}
    write(out/'verification.json',result);return result

def run(plan_path):
    audit(plan_path);p=read(plan_path);maximum=validate_budget(p);root=Path(p['output']);code=Path(p['code'])
    if read(root/'status.json')['phase']!='prepared':raise ValueError('no implicit resume/retry')
    env=dict(os.environ,PYTHONPATH=str(code/'JIT/src'),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
             JIT_AUTO_PUBLISH='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',JAX_COMPILATION_CACHE_DIR=str(root/'jax_cache'))
    start=time.monotonic();charged=0;done=0;child=None
    def status(phase,**extra):
        write(root/'status.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),stage='D0',
            charged_interactions=charged,maximum_planned_interactions=maximum,physical_hard_cap=p['budget']['physical_transitions'],
            completed_batches=done,total_batches=len(p['batches']),training_transitions=0,**extra))
    write(root/'ACTIVE_RUN.json',{'execution':str(root/'status.json'),'lineage':str(root/'status.json')})
    try:
        with (root/'watcher.log').open('x') as log:
            w=subprocess.Popen([PY,str(code/'JIT/cli/watch_run_errors.py'),'--active-run',str(root/'ACTIVE_RUN.json'),'--state-dir',str(root/'notifications')],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'launch.json',{'pid':os.getpid(),'watcher_pid':w.pid,'started_unix':time.time(),'plan':str(plan_path),'execute':True})
        for b in p['batches']:
            remaining=p['budget']['wall_seconds']-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('D0 wall budget exhausted')
            cost=b['capacity']*b['horizon']
            if charged+cost>p['budget']['physical_transitions']:raise ValueError('budget before batch')
            if (root/b['name']).exists():raise ValueError('existing output: no automatic overwrite or retry')
            status('running',batch=b['name']);charged+=cost # reserve even if child fails
            with (root/(b['name']+'.log')).open('x') as log:
                child=subprocess.Popen([PY,str(code/'JIT/cli/run_pulse_exploration.py'),'--mode','collect','--spec',b['spec'],'--output',str(root/b['name'])],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
                status('running',batch=b['name'],child_pid=child.pid)
                try:rc=child.wait(timeout=max(.1,remaining))
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise TimeoutError('D0 child exceeded remaining wall budget')
            if rc:raise RuntimeError(f"{b['name']} exit {rc}; no automatic retry")
            verify(b,root);done+=1;status('running',batch=b['name'])
        status('running',batch='D0b_and_report')
        # CPU fixed-observation diagnostics use original and new saved tapes; no physics/update.
        from .retention_repair_report import report
        subprocess.run([PY,str(code/'JIT/cli/run_retention_repair_pilot.py'),'report','--plan',str(plan_path)],cwd=code,
            env={**env,'JAX_PLATFORMS':'cpu'},check=True,timeout=max(1,p['budget']['wall_seconds']-(time.monotonic()-start)))
        status('completed',evaluation_status='D0_complete_D0b_reported',wall_seconds=time.monotonic()-start)
    except BaseException as error:
        status('failed',error=repr(error),wall_seconds=time.monotonic()-start);raise
