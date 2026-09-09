"""Resumable bounded 5 cm pilot with a measured execution-backend decision."""
from __future__ import annotations
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .jump_evidence_validation import read, write, file_sha
from .policy_comparison import bundle, verify_plan

DEFAULT_OUTPUT = 'JIT/runs/dense_tube/pi0_5cm_pilot_v2'
DEFAULT_BASELINE = 'JIT/runs/policy_comparison/expanded_pi0_pi3_20260907'


def choose_backend(serial, device):
    """Endpoint/step equality is mandatory; speed never overrides it."""
    keys = ('candidate_id', 'state_sha256', 'label', 'outcome_class',
            'environment_interactions', 'final_active_phase', 'physical_failure', 'timeout')
    if not device:
        return {'backend': 'serial', 'reason': 'device process failed; see benchmark process.log and worker_failure.json'}
    if serial['identities'] != device['identities']:
        return {'backend': 'serial', 'reason': 'benchmark policy/catalog identity mismatch'}
    if len(serial['labels']) != len(device['labels']) or any(
        any(a.get(k) != b.get(k) for k in keys) for a,b in zip(serial['labels'],device['labels'])):
        return {'backend': 'serial', 'reason': 'execution comparison differed; no new snapshot replay investigation'}
    return {'backend': 'device' if device['seconds'] < serial['seconds'] else 'serial',
            'reason': 'same small-panel endpoints/steps; use lower measured labeling time',
            'serial_seconds': serial['seconds'], 'device_seconds': device['seconds'],
            'measured_speedup': serial['seconds']/max(device['seconds'],1.e-9)}


def run(repo, output, *, baseline=None, gpu='0', budget=2_000_000, proposer='pi_0', profile=None):
    repo,output=Path(repo).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(repo,output,baseline=baseline,gpu=gpu,budget=budget,proposer=proposer,profile=profile)


def _run(repo,output,*,baseline,gpu,budget,proposer='pi_0', profile=None):
    if proposer not in {'pi_0','pi_1','pi_2','pi_3'} and (profile or {}).get('version')!='iterative_discovery_v1': raise ValueError('unknown proposer')
    from .frontier_exploration import validate_profile
    validate_profile(profile)
    request={'repo':str(repo),'baseline':str(Path(baseline or repo/DEFAULT_BASELINE).resolve()),
             'budget':int(budget),'batch_size':8,'shard_size':128,'spacing_m':0.05,
             'sampling_profile':'16_trajectories_train_v2','proposer':proposer}
    if profile is not None: request['frontier_profile']=profile
    if (output/'request.json').exists() and read(output/'request.json') != request:
        raise ValueError('request changed; use a new output directory')
    write(output/'request.json',request)
    report={'status':'running','training_transitions':0,'final_test_used':False}
    cli=repo/'JIT/cli/run_dense_tube.py'

    def account():
        items=[]
        for p in sorted((output/'tasks').glob('*/attempt_*/reservation.json')):
            reservation=read(p);done=p.parent/'completion.json'
            known=read(done).get('environment_interactions') if done.exists() else None
            measured=type(known) is int and 0<=known<=reservation['maximum_interactions']
            items.append({'attempt':str(p.parent),'charged':known if measured else reservation['maximum_interactions'],'measured':measured})
        return {'attempts':items,'charged_interactions':sum(i['charged'] for i in items),
                'budget':budget,'includes_benchmarks_acquisition_padding_retries':True,
                'historical_bootstrap_training_cost_included':False}

    def task(name,kind,args=(),maximum=0,gpu_job=False,optional=False):
        if (output/'plan.json').exists(): verify_plan(output/'plan.json')
        root=output/'tasks'/name
        reservation={'kind':kind,'args':list(args),'maximum_interactions':maximum,
                     'request':request}
        for p in sorted(root.glob('attempt_*')):
            if read(p/'reservation.json') != reservation: raise ValueError('task request drift')
            if (p/'completion.json').exists():
                done=read(p/'completion.json')
                for f,sha in done['artifacts'].items():
                    if file_sha(p/f)!=sha: raise ValueError('completed task output changed')
                print(f'[dense] reuse {name}',flush=True)
                return p
        if account()['charged_interactions']+maximum>budget:
            raise RuntimeError('pilot budget exhausted including unknown/failed attempts')
        p=root/f'attempt_{len(list(root.glob("attempt_*"))):04d}'
        write(p/'reservation.json',reservation)
        env=dict(os.environ,PYTHONPATH=str(repo/'JIT/src')+os.pathsep+os.environ.get('PYTHONPATH',''),
                 CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda,cpu' if gpu_job else 'cpu',
                 XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',
                 JAX_COMPILATION_CACHE_DIR=str(output/'compilation_cache'))
        command=[sys.executable,str(cli),'--worker',kind,'--output-dir',str(output),'--destination',str(p),*args]
        start=time.perf_counter();print(f'[dense] {name}; log={p / "process.log"}',flush=True)
        with (p/'process.log').open('w') as log:
            process=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while True:
                    try: code=process.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        print(f'[dense] {name} elapsed={time.perf_counter()-start:.0f}s; log={p / "process.log"}',flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill();process.wait()
        write(p/'exit.json',{'returncode':code,'wall_seconds':time.perf_counter()-start})
        if code:
            if optional: return None
            raise RuntimeError(f'{name} failed; preserved {p}')
        result=read(p/'worker_report.json');cost=result.get('environment_interactions',0)
        if type(cost) is not int or not 0<=cost<=maximum: raise ValueError('task cost exceeds reservation')
        artifacts={str(f.relative_to(p)):file_sha(f) for f in p.rglob('*') if f.is_file() and f.name not in {'completion.json','process.log'}}
        if result.get('figures'):
            artifacts.update({os.path.relpath(f,p):file_sha(f) for f in Path(result['figures']).rglob('*') if f.is_file()})
        write(p/'completion.json',{'artifacts':artifacts,'environment_interactions':cost})
        write(output/'cost_ledger.json',account())
        return p

    try:
        task('prepare','prepare')
        plan=verify_plan(output/'plan.json')
        decision_path=output/'backend_decision.json'
        if profile and profile['serial_only'] and not decision_path.exists():
            from .evidence_integrity import canonical_sha256
            decision={'plan_sha256':plan['plan_sha256'],'policies':{n:{'backend':'serial','reason':'locked production serial; no repeated device benchmark'} for n in plan['names']}}
            decision['decision_sha256']=canonical_sha256(decision);write(decision_path,decision)
        if not decision_path.exists():
            decisions={}
            for name in plan['names']:
                serial=task('benchmark_'+name+'_serial','benchmark',['--policy',name,'--backend','serial'],maximum=16*plan['horizon'],gpu_job=True)
                device=task('benchmark_'+name+'_device','benchmark',['--policy',name,'--backend','device'],maximum=16*plan['horizon'],gpu_job=True,optional=True)
                decisions[name]=choose_backend(read(serial/'worker_report.json'),read(device/'worker_report.json') if device else None)
            from .evidence_integrity import canonical_sha256
            decision={'plan_sha256':plan['plan_sha256'],'policies':decisions}
            decision['decision_sha256']=canonical_sha256(decision);write(decision_path,decision)
        else:
            from .jump_evidence_validation import verify_hash
            verify_hash(read(decision_path),'decision_sha256')
            if read(decision_path)['plan_sha256']!=plan['plan_sha256']:raise ValueError('backend decision plan drift')
        decisions=read(decision_path)['policies']
        if profile and any(d['backend']!='serial' for d in decisions.values()): raise ValueError('frontier requires locked serial backend')
        print('[dense] backends='+str(decisions),flush=True)
        acquired=task('acquire','acquire',maximum=plan.get('acquisition_ceiling',8000),gpu_job=True)
        catalog=acquired/'result/catalog.json'
        prepared=task('project','project',['--catalog',str(catalog)])
        panel=read(prepared/'worker_report.json')
        n=panel['candidate_count'];shards=(n+127)//128
        if n == 0:
            from .evidence_integrity import canonical_sha256
            empty={'resolution':plan['physical_resolution'],'candidate_count':0,
                   'status':'completed_empty','no_success_witness_under_declared_bank':True}
            empty['report_sha256']=canonical_sha256(empty)
            write(output/'figures/summary.json',empty)
            write(output/'analysis_inputs.json',{'projected':str(prepared/'projected.json'),
                  'merged':{},'catalog':str(catalog)})
            report.update(status='completed_empty',candidate_count=0,proposer=proposer,
                          backend_decisions=decisions,scope='no retained arrivals under declared acquisition')
            return report
        selected={}
        for name in plan['names']:
            directories=[]
            for i in range(shards):
                base,remainder=divmod(n,shards)
                lo=i*base+min(i,remainder);hi=lo+base+int(i<remainder)
                p=task(f'label_{name}_{i:03d}','label',['--policy',name,'--backend',decisions[name]['backend'],
                       '--catalog',str(catalog),'--shard-index',str(i),'--shard-count',str(shards)],
                       maximum=(hi-lo)*plan['horizon'],gpu_job=True)
                directories.append(str(p/'result'))
            # The manifest is deterministic and cannot silently change on resume.
            path=output/f'{name}_shards.json'
            if path.exists() and read(path)!=directories:raise ValueError('shard selection drift')
            write(path,directories)
            selected[name]=task('merge_'+name,'merge',['--policy',name,'--catalog',str(catalog)])
        manifest={'projected':str(prepared/'projected.json'),
                  'merged':{n:str(p/'result') for n,p in selected.items()},'catalog':str(catalog)}
        path=output/'analysis_inputs.json'
        if path.exists() and read(path)!=manifest: raise ValueError('analysis inputs drift')
        write(path,manifest)
        task('figures','analyze')
        report.update(status='completed',candidate_count=n,backend_decisions=decisions,
                      figures=str(output/'figures'),sampling_spacing_m=0.05,
                      scope=f'TRAIN {proposer} dense pilot; compare with discovery supervisor',proposer=proposer)
    except BaseException as exc:
        report.update(status='engineering_error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        write(output/'cost_ledger.json',account());report['charged_interactions']=account()['charged_interactions']
        write(output/'summary.json',report)
        print(f"[dense] {report['status']}\nReturn this file: {bundle(output)}",flush=True)
    return report
