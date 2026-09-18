"""Finite, paired full-episode policy comparisons with bounded GPU batches."""
from collections import Counter
import csv
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .rsi_comparison import read, write, file_sha, episode_results


def batch_plan(episodes, batch_size, seed):
    if any(type(v) is not int or v <= 0 for v in (episodes, batch_size)):
        raise ValueError('positive integer episodes and batch size required')
    if type(seed) is not int or not 0 <= seed < 2**32 - episodes:
        raise ValueError('seed range invalid')
    return [dict(index=i,offset=start,count=min(batch_size,episodes-start),seed=seed+i)
            for i,start in enumerate(range(0,episodes,batch_size))]


def prepare(previous, output, *, episodes=10000, batch_size=256, seed=9182702):
    previous,output=Path(previous).resolve(),Path(output).resolve()
    source=read(previous/'spec.json')
    if read(previous/'status.json')['phase'] != 'completed':
        raise ValueError('source comparison incomplete')
    plan=batch_plan(episodes,batch_size,seed)
    inputs={str(previous/'spec.json'):file_sha(previous/'spec.json')}
    methods=[]
    common=None
    for item in source['comparison_methods']:
        directory=Path(item['evaluation_dir']); key=item['key']
        template_path=directory.parent/f'{key}_random_spec.json'
        template=read(template_path)
        fields=('horizon','pulse_steps','pulse_start_schedule','pulse_batch_mode','full_episode_rollout',
                'controller_mode','delta_limit','success_criterion','reward_mode')
        contract={k:template[k] for k in fields}
        if common is None:common=contract
        if contract != common:raise ValueError('source method evaluation contracts differ')
        if not contract['full_episode_rollout'] or contract['controller_mode']!='fixed_random':
            raise ValueError('only fixed random full episodes supported')
        bank=read(template['bank'])
        member=next(m for m in bank['members'] if m['name']==template['proposer'])
        policy=member['policy']
        paths=[template_path,Path(template['bank']),Path(member['frozen_policy']),
               Path(policy['formal_config']),Path(policy['checkpoint'])/'identity.json',
               Path(policy['checkpoint'])/'payload.pkl',directory/f'{key}_nominal/prefixes.npz']
        for path in paths:inputs[str(path)]=file_sha(path)
        if template.get('phase_policy'):inputs.update(template['phase_policy']['input_files'])
        methods.append(dict(key=key,label=item['label'],template=template,
                            nominal_trace=str(directory/f'{key}_nominal/prefixes.npz')))
    spec=dict(schema='jit_batched_paired_pulse_comparison_v1',output=str(output),repo=source['repo'],
              python=source['python'],input_files=inputs,methods=methods,batches=plan,
              episodes=episodes,batch_size=batch_size,seed=seed,root_qpos_address=source['root_qpos_address'],
              new_training_transitions=0,maximum_interactions=len(methods)*episodes*common['horizon'],
              contract=common,stage_timeout_seconds=7200,resource_wait_timeout_seconds=1800,
              role='fixed policies; new paired random draws; development evaluation, not independent training seeds',
              source_comparison=str(previous))
    output.mkdir(parents=True,exist_ok=False)
    write(output/'spec.json',spec)
    write(output/'status.json',dict(phase='prepared',maximum_interactions=spec['maximum_interactions']))
    write(output/'ACTIVE_RUN.json',dict(name=f'{len(methods)}策略各{episodes}回合配对扰动',execution=str(output/'status.json')))
    (output/'INDEX.md').write_text(f'# {len(methods)}策略各{episodes}回合随机扰动\n\n'
        f'新回合共{len(methods)*episodes}，零训练；最大计费{spec["maximum_interactions"]}控制步。\n'
        '[进度](status.json) · [冻结配置](spec.json) · [逐批回执](receipts.json)\n\n'
        '相同固定起点与稳定恢复判据，四通道±0.25随机动作脉冲3步。分批同seed配对；'
        '全部原始轨迹在batches/，完成后输出comparison/三策略图、CSV、汇总和哈希索引。\n')
    return spec


def load_tape(path):
    keys=('prefix_mask','qpos','success','physical_failure','terminal','end_code','time','mask','delta',
          'front_wheel_clearance','rear_wheel_clearance','requested_delta','effective_delta')
    with np.load(path) as z:return {k:z[k] for k in keys}


def verify_batch(spec,batch,directory):
    """Check pair identity, exact denominator, and costs before accepting a batch."""
    directory=Path(directory); reference=None; rows=[]; hashes={}; charged=active=0
    for method in spec['methods']:
        key=method['key'];path=directory/key/'prefixes.npz';tape=load_tape(path)
        if reference is None:reference=tape
        else:
            np.testing.assert_array_equal(reference['delta'],tape['delta'])
            mask=reference['mask'] & tape['mask']
            np.testing.assert_array_equal(reference['requested_delta'][mask],tape['requested_delta'][mask])
        records=episode_results(tape,spec['root_qpos_address'])
        if len(records)!=batch['count']:raise ValueError('batch episode denominator drift')
        if not np.isfinite(tape['qpos'][tape['prefix_mask']]).all():raise ValueError('nonfinite trajectory')
        receipt=read(directory/key/'status.json')
        if receipt['phase']!='completed':raise ValueError('incomplete batch member')
        c,a=receipt['charged_interactions'],int(tape['prefix_mask'].sum())
        if not 0 <= a <= c <= batch['count']*spec['contract']['horizon']:
            raise ValueError('batch cost exceeds budget')
        charged+=c;active+=a
        for row in records:
            row.update(method=key,batch=batch['index'],seed=batch['seed'],lane=row['episode'],
                       episode=batch['offset']+row['episode'],trace=str(path))
        rows.extend(records);hashes[str(path)]=file_sha(path)
    return dict(rows=rows,trace_sha256=hashes,charged_interactions=charged,active_interactions=active,
                paired_random_draws_identical=True,paired_requests_identical_before_termination=True)


def run(spec_path):
    spec=read(spec_path);output=Path(spec['output']);started=time.monotonic();receipts=[]
    totals={m['key']:dict(episodes=0,successes=0) for m in spec['methods']}
    charged=0;reserved=0;stage='preflight';cli=Path(__file__).resolve().parents[2]/'cli'
    def status(phase,**extra):
        write(output/'status.json',dict(phase=phase,stage=stage,pid=os.getpid(),
            completed_batches=len(receipts),total_batches=len(spec['batches']),methods=totals,
            charged_interactions=charged,reserved_unverified_interactions=reserved,
            maximum_interactions=spec['maximum_interactions'],wall_seconds=time.monotonic()-started,**extra))
    try:
        for path,sha in {**spec['input_files'],**spec.get('source_locks',{})}.items():
            if file_sha(path)!=sha:raise ValueError('locked input drift: '+path)
        for batch in spec['batches']:
            directory=output/'batches'/f'{batch["index"]:04d}';directory.mkdir(parents=True,exist_ok=False)
            reserved=0
            for method in spec['methods']:
                import psutil
                stage=f'batch_{batch["index"]:04d}_{method["key"]}'
                waiting=time.monotonic()
                while True:
                    query=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,check=True,timeout=10)
                    gpu=int(query.stdout.strip().splitlines()[0]);host=psutil.virtual_memory().available/2**30
                    if gpu>=8192 and host>=6:break
                    status('waiting',gpu_free_mib=gpu,host_available_gib=host)
                    if time.monotonic()-waiting>spec['resource_wait_timeout_seconds']:raise TimeoutError('resource wait exceeded')
                    time.sleep(10)
                ev=dict(method['template'],seed=batch['seed'],num_envs=batch['count'],round_index=batch['offset'])
                path=directory/f'{method["key"]}_spec.json';write(path,ev)
                env=dict(os.environ,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                         JIT_AUTO_PUBLISH='0',PYTHONPATH=str(cli.parent/'src'),OMP_NUM_THREADS='4')
                reserved+=batch['count']*spec['contract']['horizon']
                with (directory/f'{method["key"]}.log').open('x') as log:
                    child=subprocess.Popen([spec['python'],str(cli/'run_pulse_exploration.py'),'--mode','collect',
                        '--spec',str(path),'--output',str(directory/method['key'])],cwd=spec['repo'],env=env,
                        stdout=log,stderr=subprocess.STDOUT)
                    status('running',child_pid=child.pid)
                    try:code=child.wait(timeout=spec['stage_timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        child.terminate();child.wait(timeout=30);raise
                if code:raise RuntimeError(f'{stage} exited {code}; see {directory / (method["key"]+".log")}')
            receipt=verify_batch(spec,batch,directory);write(directory/'verification.json',receipt)
            for row in receipt['rows']:
                totals[row['method']]['episodes']+=1;totals[row['method']]['successes']+=int(row['success'])
            charged+=receipt['charged_interactions'];reserved=0
            receipts.append(dict(batch=batch,verification=str(directory/'verification.json'),
                verification_sha256=file_sha(directory/'verification.json'),charged_interactions=receipt['charged_interactions']))
            write(output/'receipts.json',receipts);status('running')
        stage='plotting';status('running')
        from .analysis.batched_pulse_report import report
        summary=report(output)
        stage='complete';status('completed',summary=summary)
    except BaseException as exc:
        status('error',error=f'{type(exc).__name__}: {exc}',automatic_retry=False)
        raise
