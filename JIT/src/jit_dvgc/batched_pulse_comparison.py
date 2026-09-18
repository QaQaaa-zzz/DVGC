"""Finite, paired full-episode policy comparisons with bounded GPU batches."""
from collections import Counter
import copy
import csv
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .rsi_comparison import read, write, file_sha, episode_results


def normalize_pulse_start_schedule(value):
    if (not isinstance(value,(list,tuple)) or not value or
            any(type(step) is not int or step < 0 for step in value)):
        raise ValueError('pulse start schedule must be nonempty nonnegative integer ticks')
    return list(value)


def verify_initial_pulse_tape(tape,pulse_steps):
    """Require the only requested disturbance to occupy initial live ticks."""
    if type(pulse_steps) is not int or pulse_steps < 1:
        raise ValueError('positive integer pulse_steps required')
    ticks=np.arange(tape['prefix_mask'].shape[0])[:,None]
    expected=tape['prefix_mask'] & (ticks < pulse_steps)
    if not np.array_equal(tape['mask'],expected):
        raise ValueError('pulse mask differs from declared initial pulse')
    if np.any(tape['requested_delta'][ticks[:,0] >= pulse_steps] != 0):
        raise ValueError('requested disturbance exists after declared initial pulse')


def load_additional_methods(manifest_path,template,output,inputs,expected_policy):
    manifest_path=Path(manifest_path).resolve();manifest=read(manifest_path)
    if manifest.get('schema')!='jit_additional_comparison_methods_v1' or not manifest.get('methods'):
        raise ValueError('invalid additional comparison methods manifest')
    inputs[str(manifest_path)]=file_sha(manifest_path);result=[]
    identity_fields=('xml_sha256','action_order','actor_frame_fields','actor_task_fields')
    for row in manifest['methods']:
        required={'key','label','bank','proposer'}
        if (not required <= set(row) or set(row)-required-{'short_label'} or
                not all(isinstance(row[k],str) and row[k] for k in row)):
            raise ValueError('invalid additional method declaration')
        bank_path=Path(row['bank']).resolve();bank=read(bank_path)
        matches=[member for member in bank['members'] if member['name']==row['proposer']]
        if len(matches)!=1:raise ValueError('additional policy must resolve exactly once')
        member=matches[0];policy=member['policy']
        if any(policy.get(field)!=expected_policy.get(field) for field in identity_fields):
            raise ValueError('additional policy runtime contract differs')
        frozen=Path(member['frozen_policy']).resolve();formal=Path(policy['formal_config']).resolve()
        checkpoint=Path(policy['checkpoint']).resolve()
        for path in (bank_path,frozen,formal,checkpoint/'identity.json',checkpoint/'payload.pkl'):
            inputs[str(path)]=file_sha(path)
        configured=copy.deepcopy(template);configured.pop('phase_policy',None)
        configured.update(bank=str(bank_path),proposer=row['proposer'])
        key=row['key'];result.append(dict(key=key,label=row['label'],short_label=row.get('short_label',row['label']),template=configured,
            nominal_trace=str(Path(output).resolve()/'nominal'/key/'prefixes.npz'),generate_nominal=True))
    if len({row['key'] for row in result})!=len(result):raise ValueError('duplicate additional method key')
    return result


def batch_plan(episodes, batch_size, seed):
    if any(type(v) is not int or v <= 0 for v in (episodes, batch_size)):
        raise ValueError('positive integer episodes and batch size required')
    if type(seed) is not int or not 0 <= seed < 2**32 - episodes:
        raise ValueError('seed range invalid')
    return [dict(index=i,offset=start,count=min(batch_size,episodes-start),seed=seed+i)
            for i,start in enumerate(range(0,episodes,batch_size))]


def prepare(previous, output, *, episodes=10000, batch_size=256, seed=9182702,
            pulse_start_schedule=None, additional_methods=None, resource_wait_timeout_seconds=1800):
    previous,output=Path(previous).resolve(),Path(output).resolve()
    source=read(previous/'spec.json')
    if read(previous/'status.json')['phase'] != 'completed':
        raise ValueError('source comparison incomplete')
    plan=batch_plan(episodes,batch_size,seed)
    if type(resource_wait_timeout_seconds) is not int or resource_wait_timeout_seconds <= 0:
        raise ValueError('positive integer resource wait timeout required')
    inputs={str(previous/'spec.json'):file_sha(previous/'spec.json')}
    schedule=(normalize_pulse_start_schedule(pulse_start_schedule)
              if pulse_start_schedule is not None else None)
    methods=[];expected_policy=None
    common=None
    for item in source['comparison_methods']:
        directory=Path(item['evaluation_dir']); key=item['key']
        template_path=directory.parent/f'{key}_random_spec.json'
        template=read(template_path)
        if schedule is not None:template['pulse_start_schedule']=schedule
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
        if expected_policy is None:expected_policy=policy
        paths=[template_path,Path(template['bank']),Path(member['frozen_policy']),
               Path(policy['formal_config']),Path(policy['checkpoint'])/'identity.json',
               Path(policy['checkpoint'])/'payload.pkl',directory/f'{key}_nominal/prefixes.npz']
        for path in paths:inputs[str(path)]=file_sha(path)
        if template.get('phase_policy'):inputs.update(template['phase_policy']['input_files'])
        methods.append(dict(key=key,label=item['label'],template=template,
                            nominal_trace=str(directory/f'{key}_nominal/prefixes.npz')))
    if additional_methods is not None:
        methods.extend(load_additional_methods(additional_methods,methods[0]['template'],output,inputs,expected_policy))
    if len({method['key'] for method in methods})!=len(methods):raise ValueError('duplicate comparison method key')
    nominal_budget=sum(bool(method.get('generate_nominal')) for method in methods)*common['horizon']
    spec=dict(schema='jit_batched_paired_pulse_comparison_v1',output=str(output),repo=source['repo'],
              python=source['python'],input_files=inputs,methods=methods,batches=plan,
              episodes=episodes,batch_size=batch_size,seed=seed,root_qpos_address=source['root_qpos_address'],
              new_training_transitions=0,maximum_interactions=len(methods)*episodes*common['horizon']+nominal_budget,
              contract=common,stage_timeout_seconds=7200,resource_wait_timeout_seconds=resource_wait_timeout_seconds,
              role='fixed policies; new paired random draws; development evaluation, not independent training seeds',
              source_comparison=str(previous))
    output.mkdir(parents=True,exist_ok=False)
    write(output/'spec.json',spec)
    write(output/'status.json',dict(phase='prepared',maximum_interactions=spec['maximum_interactions']))
    write(output/'ACTIVE_RUN.json',dict(name=f'{len(methods)}策略各{episodes}回合配对扰动',execution=str(output/'status.json')))
    (output/'INDEX.md').write_text(f'# {len(methods)}策略各{episodes}回合随机扰动\n\n'
        f'新回合共{len(methods)*episodes}，零训练；最大计费{spec["maximum_interactions"]}控制步。\n'
        '[进度](status.json) · [冻结配置](spec.json) · [逐批回执](receipts.json)\n\n'
        f'相同固定起点与稳定恢复判据，四通道±0.25随机动作脉冲3步，起扰控制步{common["pulse_start_schedule"]}。分批同seed配对；'
        '全部原始轨迹在batches/，完成后输出comparison/多策略图、CSV、汇总和哈希索引。\n')
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
        if spec['contract'].get('pulse_start_schedule') == [0]:
            verify_initial_pulse_tape(tape,spec['contract']['pulse_steps'])
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
        nominal_receipts=[]
        for method in spec['methods']:
            if not method.get('generate_nominal'):continue
            import psutil
            stage=f'nominal_{method["key"]}';waiting=time.monotonic()
            while True:
                query=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,check=True,timeout=10)
                gpu=int(query.stdout.strip().splitlines()[0]);host=psutil.virtual_memory().available/2**30
                if gpu>=8192 and host>=6:break
                status('waiting',gpu_free_mib=gpu,host_available_gib=host)
                if time.monotonic()-waiting>spec['resource_wait_timeout_seconds']:raise TimeoutError('resource wait exceeded')
                time.sleep(10)
            directory=output/'nominal'/method['key'];directory.parent.mkdir(parents=True,exist_ok=True)
            ev=dict(method['template'],seed=spec['seed'],num_envs=1,round_index=0,delta_limit=[0.]*4)
            path=output/f'{method["key"]}_nominal_spec.json';write(path,ev)
            env=dict(os.environ,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                     JIT_AUTO_PUBLISH='0',PYTHONPATH=str(cli.parent/'src'),OMP_NUM_THREADS='4')
            reserved=spec['contract']['horizon']
            with (output/f'{method["key"]}_nominal.log').open('x') as log:
                child=subprocess.Popen([spec['python'],str(cli/'run_pulse_exploration.py'),'--mode','collect',
                    '--spec',str(path),'--output',str(directory)],cwd=spec['repo'],env=env,
                    stdout=log,stderr=subprocess.STDOUT)
                status('running',child_pid=child.pid)
                try:code=child.wait(timeout=spec['stage_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    child.terminate();child.wait(timeout=30);raise
            if code:raise RuntimeError(f'{stage} exited {code}; see {output / (method["key"]+"_nominal.log")}')
            receipt=read(directory/'status.json')
            if receipt['phase']!='completed':raise ValueError('incomplete nominal rollout')
            nominal_trace=Path(method['nominal_trace'])
            if nominal_trace!=directory/'prefixes.npz' or not nominal_trace.exists():raise ValueError('nominal trace missing')
            c=receipt['charged_interactions'];charged+=c;reserved=0
            nominal_receipts.append(dict(method=method['key'],charged_interactions=c,
                active_interactions=int(load_tape(nominal_trace)['prefix_mask'].sum()),
                trace=str(nominal_trace),trace_sha256=file_sha(nominal_trace)))
            write(output/'nominal_receipts.json',nominal_receipts);status('running')
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
