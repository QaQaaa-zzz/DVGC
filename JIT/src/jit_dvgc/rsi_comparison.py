"""Fresh RSI training and fixed-budget paired random-pulse comparison."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .jump_evidence_validation import read, write, file_sha
from .evidence_integrity import canonical_sha256


def prepare(alignment, source_spec, output, *, steps=1_000_000, episodes=100, seed=9182601):
    from .iterative_probe_training import make_config, SUPPORT_SCHEMA
    from .pulse_exploration import support_row
    alignment, source_spec, output = [Path(p).resolve() for p in (alignment, source_spec, output)]
    output.mkdir(parents=True, exist_ok=False)
    am = read(alignment/'manifest.json'); witnesses = read(alignment/'witnesses.json')
    selected = {r['context'] for r in witnesses if r['status'] == 'aligned'}
    entries = {}; locks = {}; found = set(); terminal = 0
    for name, sha in am['input_sha256'].items():
        path = Path(name)
        if path.name != 'outcomes.json':
            continue
        if file_sha(path) != sha:
            raise ValueError('aligned figure outcomes changed')
        locks[str(path)] = sha
        for row in read(path):
            key = row['snapshot_context_sha256']
            if key not in selected:
                continue
            found.add(key)
            if row['label'] != 1:
                raise ValueError('selected witness is no longer successful')
            if row.get('snapshot') is None:
                terminal += 1
                continue
            entry = support_row(row, True)
            entry['snapshot'] = str(Path(entry['snapshot']).resolve())
            entries[key] = entry
            for filename in ('identity.json', 'snapshot.pkl'):
                p = Path(entry['snapshot'])/filename; locks[str(p)] = file_sha(p)
    if found != selected:
        raise ValueError('missing aligned witness provenance')
    counts = Counter(r['phase'] for r in entries.values())
    if not all(counts[p] for p in ('upstream', 'downstream')):
        raise ValueError('RSI requires both saved phases')
    for row in entries.values():
        row['sampling_weight'] = 1. / counts[row['phase']]
    for p in (alignment/'manifest.json', alignment/'witnesses.json'):
        locks[str(p)] = file_sha(p)
    support = dict(schema=SUPPORT_SCHEMA, role='train', final_test_used=False,
                   entries=list(entries.values()), inputs=locks,
                   source_alignment=str(alignment), phase_counts=dict(counts),
                   selection='unique nonterminal original candidate snapshots from displayed witnesses; equal mass within phase',
                   coordinate_translation_applied=False, terminal_witnesses_excluded=terminal)
    support['support_sha256'] = canonical_sha256(support)
    write(output/'support.json', support)
    source = read(source_spec); bank = read(source['bank'])
    member = next(m for m in bank['members'] if m['name'] == source['proposer'])
    actual = steps // 3200 * 3200
    if actual <= 0 or type(episodes) is not int or episodes <= 0:
        raise ValueError('invalid training/evaluation budget')
    config = make_config(output/'support.json', member['frozen_policy'], source['bootstrap_config'],
                         output/'training_config.json', 'fresh_rsi', 0, actual, seed,
                         checkpoints=[actual], reward_mode='original_all_phases', initialization_mode='fresh')
    spec = dict(schema='jit_fresh_rsi_comparison_v1', output=str(output),
                repo=str(Path(source['repo']).resolve()), python=sys.executable,
                source_spec=str(source_spec), source_bank=source['bank'],
                baseline=member['name'], baseline_actor_sha256=member['policy']['actor_sha256'],
                training_config=str(output/'training_config.json'), requested_training_ceiling=steps,
                training_steps=actual, unused_training_budget=steps-actual,
                evaluation_seed=seed+1, episodes=episodes, horizon=400,
                pulse_steps=source['pulse_steps'], pulse_start_schedule=source['pulse_start_schedule'],
                delta_limit=source['delta_limit'], success_criterion=config['success_criterion'],
                reward_mode='original_all_phases', root_qpos_address=am['root_qpos_address'],
                maximum_interactions=actual+1600+2*(episodes+1)*400,
                selection='final checkpoint only; no selection using comparison outcomes',
                role='development paired random-pulse evaluation; not final TEST',
                input_files={str(p):file_sha(p) for p in (source_spec, output/'support.json', output/'training_config.json', Path(source['bank']))},
                resource_policy='user requested launch; preserve other jobs; wait for 8GiB GPU and 6GiB host available',
                stage_timeout_seconds=7200, resource_wait_timeout_seconds=1800)
    write(output/'spec.json', spec)
    write(output/'status.json', dict(phase='prepared', maximum_interactions=spec['maximum_interactions']))
    write(output/'ACTIVE_RUN.json', dict(name='全新RSI与初始策略100回合对照', execution=str(output/'status.json')))
    (output/'INDEX.md').write_text('# 全新 RSI 与初始策略对照\n\n'
        f'基线：`{spec["baseline"]}`。全新训练预算{steps}，整批实际目标{actual}。'
        f'支持{len(entries)}个原始完整快照，phase计数{dict(counts)}；未平移状态。\n\n'
        '- [流水线状态](status.json) · [配置与预算](spec.json) · [训练配置](training_config.json)\n'
        '- [训练状态](training/fresh_rsi/status.json) · [训练日志](training.log)\n'
        '- 完成后生成 `comparison/comparison.png`、PDF/SVG、逐回合CSV与summary.json。\n\n'
        'Actor、Critic、优化器和归一化均从零开始；原奖励不改；20%固定起点、80%RSI。'
        '两策略各100个随机动作脉冲回合+1个无扰动回合，固定起点连续闭环，不在途中换策略。'
        '所有失败、未离地与超时保留在分母和世界坐标图。\n')
    return spec


def run(spec_path):
    from .probe_bank import lock_probe_bank
    from .unified_policy_freeze import freeze_development_checkpoint
    spec = read(spec_path); output = Path(spec['output']); started = time.monotonic()
    cli = Path(__file__).resolve().parents[2]/'cli'
    receipts = []

    def status(phase, stage, **extra):
        write(output/'status.json', dict(phase=phase, stage=stage, pid=os.getpid(),
              wall_seconds=time.monotonic()-started, stages=receipts, **extra))

    def child(name, argv):
        waiting = time.monotonic()
        while True:
            import psutil
            query = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, check=True, timeout=10)
            gpu = int(query.stdout.strip().splitlines()[0])
            host = psutil.virtual_memory().available / 2**30
            if gpu >= 8192 and host >= 6:
                break
            status('waiting', name, gpu_free_mib=gpu, host_available_gib=host)
            if time.monotonic()-waiting > spec['resource_wait_timeout_seconds']:
                raise TimeoutError('resource wait exceeded declared limit')
            time.sleep(10)
        env = dict(os.environ, JAX_PLATFORMS='cuda,cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   JIT_AUTO_PUBLISH='0', JIT_RUN_ROOT=str(output/'training'),
                   PYTHONPATH=str(cli.parent/'src'), OMP_NUM_THREADS='4')
        t = time.monotonic()
        with (output/f'{name}.log').open('w') as log:
            process = subprocess.Popen([spec['python'], *map(str, argv)], cwd=spec['repo'], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            status('running', name, child_pid=process.pid, gpu_free_mib=gpu, host_available_gib=host)
            try:
                code = process.wait(timeout=spec['stage_timeout_seconds'])
            except subprocess.TimeoutExpired:
                process.terminate(); process.wait(timeout=30)
                raise
        receipts.append(dict(name=name, returncode=code, wall_seconds=time.monotonic()-t))
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; see {output / (name + ".log")}')

    try:
        for path, sha in spec.get('source_locks', {}).items():
            if file_sha(path) != sha:
                raise ValueError('frozen comparison source changed: '+path)
        for path, sha in spec['input_files'].items():
            if file_sha(path) != sha:
                raise ValueError('comparison input changed: '+path)
        child('training', [cli/'train_unified.py', '--config', spec['training_config'], '--run-id', 'fresh_rsi'])
        checkpoint = output/'training/fresh_rsi/checkpoints'/f'transition_{spec["training_steps"]}'
        frozen = freeze_development_checkpoint(output/'frozen', config_path=Path(spec['training_config']),
                                              checkpoint=checkpoint, name='fresh_rsi')
        source_bank = read(spec['source_bank'])
        new_bank = output/'trained_bank.json'
        lock_probe_bank(dict(version='fresh_rsi_evaluation', task=source_bank['task'],
                             max_ticks=spec['horizon'], label_interaction_budget=spec['episodes']*spec['horizon'],
                             max_candidates_per_process=spec['episodes'],
                             members=[dict(frozen_policy=str(output/'frozen/frozen_unified_policy.json'),
                                           roles=['proposer', 'evaluator'])]), new_bank)
        for method, bank, proposer in [('baseline', spec['source_bank'], spec['baseline']),
                                       ('fresh_rsi', str(new_bank), frozen['policy']['name'])]:
            for condition, count in [('nominal', 1), ('random', spec['episodes'])]:
                ev = dict(bank=bank, proposer=proposer, seed=spec['evaluation_seed'],
                          num_envs=count, horizon=spec['horizon'], pulse_steps=spec['pulse_steps'],
                          pulse_start_schedule=spec['pulse_start_schedule'], pulse_batch_mode='mixed',
                          full_episode_rollout=True, controller_mode='fixed_random', round_index=0,
                          delta_limit=spec['delta_limit'] if condition=='random' else [0.]*4,
                          success_criterion=spec['success_criterion'], reward_mode=spec['reward_mode'],
                          learning_rate=3e-5, max_grad_norm=.75)
                name=method+'_'+condition; path=output/(name+'_spec.json'); write(path,ev)
                child(name, [cli/'run_pulse_exploration.py', '--mode', 'collect', '--spec', path,
                             '--output', output/'evaluation'/name])
        status('running', 'plotting')
        summary = report(output)
        status('completed', 'complete', summary=summary)
    except BaseException as exc:
        status('error', 'pipeline', error=str(exc))
        raise


def episode_results(tape, q0):
    from .constants import END_TIMEOUT, END_REASONS
    result = []
    for lane in range(tape['prefix_mask'].shape[1]):
        ticks = np.flatnonzero(tape['prefix_mask'][:, lane])
        last = int(ticks[-1])
        success = bool(tape['success'][last, lane]); failure = bool(tape['physical_failure'][last, lane])
        code = int(tape['end_code'][last, lane]) if 'end_code' in tape else None
        result.append(dict(episode=lane, success=success and not failure,
                           physical_failure=failure, conflict=success and failure,
                           end_code=code, terminal_reason=END_REASONS.get(code, 'not_recorded'),
                           environment_timeout=code == END_TIMEOUT,
                           horizon_exhausted=not bool(tape['terminal'][last, lane]),
                           control_steps=len(ticks), end_time_s=float(tape['time'][last, lane]),
                           pulse_applied_steps=int(tape['mask'][:, lane].sum()),
                           peak_root_z_m=float(tape['qpos'][ticks, lane, q0+2].max())))
    return result


def report(output, destination=None):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from .analysis.liftoff_alignment import liftoff_index, align_xz
    output=Path(output); spec=read(output/'spec.json')
    destination=Path(destination) if destination else output/'comparison'
    destination.mkdir(exist_ok=False)
    q0=spec['root_qpos_address']; tapes={}; rows=[]; summary={}; hashes={}
    for method in ('baseline','fresh_rsi'):
        for condition in ('nominal','random'):
            path=output/'evaluation'/f'{method}_{condition}'/'prefixes.npz'
            with np.load(path) as z:
                keys=['prefix_mask','qpos','success','physical_failure','terminal','end_code','time','mask','delta',
                      'front_wheel_clearance','rear_wheel_clearance','requested_delta','effective_delta']
                tapes[method,condition]={k:z[k] for k in keys}
            hashes[str(path)]=file_sha(path)
            records=episode_results(tapes[method,condition],q0)
            rows.extend(dict(method=method,condition=condition,**r) for r in records)
            if condition=='random':
                summary[method]=dict(successes=sum(r['success'] for r in records),episodes=len(records),
                                     physical_failures=sum(r['physical_failure'] for r in records),
                                     conflicts=sum(r['conflict'] for r in records),
                                     environment_timeouts=sum(r['environment_timeout'] for r in records),
                                     terminal_reasons=dict(Counter(r['terminal_reason'] for r in records)),
                                     horizon_exhausted=sum(r['horizon_exhausted'] for r in records))
                if len(records)!=spec['episodes']:
                    raise ValueError('comparison denominator drift')
    # Random draws are policy-independent, including unused draws after early termination.
    np.testing.assert_array_equal(tapes['baseline','random']['delta'],tapes['fresh_rsi','random']['delta'])
    a,b=tapes['baseline','random'],tapes['fresh_rsi','random']
    common=a['mask'] & b['mask']
    np.testing.assert_array_equal(a['requested_delta'][common],b['requested_delta'][common])
    font=Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    colors={'baseline':'#2369a1','fresh_rsi':'#d15a28'}
    labels={'baseline':spec['baseline'],'fresh_rsi':f'全新RSI（{spec["training_steps"]:,}步）'}
    exclusions=Counter()
    for method in colors:
        for condition in ('nominal','random'):
            tape=tapes[method,condition]
            for lane in range(tape['prefix_mask'].shape[1]):
                ticks=np.flatnonzero(tape['prefix_mask'][:,lane]);pts=tape['qpos'][ticks,lane][:,[q0,q0+2]]
                good=bool(tape['success'][ticks[-1],lane]) and not bool(tape['physical_failure'][ticks[-1],lane])
                ax=axes[0,0] if condition=='nominal' else axes[0,1]
                ax.plot(pts[:,0],pts[:,1],color=colors[method],alpha=1 if condition=='nominal' else .23,
                        lw=2 if condition=='nominal' else .8,ls='-' if good else '--',
                        label=labels[method] if lane==0 else None)
                if not good:ax.scatter(*pts[-1],marker='x',s=15,color=colors[method])
                if condition=='random':
                    k=liftoff_index(tape['front_wheel_clearance'][ticks,lane],tape['rear_wheel_clearance'][ticks,lane])
                    if k is None:exclusions[method]+=1
                    else:
                        aligned=align_xz(pts,pts[k,0]);axes[1,0].plot(aligned[:,0],aligned[:,1],color=colors[method],alpha=.23,lw=.8,ls='-' if good else '--')
    for ax,title in zip(axes.flat[:3],['无扰动完整闭环',f'{spec["episodes"]}个配对随机扰动回合：全部轨迹','诊断离地点对齐：保留真实高度']):
        ax.set_title(title);ax.set_ylabel('根部高度 z（m）');ax.set_xlabel('世界位置 x（m）' if ax is not axes[1,0] else 'x − x_LO（m）');ax.grid(alpha=.2)
    axes[0,0].legend();axes[0,1].legend();axes[1,0].axvline(0,color='gray',ls='--',lw=.8)
    values=[100*summary[m]['successes']/summary[m]['episodes'] for m in colors]
    bars=axes[1,1].bar([labels[m] for m in colors],values,color=list(colors.values()),width=.5)
    axes[1,1].set(ylim=(0,112),ylabel='成功率（%）',title=f'相同稳定恢复判据；全部{spec["episodes"]}回合为分母')
    for bar,method,value in zip(bars,colors,values):
        axes[1,1].text(bar.get_x()+bar.get_width()/2,value+2,f'{summary[method]["successes"]}/{spec["episodes"]} = {value:.0f}%',ha='center')
    fig.suptitle('初始策略与全新RSI策略：固定起点、配对随机扰动对照',fontsize=17)
    fig.supxlabel('实线：成功；虚线/叉号：未成功轨迹及真实终点。对齐图未检出离地：'
                    f'基线 {exclusions["baseline"]}，RSI {exclusions["fresh_rsi"]}；这些回合仍计入成功率。\n'
                    '随机动作残差3步，四通道±0.25；本次为开发评估、单训练种子，不是独立最终测试。',fontsize=10)
    for ext in ('png','pdf','svg'):fig.savefig(destination/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    with (destination/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary.update(paired_random_draws_identical=True,paired_requested_pulses_identical_before_termination=True,
                   aligned_exclusions=dict(exclusions),input_sha256=hashes,baseline=spec['baseline'],
                   training_steps=spec['training_steps'],evaluation_role=spec['role'],
                   report_source_sha256=file_sha(__file__))
    report_path=output/'training/fresh_rsi/formal_report.json'
    summary['training_report']=str(report_path)
    summary['evaluation_charged_interactions']=sum(read(output/'evaluation'/f'{m}_{c}'/'status.json')['charged_interactions'] for m in colors for c in ('nominal','random'))
    write(destination/'summary.json',summary)
    link=os.path.relpath(destination,output)
    with (output/'INDEX.md').open('a') as f:
        f.write(f'\n## 已完成的对照\n\n[同图对比]({link}/comparison.png) · [PDF]({link}/comparison.pdf) · [SVG]({link}/comparison.svg) · [逐回合结果]({link}/episodes.csv) · [汇总]({link}/summary.json)\n\n')
        for m in colors:f.write(f'- {labels[m]}：成功 {summary[m]["successes"]}/{summary[m]["episodes"]}。\n')
    return summary


def launch(spec_path, repository, snapshot):
    """Freeze committed sources and start the finite pipeline and notification watcher."""
    import tarfile
    spec_path, repository, snapshot = [Path(p).resolve() for p in (spec_path, repository, snapshot)]
    spec=read(spec_path);output=Path(spec['output'])
    if (output/'launch_receipt.json').exists():
        raise ValueError('comparison already launched')
    subprocess.run(['git','diff','--exit-code','HEAD','--','JIT/src','JIT/cli'],cwd=repository,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    snapshot.mkdir(parents=True,exist_ok=False)
    archive=output/'source.tar'
    subprocess.run(['git','archive','--output',str(archive),commit],cwd=repository,check=True)
    with tarfile.open(archive) as tar:tar.extractall(snapshot,filter='data')
    archive.unlink()
    spec['repo']=str(snapshot);spec['code_commit']=commit
    spec['source_locks']={str(p):file_sha(p) for folder in ('JIT/src','JIT/cli') for p in sorted((snapshot/folder).rglob('*.py'))}
    write(spec_path,spec)
    write(output/'freeze.json',dict(code_commit=commit,snapshot=str(snapshot),spec_sha256=file_sha(spec_path)))
    env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONPATH=str(snapshot/'JIT/src'),JIT_AUTO_PUBLISH='0')
    env.setdefault('DBUS_SESSION_BUS_ADDRESS',f'unix:path=/run/user/{os.getuid()}/bus')
    env.setdefault('XDG_RUNTIME_DIR',f'/run/user/{os.getuid()}')
    with (output/'supervisor.log').open('x') as log:
        process=subprocess.Popen([spec['python'],str(snapshot/'JIT/cli/run_rsi_comparison.py'),'run','--spec',str(spec_path)],
                                 cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    with (output/'watcher.log').open('x') as log:
        watcher=subprocess.Popen([spec['python'],str(snapshot/'JIT/cli/watch_run_errors.py'),
                                  '--active-run',str(output/'ACTIVE_RUN.json'),'--state-dir',str(output/'notifications')],
                                 cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(supervisor_pid=process.pid,watcher_pid=watcher.pid,code_commit=commit,launched_unix=time.time())
    write(output/'launch_receipt.json',receipt)
    return receipt
