#!/usr/bin/env python3
"""Queue a locked pulse recovery; yield GPU at completed child boundaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from jit_dvgc.current_policy_iteration import recovery_charge, verify_stage_reuse
from jit_dvgc.pulse_exploration import budget_contract


def read(p):
    return json.loads(Path(p).read_text())


def write(p,value):
    Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--rounds',type=int,required=True)
    parser.add_argument('--active-run',type=Path,required=True)
    args=parser.parse_args()
    old=args.previous.resolve();out=args.output.resolve();repo=args.repository.resolve()
    state=read(old/'status.json');spec=read(old/'declaration.json')['spec']
    inherited=recovery_charge(state)
    if state['phase'] not in ('error','failed','blocked'):
        raise ValueError('only a stopped lineage can be recovered')
    completed=len(read(old/'training_metrics.json'))
    if not 0<completed<args.rounds:
        raise ValueError('new total round limit must exceed completed rounds')
    boundary=read(old/'recovery.json')
    if completed!=boundary['completed_rounds']:
        source=read(old/'current_source.json');prior=old/f'round_{completed-1:04d}'
        if source['completed_rounds'] != completed:
            raise ValueError('round metrics precede source adoption; a completed boundary is required')
        support=prior/'next_training_support.json'
        if not support.exists():support=prior/'witnessed_support.json'
        bank=prior/'expanded_bank.json'
        if not bank.exists():bank=Path(read(prior/'collection_spec.json')['bank'])
        boundary=dict(previous=str(old),completed_rounds=completed,
            source=source['source'],explorer_checkpoint=source['explorer_checkpoint'],
            bank=str(bank),support=str(support))
    boundary['charged_interactions']=inherited
    out.mkdir(parents=True,exist_ok=False)
    view=out/'reuse_view';view.mkdir()
    for filename in ('status.json','declaration.json'):
        (view/filename).symlink_to(old/filename)
    current=old/f'round_{completed:04d}';target=view/current.name;target.mkdir()
    for p in current.iterdir():
        (target/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
    # Whole-stage reuse receipts point at canonical immutable data, not copies.
    for p in current.glob('*_reuse.json'):
        receipt=read(p);name=p.name.removesuffix('_reuse.json')
        previous=receipt.get('previous')
        if previous and not (target/name).exists():
            source=Path(previous).resolve()
            (target/name).symlink_to(source,target_is_directory=True)
            stage_spec=source.parent/(name+'_spec.json')
            if stage_spec.exists() and not (target/stage_spec.name).exists():
                (target/stage_spec.name).symlink_to(stage_spec)
    write(out/'boundary.json',boundary)
    new={**spec,'rounds':args.rounds,'resume_boundary':str(out/'boundary.json'),
         'resume_stage_root':str(view),'gate':{'kind':'gpu_idle','minimum_free_mib':20000,'wait_until_idle':True}}
    new['maximum_interactions']=budget_contract(new,len(new['order']))['maximum_interactions']
    new.pop('maximum_actual_interactions',None);verify_stage_reuse(spec,new)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    snapshot=out/'code';snapshot.mkdir();archive=out/'source.tar'
    with archive.open('wb') as stream:
        subprocess.run(['git','archive',commit],cwd=repo,stdout=stream,check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(snapshot,filter='data')
    archive.unlink()
    new.update(repo=str(snapshot),code_commit=commit)
    new['source_locks']={str(p):sha(p) for base in ('JIT/src','JIT/cli') for p in sorted((snapshot/base).rglob('*.py'))}
    locks=dict(spec['input_files'])
    paths=[out/'boundary.json',old/'status.json',old/'declaration.json',
           Path(boundary['previous'])/'training_metrics.json',Path(boundary['previous'])/'visited_cells.json']
    paths.extend(p for p in current.rglob('*') if p.is_file())
    for key in ('bank','support','explorer_checkpoint'):
        paths.append(Path(boundary[key]))
    for p in paths:
        locks[str(p.resolve())]=sha(p)
    new['input_files']=locks;write(out/'spec.json',new)
    remaining=new['maximum_interactions']-inherited
    if remaining<=0:
        raise ValueError('no remaining interaction budget')
    env=dict(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(snapshot/'JIT/src'),
             XLA_PYTHON_CLIENT_PREALLOCATE='false',JIT_AUTO_PUBLISH='0')
    plan=dict(schema='jit_gated_plan_v1',gate=new['gate'],input_files={**locks,str(out/'spec.json'):sha(out/'spec.json')},
        source_locks=new['source_locks'],max_interactions=remaining,wait_timeout_seconds=new['wait_timeout_seconds'],
        stages=[dict(name='resume_when_idle',argv=[new['python'],'JIT/cli/run_pulse_exploration.py','--mode','loop',
            '--spec',str(out/'spec.json'),'--output',str(out/'lineage')],cwd=str(snapshot),env=env,
            execution_backend='cpu',resource_supervisor=True,timeout_seconds=1209600,max_interactions=remaining)])
    write(out/'plan.json',plan)
    manifest=dict(name=f'空闲自动接续：累计{args.rounds}轮',execution=str(out/'execution/status.json'),lineage=str(out/'lineage/status.json'))
    write(out/'ACTIVE_RUN.json',manifest)
    with (out/'supervisor.log').open('xb') as log:
        process=subprocess.Popen([new['python'],str(snapshot/'JIT/cli/run_gated_plan.py'),'--plan',str(out/'plan.json'),
            '--output-dir',str(out/'execution'),'--wait','--poll-seconds','5'],cwd=snapshot,env={**os.environ,**env},
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(args.active_run,manifest)
    receipt=dict(supervisor_pid=process.pid,code_commit=commit,completed_rounds=completed,target_total_rounds=args.rounds,
        inherited_interactions=inherited,maximum_total_interactions=new['maximum_interactions'],launched_unix=time.time(),
        pause_semantics='finish active child, release GPU, wait before next child; no SIGSTOP or forced preemption')
    write(out/'launch_receipt.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
