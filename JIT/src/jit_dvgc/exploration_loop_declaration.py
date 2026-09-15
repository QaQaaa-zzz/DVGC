"""CPU-only immutable preparation of a bounded delayed exploration queue."""
from pathlib import Path

from .exploration_loop import read, write, budget_contract
from .probe_bank import load_probe_bank, _file_sha
from .gated_execution import _validate


def prepare(template_path, output):
    template_path,output=Path(template_path).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    spec=read(template_path)
    bank=load_probe_bank(Path(spec['bank']))
    repo=Path(spec['repo']).resolve()
    inputs={str(template_path):_file_sha(template_path)}
    for path in (spec['bank'],spec['witnessed_support'],spec['bootstrap_config']):
        inputs[str(Path(path).resolve())]=_file_sha(path)
    for member in bank['members']:
        record=member['policy']
        for path in (member['frozen_policy'],record['formal_config'],
                     str(Path(record['checkpoint'])/'identity.json'),str(Path(record['checkpoint'])/'payload.pkl')):
            inputs[str(Path(path).resolve())]=_file_sha(path)
    # The support manifest already binds all its raw evidence; preserve those
    # preexisting digests instead of trusting freshly rehashed changed evidence.
    for path,sha in read(spec['witnessed_support'])['inputs'].items():
        path=str(Path(path).resolve())
        if path in inputs and inputs[path]!=sha:raise ValueError('conflicting inherited input lock')
        inputs[path]=sha
    for path,sha in inputs.items():
        if _file_sha(path)!=sha:raise ValueError('inherited input drift: '+path)
    if spec.get('reuse_first_learned_arrivals'):
        source=Path(spec['reuse_first_learned_arrivals']).resolve()
        for path in source.rglob('*'):
            if path.is_file():inputs[str(path)]=_file_sha(path)
    sources={str(path.resolve()):_file_sha(path) for base in (repo/'JIT/src',repo/'JIT/cli') for path in base.rglob('*.py')}
    for base in (repo/'assets',repo/'JIT/configs'):
        for path in base.rglob('*'):
            if path.is_file():sources[str(path.resolve())]=_file_sha(path)
    budget=budget_contract(spec,sum('evaluator' in m['roles'] for m in bank['members']))
    spec.update(input_files=inputs,source_locks=sources,maximum_interactions=budget['maximum_interactions'])
    write(output,spec)
    return spec


def prepare_queue(templates, output):
    """Engineering must return completed before the pilot command can launch."""
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    stages=[];all_inputs={};all_sources={};total=0;gate=None
    for index,template in enumerate(templates):
        name='stage_'+str(index)
        spec_path=output/(name+'_spec.json');spec=prepare(template,spec_path)
        if gate is not None and gate!=spec['gate']:raise ValueError('queue gate disagreement')
        gate=spec['gate'];total+=spec['maximum_interactions']
        for destination,values in ((all_inputs,spec['input_files']),(all_sources,spec['source_locks'])):
            for path,sha in values.items():
                if path in destination and destination[path]!=sha:raise ValueError('queue lock disagreement')
                destination[path]=sha
        all_inputs[str(spec_path)]=_file_sha(spec_path)
        stages.append(dict(name=name,execution_backend='cpu',
            argv=[spec['python'],'JIT/cli/run_exploration_loop.py','--spec',str(spec_path),'--output',str(output/(name+'_result'))],
            cwd=spec['repo'],env=dict(JAX_PLATFORMS='cpu',PYTHONPATH=str(Path(spec['repo'])/'JIT/src'),JIT_AUTO_PUBLISH='0'),
            timeout_seconds=spec['rounds']*7*spec['stage_timeout_seconds']+600,
            max_interactions=spec['maximum_interactions']))
    plan=dict(schema='jit_gated_plan_v1',max_interactions=total,wait_timeout_seconds=43200,
        gate=gate,input_files=all_inputs,source_locks=all_sources,stages=stages)
    _validate(plan);write(output/'queue_plan.json',plan)
    return plan
