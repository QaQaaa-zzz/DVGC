"""Immutable queue preparation and conditional CLI exit, without execution."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from jit_dvgc import exploration_loop_declaration as declaration
from jit_dvgc import exploration_loop


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_queue_preserves_order_locks_and_budgets(tmp_path,monkeypatch):
    repo=tmp_path/'repo'
    for path in ('JIT/src/module.py','JIT/cli/run.py','assets/model.xml','JIT/configs/config.json'):
        target=repo/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture')
    paths={name:tmp_path/name for name in ('bank.json','bootstrap.json','manifest.json','formal.json','raw.json')}
    for path in paths.values():path.write_text('{}')
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    for name in ('identity.json','payload.pkl'):(checkpoint/name).write_text('fixture')
    support=tmp_path/'support.json'
    support.write_text(json.dumps({'inputs':{str(paths['raw.json']):sha(paths['raw.json'])}}))
    bank={'members':[{'roles':['evaluator'], 'frozen_policy':str(paths['manifest.json']),
        'policy':{'formal_config':str(paths['formal.json']),'checkpoint':str(checkpoint)}}]}
    monkeypatch.setattr(declaration,'load_probe_bank',lambda _:bank)
    templates=[]
    for index in range(2):
        spec=dict(repo=str(repo),python='/fake/python',bank=str(paths['bank.json']),
            witnessed_support=str(support),bootstrap_config=str(paths['bootstrap.json']),
            gate={'status_path':'/external/status.json'},rounds=index+1,policy_steps=3200,
            pending_limit=1,stage_timeout_seconds=10,
            explorer=dict(num_envs=1,batches=1,max_candidates_per_episode=1,horizon=400,evaluation_episodes=1))
        path=tmp_path/f'template{index}.json';path.write_text(json.dumps(spec));templates.append(path)
    output=tmp_path/'queue'
    plan=declaration.prepare_queue(templates,output)
    assert [stage['name'] for stage in plan['stages']]==['stage_0','stage_1']
    assert all(stage['execution_backend']=='cpu' and stage['env']['JAX_PLATFORMS']=='cpu' for stage in plan['stages'])
    assert plan['max_interactions']==sum(stage['max_interactions'] for stage in plan['stages'])
    assert all(stage['timeout_seconds']==(i+1)*7*10+600 for i,stage in enumerate(plan['stages']))
    for name in ('stage_0_spec.json','stage_1_spec.json'):
        assert plan['input_files'][str(output/name)]==sha(output/name)
    assert str(checkpoint/'payload.pkl') in plan['input_files']
    assert plan['input_files'][str(paths['raw.json'])]==sha(paths['raw.json'])
    assert len(plan['source_locks'])==4
    assert json.loads((output/'queue_plan.json').read_text())==plan
    with pytest.raises(FileExistsError):declaration.prepare_queue(templates,output)
    paths['raw.json'].write_text('changed')
    with pytest.raises(ValueError,match='inherited input drift'):
        declaration.prepare_queue(templates,tmp_path/'changed_queue')


def test_no_pending_cli_exits_nonzero_to_stop_queue(tmp_path,monkeypatch):
    cli_path=Path(__file__).resolve().parents[1]/'cli/run_exploration_loop.py'
    module_spec=importlib.util.spec_from_file_location('test_loop_cli',cli_path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    monkeypatch.setattr(module,'run',lambda *args:{'phase':'stopped_no_pending'})
    monkeypatch.setattr(module.signal,'signal',lambda *args:None)
    monkeypatch.setattr(sys,'argv',['run_exploration_loop.py','--spec',str(tmp_path/'spec'),'--output',str(tmp_path/'output')])
    with pytest.raises(SystemExit) as exc:module.main()
    assert exc.value.code==2
