import json
import pytest
from jit_dvgc.generative_bridge.contracts import file_sha


def test_reuse_checks_config_cost_and_evidence(tmp_path):
    from jit_dvgc.generative_bridge.closed_loop import baseline_reuse
    def write(name, value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    cfg={'gate':{'kind':'gpu_idle'},'seed':2,'bank':'frozen'}
    spec=write('spec.json',cfg)
    status=write('status.json',{'phase':'completed','charged_interactions':12800})
    rows=write('candidates.json',[{'prefix_label':1}]*32)
    execution=write('execution.json',{'phase':'completed','stages':[{'phase':'completed','returncode':0}]})
    entry=dict(output=str(tmp_path),spec=spec,execution=execution,cost=dict(stage='baseline_stress_00',phase='completed',accounting='measured',charged_interactions=12800,charged_updates=0),locks={p:file_sha(p) for p in (spec,status,rows,execution)})
    changed={**cfg,'gate':{'kind':'gpu_shared'}}
    assert baseline_reuse('baseline_stress_00','collect',changed,12800,entry)==tmp_path
    with pytest.raises(ValueError):baseline_reuse('baseline_stress_00','collect',{**changed,'seed':3},12800,entry)
    with pytest.raises(ValueError):baseline_reuse('seed_support','collect',changed,12800,entry)
    entry['cost']['charged_interactions']=0
    with pytest.raises(ValueError):baseline_reuse('baseline_stress_00','collect',changed,12800,entry)
    entry['cost']['charged_interactions']=12800
    (tmp_path/'candidates.json').write_text('[]')
    with pytest.raises(ValueError):baseline_reuse('baseline_stress_00','collect',changed,12800,entry)
