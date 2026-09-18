from types import SimpleNamespace
from jit_dvgc.execution_gate import check_execution_gate


def test_gpu_gate_requires_idle_and_free_memory(monkeypatch):
    state={'compute':'','memory':'23300, 24564\n'}
    def query(argv,**kwargs):
        return SimpleNamespace(stdout=state['compute'] if any('query-compute' in a for a in argv) else state['memory'])
    monkeypatch.setattr('subprocess.run',query)
    gate={'kind':'gpu_idle','minimum_free_mib':20000}
    assert check_execution_gate(gate)['ready']
    state['memory']='19000, 24564\n'
    assert not check_execution_gate(gate)['ready']
    state['memory']='23300, 24564\n';state['compute']='123, 100\n'
    assert not check_execution_gate(gate)['ready']
    state['compute']='';state['memory']='N/A, 24564\n'
    assert not check_execution_gate(gate)['ready']
