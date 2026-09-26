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

def test_only_explicit_small_desktop_executable_is_exempt(monkeypatch):
    state={'compute':'123, 502\n','memory':'23000, 24564\n'}
    monkeypatch.setattr('subprocess.run',lambda argv,**kw:SimpleNamespace(stdout=state['compute'] if any('query-compute' in a for a in argv) else state['memory']))
    monkeypatch.setattr('os.readlink',lambda path:'/desktop/viewer')
    gate=dict(kind='gpu_idle',minimum_free_mib=20000,allowed_compute_executables=['/desktop/viewer'])
    assert check_execution_gate(gate)['ready']
    assert check_execution_gate(gate)['exempt_compute_processes']==['123, 502']
    monkeypatch.setattr('os.readlink',lambda path:'/venv/bin/python')
    assert not check_execution_gate(gate)['ready']
    monkeypatch.setattr('os.readlink',lambda path:'/desktop/viewer')
    state['compute']='123, 1500\n'
    assert not check_execution_gate(gate)['ready']
    state['compute']='123, 502\n';state['memory']='19000, 24564\n'
    assert not check_execution_gate(gate)['ready']
