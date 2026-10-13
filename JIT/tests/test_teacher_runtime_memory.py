"""Transient Warp graph ownership must end without discarding explicit kernels."""
from types import SimpleNamespace
import json
import weakref

import jax.numpy as jp
import numpy as np
import pytest

from jit_dvgc.generative_bridge.teacher_runtime import TeacherEvaluationSession


def graphs(monkeypatch):
    from mujoco.mjx.warp.ffi import ffi
    import warp._src.jax_experimental.ffi as external
    class Capture:
        pass
    capture=Capture();reference=weakref.ref(capture)
    monkeypatch.setattr(ffi,'_FFI_CALLABLE_REGISTRY',{'temporary':SimpleNamespace(captures={1:capture})})
    monkeypatch.setattr(external,'_FFI_CALLABLE_REGISTRY',{})
    return ffi,reference


def test_boundary_releases_registered_graphs_but_reuses_exact_compiled_kernel(tmp_path,monkeypatch):
    ffi,reference=graphs(monkeypatch)
    session=TeacherEvaluationSession()
    function=lambda state,actions:state+actions
    first=session.execute('fixed',function,jp.zeros(2),jp.array([1.,2.]))
    assert isinstance(first,np.ndarray)
    report=session.memory_boundary('review',tmp_path,max_rss_bytes=32*1024**3)
    assert reference() is None and not ffi._FFI_CALLABLE_REGISTRY['temporary'].captures
    np.testing.assert_array_equal(session.execute('fixed',function,jp.zeros(2),jp.array([1.,2.])),first)
    assert session.compile_count==1 and len(session.kernels)==1
    assert report['before']['warp_graph_captures']==1 and report['after']['warp_graph_captures']==0
    assert report['after']['rss_bytes']>0
    assert json.loads((tmp_path/'memory_status.json').read_text())['stage']=='review'


def test_cleanup_reaches_actual_mjx_backend_and_separate_external_registry(tmp_path,monkeypatch):
    ffi,reference=graphs(monkeypatch)
    import warp._src.jax_experimental.ffi as external
    class Capture:
        pass
    extra=Capture();other=weakref.ref(extra)
    monkeypatch.setattr(external,'_FFI_CALLABLE_REGISTRY',{'other':SimpleNamespace(captures={1:extra})})
    del extra
    report=TeacherEvaluationSession().memory_boundary('restore',tmp_path,max_rss_bytes=32*1024**3)
    assert reference() is None and other() is None
    assert report['before']['warp_graph_captures']==2
    assert report['before']['warp_ffi_backends'][ffi.__name__]['graph_captures']==1
    assert report['after']['warp_graph_captures']==0


def test_boundary_guard_records_an_auditable_stop_after_release(tmp_path,monkeypatch):
    ffi,reference=graphs(monkeypatch)
    with pytest.raises(MemoryError,match='RSS'):
        TeacherEvaluationSession().memory_boundary('restore',tmp_path,max_rss_bytes=1)
    record=json.loads((tmp_path/'memory_status.json').read_text())
    assert record['state']=='host_rss_limit_exceeded'
    assert record['after']['warp_graph_captures']==0 and reference() is None
    assert (tmp_path/'memory.jsonl').is_file()


def test_persistent_child_releases_graphs_after_host_trace_materialization(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.teacher_worker import PersistentTeacherRunner
    import jit_dvgc.pulse_exploration_runtime as runtime
    from jit_dvgc.generative_bridge.protocol import atomic_json
    ffi,reference=graphs(monkeypatch)
    runner=PersistentTeacherRunner.__new__(PersistentTeacherRunner)
    runner.root=tmp_path;runner.costs=[];runner.session=TeacherEvaluationSession()
    runner.spec={'budgets':{'max_physics':10},'teacher_host_rss_limit_bytes':32*1024**3}
    runner.status=lambda *args,**kw:None
    config=tmp_path/'spec.json';atomic_json(config,{})
    observed=[]
    def evaluate(spec,out,*,session):
        trace=session.execute('lane',lambda x:x*2,jp.ones(3))
        assert isinstance(trace,np.ndarray) and reference() is not None
        observed.append(trace.tolist())
    monkeypatch.setattr(runtime,'evaluate',evaluate)
    cost=runner.child('review',['JIT/cli/run_pulse_exploration.py','--mode','evaluate',
                               '--spec',str(config),'--output',str(tmp_path/'physical')],10)
    assert observed==[[2.,2.,2.]] and reference() is None
    assert cost['phase']=='completed' and runner.session.compile_count==1
    assert json.loads((tmp_path/'memory_status.json').read_text())['stage']=='review'
