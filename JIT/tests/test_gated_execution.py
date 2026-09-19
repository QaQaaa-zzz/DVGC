import hashlib
import json
import subprocess

import pytest

from jit_dvgc import gated_execution as runner


def plan(tmp_path, stages=1):
    source = tmp_path / "source.py"
    source.write_text("locked fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = {"schema": "jit_gated_plan_v1", "max_interactions": 20, "wait_timeout_seconds": 1,
            "gate": {"test": True}, "input_files": {str(source): digest},
            "source_locks": {str(source): digest}, "stages": [
                {"name": f"stage{i}", "argv": ["/fake/python", "test.py"],
                 "cwd": str(tmp_path), "env": {"JAX_PLATFORMS": "cuda,cpu"},
                 "timeout_seconds": 10, "max_interactions": 10} for i in range(stages)]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data))
    return path, data


@pytest.fixture
def children(monkeypatch):
    calls = []
    class FakeProcess:
        pid = 123456
        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
        def wait(self, timeout):
            return 0
    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    return calls


def test_completed_plan_records_provenance_and_checks_every_child(tmp_path, monkeypatch, children):
    path, _ = plan(tmp_path, stages=2)
    checks = []
    def gate(config):
        checks.append(config)
        return {"ready": True}
    monkeypatch.setattr(runner, "check_execution_gate", gate)
    result = runner.run_gated_plan(path, tmp_path / "output")
    assert result["phase"] == "completed"
    assert len(children) == 2 and len(checks) == 4
    assert all(call[1]["start_new_session"] for call in children)
    assert result["reserved_interactions"] == 20
    assert result["plan_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(stage["wall_seconds"] >= 0 for stage in result["stages"])
    with pytest.raises(FileExistsError):
        runner.run_gated_plan(path, tmp_path / "output")
    assert len(children) == 2


def test_reclosed_gate_prevents_child(tmp_path, monkeypatch, children):
    path, _ = plan(tmp_path)
    assessments = iter([{"ready": True}, {"ready": False}])
    monkeypatch.setattr(runner, "check_execution_gate", lambda _: next(assessments))
    result = runner.run_gated_plan(path, tmp_path / "output")
    assert result["phase"] == "blocked" and not children


def test_reclosed_gate_waits_and_resumes_without_duplicate_launch(tmp_path, monkeypatch, children):
    path, _ = plan(tmp_path, stages=2)
    assessments=iter([True,False,False,True,False,True,True])
    monkeypatch.setattr(runner,'check_execution_gate',lambda _: {'ready':next(assessments)})
    monkeypatch.setattr(runner.time,'sleep',lambda _: None)
    result=runner.run_gated_plan(path,tmp_path/'output',wait=True)
    assert result['phase']=='completed' and len(children)==2
    assert result['reserved_interactions']==20


def test_opt_in_resource_wait_outlives_wait_deadline(tmp_path,monkeypatch,children):
    path,data=plan(tmp_path)
    data['gate']={'kind':'gpu_idle','wait_until_idle':True}
    path.write_text(json.dumps(data))
    clock=[0.]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(runner.time,'sleep',lambda s:clock.__setitem__(0,clock[0]+s))
    monkeypatch.setattr(runner,'check_execution_gate',lambda _: {'ready':clock[0]>=3})
    result=runner.run_gated_plan(path,tmp_path/'output',wait=True,poll_seconds=1)
    assert result['phase']=='completed' and len(children)==1 and clock[0]>=3


def test_only_cpu_resource_supervisor_can_wait_without_execution_timeout(tmp_path,monkeypatch):
    path,data=plan(tmp_path)
    data['gate']={'kind':'gpu_idle','wait_until_idle':True}
    data['stages'][0].update(resource_supervisor=True)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='CPU'):
        runner.run_gated_plan(path,tmp_path/'invalid')
    data['stages'][0].update(execution_backend='cpu',env={'JAX_PLATFORMS':'cpu'})
    path.write_text(json.dumps(data))
    waits=[]
    class Process:
        pid=123
        def __init__(self,*args,**kwargs):pass
        def wait(self,timeout):waits.append(timeout);return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner,'check_execution_gate',lambda _: {'ready':True})
    assert runner.run_gated_plan(path,tmp_path/'valid')['phase']=='completed'
    assert waits==[None]


def test_blocked_without_wait_never_launches(tmp_path, monkeypatch, children):
    path, _ = plan(tmp_path)
    monkeypatch.setattr(runner, "check_execution_gate", lambda _: {"ready": False})
    assert runner.run_gated_plan(path, tmp_path / "output")["phase"] == "blocked"
    assert not children


def test_wait_is_bounded_and_polls_at_most_30_seconds(tmp_path, monkeypatch, children):
    path, _ = plan(tmp_path)
    monkeypatch.setattr(runner, "check_execution_gate", lambda _: {"ready": False})
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(runner.time, "sleep", sleep)
    result = runner.run_gated_plan(path, tmp_path / "output", wait=True)
    assert result["phase"] == "gate_timeout" and not children
    assert sleeps and max(sleeps) <= 30


def test_source_change_while_waiting_aborts(tmp_path, monkeypatch, children):
    path, _ = plan(tmp_path)
    monkeypatch.setattr(runner, "check_execution_gate", lambda _: {"ready": False})
    monkeypatch.setattr(runner.time, "sleep", lambda _: (tmp_path / "source.py").write_text("changed"))
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run_gated_plan(path, tmp_path / "output", wait=True)
    assert not children
    assert json.loads((tmp_path / "output/status.json").read_text())["phase"] == "error"


def test_timeout_terminates_only_own_group_no_retry(tmp_path, monkeypatch):
    path, _ = plan(tmp_path, stages=2)
    monkeypatch.setattr(runner, "check_execution_gate", lambda _: {"ready": True})
    calls, signals = [], []
    class TimeoutProcess:
        pid = 987654
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            self.waits = 0
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fake", timeout)
            return -15
    monkeypatch.setattr(runner.subprocess, "Popen", TimeoutProcess)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    result = runner.run_gated_plan(path, tmp_path / "output")
    assert result["phase"] == "timeout"
    assert len(calls) == 1 and signals == [(987654, runner.signal.SIGTERM), (987654, runner.signal.SIGKILL)]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(max_interactions=1),
    lambda p: p.update(source_locks={}),
    lambda p: p["stages"][0].update(env={"JAX_PLATFORMS": "cuda"}),
    lambda p: p["stages"][0].update(timeout_seconds=float("inf")),
])
def test_invalid_plan_rejected_without_child(tmp_path, monkeypatch, children, mutation):
    path, data = plan(tmp_path)
    mutation(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runner.run_gated_plan(path, tmp_path / "output")
    assert not children and not (tmp_path / "output").exists()


def test_cpu_orchestrator_stage_preserves_cpu_environment(tmp_path, monkeypatch, children):
    path, data = plan(tmp_path)
    data["stages"][0].update(execution_backend="cpu", env={"JAX_PLATFORMS": "cpu"})
    path.write_text(json.dumps(data))
    monkeypatch.setattr(runner, "check_execution_gate", lambda _: {"ready": True})
    assert runner.run_gated_plan(path, tmp_path / "output")["phase"] == "completed"
    assert children[0][1]["env"]["JAX_PLATFORMS"] == "cpu"


def test_gate_block_does_not_charge_unlaunched_budget(tmp_path, monkeypatch, children):
    path, _ = plan(tmp_path)
    assessments = iter([{'ready': True}, {'ready': False}])
    monkeypatch.setattr(runner, 'check_execution_gate', lambda _: next(assessments))
    result=runner.run_gated_plan(path,tmp_path/'output')
    assert not children and result['reserved_interactions']==0


def test_interrupted_cpu_orchestrator_gets_cleanup_grace(tmp_path, monkeypatch):
    path,data=plan(tmp_path)
    data['stages'][0].update(execution_backend='cpu',env={'JAX_PLATFORMS':'cpu'})
    path.write_text(json.dumps(data))
    monkeypatch.setattr(runner,'check_execution_gate',lambda _: {'ready':True})
    waits,signals=[],[]
    class InterruptedProcess:
        pid=1234567
        def __init__(self,*args,**kwargs):
            pass
        def wait(self,timeout):
            waits.append(timeout)
            if len(waits)==1:
                raise SystemExit(143)
            return -15
    monkeypatch.setattr(runner.subprocess,'Popen',InterruptedProcess)
    monkeypatch.setattr(runner.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(SystemExit):
        runner.run_gated_plan(path,tmp_path/'output')
    assert waits==[10,30,5]
    assert signals==[(1234567,runner.signal.SIGTERM),(1234567,runner.signal.SIGKILL)]
    status=json.loads((tmp_path/'output/status.json').read_text())
    assert status['phase']=='error' and status['reserved_interactions']==10
