import json

import pytest

from jit_dvgc.execution_gate import check_execution_gate


def gate(tmp_path, phase="completed"):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"phase": phase}))
    return {"status_path": str(status), "process_run_path": str(tmp_path / "runX"),
            "complete_phases": ["completed", "complete"]}


def process(config, executable="python", path=None):
    return {"pid": 123, "cmdline": [executable, "train.py", "--output", path or config["process_run_path"]],
            "cwd": "/repo", "start_time": "1234"}


def test_completed_with_live_python_is_blocked(tmp_path):
    config = gate(tmp_path)
    result = check_execution_gate(config, process_inventory=lambda: [process(config)])
    assert not result["ready"]
    assert result["matching_processes"][0]["pid"] == 123


def test_running_without_pid_is_blocked(tmp_path):
    assert not check_execution_gate(gate(tmp_path, "running"), process_inventory=lambda: [])["ready"]


@pytest.mark.parametrize("phase", ["unknown", "error", "failed", None])
def test_nonpositive_status_fails_closed(tmp_path, phase):
    assert not check_execution_gate(gate(tmp_path, phase), process_inventory=lambda: [])["ready"]


@pytest.mark.parametrize("contents", [None, "{", "[]", '{"phase":"completed","error":"failure"}'])
def test_missing_malformed_or_error_fails_closed(tmp_path, contents):
    config = gate(tmp_path)
    status = tmp_path / "status.json"
    if contents is None:
        status.unlink()
    else:
        status.write_text(contents)
    assert not check_execution_gate(config, process_inventory=lambda: [])["ready"]


def test_viewer_and_path_prefix_do_not_block(tmp_path):
    config = gate(tmp_path)
    processes = [process(config, "eog"), process(config, path=config["process_run_path"] + "extra")]
    assert check_execution_gate(config, process_inventory=lambda: processes)["ready"]


def test_child_path_and_equals_argument_block(tmp_path):
    config = gate(tmp_path)
    proc = process(config)
    proc["cmdline"] = ["/usr/bin/python3.11", "train.py", "--output=" + config["process_run_path"] + "/child"]
    assert not check_execution_gate(config, process_inventory=lambda: [proc])["ready"]


def test_scan_permission_denial_fails_closed(tmp_path):
    def denied():
        raise PermissionError("denied")
    assert not check_execution_gate(gate(tmp_path), process_inventory=denied)["ready"]


def test_status_pid_reuse_does_not_block_unrelated_python(tmp_path):
    config = gate(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"phase": "completed", "pid": 123}))
    assert check_execution_gate(config, process_inventory=lambda: [process(config, path="/other/run")])["ready"]


def test_status_is_read_each_call(tmp_path):
    config = gate(tmp_path)
    assert check_execution_gate(config, process_inventory=lambda: [])["ready"]
    (tmp_path / "status.json").write_text('{"phase":"running"}')
    assert not check_execution_gate(config, process_inventory=lambda: [])["ready"]


def test_repository_pipeline_process_blocks(tmp_path):
    config = gate(tmp_path)
    config["process_repository_path"] = "/repo"
    proc = process(config, path="relative-config.json")
    proc["cmdline"][1] = "learning/cli/run_pipeline.py"
    assert not check_execution_gate(config, process_inventory=lambda: [proc])["ready"]


def test_declared_uid_filters_other_users(tmp_path):
    config = gate(tmp_path)
    config["process_uid"] = 1001
    proc = process(config)
    proc["uid"] = 0
    result = check_execution_gate(config, process_inventory=lambda: [proc])
    assert result["ready"]
    assert result["process_scope"]["uid"] == 1001
    proc["uid"] = 1001
    assert not check_execution_gate(config, process_inventory=lambda: [proc])["ready"]


@pytest.mark.parametrize("uid", [-1, "1000", True, None])
def test_invalid_uid_fails_closed(tmp_path, uid):
    config = gate(tmp_path)
    config["process_uid"] = uid
    assert not check_execution_gate(config, process_inventory=lambda: [])["ready"]


def test_scanner_skips_other_uid_before_cwd_and_rejects_same_uid_denial(tmp_path, monkeypatch):
    from jit_dvgc import execution_gate
    proc_root = tmp_path / "proc"
    directory = proc_root / "123"
    directory.mkdir(parents=True)
    (directory / "status").write_text("Uid:\t0\t0\t0\t0\n")
    real_path = execution_gate.Path
    monkeypatch.setattr(execution_gate, "Path", lambda value: proc_root if value == "/proc" else real_path(value))
    assert execution_gate._process_inventory(1001) == []
    (directory / "status").write_text("Uid:\t1001\t1001\t1001\t1001\n")
    (directory / "stat").write_text("123 (python) " + " ".join(["0"] * 30))
    (directory / "cmdline").write_bytes(b"python\0train.py\0")
    def denied(path):
        raise PermissionError("same UID cwd denied")
    monkeypatch.setattr(execution_gate.os, "readlink", denied)
    config = gate(tmp_path)
    config["process_uid"] = 1001
    result = check_execution_gate(config)
    assert not result["ready"]
    assert any("same UID cwd denied" in reason for reason in result["reasons"])
