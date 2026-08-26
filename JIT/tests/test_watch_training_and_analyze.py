from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def _write_status(run_dir: Path, status: str) -> None:
    (run_dir / "status.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )


def _make_fake_codex(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "codex"
    invocation_path = tmp_path / "codex_invocations.jsonl"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

arguments = sys.argv[1:]
invocations = Path(os.environ["FAKE_CODEX_INVOCATIONS"])
with invocations.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\\n")
print("fake codex stdout", flush=True)
print("fake codex stderr", file=sys.stderr, flush=True)
time.sleep(float(os.environ.get("FAKE_CODEX_SLEEP_SECONDS", "0")))
exit_code = int(os.environ.get("FAKE_CODEX_EXIT", "0"))
if exit_code == 0:
    output_index = arguments.index("--output-last-message") + 1
    Path(arguments[output_index]).write_text(
        "fake read-only analysis\\n", encoding="utf-8"
    )
raise SystemExit(exit_code)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir, invocation_path


def _watcher_command(
    jit_root: Path,
    *,
    run_dir: Path,
    pid_file: Path,
    launch_log: Path,
    poll_seconds: str = "0.02",
    codex_timeout_seconds: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(jit_root / "cli/watch_training_and_analyze.py"),
        "--run-dir",
        str(run_dir),
        "--pid-file",
        str(pid_file),
        "--launch-log",
        str(launch_log),
        f"--poll-seconds={poll_seconds}",
    ]
    if codex_timeout_seconds is not None:
        command.append(f"--codex-timeout-seconds={codex_timeout_seconds}")
    return command


def _watcher_environment(bin_dir: Path, invocation_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    environment["FAKE_CODEX_INVOCATIONS"] = str(invocation_path)
    return environment


def _read_invocations(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_completed_run_invokes_read_only_ephemeral_analysis_once(
    jit_root: Path, repository_root: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "formal_run"
    run_dir.mkdir()
    _write_status(run_dir, "completed")
    pid_file = tmp_path / "formal_run.pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    launch_log = tmp_path / "formal_run.launch.log"
    launch_log.write_text(
        "startup\nCCD overflow first\nCCD overflow second\n", encoding="utf-8"
    )
    bin_dir, invocation_path = _make_fake_codex(tmp_path)
    command = _watcher_command(
        jit_root, run_dir=run_dir, pid_file=pid_file, launch_log=launch_log
    )
    environment = _watcher_environment(bin_dir, invocation_path)

    first = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    invocations = _read_invocations(invocation_path)
    assert len(invocations) == 1
    arguments = invocations[0]
    assert arguments[:2] == ["exec", "--ephemeral"]
    assert arguments[arguments.index("--model") + 1] == "gpt-5.6-luna"
    assert arguments[arguments.index("--sandbox") + 1] == "read-only"
    assert Path(arguments[arguments.index("--cd") + 1]) == repository_root
    assert Path(arguments[arguments.index("--output-last-message") + 1]) == (
        run_dir / "AUTO_ANALYSIS.md"
    )
    prompt = arguments[-1]
    assert str(run_dir) in prompt
    assert str(run_dir / "status.json") in prompt
    assert str(launch_log) in prompt
    assert "Do not edit" in prompt
    assert "Do not start or resume training" in prompt
    assert "verify-run" in prompt
    assert "natural-start" in prompt
    assert "airborne-RSI" in prompt
    assert "training curves" in prompt
    assert "termination" in prompt
    assert "CCD overflow" in prompt
    assert "evidence paths" in prompt
    assert (run_dir / "AUTO_ANALYSIS.md").read_text(encoding="utf-8")
    assert json.loads(
        (run_dir / "codex_analysis.started.json").read_text(encoding="utf-8")
    )["terminal_status"] == "completed"
    completed = json.loads(
        (run_dir / "codex_analysis.completed.json").read_text(encoding="utf-8")
    )
    assert completed["returncode"] == 0
    assert completed["terminal_status"] == "completed"
    codex_log = (run_dir / "codex_exec.log").read_text(encoding="utf-8")
    assert "fake codex stdout" in codex_log
    assert "fake codex stderr" in codex_log

    second = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert second.returncode == 0, second.stderr
    assert len(_read_invocations(invocation_path)) == 1


def test_dead_training_pid_does_not_trigger_analysis_before_terminal_status(
    jit_root: Path, repository_root: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "dead_process_run"
    run_dir.mkdir()
    _write_status(run_dir, "running")
    pid_file = tmp_path / "dead_process_run.pid"
    pid_file.write_text("999999999\n", encoding="utf-8")
    launch_log = tmp_path / "dead_process_run.launch.log"
    launch_log.write_text("process exited\n", encoding="utf-8")
    bin_dir, invocation_path = _make_fake_codex(tmp_path)
    process = subprocess.Popen(
        _watcher_command(
            jit_root, run_dir=run_dir, pid_file=pid_file, launch_log=launch_log
        ),
        cwd=repository_root,
        env=_watcher_environment(bin_dir, invocation_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.12)
        assert process.poll() is None
        assert _read_invocations(invocation_path) == []
        assert not (run_dir / "codex_analysis.started.json").exists()

        _write_status(run_dir, "completed")
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    assert process.returncode == 0, (stdout, stderr)
    assert len(_read_invocations(invocation_path)) == 1
    completed = json.loads(
        (run_dir / "codex_analysis.completed.json").read_text(encoding="utf-8")
    )
    assert completed["terminal_status"] == "completed"


def test_failed_codex_attempt_is_logged_completed_and_never_retried(
    jit_root: Path, repository_root: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "failed_analysis_run"
    run_dir.mkdir()
    _write_status(run_dir, "aborted")
    pid_file = tmp_path / "failed_analysis_run.pid"
    pid_file.write_text("999999999\n", encoding="utf-8")
    launch_log = tmp_path / "failed_analysis_run.launch.log"
    launch_log.write_text("aborted\n", encoding="utf-8")
    bin_dir, invocation_path = _make_fake_codex(tmp_path)
    environment = _watcher_environment(bin_dir, invocation_path)
    environment["FAKE_CODEX_EXIT"] = "7"
    command = _watcher_command(
        jit_root, run_dir=run_dir, pid_file=pid_file, launch_log=launch_log
    )

    first = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 7
    assert len(_read_invocations(invocation_path)) == 1
    completed = json.loads(
        (run_dir / "codex_analysis.completed.json").read_text(encoding="utf-8")
    )
    assert completed["returncode"] == 7
    codex_log = (run_dir / "codex_exec.log").read_text(encoding="utf-8")
    assert "fake codex stdout" in codex_log
    assert "fake codex stderr" in codex_log

    second = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert second.returncode == 0, second.stderr
    assert len(_read_invocations(invocation_path)) == 1


def test_codex_start_failure_still_writes_log_and_completed_marker(
    jit_root: Path, repository_root: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "missing_codex_run"
    run_dir.mkdir()
    _write_status(run_dir, "engineering_error")
    pid_file = tmp_path / "missing_codex_run.pid"
    pid_file.write_text("999999999\n", encoding="utf-8")
    launch_log = tmp_path / "missing_codex_run.launch.log"
    launch_log.write_text("startup failure\n", encoding="utf-8")
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()
    environment = os.environ.copy()
    environment["PATH"] = str(empty_bin)
    command = _watcher_command(
        jit_root, run_dir=run_dir, pid_file=pid_file, launch_log=launch_log
    )

    first = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 127
    assert "FileNotFoundError" in (run_dir / "codex_exec.log").read_text(
        encoding="utf-8"
    )
    completed = json.loads(
        (run_dir / "codex_analysis.completed.json").read_text(encoding="utf-8")
    )
    assert completed["returncode"] == 127
    assert completed["terminal_status"] == "engineering_error"

    second = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert second.returncode == 0, second.stderr


def test_blocking_codex_times_out_with_partial_output_and_is_never_retried(
    jit_root: Path, repository_root: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "timed_out_analysis_run"
    run_dir.mkdir()
    _write_status(run_dir, "completed")
    pid_file = tmp_path / "timed_out_analysis_run.pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    launch_log = tmp_path / "timed_out_analysis_run.launch.log"
    launch_log.write_text("training completed\n", encoding="utf-8")
    bin_dir, invocation_path = _make_fake_codex(tmp_path)
    environment = _watcher_environment(bin_dir, invocation_path)
    environment["FAKE_CODEX_SLEEP_SECONDS"] = "2"
    command = _watcher_command(
        jit_root,
        run_dir=run_dir,
        pid_file=pid_file,
        launch_log=launch_log,
        codex_timeout_seconds="0.05",
    )

    started_at = time.monotonic()
    first = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started_at

    assert first.returncode == 124
    assert elapsed < 1.5
    assert len(_read_invocations(invocation_path)) == 1
    codex_log = (run_dir / "codex_exec.log").read_text(encoding="utf-8")
    assert "fake codex stdout" in codex_log
    assert "fake codex stderr" in codex_log
    assert "timed out after 0.05 seconds" in codex_log
    completed = json.loads(
        (run_dir / "codex_analysis.completed.json").read_text(encoding="utf-8")
    )
    assert completed["returncode"] == 124
    assert completed["terminal_status"] == "completed"

    second = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert second.returncode == 0, second.stderr
    assert len(_read_invocations(invocation_path)) == 1


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--poll-seconds", "nan"),
        ("--poll-seconds", "inf"),
        ("--poll-seconds", "-inf"),
        ("--codex-timeout-seconds", "nan"),
        ("--codex-timeout-seconds", "inf"),
        ("--codex-timeout-seconds", "-inf"),
    ],
)
def test_cli_rejects_nonfinite_intervals(
    jit_root: Path,
    repository_root: Path,
    tmp_path: Path,
    option: str,
    value: str,
) -> None:
    run_dir = tmp_path / f"invalid_{option.removeprefix('--').replace('-', '_')}_{value}"
    run_dir.mkdir()
    _write_status(run_dir, "completed")
    pid_file = tmp_path / f"{run_dir.name}.pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    launch_log = tmp_path / f"{run_dir.name}.launch.log"
    launch_log.write_text("completed\n", encoding="utf-8")
    command_options = (
        {"poll_seconds": value}
        if option == "--poll-seconds"
        else {"codex_timeout_seconds": value}
    )
    command = _watcher_command(
        jit_root,
        run_dir=run_dir,
        pid_file=pid_file,
        launch_log=launch_log,
        **command_options,
    )

    result = subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "finite positive number" in result.stderr
    assert not (run_dir / "codex_analysis.started.json").exists()


@pytest.mark.parametrize("wrong_input", ["pid", "launch_log"])
def test_wrong_sibling_input_is_rejected_before_analysis(
    jit_root: Path,
    repository_root: Path,
    tmp_path: Path,
    wrong_input: str,
) -> None:
    run_dir = tmp_path / "exact_run"
    run_dir.mkdir()
    _write_status(run_dir, "completed")
    expected_pid = tmp_path / "exact_run.pid"
    expected_pid.write_text(f"{os.getpid()}\n", encoding="utf-8")
    expected_log = tmp_path / "exact_run.launch.log"
    expected_log.write_text("completed\n", encoding="utf-8")
    wrong_pid = tmp_path / "different.pid"
    wrong_pid.write_text(f"{os.getpid()}\n", encoding="utf-8")
    wrong_log = tmp_path / "different.launch.log"
    wrong_log.write_text("completed\n", encoding="utf-8")
    pid_file = wrong_pid if wrong_input == "pid" else expected_pid
    launch_log = wrong_log if wrong_input == "launch_log" else expected_log
    bin_dir, invocation_path = _make_fake_codex(tmp_path)

    result = subprocess.run(
        _watcher_command(
            jit_root,
            run_dir=run_dir,
            pid_file=pid_file,
            launch_log=launch_log,
        ),
        cwd=repository_root,
        env=_watcher_environment(bin_dir, invocation_path),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "exact run sibling" in result.stderr
    assert _read_invocations(invocation_path) == []
    assert not (run_dir / "codex_analysis.started.json").exists()


@pytest.mark.parametrize("missing_input", ["pid", "launch_log"])
def test_missing_exact_input_is_rejected_before_analysis(
    jit_root: Path,
    repository_root: Path,
    tmp_path: Path,
    missing_input: str,
) -> None:
    run_dir = tmp_path / "missing_input_run"
    run_dir.mkdir()
    _write_status(run_dir, "completed")
    pid_file = tmp_path / "missing_input_run.pid"
    launch_log = tmp_path / "missing_input_run.launch.log"
    if missing_input != "pid":
        pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    if missing_input != "launch_log":
        launch_log.write_text("completed\n", encoding="utf-8")
    bin_dir, invocation_path = _make_fake_codex(tmp_path)

    result = subprocess.run(
        _watcher_command(
            jit_root,
            run_dir=run_dir,
            pid_file=pid_file,
            launch_log=launch_log,
        ),
        cwd=repository_root,
        env=_watcher_environment(bin_dir, invocation_path),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "does not exist" in result.stderr
    assert _read_invocations(invocation_path) == []
    assert not (run_dir / "codex_analysis.started.json").exists()
