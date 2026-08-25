#!/usr/bin/env python3
"""Wait for one terminal JIT run, then request one read-only Codex analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


TERMINAL_STATUSES = frozenset({"completed", "engineering_error", "aborted"})


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one read-only Codex analysis after a JIT run closes."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("--launch-log", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=_positive_seconds, default=30.0)
    parser.add_argument(
        "--codex-timeout-seconds", type=_positive_seconds, default=3600.0
    )
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status(status_path: Path) -> str | None:
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def _process_state(pid_file: Path) -> str:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return "unavailable"
    if pid <= 0:
        return "unavailable"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "not_running"
    except PermissionError:
        return "running"
    except OSError:
        return "unavailable"
    return "running"


def _wait_for_terminal_status(
    status_path: Path, pid_file: Path, poll_seconds: float
) -> str:
    reported_process_state: str | None = None
    while True:
        status = _read_status(status_path)
        if status in TERMINAL_STATUSES:
            return status
        process_state = _process_state(pid_file)
        if process_state != "running" and process_state != reported_process_state:
            print(
                f"training process state is {process_state}; waiting for terminal "
                f"status in {status_path}",
                flush=True,
            )
            reported_process_state = process_state
        time.sleep(poll_seconds)


def _create_started_marker(path: Path, *, terminal_status: str) -> bool:
    payload = {
        "terminal_status": terminal_status,
        "started_at_utc": _utc_now(),
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _captured_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _analysis_prompt(
    *,
    repository_root: Path,
    run_dir: Path,
    launch_log: Path,
    terminal_status: str,
) -> str:
    status_path = run_dir / "status.json"
    source_root = repository_root / "JIT/src"
    python = Path("/home/qy/mujoco_playground/.venv/bin/python")
    return f"""Perform one read-only post-run analysis of this exact JIT training run.

Repository root: {repository_root}
Run directory: {run_dir}
Terminal status file: {status_path}
Observed terminal status: {terminal_status}
Declared training launch log: {launch_log}

Do not edit source, configuration, documentation, run artifacts, or Git state.
Do not start or resume training, and do not load a checkpoint to continue work.
Use only read-only inspection commands. Return a concise Markdown analysis; the
Codex CLI will place your final response in the declared AUTO_ANALYSIS.md.

For a completed status, run the strict provenance verifier and report its exact
result using this repository command:
PYTHONPATH={source_root} {python} -m jit_dvgc.provenance verify-run {run_dir}

Analyze interaction accounting and terminal status. Summarize all available
training curves, including reward, episode length, RSI fraction, KL, losses,
policy standard deviation, and throughput, and call out non-finite values or
other PPO anomalies. Compare the natural-start and airborne-RSI held-out panels
at every checkpoint; keep the two routes separate. Inspect termination causes,
Apex and post-Apex behavior, final diagnostic PNG/NPZ/MP4 artifacts, and traces.
Count every line containing the literal text `CCD overflow` in the declared
training launch log. Support each conclusion with exact evidence paths. Do not
make a Tube, JCE, JEL, promotion, or trained-expert claim beyond the evidence.
"""


def _run_codex_once(
    *,
    repository_root: Path,
    run_dir: Path,
    launch_log: Path,
    terminal_status: str,
    codex_timeout_seconds: float,
) -> int:
    analysis_path = run_dir / "AUTO_ANALYSIS.md"
    log_path = run_dir / "codex_exec.log"
    completed_path = run_dir / "codex_analysis.completed.json"
    prompt = _analysis_prompt(
        repository_root=repository_root,
        run_dir=run_dir,
        launch_log=launch_log,
        terminal_status=terminal_status,
    )
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--cd",
        str(repository_root),
        "--output-last-message",
        str(analysis_path),
        prompt,
    ]
    stdout = ""
    stderr = ""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=codex_timeout_seconds,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as error:
        returncode = 124
        stdout = _captured_text(error.stdout)
        stderr = _captured_text(error.stderr)
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += (
            f"Codex analysis timed out after {codex_timeout_seconds:g} seconds.\n"
        )
    except OSError as error:
        returncode = 127
        stderr = f"{type(error).__name__}: {error}\n"
    log_path.write_text(
        f"STDOUT\n{stdout}\nSTDERR\n{stderr}", encoding="utf-8"
    )
    _write_json(
        completed_path,
        {
            "terminal_status": terminal_status,
            "returncode": returncode,
            "completed_at_utc": _utc_now(),
        },
    )
    return returncode


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"run directory does not exist: {run_dir}")
    pid_file = args.pid_file.resolve()
    launch_log = args.launch_log.resolve()
    expected_pid_file = (run_dir.parent / f"{run_dir.name}.pid").resolve()
    expected_launch_log = (
        run_dir.parent / f"{run_dir.name}.launch.log"
    ).resolve()
    if pid_file != expected_pid_file:
        parser.error(f"pid file must be exact run sibling: {expected_pid_file}")
    if launch_log != expected_launch_log:
        parser.error(
            f"launch log must be exact run sibling: {expected_launch_log}"
        )
    if not pid_file.is_file():
        parser.error(f"pid file does not exist as a file: {pid_file}")
    if not launch_log.is_file():
        parser.error(f"launch log does not exist as a file: {launch_log}")
    terminal_status = _wait_for_terminal_status(
        run_dir / "status.json", pid_file, args.poll_seconds
    )
    started_path = run_dir / "codex_analysis.started.json"
    if not _create_started_marker(started_path, terminal_status=terminal_status):
        return 0
    repository_root = Path(__file__).resolve().parents[2]
    return _run_codex_once(
        repository_root=repository_root,
        run_dir=run_dir,
        launch_log=launch_log,
        terminal_status=terminal_status,
        codex_timeout_seconds=args.codex_timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
