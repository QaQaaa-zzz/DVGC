"""Finite, provenance-locked external-run-gated subprocess execution.

Interaction accounting reserves declared stage maxima; each scientific child
must enforce its own interaction cap. This runner never retries a child.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any

from .execution_gate import check_execution_gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate(plan: dict[str, Any]) -> None:
    if plan.get("schema") != "jit_gated_plan_v1":
        raise ValueError("unsupported gated plan schema")
    if type(plan.get("max_interactions")) is not int or plan["max_interactions"] < 1:
        raise ValueError("positive integer total interaction budget required")
    if not _positive(plan.get("wait_timeout_seconds")):
        raise ValueError("finite positive wait_timeout_seconds required")
    if not isinstance(plan.get("gate"), dict) or not plan["gate"]:
        raise ValueError("external execution gate required")
    for key in ("input_files", "source_locks"):
        locks = plan.get(key)
        if not isinstance(locks, dict) or not locks:
            raise ValueError(f"nonempty {key} hash mapping required")
        for path, digest in locks.items():
            if not Path(path).is_absolute() or not re.fullmatch("[0-9a-f]{64}", str(digest)):
                raise ValueError(f"invalid {key} entry: {path}")
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("nonempty finite stage list required")
    names = set()
    for stage in stages:
        name = stage.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("unique filesystem-safe stage names required")
        names.add(name)
        argv = stage.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in argv):
            raise ValueError("stage argv must be a nonempty exact string list")
        if not Path(argv[0]).is_absolute() or not Path(stage.get("cwd", "")).is_absolute():
            raise ValueError("absolute child executable and cwd required")
        env = stage.get("env")
        backend = stage.get("execution_backend", "gpu")
        if backend not in ("cpu", "gpu"):
            raise ValueError("execution_backend must be cpu or gpu")
        expected_platforms = "cpu" if backend == "cpu" else "cuda,cpu"
        if not isinstance(env, dict) or env.get("JAX_PLATFORMS") != expected_platforms or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
        ):
            raise ValueError(f"explicit child environment with JAX_PLATFORMS={expected_platforms} required")
        if not _positive(stage.get("timeout_seconds")):
            raise ValueError("finite positive stage timeout required")
        if type(stage.get("max_interactions")) is not int or stage["max_interactions"] < 0:
            raise ValueError("nonnegative integer stage interaction maximum required")
    if sum(stage["max_interactions"] for stage in stages) > plan["max_interactions"]:
        raise ValueError("stage interaction maxima exceed declared plan budget")


def _verify_locks(plan: dict[str, Any], plan_path: Path, digest: str) -> None:
    if _sha(plan_path) != digest:
        raise ValueError("immutable plan changed")
    for kind in ("input_files", "source_locks"):
        for path, expected in plan[kind].items():
            if _sha(Path(path)) != expected:
                raise ValueError(f"{kind} hash mismatch: {path}")


def _write_status(output: Path, status: dict[str, Any]) -> None:
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "status.json")


def _stop_owned_process(process: subprocess.Popen, grace_seconds: float = 5) -> None:
    """Only the process group we created with start_new_session is signalled."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit while its children remain; terminate that same owned
    # session group as well, never another project's process or status-file PID.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_gated_plan(plan_path: Path, output_dir: Path, *, wait: bool = False,
                   poll_seconds: float = 30.0) -> dict[str, Any]:
    """Execute one fresh output once; record blocked/failure state without retry."""
    plan_path, output = Path(plan_path).resolve(), Path(output_dir).resolve()
    if not _positive(poll_seconds) or poll_seconds > 30:
        raise ValueError("poll_seconds must be positive and at most 30")
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    _validate(plan)
    digest = hashlib.sha256(raw).hexdigest()
    _verify_locks(plan, plan_path, digest)
    # Exclusive creation is also the no-rerun lock, including failed/blocked runs.
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(raw)
    started = time.monotonic()
    status: dict[str, Any] = {"schema": "jit_gated_execution_v1", "phase": "prepared",
        "plan_path": str(plan_path), "plan_sha256": digest, "started_unix": time.time(),
        "max_interactions": plan["max_interactions"], "reserved_interactions": 0,
        "interaction_accounting": "declared_maxima_reserved_not_measured_actuals", "stages": []}
    _write_status(output, status)
    try:
        for stage in plan["stages"]:
            gate_start = time.monotonic()
            while True:
                _verify_locks(plan, plan_path, digest)
                assessment = check_execution_gate(plan["gate"])
                status["gate"] = assessment
                if assessment["ready"]:
                    break
                status["phase"] = "waiting" if wait else "blocked"
                _write_status(output, status)
                if not wait:
                    return status
                remaining = plan["wait_timeout_seconds"] - (time.monotonic() - gate_start)
                if remaining <= 0:
                    status["phase"] = "gate_timeout"
                    return status
                time.sleep(min(poll_seconds, remaining))
            record = {"name": stage["name"], "argv": stage["argv"], "cwd": stage["cwd"],
                      "env_overrides": stage["env"], "execution_backend": stage.get("execution_backend", "gpu"),
                      "max_interactions": stage["max_interactions"],
                      "timeout_seconds": stage["timeout_seconds"], "phase": "launching",
                      "started_unix": time.time()}
            status["stages"].append(record)
            status["phase"] = "running"
            _write_status(output, status)
            stage_start = time.monotonic()
            process = None
            try:
                with (output / f"{stage['name']}.log").open("xb") as log:
                    # Recheck immediately before each process creation.
                    _verify_locks(plan, plan_path, digest)
                    assessment = check_execution_gate(plan["gate"])
                    status["gate"] = assessment
                    if not assessment["ready"]:
                        record["phase"] = status["phase"] = "blocked"
                        return status
                    process = subprocess.Popen(stage["argv"], cwd=stage["cwd"],
                        env={**os.environ, **stage["env"]}, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    status["reserved_interactions"] += stage["max_interactions"]
                    record["pid"] = process.pid
                    _write_status(output, status)
                    record["returncode"] = process.wait(timeout=stage["timeout_seconds"])
                    record["phase"] = "completed" if record["returncode"] == 0 else "failed"
            except subprocess.TimeoutExpired:
                if process is not None:
                    _stop_owned_process(process, 30 if stage.get("execution_backend") == "cpu" else 5)
                record["phase"] = "timeout"
            except BaseException:
                if process is not None:
                    _stop_owned_process(process, 30 if stage.get("execution_backend") == "cpu" else 5)
                record["phase"] = "error"
                raise
            finally:
                record["wall_seconds"] = time.monotonic() - stage_start
                record["finished_unix"] = time.time()
                _write_status(output, status)
            if record["phase"] != "completed":
                status["phase"] = record["phase"]
                return status
        status["phase"] = "completed"
        return status
    except BaseException as exc:
        status["phase"] = "error"
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - started
        status["finished_unix"] = time.time()
        _write_status(output, status)
