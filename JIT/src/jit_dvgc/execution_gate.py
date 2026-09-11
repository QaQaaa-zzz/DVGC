"""Read-only, fail-closed gate for sharing an accelerator with an external run.

This check is a launch-time observation, not a cross-project resource lock. Every
GPU child must recheck it. No PID is signalled and a status-file PID is never
trusted as evidence that a process is alive or dead.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping, Any


def _process_inventory(process_uid: int) -> list[dict[str, Any]]:
    """Read current identities; permission errors propagate rather than hide work."""
    result = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            uid_lines = [line for line in (directory / "status").read_text().splitlines()
                         if line.startswith("Uid:")]
            if len(uid_lines) != 1:
                raise ValueError(f"missing process UID: {directory.name}")
            uid = int(uid_lines[0].split()[1])
            if uid != process_uid:
                continue
            before = (directory / "stat").read_text()
            cmdline = (directory / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
            # Kernel threads and zombies have no executable command.
            if not cmdline or not cmdline[0]:
                continue
            if not _is_python(cmdline):
                continue
            cwd = os.readlink(directory / "cwd")
            after = (directory / "stat").read_text()
            start = before.rsplit(")", 1)[1].split()[19]
            if start != after.rsplit(")", 1)[1].split()[19]:
                raise RuntimeError(f"process identity changed during scan: {directory.name}")
            result.append({"pid": int(directory.name), "cmdline": cmdline,
                           "cwd": cwd, "start_time": start, "uid": uid})
        except (FileNotFoundError, ProcessLookupError):
            # A process that disappeared cannot still occupy the accelerator.
            continue
    return result


def _is_python(cmdline: list[str]) -> bool:
    return bool(cmdline and re.fullmatch(r"(?:python(?:\d+(?:\.\d+)*)?|pypy\d*)", Path(cmdline[0]).name))


def _within(value: str, root: Path, cwd: str) -> bool:
    value = value.split("=", 1)[-1]
    if not value or value.startswith("-"):
        return False
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    return candidate.resolve().is_relative_to(root)


def check_execution_gate(
    gate: Mapping[str, Any], *,
    process_inventory: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Return readiness and explicit reasons, freshly reading status and processes.

    Required config keys are ``status_path`` and ``process_run_path`` (absolute
    paths). ``complete_phases`` defaults to completed/complete. Optionally set
    ``process_repository_path`` to also block training/pipeline Python CLIs in
    that repository, including ones whose command only names an indirect config.
    Injected process records contain pid, cmdline (argv list), cwd and optionally
    start_time and uid. ``process_uid`` defaults to the current real UID; only
    that user's processes are monitored, not global accelerator use. Inventory
    errors within that declared scope always close the gate.
    """
    result: dict[str, Any] = {"ready": False, "reasons": [], "phase": None,
                              "matching_processes": [], "status_path": None}
    reasons = result["reasons"]
    try:
        status_path = Path(gate["status_path"])
        run_path = Path(gate["process_run_path"])
        if not status_path.is_absolute() or not run_path.is_absolute():
            raise ValueError("gate paths must be absolute")
        run_path = run_path.resolve()
        phases = gate.get("complete_phases", ["completed", "complete"])
        if not isinstance(phases, (list, tuple)) or not phases or any(
            phase not in {"completed", "complete"} for phase in phases
        ):
            raise ValueError("complete_phases must contain only completed/complete")
        repository = gate.get("process_repository_path")
        if repository is not None and not Path(repository).is_absolute():
            raise ValueError("process_repository_path must be absolute")
        process_uid = gate.get("process_uid", os.getuid())
        if type(process_uid) is not int or process_uid < 0:
            raise ValueError("process_uid must be a nonnegative integer")
        result["process_scope"] = {"uid": process_uid, "kind": "declared_user_python_processes"}
        result["status_path"] = str(status_path)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        reasons.append(f"invalid_gate_configuration: {exc}")
        return result
    try:
        status = json.loads(status_path.read_text())
        if not isinstance(status, dict):
            raise ValueError("status must be a JSON object")
        result["phase"] = status.get("phase")
        if status.get("phase") not in phases:
            reasons.append(f"external_run_not_completed: {status.get('phase')!r}")
        if any(status.get(key) for key in ("error", "errors", "exception")):
            reasons.append("external_status_reports_error")
        if status.get("status") in ("error", "failed", "failure"):
            reasons.append("external_status_reports_failure")
    except (OSError, ValueError, TypeError) as exc:
        reasons.append(f"external_status_unreadable: {exc}")
    try:
        inventory = process_inventory() if process_inventory is not None else _process_inventory(process_uid)
        for process in inventory:
            if process.get("uid", process_uid) != process_uid:
                continue
            cmdline = process["cmdline"]
            if not isinstance(cmdline, (list, tuple)) or any(not isinstance(arg, str) for arg in cmdline):
                raise ValueError("process cmdline must be an argv sequence")
            if not _is_python(cmdline):
                continue
            cwd = process["cwd"]
            matches_run = any(_within(arg, run_path, cwd) for arg in cmdline[1:])
            matches_repository = bool(repository and _within(cwd, Path(repository).resolve(), "/") and any(
                re.search(r"(?:train|pipeline)", Path(arg).stem, re.IGNORECASE)
                for arg in cmdline[1:] if not arg.startswith("-")
            ))
            if matches_run or matches_repository:
                result["matching_processes"].append(dict(process))
        if result["matching_processes"]:
            reasons.append("external_run_python_processes_alive")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        reasons.append(f"process_inventory_unreadable: {exc}")
    result["ready"] = not reasons
    return result
