"""Low-overhead, out-of-process watchdog for the persistent DVGC pipeline."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/qy/DVGC")
DEFAULT_RUN = ROOT / "runs/stage_experts/descent_tube_seed0_20260716T2330"
CONTROLLER_UNIT = "dvgc-descent-tube-controller.service"
RESUME_UNIT = "dvgc-descent-tube-postaudit-resume.service"
STATUS_JSON = ROOT / "runs/CURRENT_PIPELINE_STATUS.json"
STATUS_MD = ROOT / "runs/CURRENT_PIPELINE_STATUS.md"
WATCHDOG_STATE = ROOT / "runs/DVGC_WATCHDOG_STATE.json"
NOTIFICATION_STATE = ROOT / "runs/DVGC_NOTIFICATION_STATE.json"
PENDING_NOTIFICATION = ROOT / "runs/PENDING_NOTIFICATION.json"
HEARTBEAT_STALE_SECONDS = 600.0
RECOVERY_COOLDOWN_SECONDS = 120.0
MAX_RECOVERIES = 3


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {} if default is None else default


def unit_properties(unit: str) -> dict:
    names = ("LoadState", "ActiveState", "SubState", "MainPID", "Result",
             "ExecMainCode", "ExecMainStatus", "FragmentPath")
    command = ["systemctl", "--user", "show", unit]
    for name in names:
        command.extend(["--property", name])
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    values = {name: "" for name in names}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    values["unit"] = unit
    values["pid"] = int(values.get("MainPID") or 0)
    values["active"] = values.get("ActiveState") == "active"
    return values


def active_worker_units() -> list[str]:
    result = subprocess.run(
        ["systemctl", "--user", "list-units", "dvgc-descent-worker-*.service",
         "--state=active", "--no-legend", "--plain"],
        capture_output=True, text=True, timeout=10,
    )
    return sorted(line.split()[0] for line in result.stdout.splitlines() if line.strip())


def process_count(token: str) -> int:
    count = 0
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        count += token in command
    return count


def pid_alive(pid: int) -> bool:
    return int(pid) > 0 and Path(f"/proc/{int(pid)}").exists()


def valid_completed_marker(path: Path) -> tuple[bool, dict]:
    payload = load_json(path, {})
    valid = bool(payload.get("status") == "PASS" and isinstance(payload.get("rows"), list))
    return valid, payload


def progress_snapshot(state: dict) -> dict:
    expected = [Path(value) for value in state.get("expected_outputs", [])]
    shard = next((path for path in expected if "shard_" in path.name), None)
    root = shard.parent if shard else None
    completed_paths = sorted(root.glob("shard_*.completed.json")) if root and root.exists() else []
    completed_indices = []
    invalid_markers = []
    latest_mtime = 0.0
    total = None
    for path in completed_paths:
        valid, payload = valid_completed_marker(path)
        latest_mtime = max(latest_mtime, path.stat().st_mtime)
        if not valid:
            invalid_markers.append(str(path))
            continue
        completed_indices.extend(int(row["candidate_index"]) for row in payload["rows"])
        total = int(payload.get("total_states", total or 0)) or total
    if root:
        manifest = load_json(root / "pointwise_audit_manifest.json", {})
        total = int(manifest.get("states", total or 0)) or total
    current_range = None
    if shard:
        match = re.search(r"shard_(\d+)_(\d+)", shard.name)
        if match:
            current_range = [int(match.group(1)), int(match.group(2))]
    partial = []
    if root and root.exists():
        partial.extend(str(path) for pattern in ("*.partial", "*.tmp") for path in root.glob(pattern))
    completed_unique = sorted(set(completed_indices))
    return {
        "root": str(root) if root else None,
        "completed": len(completed_unique),
        "total": total,
        "completed_fraction": (len(completed_unique) / total if total else None),
        "completed_index_min": min(completed_unique) if completed_unique else None,
        "completed_index_max": max(completed_unique) if completed_unique else None,
        "current_index_range": current_range,
        "completed_markers": len(completed_paths) - len(invalid_markers),
        "partial_files": partial,
        "invalid_completed_markers": invalid_markers,
        "latest_marker_mtime": latest_mtime or None,
    }


def result_paths(state: dict, progress: dict) -> list[str]:
    paths = []
    paths.extend(str(value) for value in state.get("expected_outputs", []))
    history = state.get("history", [])
    if history:
        paths.extend(str(value) for value in history[-1].get("outputs", []))
    root = Path(progress["root"]) if progress.get("root") else None
    if root:
        for name in ("analysis.json", "merged.json", "pointwise_audit_manifest.json",
                     "seed_intersection_proof.json"):
            if (root / name).exists():
                paths.append(str(root / name))
    return list(dict.fromkeys(paths))


def core_metrics(paths: list[str]) -> dict:
    metrics = {}
    for value in reversed(paths):
        path = Path(value)
        if path.suffix != ".json" or not path.exists():
            continue
        payload = load_json(path, {})
        pointwise = payload.get("pointwise", {})
        calibration = payload.get("calibration", {})
        for name, item in (
            ("pointwise_precision", pointwise.get("precision")),
            ("pointwise_recall", pointwise.get("recall")),
            ("coverage", pointwise.get("candidate_mass_coverage")),
            ("aggregate_final_rate", payload.get("aggregate_final_rate")),
            ("physical_failure_rate", payload.get("physical_failure_rate")),
            ("timeout_rate", payload.get("timeout_rate")),
            ("brier", calibration.get("brier")),
            ("ece", calibration.get("ece")),
        ):
            if item is not None and name not in metrics:
                metrics[name] = item
    return metrics


def collect_status(run: Path, now: float | None = None) -> dict:
    now = time.time() if now is None else float(now)
    state_path = run / "controller_state.json"
    lock_path = run / "controller.lock"
    state = load_json(state_path, {})
    lock = load_json(lock_path, {})
    controller = unit_properties(CONTROLLER_UNIT)
    workers = active_worker_units()
    named_worker = str(state.get("active_worker_unit") or "")
    worker = unit_properties(named_worker) if named_worker else {
        "unit": None, "pid": 0, "active": False, "ActiveState": "inactive", "SubState": "dead"
    }
    progress = progress_snapshot(state)
    heartbeat = float(state.get("heartbeat", 0.0) or 0.0)
    latest_progress = max(float(progress.get("latest_marker_mtime") or 0.0),
                          float(state.get("history", [{}])[-1].get("completed_at", 0.0) if state.get("history") else 0.0))
    provenance = state.get("provenance", {})
    expected_policy = provenance.get("current_policy_hash") or provenance.get("policy_hash")
    lock_pid = int(lock.get("pid", 0) or 0)
    lock_policy = lock.get("policy_hash")
    terminal = None
    stage = str(state.get("current_stage") or "unknown")
    transition_pending = bool(
        stage == "gate_pause"
        and "Round-2 exact pointwise Tube precision" in str(state.get("stop_reason") or "")
        and unit_properties(RESUME_UNIT).get("active")
    )
    if stage == "pipeline_complete":
        terminal = "pipeline_complete"
    elif stage == "gate_pause" and not transition_pending:
        terminal = "gate_pause"
    elif stage == "authorized_stop":
        terminal = "engineering_failure_after_retries"
    reports = result_paths(state, progress)
    log_paths = [str(run / "persistent_controller.log"), str(run / "systemd-controller.stdout.log")]
    status = {
        "timestamp": dt.datetime.fromtimestamp(now, dt.timezone.utc).astimezone().isoformat(),
        "timestamp_epoch": now,
        "run_id": run.name,
        "run_path": str(run),
        "controller_unit": CONTROLLER_UNIT,
        "controller_pid": controller["pid"],
        "controller_active": controller["active"],
        "controller_active_state": controller.get("ActiveState"),
        "controller_substate": controller.get("SubState"),
        "controller_result": controller.get("Result"),
        "controller_exit_code": controller.get("ExecMainCode"),
        "controller_exit_status": controller.get("ExecMainStatus"),
        "worker_unit": worker.get("unit"),
        "worker_pid": worker.get("pid", 0),
        "worker_active": worker.get("active", False),
        "worker_active_state": worker.get("ActiveState"),
        "worker_substate": worker.get("SubState"),
        "active_worker_units": workers,
        "current_stage": stage,
        "progress": progress,
        "heartbeat_age_seconds": now - heartbeat if heartbeat else None,
        "progress_age_seconds": now - latest_progress if latest_progress else None,
        "policy_hash": expected_policy or lock_policy,
        "tube_hash": provenance.get("exact_safe_hash") or provenance.get("construction_certification_hash"),
        "candidate_hash": provenance.get("candidate_hash") or provenance.get("candidate_bank_sha256"),
        "last_completed_action": state.get("last_completed_action"),
        "next_automatic_action": state.get("next_decision"),
        "in_progress_action": state.get("in_progress_action"),
        "retry_count": int(state.get("retry_count", 0) or 0),
        "last_error": state.get("stop_reason"),
        "terminal_state": terminal,
        "automatic_transition_pending": transition_pending,
        "result_report_paths": reports,
        "core_metrics": core_metrics(reports),
        "log_paths": [path for path in log_paths if Path(path).exists()],
        "state_json": str(state_path),
        "experiment_state": str(ROOT / "docs/EXPERIMENT_STATE.md"),
        "lock": {
            "path": str(lock_path), "pid": lock_pid, "pid_alive": pid_alive(lock_pid),
            "policy_hash": lock_policy,
            "policy_hash_matches": bool(not expected_policy or lock_policy == expected_policy),
        },
        "duplicate_controller_processes": max(0, process_count("cli.descent_tube_controller") - 1),
        "duplicate_worker_units": max(0, len(workers) - (1 if worker.get("active") else 0)),
    }
    return status


def recovery_decision(status: dict, monitor: dict, now: float | None = None) -> str:
    now = time.time() if now is None else float(now)
    if status.get("terminal_state") or status.get("automatic_transition_pending"):
        return "none"
    if status.get("duplicate_controller_processes") or status.get("duplicate_worker_units"):
        return "none"
    active = bool(status.get("controller_active"))
    heartbeat_stale = float(status.get("heartbeat_age_seconds") or 0.0) > HEARTBEAT_STALE_SECONDS
    progress_stale = float(status.get("progress_age_seconds") or 0.0) > HEARTBEAT_STALE_SECONDS
    worker_active = bool(status.get("worker_active"))
    lock_alive = bool(status.get("lock", {}).get("pid_alive"))
    if active:
        if heartbeat_stale and progress_stale and not worker_active:
            return "restart_stale_controller"
        return "none"
    if lock_alive:
        return "none"
    last_attempt = float(monitor.get("last_recovery_at", 0.0) or 0.0)
    if now - last_attempt < RECOVERY_COOLDOWN_SECONDS:
        return "none"
    if int(monitor.get("consecutive_recoveries", 0)) >= MAX_RECOVERIES:
        return "engineering_failure_after_retries"
    return "start_controller"


def terminal_notification(status: dict) -> tuple[str, str, str, str] | None:
    terminal = status.get("terminal_state")
    policy = str(status.get("policy_hash") or "unknown")
    event = f"{status.get('run_id')}:{status.get('current_stage')}:{terminal}:{policy}"
    if terminal == "pipeline_complete":
        body = (f"最终阶段：{status.get('current_stage')}\nPolicy/Tube：{policy} / "
                f"{status.get('tube_hash') or 'unknown'}\n报告：{', '.join(status.get('result_report_paths', [])[-2:])}")
        return event, "DVGC：流水线完成", body, "normal"
    if terminal == "gate_pause":
        metrics = json.dumps(status.get("core_metrics", {}), ensure_ascii=False, sort_keys=True)
        body = (f"阶段：{status.get('current_stage')}\nGate：{status.get('last_error') or 'unknown'}\n"
                f"指标：{metrics}\n状态：{status.get('experiment_state')}")
        return event, "DVGC：需要研究决策", body, "critical"
    if terminal == "engineering_failure_after_retries":
        body = (f"进程：{status.get('controller_unit')} / {status.get('worker_unit')}\n"
                f"退出：{status.get('controller_exit_code')}/{status.get('controller_exit_status')}\n"
                f"重试：{status.get('retry_count')}\n日志：{', '.join(status.get('log_paths', []))}\n"
                f"状态：{status.get('state_json')}")
        return event, "DVGC：自动恢复失败", body, "critical"
    return None


MAJOR_STAGES = {
    "viability_train": "Pointwise Tube PASS，进入 Viability",
    "support_repair_build": "Pointwise Tube 未通过，进入有界支持修复",
    "acquisition": "Viability 完成，进入 acquisition",
    "apex": "Continuous C_D PASS，启动 apex",
    "takeoff": "Flight 完成，启动 Takeoff",
    "flight_complete": "Flight 完成",
    "approach": "Takeoff 完成，启动 Approach",
    "takeoff_complete": "Takeoff 完成",
    "approach_complete": "Approach 完成",
    "shared_policy": "开始 final shared policy",
}


def stage_notification(status: dict, monitor: dict) -> tuple[str, str, str, str] | None:
    stage = str(status.get("current_stage"))
    previous = str(monitor.get("last_stage") or "")
    if stage == previous or stage not in MAJOR_STAGES:
        return None
    policy = str(status.get("policy_hash") or "unknown")
    event = f"{status.get('run_id')}:{stage}:stage_transition:{policy}"
    return event, "DVGC：阶段进展", MAJOR_STAGES[stage] + "；流水线将自动继续。", "normal"


def send_notification(event: tuple[str, str, str, str], notification_state: dict) -> bool:
    event_id, title, body, urgency = event
    sent = set(notification_state.get("sent_event_ids", []))
    if event_id in sent:
        return False
    command = ["notify-send", "--app-name=DVGC", f"--urgency={urgency}", title, body]
    try:
        result = subprocess.run(command, capture_output=True, timeout=10)
        delivered = result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        delivered = False
    if delivered:
        sent.add(event_id)
        notification_state["sent_event_ids"] = sorted(sent)
        notification_state["last_sent_at"] = time.time()
        atomic_json(NOTIFICATION_STATE, notification_state)
    else:
        atomic_json(PENDING_NOTIFICATION, {
            "event_id": event_id, "title": title, "body": body, "urgency": urgency,
            "created_at": time.time(),
        })
        fallback = set(notification_state.get("fallback_event_ids", []))
        if urgency == "critical" and event_id not in fallback:
            try:
                subprocess.run(["wall", f"{title}: {body}"], capture_output=True, timeout=10)
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
            fallback.add(event_id)
            notification_state["fallback_event_ids"] = sorted(fallback)
            atomic_json(NOTIFICATION_STATE, notification_state)
    return delivered


def perform_recovery(action: str) -> tuple[bool, str]:
    if action == "start_controller":
        props = unit_properties(CONTROLLER_UNIT)
        if props.get("LoadState") == "not-found":
            command = ["bash", str(ROOT / "scripts/start_descent_tube_controller.sh")]
        else:
            command = ["systemctl", "--user", "start", CONTROLLER_UNIT]
    elif action == "restart_stale_controller":
        command = ["systemctl", "--user", "restart", CONTROLLER_UNIT]
    else:
        return False, "unsupported recovery action"
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    detail = (result.stdout + result.stderr).strip()
    return result.returncode == 0, detail


def status_markdown(status: dict) -> str:
    progress = status["progress"]
    completed = f"{progress.get('completed')}/{progress.get('total') or '?'}"
    lines = [
        "# DVGC Pipeline Status", "",
        f"- Updated: {status['timestamp']}",
        f"- Controller: {status['controller_active_state']}/{status['controller_substate']} PID {status['controller_pid']}",
        f"- Worker: {status.get('worker_unit') or 'none'} PID {status.get('worker_pid', 0)}",
        f"- Stage: {status['current_stage']}",
        f"- Progress: {completed}; current range {progress.get('current_index_range')}",
        f"- Heartbeat age: {status.get('heartbeat_age_seconds')}",
        f"- Policy: {status.get('policy_hash')}",
        f"- Last action: {status.get('last_completed_action')}",
        f"- Next action: {status.get('next_automatic_action')}",
        f"- Terminal: {status.get('terminal_state') or 'no'}", "",
    ]
    return "\n".join(lines)


def concise(status: dict) -> str:
    progress = status["progress"]
    completed = f"{progress.get('completed')}/{progress.get('total') or '?'}"
    worker = f"{status.get('worker_unit') or 'none'} pid={status.get('worker_pid', 0)}"
    return (f"DVGC {status['current_stage']} | progress={completed} range={progress.get('current_index_range')}\n"
            f"controller={status['controller_active_state']}/{status['controller_substate']} pid={status['controller_pid']} | worker={worker}\n"
            f"heartbeat={status.get('heartbeat_age_seconds', 0):.1f}s | last={status.get('last_completed_action')} | next={status.get('next_automatic_action')}\n"
            f"terminal={status.get('terminal_state') or 'no'} | status={STATUS_JSON}")


def run_watchdog(run: Path) -> int:
    now = time.time()
    monitor = load_json(WATCHDOG_STATE, {})
    notifications = load_json(NOTIFICATION_STATE, {"sent_event_ids": []})
    status = collect_status(run, now)
    decision = recovery_decision(status, monitor, now)
    if decision in {"start_controller", "restart_stale_controller"}:
        ok, detail = perform_recovery(decision)
        monitor["consecutive_recoveries"] = int(monitor.get("consecutive_recoveries", 0)) + 1
        monitor["last_recovery_at"] = now
        monitor["last_recovery_action"] = decision
        monitor["last_recovery_ok"] = ok
        monitor["last_recovery_detail"] = detail
    elif decision == "engineering_failure_after_retries":
        status["terminal_state"] = "engineering_failure_after_retries"
        status["retry_count"] = int(monitor.get("consecutive_recoveries", 0))
    elif status.get("controller_active") and float(status.get("heartbeat_age_seconds") or 1e9) < HEARTBEAT_STALE_SECONDS:
        monitor["consecutive_recoveries"] = 0
    event = terminal_notification(status) or stage_notification(status, monitor)
    if event:
        send_notification(event, notifications)
    monitor["last_stage"] = status.get("current_stage")
    monitor["last_status_at"] = now
    monitor["last_decision"] = decision
    status["watchdog_recovery_count"] = int(monitor.get("consecutive_recoveries", 0))
    atomic_json(WATCHDOG_STATE, monitor)
    atomic_json(STATUS_JSON, status)
    atomic_text(STATUS_MD, status_markdown(status))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--test-notification", action="store_true")
    parser.add_argument("--flush-pending", action="store_true")
    args = parser.parse_args()
    if args.print_only:
        print(concise(collect_status(args.run)))
        return
    if args.test_notification:
        state = load_json(NOTIFICATION_STATE, {"sent_event_ids": []})
        event_id = f"{args.run.name}:watchdog:enabled"
        send_notification((event_id, "DVGC监控已启用", "后台流水线将自动运行，仅在完成或需要决策时通知。", "normal"), state)
        return
    if args.flush_pending:
        pending = load_json(PENDING_NOTIFICATION, {})
        if pending:
            state = load_json(NOTIFICATION_STATE, {"sent_event_ids": []})
            send_notification((pending["event_id"], pending["title"], pending["body"], pending["urgency"]), state)
            refreshed = load_json(NOTIFICATION_STATE, {})
            if pending["event_id"] in set(refreshed.get("sent_event_ids", [])):
                PENDING_NOTIFICATION.unlink(missing_ok=True)
        return
    raise SystemExit(run_watchdog(args.run))


if __name__ == "__main__":
    main()
