from pathlib import Path
from types import SimpleNamespace

from cli import pipeline_watchdog as watchdog


def _status(**updates):
    value = {
        "run_id": "run", "current_stage": "pointwise_audit", "controller_active": True,
        "worker_active": True, "heartbeat_age_seconds": 20.0, "progress_age_seconds": 20.0,
        "terminal_state": None, "automatic_transition_pending": False,
        "duplicate_controller_processes": 0, "duplicate_worker_units": 0,
        "lock": {"pid_alive": True}, "policy_hash": "policy", "tube_hash": "tube",
        "last_error": None, "experiment_state": "docs/EXPERIMENT_STATE.md",
        "result_report_paths": [], "controller_unit": "controller", "worker_unit": "worker",
        "controller_exit_code": "", "controller_exit_status": "", "retry_count": 0,
    }
    value.update(updates)
    return value


def test_healthy_controller_does_not_recover_or_notify():
    status = _status()
    assert watchdog.recovery_decision(status, {}, now=1000) == "none"
    assert watchdog.terminal_notification(status) is None
    assert watchdog.stage_notification(status, {"last_stage": "pointwise_audit"}) is None


def test_partial_shard_recovers_only_after_controller_exit():
    status = _status(controller_active=False, worker_active=False, lock={"pid_alive": False},
                     progress={"partial_files": ["shard.partial"]})
    assert watchdog.recovery_decision(status, {}, now=1000) == "start_controller"


def test_worker_oom_and_normal_long_worker_do_not_interrupt_parent():
    oom = _status(last_error="worker OOM", heartbeat_age_seconds=900, progress_age_seconds=900)
    assert watchdog.recovery_decision(oom, {}, now=1000) == "none"
    long_gpu = _status(heartbeat_age_seconds=1200, progress_age_seconds=1200, worker_active=True)
    assert watchdog.recovery_decision(long_gpu, {}, now=1000) == "none"


def test_seed_conflict_is_left_to_live_controller_allocator():
    status = _status(last_error="seed conflict")
    assert watchdog.recovery_decision(status, {}, now=1000) == "none"


def test_stale_heartbeat_requires_stale_progress_and_no_worker():
    base = _status(heartbeat_age_seconds=1200, progress_age_seconds=1200, worker_active=False)
    assert watchdog.recovery_decision(base, {}, now=1000) == "restart_stale_controller"
    assert watchdog.recovery_decision({**base, "progress_age_seconds": 30}, {}, now=1000) == "none"


def test_inactive_controller_retries_three_times_then_terminal():
    status = _status(controller_active=False, worker_active=False, lock={"pid_alive": False})
    assert watchdog.recovery_decision(status, {"consecutive_recoveries": 2}, now=1000) == "start_controller"
    assert watchdog.recovery_decision(status, {"consecutive_recoveries": 3}, now=1000) == "engineering_failure_after_retries"


def test_gate_pause_and_pipeline_complete_notify_without_restart():
    gate = _status(current_stage="gate_pause", terminal_state="gate_pause")
    complete = _status(current_stage="pipeline_complete", terminal_state="pipeline_complete")
    assert watchdog.recovery_decision(gate, {}, now=1000) == "none"
    assert watchdog.terminal_notification(gate)[1] == "DVGC：需要研究决策"
    assert watchdog.terminal_notification(complete)[1] == "DVGC：流水线完成"


def test_collect_status_honors_explicit_terminal_without_rewriting_stage(monkeypatch, tmp_path):
    (tmp_path / "controller_state.json").write_text(
        '{"current_stage":"takeoff_controller_pilot","terminal_state":"gate_pause",'
        '"blocked_stage":"takeoff","stop_reason":"takeoff_stage_controller_blocker",'
        '"heartbeat":100,"history":[],"provenance":{}}'
    )
    monkeypatch.setattr(watchdog, "unit_properties", lambda unit: {
        "unit": unit, "pid": 0, "active": False, "ActiveState": "failed",
        "SubState": "failed", "Result": "exit-code", "ExecMainCode": "1",
        "ExecMainStatus": "40",
    })
    monkeypatch.setattr(watchdog, "active_worker_units", lambda: [])
    monkeypatch.setattr(watchdog, "process_count", lambda value: 0)
    status = watchdog.collect_status(tmp_path, now=101, controller_unit="controller.service")
    assert status["terminal_state"] == "gate_pause"
    assert status["current_stage"] == "takeoff_controller_pilot"
    assert status["failed_stage"] == "takeoff"
    assert status["research_gate_valid"] is True


def test_duplicate_notification_event_is_not_sent_twice(monkeypatch, tmp_path):
    monkeypatch.setattr(watchdog, "NOTIFICATION_STATE", tmp_path / "notifications.json")
    monkeypatch.setattr(watchdog, "PENDING_NOTIFICATION", tmp_path / "pending.json")
    calls = []
    monkeypatch.setattr(watchdog.subprocess, "run", lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0))
    state = {"sent_event_ids": []}
    event = ("event", "title", "body", "normal")
    assert watchdog.send_notification(event, state)
    assert not watchdog.send_notification(event, state)
    assert len(calls) == 1


def test_status_script_is_read_only():
    text = Path("scripts/dvgc_status.sh").read_text(encoding="utf-8")
    assert "--print-only" in text
    assert "systemctl" not in text
    assert "start" not in text and "restart" not in text


def test_status_policy_comes_from_current_frozen_manifest(tmp_path):
    manifest = tmp_path / "round_3/frozen/discrete_tube_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"status":"PASS","policy_hash":"current","sets":{"all":{"sha256":"all"},"safe":{"sha256":"safe"}}}')
    state = {"active_round": 3, "provenance": {"policy_hash": "old-source"}}
    assert watchdog.frozen_manifest_provenance(tmp_path, state) == {
        "manifest": str(manifest), "policy_hash": "current",
        "candidate_hash": "all", "tube_hash": "safe",
    }


def test_active_pipeline_pointer_selects_new_controller(monkeypatch, tmp_path):
    pointer=tmp_path/"ACTIVE_PIPELINE.json"
    pointer.write_text('{"run_path":"runs/new","controller_unit":"new.service","start_script":"scripts/start-new.sh"}')
    monkeypatch.setattr(watchdog,"ACTIVE_PIPELINE",pointer)
    run,unit,start=watchdog.pipeline_target()
    assert str(run)=="runs/new" and unit=="new.service" and start=="scripts/start-new.sh"


def test_concise_terminal_status_names_failure_and_resume_action():
    status=_status(current_stage="stable_analyze",terminal_state="engineering_failure_after_retries",
        failed_stage="stable_analyze",last_valid_checkpoint="checkpoint",recommended_resume_action="repair_selector",
        last_error="duplicate snapshots",progress={"completed":0,"total":None,"current_index_range":None},
        controller_active_state="failed",controller_substate="failed",controller_pid=0,heartbeat_age_seconds=1)
    text=watchdog.concise(status)
    assert "terminal=yes type=engineering_failure_after_retries" in text
    assert "failed_stage=stable_analyze" in text and "last_valid_checkpoint=checkpoint" in text
    assert "recommended_resume=repair_selector" in text


def test_concise_status_exposes_cycle_and_gate_validity():
    status=_status(current_stage="stable_stage_a",cycle=5,research_gate_valid=False,
        progress={"completed":12,"total":153,"current_index_range":[12,24]},controller_active_state="active",
        controller_substate="running",controller_pid=1,heartbeat_age_seconds=2)
    text=watchdog.concise(status)
    assert "cycle=5" in text and "gate_valid=False" in text
