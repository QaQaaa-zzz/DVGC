import json
import subprocess

from jit_dvgc.error_notifications import check_once


def test_queued_run_missing_lineage_is_not_notification_error(tmp_path):
    execution=tmp_path/'execution.json';execution.write_text(json.dumps({'phase':'waiting'}))
    manifest=tmp_path/'ACTIVE_RUN.json'
    manifest.write_text(json.dumps({'execution':str(execution),'lineage':str(tmp_path/'not_started.json')}))
    state={};alerts=[]
    check_once([manifest],state,lambda *args:alerts.append(args))
    assert not alerts and not state['delivery_errors']


def test_active_only_dedup_and_pointer_change(tmp_path):
    status = tmp_path / "status.json"
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"phase": "error"}))
    manifest = tmp_path / "ACTIVE_RUN.json"
    manifest.write_text(json.dumps({"lineage": str(status), "previous_failed_attempt": str(old)}))
    state, alerts = {}, []
    send = lambda *args: alerts.append(args)
    for phase in ("running", "waiting", "round_completed", "blocked"):
        status.write_text(json.dumps({"phase": phase}))
        check_once([manifest], state, send)
    assert not alerts
    for wall in (1, 2):
        status.write_text(json.dumps({"phase": "error", "error": "boom", "wall_seconds": wall}))
        check_once([manifest], state, send)
    assert len(alerts) == 1
    manifest.write_text(json.dumps({"execution": str(old)}))
    check_once([manifest], state, send)
    assert len(alerts) == 2


def test_delivery_failure_retries(tmp_path):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"phase": "failed"}))
    manifest = tmp_path / "ACTIVE_RUN.json"
    manifest.write_text(json.dumps({"execution": str(status)}))
    state = {}
    def fail(*args):
        raise subprocess.CalledProcessError(1, "notify-send")
    check_once([manifest], state, fail)
    assert state["delivery_errors"] and not state["delivered"]
    check_once([manifest], state, lambda *args: None)
    assert len(state["delivered"]) == 1
    assert not state["delivery_errors"]


def test_completion_waits_for_whole_run_and_deduplicates(tmp_path):
    lineage, execution = tmp_path / "lineage.json", tmp_path / "execution.json"
    manifest = tmp_path / "ACTIVE_RUN.json"
    manifest.write_text(json.dumps({"lineage": str(lineage), "execution": str(execution)}))
    lineage.write_text(json.dumps({"phase": "completed"}))
    execution.write_text(json.dumps({"phase": "running"}))
    state, alerts = {}, []
    send = lambda *args: alerts.append(args)
    check_once([manifest], state, send)
    assert not alerts
    for wall in (1, 2):
        execution.write_text(json.dumps({"phase": "completed", "wall_seconds": wall}))
        check_once([manifest], state, send)
    assert len(alerts) == 1
    assert alerts[0][0] == "JIT 实验正常结束"


def test_named_run_distinguishes_validation_from_training(tmp_path):
    status=tmp_path/'status.json';status.write_text(json.dumps({'phase':'completed'}))
    manifest=tmp_path/'ACTIVE_RUN.json';manifest.write_text(json.dumps({'name':'工程验证','execution':str(status)}))
    alerts=[];check_once([manifest],{},lambda *a:alerts.append(a))
    assert '工程验证' in alerts[0][0]
