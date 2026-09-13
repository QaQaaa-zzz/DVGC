import json
import os

import pytest

from dvgc.construction_lifecycle import (
    PROTOCOL_VERSION,
    completed_coverage,
    failure_fuse_update,
    is_stale_lock,
    split_range_after_oom,
    validate_artifact,
    validate_policy_update_gate,
    worker_log_is_oom,
)


def _identity(policy="policy",bank="bank",cycle=5):
    return {"policy_hash":policy,"candidate_bank_sha256":bank,"xml_sha256":"xml",
        "landing_entry_set_sha256":"entry","landing_policy_hash":"landing","config_hash":"config",
        "protocol":{"stage_branches":32},"certification_protocol_version":PROTOCOL_VERSION,
        "construction_seed_epoch":cycle,"runtime_source_fingerprint":"runtime"}


def _payload(identity,stage="stage_a",seed=10):
    return {"status":"PASS","complete":True,"stage":stage,"seed":seed,
        "descent_policy_hash":identity["policy_hash"],"candidate_bank_sha256":identity["candidate_bank_sha256"],
        "xml_sha256":identity["xml_sha256"],"landing_entry_set_sha256":identity["landing_entry_set_sha256"],
        "landing_policy_hash":identity["landing_policy_hash"],"config_hash":identity["config_hash"],
        "protocol":identity["protocol"],"certification_protocol_version":identity["certification_protocol_version"],
        "construction_seed_epoch":identity["construction_seed_epoch"],"runtime_source_fingerprint":"runtime"}


def test_same_policy_bank_and_full_provenance_can_resume(tmp_path):
    identity=_identity();path=tmp_path/"report.json";path.write_text(json.dumps(_payload(identity)))
    assert validate_artifact(path,identity,checkpoint_mtime=0,stage="stage_a",seed=10,require_complete=True)["status"]=="PASS"


def test_same_bank_new_policy_rejects_old_cached_report(tmp_path):
    path=tmp_path/"report.json";path.write_text(json.dumps(_payload(_identity(policy="old"))))
    with pytest.raises(RuntimeError,match="Stale construction artifact"):
        validate_artifact(path,_identity(policy="new"),checkpoint_mtime=0,stage="stage_a",seed=10,require_complete=True)


def test_existing_report_with_manifest_or_seed_epoch_mismatch_is_rejected(tmp_path):
    path=tmp_path/"report.json";path.write_text(json.dumps(_payload(_identity(cycle=4))))
    with pytest.raises(RuntimeError):validate_artifact(path,_identity(cycle=5),checkpoint_mtime=0,stage="stage_a",seed=10)


def test_report_must_be_generated_after_checkpoint(tmp_path):
    path=tmp_path/"report.json";path.write_text(json.dumps(_payload(_identity())));os.utime(path,(10,10))
    with pytest.raises(RuntimeError):validate_artifact(path,_identity(),checkpoint_mtime=11,stage="stage_a",seed=10)


def test_policy_update_gate_rejects_same_before_after_report(tmp_path):
    path=tmp_path/"stable.json";path.write_text('{"policy_hash":"old"}')
    with pytest.raises(RuntimeError,match="Invalid before/after"):
        validate_policy_update_gate(path,path,before_policy_hash="old",after_policy_hash="new",before_cycle=4,after_cycle=5)


def test_policy_update_gate_accepts_fresh_cycle5_report(tmp_path):
    before=tmp_path/"before.json";after=tmp_path/"after.json";before.write_text('{"policy_hash":"old"}');after.write_text('{"policy_hash":"new"}')
    os.utime(before,(10,10));os.utime(after,(11,11))
    _,_,checks=validate_policy_update_gate(before,after,before_policy_hash="old",after_policy_hash="new",before_cycle=4,after_cycle=5)
    assert all(checks.values())


def test_completed_coverage_requires_every_index_exactly_once():
    complete = [
        {"rows": [{"candidate_index": 0}, {"candidate_index": 1}]},
        {"rows": [{"candidate_index": 2}]},
    ]
    assert completed_coverage(complete, 3) == [0, 1, 2]
    with pytest.raises(ValueError, match="every global index exactly once"):
        completed_coverage(complete + [{"rows": [{"candidate_index": 2}]}], 3)


def test_oom_backoff_reaches_single_state_shards():
    assert split_range_after_oom(0, 12) == [(0, 6), (6, 12)]
    assert split_range_after_oom(0, 6) == [(0, 3), (3, 6)]
    assert split_range_after_oom(0, 3) == [(0, 1), (1, 2), (2, 3)]
    assert split_range_after_oom(4, 5) == [(4, 5)]


def test_oom_detection_does_not_confuse_domain_failure_with_allocation_failure():
    assert worker_log_is_oom("RuntimeError: Failed to allocate 8192 bytes")
    assert worker_log_is_oom("RESOURCE_EXHAUSTED")
    assert not worker_log_is_oom("No Landing-entry radius meets calibration precision")


def test_failure_fuse_normalizes_worker_launch_timestamp_and_resets_on_stage_change():
    state = {}
    signature, count = failure_fuse_update(
        state, "audit", RuntimeError("worker failed unit=dvgc-worker-audit-1784275003.service")
    )
    state.update(failure_signature=signature, consecutive_failure_count=count)
    _, count = failure_fuse_update(
        state, "audit", RuntimeError("worker failed unit=dvgc-worker-audit-1784275093.service")
    )
    assert count == 2
    _, count = failure_fuse_update(state, "merge", RuntimeError("different"))
    assert count == 1


def test_stale_lock_requires_every_liveness_signal_to_be_absent():
    lock = {"pid": 99_999_999}
    assert is_stale_lock(lock, unit_active=False, worker_pids=[], heartbeat_age=61)
    assert not is_stale_lock(lock, unit_active=True, worker_pids=[], heartbeat_age=61)
    assert not is_stale_lock(lock, unit_active=False, worker_pids=[7], heartbeat_age=61)
    assert not is_stale_lock(lock, unit_active=False, worker_pids=[], heartbeat_age=59)
