import json

import pytest

from cli.descent_local_controller import (
    completed_coverage,
    is_stale_lock,
    split_range_after_oom,
    worker_log_is_oom,
)


def _shard(start, end):
    return {"complete": True, "start_index": start, "end_index": end,
            "rows": [{"candidate_index": index} for index in range(start, end)]}


def test_three_shards_must_all_complete_before_merge():
    first, second, third = _shard(0, 3), _shard(3, 6), _shard(6, 9)
    with pytest.raises(ValueError, match="every global index"):
        completed_coverage([first], 9)
    with pytest.raises(ValueError, match="every global index"):
        completed_coverage([first, third], 9)
    assert completed_coverage([first, second, third], 9) == list(range(9))


def test_interrupted_second_shard_resumes_without_reusing_first():
    completed = [_shard(0, 3)]
    with pytest.raises(ValueError):
        completed_coverage(completed, 9)
    completed.append(_shard(3, 6))
    completed.append(_shard(6, 9))
    indices = completed_coverage(completed, 9)
    assert indices.count(0) == 1 and indices.count(3) == 1


def test_duplicate_or_missing_global_index_cannot_merge():
    with pytest.raises(ValueError):
        completed_coverage([_shard(0, 3), _shard(2, 5)], 5)


def test_worker_oom_backoff_is_12_to_6_to_3_to_1():
    assert split_range_after_oom(0, 12) == [(0, 6), (6, 12)]
    assert split_range_after_oom(0, 6) == [(0, 3), (3, 6)]
    assert split_range_after_oom(0, 3) == [(0, 1), (1, 2), (2, 3)]
    assert split_range_after_oom(4, 5) == [(4, 5)]
    assert worker_log_is_oom("RuntimeError: Failed to allocate 8192 bytes")
    assert worker_log_is_oom("RESOURCE_EXHAUSTED")
    assert not worker_log_is_oom("No Landing-entry radius meets calibration precision")


def test_stale_lock_requires_all_liveness_checks_to_fail():
    lock = {"pid": 99999999}
    assert is_stale_lock(lock, unit_active=False, worker_pids=[], heartbeat_age=61)
    assert not is_stale_lock(lock, unit_active=True, worker_pids=[], heartbeat_age=61)
    assert not is_stale_lock(lock, unit_active=False, worker_pids=[7], heartbeat_age=61)
    assert not is_stale_lock(lock, unit_active=False, worker_pids=[], heartbeat_age=59)


def test_controller_service_has_restart_and_terminal_exit_guards():
    text = open("scripts/start_descent_local_controller.sh", encoding="utf-8").read()
    assert "Restart=on-failure" in text
    assert 'RestartPreventExitStatus="40 41"' in text
    assert "StartLimitIntervalSec=0" in text
    assert "RuntimeMaxSec=infinity" in text
