"""Fail-closed identity rules for immutable stable-construction cycles."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


PROTOCOL_VERSION="stable-descent-cross-seed-v1"


def is_stale_lock(lock_payload, *, unit_active, worker_pids, heartbeat_age):
    """Return true only when every independent liveness signal is absent."""
    pid = int(lock_payload.get("pid", 0) or 0)
    try:
        os.kill(pid, 0)
        pid_alive = pid > 0
    except OSError:
        pid_alive = False
    return bool(not pid_alive and not unit_active and not worker_pids and heartbeat_age > 60.0)


def split_range_after_oom(start, end):
    """Apply the declared 12 -> 6 -> 3 -> 1 state OOM backoff."""
    width = int(end) - int(start)
    if width <= 1:
        return [(int(start), int(end))]
    target = 6 if width > 6 else 3 if width > 3 else 1
    return [
        (left, min(left + target, int(end)))
        for left in range(int(start), int(end), target)
    ]


def completed_coverage(payloads, total):
    """Return exact completed coverage, rejecting gaps and overlaps."""
    indices = sorted(
        int(row["candidate_index"])
        for payload in payloads
        for row in payload["rows"]
    )
    if indices != list(range(int(total))):
        raise ValueError(
            "Completed shard markers do not cover every global index exactly once"
        )
    return indices


def worker_log_is_oom(text):
    """Recognize allocation failure without conflating domain-level failures."""
    lowered = str(text).lower()
    return any(
        token in lowered
        for token in ("out of memory", "failed to allocate", "resource_exhausted")
    )


def failure_fuse_update(state, stage: str, exc: Exception):
    """Count identical deterministic failures across timestamped worker restarts."""
    normalized = re.sub(r"-\d{10}\.service", "-<launch>.service", str(exc))
    message = f"{stage}|{type(exc).__name__}|{normalized}"
    signature = hashlib.sha256(message.encode("utf-8")).hexdigest()
    previous = state.get("failure_signature")
    count = (
        int(state.get("consecutive_failure_count", 0)) + 1
        if previous == signature
        else 1
    )
    return signature, count


def artifact_checks(payload,identity,*,stage=None,seed=None):
    mapping={"policy_hash":"descent_policy_hash" if stage else "policy_hash",
        "candidate_bank_sha256":"candidate_bank_sha256","xml_sha256":"xml_sha256",
        "landing_entry_set_sha256":"landing_entry_set_sha256","landing_policy_hash":"landing_policy_hash",
        "config_hash":"config_hash","certification_protocol_version":"certification_protocol_version",
        "construction_seed_epoch":"construction_seed_epoch","runtime_source_fingerprint":"runtime_source_fingerprint"}
    checks={name:payload.get(key)==identity.get(name) for name,key in mapping.items()}
    checks["protocol"]=payload.get("protocol")==identity.get("protocol")
    if stage is not None:checks.update({"stage":payload.get("stage")==stage,"seed":int(payload.get("seed",-1))==int(seed)})
    return checks


def validate_artifact(path,identity,*,checkpoint_mtime,stage=None,seed=None,require_complete=False):
    path=Path(path);payload=json.loads(path.read_text())
    checks=artifact_checks(payload,identity,stage=stage,seed=seed)
    checks["status_pass"]=payload.get("status")=="PASS"
    if require_complete:checks["complete"]=payload.get("complete") is True
    checks["generated_after_checkpoint"]=path.stat().st_mtime>float(checkpoint_mtime)
    if not all(checks.values()):raise RuntimeError(f"Stale construction artifact {path}: {checks}")
    return payload


def validate_policy_update_gate(before_path,after_path,*,before_policy_hash,after_policy_hash,before_cycle,after_cycle):
    before_path=Path(before_path);after_path=Path(after_path);before=json.loads(before_path.read_text());after=json.loads(after_path.read_text())
    checks={"policy_changed":str(before_policy_hash)!=str(after_policy_hash),
        "cycle_changed":int(before_cycle)!=int(after_cycle),
        "before_policy_matches":before.get("policy_hash")==str(before_policy_hash),
        "after_policy_matches":after.get("policy_hash")==str(after_policy_hash),
        "report_paths_differ":before_path.resolve()!=after_path.resolve(),
        "report_payloads_differ":before_path.read_bytes()!=after_path.read_bytes(),
        "after_generated_later":after_path.stat().st_mtime>before_path.stat().st_mtime}
    if not all(checks.values()):raise RuntimeError(f"Invalid before/after stable gate lifecycle: {checks}")
    return before,after,checks
