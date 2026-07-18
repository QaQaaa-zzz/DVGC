"""Fail-closed identity rules for immutable stable-construction cycles."""
from __future__ import annotations

import json
from pathlib import Path


PROTOCOL_VERSION="stable-descent-cross-seed-v1"


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
