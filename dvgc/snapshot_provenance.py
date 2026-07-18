"""Per-snapshot policy-state provenance for mixed-source candidate banks."""
from __future__ import annotations

from collections.abc import Iterable


def declared_snapshot_source_hashes(metadata: dict) -> tuple[str, ...]:
    values=metadata.get("snapshot_source_policy_hashes")
    if values is None:
        one=metadata.get("snapshot_source_policy_hash",metadata.get("descent_policy_hash"))
        values=[] if one is None else [one]
    result=tuple(sorted({str(value) for value in values if value}))
    if not result:raise ValueError("Candidate bank has no snapshot-source policy provenance")
    return result


def validate_snapshot_source_records(records: Iterable[dict],metadata: dict) -> tuple[str,...]:
    declared=declared_snapshot_source_hashes(metadata)
    fallback=declared[0] if len(declared)==1 else None
    observed=set()
    for row in records:
        value=row.get("snapshot_source_policy_hash",fallback)
        if not value:raise ValueError(f"Snapshot {row.get('id')} lacks source-policy hash")
        observed.add(str(value))
    if observed-set(declared):raise ValueError("Snapshot record source-policy hash is undeclared")
    if observed!=set(declared):raise ValueError("Declared snapshot-source policies do not match records")
    return declared


def verify_source_policy_paths(declared: Iterable[str],paths: Iterable[str],hash_fn) -> tuple[str,...]:
    supplied=tuple(sorted({str(hash_fn(path)) for path in paths}))
    expected=tuple(sorted({str(value) for value in declared}))
    if supplied!=expected:raise ValueError(f"Snapshot-source policy mismatch: expected {expected}, got {supplied}")
    return supplied
