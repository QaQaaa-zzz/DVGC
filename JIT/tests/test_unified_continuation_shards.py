from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jit_dvgc.unified_continuation_shards import (
    POLICY_KEY_SCHEME,
    contiguous_shard_bounds,
    merge_unified_continuation_shards,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_balanced_shard_bounds_1901_four() -> None:
    assert [contiguous_shard_bounds(1901, i, 4) for i in range(4)] == [
        (0, 476),
        (476, 951),
        (951, 1426),
        (1426, 1901),
    ]


def _make_merge_fixture(tmp_path: Path, *, shard_count: int = 2):
    rows = []
    for index in range(5):
        rows.append(
            {
                "candidate_id": f"c{index}",
                "state_sha256": f"{index + 1:064x}",
                "phase": "upstream" if index < 3 else "downstream",
                "phase_index": 0 if index < 3 else 1,
                "parent_group_id": f"p{index}",
                "parent_state_sha256": f"{index + 101:064x}",
                "source_bank": "bank",
                "snapshot": f"s{index}",
            }
        )
    catalog = {"candidate_count": len(rows), "entries": rows}
    catalog_path = tmp_path / "catalog.json"
    _write(catalog_path, catalog)
    import jit_dvgc.unified_continuation_shards as shards

    catalog_sha = shards.file_sha256(catalog_path)
    protocol_base = {
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": "a" * 64,
        "policy_payload_sha256": "b" * 64,
        "frozen_unified_manifest_sha256": "c" * 64,
        "candidate_catalog_file_sha256": catalog_sha,
        "candidate_catalog_protocol_sha256": "d" * 64,
        "candidate_count": len(rows),
        "maximum_environment_interactions": len(rows) * 400,
    }
    protocol_sha = _sha(protocol_base)
    protocol = {**protocol_base, "protocol_sha256": protocol_sha}

    shard_dirs = []
    for shard_index in range(shard_count):
        start, stop = contiguous_shard_bounds(len(rows), shard_index, shard_count)
        root = tmp_path / f"shard_{shard_index}"
        labels = []
        for index in range(start, stop):
            source = rows[index]
            labels.append(
                {
                    **source,
                    "candidate_index": index,
                    "policy_key_candidate_index": index,
                    "policy_key_scheme": POLICY_KEY_SCHEME,
                    "label_protocol_sha256": protocol_sha,
                    "label": int(index % 2 == 0),
                    "outcome_class": "success" if index % 2 == 0 else "physical_failure",
                    "environment_interactions": 10 + index,
                }
            )
        _write(root / "protocol.json", protocol)
        _write(
            root / "execution.json",
            {
                "status": "completed",
                "logical_protocol_sha256": protocol_sha,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "candidate_start_index": start,
                "candidate_stop_index_exclusive": stop,
            },
        )
        _write(
            root / "summary.json",
            {
                "status": "completed_shard",
                "logical_protocol_sha256": protocol_sha,
                "candidate_count": len(labels),
            },
        )
        _write(root / "labels.json", labels)
        shard_dirs.append(root)
    return catalog_path, shard_dirs


def test_merge_restores_global_order_and_full_coverage(tmp_path: Path) -> None:
    catalog_path, shard_dirs = _make_merge_fixture(tmp_path)
    output = tmp_path / "merged"
    report = merge_unified_continuation_shards(catalog_path, shard_dirs, output)
    labels = json.loads((output / "labels.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["candidate_count"] == 5
    assert [row["candidate_index"] for row in labels] == list(range(5))


def test_merge_rejects_missing_shard(tmp_path: Path) -> None:
    catalog_path, shard_dirs = _make_merge_fixture(tmp_path)
    with pytest.raises(ValueError, match="every index"):
        merge_unified_continuation_shards(
            catalog_path, shard_dirs[:-1], tmp_path / "merged"
        )
