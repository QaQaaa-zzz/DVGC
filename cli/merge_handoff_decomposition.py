"""Merge complete handoff shards, prove coverage, and isolate H1 proposals."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.research_semantics import summarize_handoff
from dvgc.runtime import save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", action="append", required=True)
    p.add_argument("--proposal-shard", action="append", required=True)
    p.add_argument("--entry-bank", required=True)
    p.add_argument("--output", required=True); p.add_argument("--proposal-bank", required=True)
    p.add_argument("--dedup-distance", type=float, default=.15)
    a = p.parse_args(); out, proposal_out = Path(a.output), Path(a.proposal_bank)
    if out.exists() or proposal_out.exists(): raise SystemExit("Merged handoff output exists")
    reports = [json.loads(Path(path).read_text()) for path in a.shard]
    intervals = sorted((r["start_event"], r["end_event"]) for r in reports)
    expected = 0
    for start, end in intervals:
        if start != expected: raise SystemExit(f"Handoff shard gap/overlap at {expected}: [{start},{end})")
        expected = end
    if not reports or expected != reports[0]["total_events"] or any(r["total_events"] != expected for r in reports):
        raise SystemExit("Handoff shard coverage is incomplete")
    rows = [row for report in reports for row in report["rows"]]
    if len({(row["policy_label"], row["event_index"]) for row in rows}) != len(rows):
        raise SystemExit("Duplicate handoff event index")
    entry = SnapshotBank.load(a.entry_bank); matcher = entry.metadata["entry_matcher"]
    scale = np.asarray(matcher["scale"], np.float64)
    proposals = [copy.deepcopy(row) for path in a.proposal_shard for row in SnapshotBank.load(path).records]
    unique = []
    for row in proposals:
        feature = np.asarray(row["entry_feature"], np.float64)
        if any(np.linalg.norm((feature - np.asarray(old["entry_feature"], np.float64)) / scale) < a.dedup_distance for old in unique):
            continue
        unique.append(row)
    parents = {row.get("entry_source_parent") or row["entry_source_id"] for row in unique}
    metadata = {"artifact_role": "pending_entry_proposals", "active_for_matching": False,
                "safe_claim_allowed": False, "entry_bank_sha256": file_sha256(a.entry_bank),
                "entry_matcher_radius": matcher["radius"], "entry_matcher_unchanged": True,
                "dedup_distance": a.dedup_distance, "proposal_count_before_dedup": len(proposals),
                "unique_parents": len(parents), "source_shards": [str(Path(x).resolve()) for x in a.proposal_shard]}
    SnapshotBank(unique, metadata).save(proposal_out)
    summary = summarize_handoff(rows)
    payload = {"status": "PASS", "artifact_role": "handoff_decomposition",
               "summary": summary, "rows": rows, "events": len(rows),
               "pending_entry_proposals": len(unique), "pending_entry_parents": len(parents),
               "new_entry_extension_eligible": len(unique) >= 4 and len(parents) >= 2,
               "entry_bank_sha256": file_sha256(a.entry_bank), "entry_matcher_radius": matcher["radius"],
               "entry_matcher_unchanged": True, "proposal_bank": str(proposal_out.resolve()),
               "proposal_bank_sha256": file_sha256(proposal_out)}
    save_json(out, payload); print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
