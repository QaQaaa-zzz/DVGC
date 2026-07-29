"""Select a diverse candidate bank from reachability scores.

Predictions are proposal priorities only.  This command deliberately strips
observed outcomes and never emits a safety label; every selected state must be
evaluated again by the branch labeler/certifier.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def parent_key(row: dict, proposal: dict) -> str:
    return str(
        proposal.get("parent")
        or row.get("trajectory_parent_id")
        or row.get("parent_candidate_id")
        or row.get("parent_anchor_pair")
        or f"reference:{row.get('reference_index')}"
    )


def select(records: list[dict], proposals: list[dict], target: int,
           max_per_parent: int, minimum_per_kind: int) -> list[tuple[dict, dict, str]]:
    if target <= 0 or max_per_parent <= 0 or minimum_per_kind < 0:
        raise ValueError("selection limits must be positive")
    by_id = {str(row["id"]): row for row in records}
    ranked = []
    for proposal in proposals:
        candidate_id = str(proposal["candidate_id"])
        if candidate_id not in by_id:
            raise ValueError(f"ranked proposal is absent from source bank: {candidate_id}")
        if "predicted_p_next" not in proposal and "reachability_score" not in proposal:
            raise ValueError(f"proposal lacks a reachability score: {candidate_id}")
        score = float(proposal.get("predicted_p_next", proposal.get("reachability_score")))
        ranked.append((score, candidate_id, by_id[candidate_id], proposal))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected: list[tuple[dict, dict, str]] = []
    selected_ids: set[str] = set()
    parent_counts: Counter[str] = Counter()

    def add(item) -> bool:
        _score, candidate_id, row, proposal = item
        parent = parent_key(row, proposal)
        if candidate_id in selected_ids or parent_counts[parent] >= max_per_parent:
            return False
        selected.append((row, proposal, parent))
        selected_ids.add(candidate_id)
        parent_counts[parent] += 1
        return True

    kinds = sorted({str(item[2].get("candidate_kind") or "unspecified") for item in ranked})
    for kind in kinds:
        need = minimum_per_kind
        for item in ranked:
            if need == 0:
                break
            if str(item[2].get("candidate_kind") or "unspecified") == kind and add(item):
                need -= 1
        if need:
            raise ValueError(f"cannot satisfy minimum coverage for candidate_kind={kind!r}")
    for item in ranked:
        if len(selected) >= target:
            break
        add(item)
    if len(selected) != target:
        raise ValueError(
            f"diversity limits allow only {len(selected)}/{target} candidates; "
            "do not relax them implicitly"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--target", type=int, default=16)
    parser.add_argument("--max-per-parent", type=int, default=2)
    parser.add_argument("--minimum-per-kind", type=int, default=4)
    args = parser.parse_args()
    output_bank, output_report = Path(args.output_bank), Path(args.output_report)
    if output_bank.exists() or output_report.exists():
        raise SystemExit("refusing to overwrite ranked selection")
    bank = SnapshotBank.load(args.bank)
    payload = json.loads(Path(args.proposals).read_text())
    if payload.get("not_certified_tube") is not True or payload.get("not_safe_labels") is not True:
        raise SystemExit("proposal artifact does not declare ranking-only semantics")
    chosen = select(bank.records, payload["records"], args.target,
                    args.max_per_parent, args.minimum_per_kind)
    rows = []
    for rank, (source, proposal, parent) in enumerate(chosen):
        row = copy.deepcopy(source)
        row.update({
            "reachability_rank": rank,
            "reachability_score": float(proposal.get("predicted_p_next", proposal.get("reachability_score"))),
            "reachability_parent": parent,
            "artifact_role": "proposal_support_bank",
            "safe_claim_allowed": False,
            "requires_fresh_branch_certification": True,
        })
        # Old outcomes informed model fitting, but are not carried into the
        # prospective certification bank as labels.
        for key in ("final", "certified_safe", "tube_version", "branch_evidence"):
            row.pop(key, None)
        rows.append(row)
    metadata = {
        "artifact_role": "reachability_ranked_proposal_bank",
        "safe_claim_allowed": False,
        "not_certified_tube": True,
        "requires_fresh_branch_certification": True,
        "source_bank_sha256": file_sha256(args.bank),
        "proposal_report_sha256": file_sha256(args.proposals),
        "model_sha256": payload.get("model_sha256"),
        "selection": {
            "target": args.target,
            "max_per_parent": args.max_per_parent,
            "minimum_per_kind": args.minimum_per_kind,
        },
    }
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    SnapshotBank(rows, metadata).save(output_bank)
    report = {
        "status": "PASS",
        **metadata,
        "output_bank": str(output_bank),
        "output_bank_sha256": file_sha256(output_bank),
        "selected": len(rows),
        "unique_parents": len({row["reachability_parent"] for row in rows}),
        "max_parent_contribution": max(Counter(row["reachability_parent"] for row in rows).values()),
        "candidate_kind_counts": dict(Counter(str(row.get("candidate_kind") or "unspecified") for row in rows)),
        "score_range": [min(row["reachability_score"] for row in rows),
                        max(row["reachability_score"] for row in rows)],
    }
    save_json(output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
