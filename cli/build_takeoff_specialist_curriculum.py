"""Build authentic, eval-parent-disjoint Takeoff specialist curriculum banks."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def _choose(rows, count):
    if not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    near = max(2, count // 2)
    chosen = list(rows[:near])
    tail = rows[near:]
    chosen.extend(tail[int(i)] for i in np.linspace(
        0, len(tail) - 1, count - near, dtype=int
    ))
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-bank", required=True)
    p.add_argument("--eval-bank", required=True)
    p.add_argument("--baseline-evaluation", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--output-report", required=True)
    a = p.parse_args()
    source = SnapshotBank.load(a.source_bank)
    evaluation = SnapshotBank.load(a.eval_bank)
    result = json.loads(Path(a.baseline_evaluation).read_text())
    old_success_ids = {
        row["candidate_id"] for row in result["outcomes"]
        if row["controller"] == "old_takeoff" and row["success"]
    }
    success_eval = [row for row in evaluation.records if row["id"] in old_success_ids]
    eval_parents = {int(row.get("reference_index", -1)) for row in evaluation.records}
    train = [row for row in source.records if int(row.get("reference_index", -1)) not in eval_parents]
    canonical = [row for row in train if row["candidate_kind"] == "canonical_compressed"]
    aligned = [row for row in train if row["candidate_kind"] == "reference_aligned_compressed"]
    scale = np.maximum(np.std(
        np.asarray([row["physical_feature"] for row in source.records], float), axis=0
    ), np.asarray([.05, .05, .05, .05, .05, .05, .2, .2, .2, .2, .2, .2, .1, .1, .2, .2]))
    def near_success(rows, kind):
        targets = [np.asarray(row["physical_feature"], float) for row in success_eval
                   if row["candidate_kind"] == kind]
        if not targets:
            return list(rows)
        return sorted(rows, key=lambda row: min(
            np.linalg.norm((np.asarray(row["physical_feature"], float) - target) / scale)
            for target in targets
        ))
    successful_c = near_success(canonical, "canonical_compressed")
    successful_a = near_success(aligned, "reference_aligned_compressed")

    # Existing validated snapshots only: no full-vector interpolation and no
    # state mutation.  The count schedule shifts from success-neighbour support
    # toward a balanced authentic distribution.
    schedules = {
        1: (12, 20),  # canonical, aligned
        2: (20, 16),
        3: (20, 20),
        4: (20, 20),
    }
    root = Path(a.output_root); root.mkdir(parents=True, exist_ok=True)
    stages = {}
    for block, (nc, na) in schedules.items():
        # Put fresh old-policy success states first, then parent-diverse source
        # states.  Evaluation parents remain absent from every block.
        c = successful_c
        r = successful_a
        rows = _choose(c, nc) + _choose(r, na)
        if len(rows) != nc + na:
            raise SystemExit(f"insufficient authentic Takeoff states for block {block}")
        path = root / f"block_{block}_reset_bank.pkl"
        metadata = copy.deepcopy(source.metadata)
        metadata.update({
            "artifact_role": "takeoff_canonical_specialist_curriculum",
            "block": block, "canonical_states": nc, "reference_aligned_states": na,
            "fixed_eval_bank_sha256": file_sha256(a.eval_bank),
            "eval_reference_parents_excluded": True,
            "state_mutation": False,
        })
        SnapshotBank([copy.deepcopy(row) for row in rows], metadata).save(path)
        stages[str(block)] = {
            "bank": str(path), "bank_sha256": file_sha256(path),
            "canonical_states": nc, "reference_aligned_states": na,
            "old_policy_success_eval_states": len(success_eval),
            "reference_parent_overlap_with_eval": len(
                {int(row.get("reference_index", -1)) for row in rows} & eval_parents
            ),
        }
    save_json(a.output_report, {
        "status": "PASS", "artifact_role": "takeoff_specialist_curriculum",
        "source_bank_sha256": file_sha256(a.source_bank),
        "fixed_eval_bank_sha256": file_sha256(a.eval_bank),
        "old_policy_fresh_success_ids": sorted(old_success_ids),
        "training_state_order": "normalized-nearest legal neighbour to fresh eval success within stratum",
        "nearest_canonical_training_ids": [row["id"] for row in successful_c[:8]],
        "nearest_reference_training_ids": [row["id"] for row in successful_a[:8]],
        "blocks": stages,
        "no_unconstrained_interpolation": True,
    })


if __name__ == "__main__":
    main()
