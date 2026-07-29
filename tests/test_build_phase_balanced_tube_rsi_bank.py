import numpy as np
import pytest

from cli.build_phase_balanced_tube_rsi_bank import STAGES, build_balanced_records
from dvgc.bank import SnapshotBank


def _row(stage, index):
    phase = "takeoff" if stage == "takeoff" else ("landing" if stage == "landing" else "flight")
    return {"id": f"{stage}-{index}", "source_phase": phase, "qpos": np.zeros(13),
            "qvel": np.zeros(12), "ctrl": np.zeros(4), "physical_feature": np.zeros(16),
            "trajectory_parent_id": f"{stage}-parent-{index}",
            "final": {"label": "safe"}, "certified_safe": True}


def _banks():
    banks = {}
    for stage in STAGES:
        if stage in {"takeoff", "ascent", "apex"}:
            metadata = {"artifact_role": "stage_entry_certified_proposal_support",
                        "evidence_scope": "local_next_stage", "independent_audit": True,
                        "branch_count": 32, "certified_tube": False}
        elif stage == "descent":
            metadata = {"artifact_role": "certified_tube", "independent_audit": True,
                        "branches_per_state": 32}
        else:
            metadata = {}
        banks[stage] = SnapshotBank([_row(stage, 0), _row(stage, 1)], metadata)
    return banks


def test_phase_and_parent_balancing_preserves_semantic_roles():
    banks = _banks(); hashes = {stage: f"hash-{stage}" for stage in STAGES}
    rows, counts, parents = build_balanced_records(banks, hashes)
    assert counts == {stage: 2 for stage in STAGES}
    assert parents == {stage: 2 for stage in STAGES}
    assert sum(row["reset_weight"] for row in rows) == pytest.approx(1.0)
    for stage in STAGES:
        assert sum(row["reset_weight"] for row in rows if row["phase_rsi_stage"] == stage) == pytest.approx(.2)
    assert all(row["artifact_role"] == "proposal_support_bank" for row in rows)
    assert all("final" not in row and "certified_safe" not in row for row in rows)


def test_local_support_must_be_independently_audited_at_32_branches():
    banks = _banks(); banks["apex"].metadata["branch_count"] = 8
    with pytest.raises(ValueError, match="32-branch local support"):
        build_balanced_records(banks, {stage: stage for stage in STAGES})
