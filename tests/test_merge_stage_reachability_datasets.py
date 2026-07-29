import json

import numpy as np
import pytest

from cli.merge_stage_reachability_datasets import merge_datasets
from dvgc.bank import SnapshotBank


def _record(candidate_id, parent):
    return {"id": candidate_id, "source_phase": "flight", "qpos": np.zeros(13),
            "qvel": np.zeros(12), "ctrl": np.zeros(4), "physical_feature": np.zeros(16),
            "trajectory_parent_id": parent}


def _write(tmp_path, name, candidate_id, parent, stage="apex"):
    bank = tmp_path / f"{name}.pkl"
    labels = tmp_path / f"{name}.json"
    SnapshotBank([_record(candidate_id, parent)]).save(bank)
    labels.write_text(json.dumps({"labels": [{"candidate_id": candidate_id,
        "trajectory_parent": parent, "stage": stage, "s": 3, "n": 4,
        "label": "positive", "formal_successes": 0, "final_successes": 0}]}))
    return str(bank), str(labels)


def test_merge_preserves_distinct_states_parents_and_branch_evidence(tmp_path):
    b0, l0 = _write(tmp_path, "a", "s0", "p0")
    b1, l1 = _write(tmp_path, "b", "s1", "p1")
    records, labels, parents, sources = merge_datasets([b0, b1], [l0, l1], "apex")
    assert [row["id"] for row in records] == ["s0", "s1"]
    assert sum(row["s"] for row in labels) == 6
    assert parents == {"p0", "p1"}
    assert len(sources) == 2


def test_merge_rejects_stage_or_parent_mismatch(tmp_path):
    bank, labels = _write(tmp_path, "bad", "s0", "p0", stage="ascent")
    with pytest.raises(ValueError, match="stage mismatch"):
        merge_datasets([bank], [labels], "apex")
    payload = json.loads(open(labels).read())
    payload["labels"][0]["stage"] = "apex"
    payload["labels"][0]["trajectory_parent"] = "wrong"
    open(labels, "w").write(json.dumps(payload))
    with pytest.raises(ValueError, match="parent mismatch"):
        merge_datasets([bank], [labels], "apex")


def test_merge_rejects_duplicate_state_across_inputs(tmp_path):
    bank, labels = _write(tmp_path, "same", "s0", "p0")
    with pytest.raises(ValueError, match="duplicate state across inputs"):
        merge_datasets([bank, bank], [labels, labels], "apex")
