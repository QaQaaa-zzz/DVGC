import pytest

from dvgc.audit import build_audit_report, merge_audit_reports
from dvgc.certification import branch_evidence, branch_seed


def _state(index, final, label):
    branches=[]
    for branch_index,value in enumerate(final):
        branches.append(branch_evidence(
            branch_index=branch_index,
            seed=branch_seed(1_000_000,index,branch_index),
            seed_namespace="audit:landing",
            dynamics_variant="nominal",
            outcome={"chain":value,"final":value,"terminated":not value,"truncated":False,"steps":12},
        ))
    return {
        "id":f"state-{index}","state_index":index,"predicted_label":label,
        "predicted_mean":sum(final)/len(final),"audit_chain":sum(final)/len(final),
        "audit_final":sum(final)/len(final),"branches":branches,
        "terminal_summary":{},
    }


def _report(rows,start,stop,total=2):
    report=build_audit_report(
        rows,policy_version="policy-v1",phase="landing",
        seed_namespace="audit:landing",branches_per_state=2,
        safe_threshold=.7,dynamics_variants=[{"id":"nominal"}],
    )
    report.update({"state_index_start":start,"state_index_end_exclusive":stop,"total_bank_states":total})
    return report


def test_chunked_audit_merge_preserves_global_seed_evidence():
    first=_report([_state(0,[1,1],"safe")],0,1)
    second=_report([_state(1,[0,0],"dead")],1,2)
    merged=merge_audit_reports([second,first])
    assert merged["states"]==2
    assert merged["terminal_summary"]["branches"]==4
    assert merged["terminal_summary"]["final_recoveries"]==2
    assert merged["terminal_summary"]["physical_failures"]==2
    assert [row["state_index"] for row in merged["rows"]]==[0,1]


def test_chunked_audit_merge_rejects_incomplete_coverage():
    with pytest.raises(ValueError,match="global state index"):
        merge_audit_reports([_report([_state(1,[1,1],"safe")],1,2)])


def test_chunked_audit_merge_rejects_false_part_range():
    false_range=_report([_state(0,[1,1],"safe")],0,2)
    with pytest.raises(ValueError,match="declared global state range"):
        merge_audit_reports([false_range])
