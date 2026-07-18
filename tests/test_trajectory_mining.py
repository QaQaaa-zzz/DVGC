from types import SimpleNamespace

import numpy as np
import pytest

from cli.analyze_stable_descent_construction import validate_unique_candidates
from dvgc.roll_controllability import audit_decision
from dvgc.snapshot_provenance import validate_snapshot_source_records,verify_source_policy_paths
from dvgc.trajectory_mining import canonical_state_byte_hash,layer_targets,select_parent_balanced,select_parent_balanced_with_report,select_trace_records,trajectory_parent_id


def _row(identifier,parent=None,layer="middle",state=None,**extra):
    value=float(state if state is not None else sum(identifier.encode()))
    row={"id":identifier,"trajectory_parent_id":parent or f"p-{identifier}","descent_layer":layer,
        "mining_rank_score":10.0,"qpos":np.array([value],np.float32),"qvel":np.array([value+.1],np.float32),
        "ctrl":np.array([value+.2],np.float32),"qacc_warmstart":np.array([value+.3],np.float32),
        "policy_state":{"last_action":np.array([value+.4],np.float32)},"oracle_phase":2,
        "had_airborne":1,"had_valid_landing":0,"contact_age":0,"airborne_count":1,"recovery_count":0,
        "bootstrap_eligible":True,"training_only":False}
    row.update(extra);return row


def test_each_successful_branch_is_a_distinct_trajectory_parent():
    a=trajectory_parent_id("policy","candidate",101);b=trajectory_parent_id("policy","candidate",102)
    assert a!=b and a==trajectory_parent_id("policy","candidate",101)


def test_trace_selection_is_sparse_and_capped_at_four():
    trace=[{"step":i,"record":{"physical_feature":[0,0,0,.1+i*.01,0,0,0,0,0,.2-i*.01,0,0,0,0,0,0]}} for i in range(12)]
    selected=select_trace_records(trace,max_children=4,min_step_gap=2)
    assert len(selected)<=4
    assert {role for role,_ in selected}>={"earliest","middle","late"}
    assert len({item["step"] for _,item in selected})==len(selected)


def test_parent_balanced_selection_uses_trajectory_parent_not_original_parent():
    rows=[]
    for parent in range(10):
        for child in range(4):rows.append(_row(f"{parent}-{child}",f"t{parent}",("middle","late","early")[parent%3],state=parent*10+child,
            original_candidate_parent="same",mining_rank_score=100-child))
    chosen=select_parent_balanced(rows,target=20,masses={"middle":.4,"late":.4,"early":.2},parent_cap=4)
    assert len(chosen)==20 and len({x["trajectory_parent_id"] for x in chosen})==10
    assert max(sum(x["trajectory_parent_id"]==parent for x in chosen) for parent in {x["trajectory_parent_id"] for x in chosen})<=4
    assert layer_targets(20,{"middle":.4,"late":.4,"early":.2})=={"middle":8,"late":8,"early":4}


def test_layer_targets_do_not_block_validated_surplus_support():
    rows=[_row(str(i),f"t{i}",state=i,mining_rank_score=10-i) for i in range(8)]
    selected=select_parent_balanced(rows,target=8,masses={"middle":.4,"late":.4,"early":.2},parent_cap=4)
    assert len(selected)==8


def test_seven_unique_candidates_do_not_repeat_across_four_quota_rounds():
    rows=[_row(str(i),f"p{i}",("middle","late")[i%2],state=i) for i in range(7)]
    selected,report=select_parent_balanced_with_report(rows,target=28,masses={"middle":.4,"late":.4,"early":.2},parent_cap=4)
    assert len(selected)==7 and report["quota_target"]==28 and report["quota_shortfall"]==21
    assert report["exhausted_unique_support"] and report["unique_state_byte_hashes"]==7


def test_different_candidate_ids_with_same_snapshot_keep_one():
    rows=[_row("a",state=1),_row("b",state=1)]
    selected,report=select_parent_balanced_with_report(rows,target=2,masses={"middle":1,"late":0,"early":0})
    assert len(selected)==1 and report["duplicate_rejected"]["state_byte_hash"]==1


def test_same_candidate_id_with_different_wrappers_keeps_one():
    rows=[_row("a",state=1),_row("a",state=2,mining_rank_score=9)]
    selected,report=select_parent_balanced_with_report(rows,target=2,masses={"middle":1,"late":0,"early":0})
    assert len(selected)==1 and report["duplicate_rejected"]["candidate_id"]==1


def test_duplicate_across_source_groups_and_layer_fallback_keeps_one():
    rows=[_row("a",state=1,source_group="one"),_row("b",layer="late",state=1,source_group="two")]
    selected,_=select_parent_balanced_with_report(rows,target=2,masses={"middle":0,"late":0,"early":1})
    assert len(selected)==1


def test_parent_cap_counts_only_unique_states():
    rows=[_row(f"a{i}",parent="p",state=i) for i in range(6)]+[_row("duplicate",parent="p",state=0)]
    selected,report=select_parent_balanced_with_report(rows,target=6,masses={"middle":1,"late":0,"early":0},parent_cap=4)
    assert len(selected)==4 and report["maximum_children_per_parent"]==4


def test_selection_is_deterministic_in_order_and_hash():
    rows=[_row(str(i),f"p{i%3}",("middle","late","early")[i%3],state=i,mining_rank_score=i%2) for i in range(9)]
    first,one=select_parent_balanced_with_report(rows,target=8,masses={"middle":.4,"late":.4,"early":.2},parent_cap=4)
    second,two=select_parent_balanced_with_report(list(reversed(rows)),target=8,masses={"middle":.4,"late":.4,"early":.2},parent_cap=4)
    assert [x["id"] for x in first]==[x["id"] for x in second]
    assert one["selected_order"]==two["selected_order"]


def test_unique_shortfall_is_valid_for_construction_but_duplicate_bank_is_rejected():
    unique=[_row(str(i),state=i) for i in range(7)]
    selected,report=select_parent_balanced_with_report(unique,target=28,masses={"middle":.4,"late":.4,"early":.2})
    assert report["status"]=="PASS" and validate_unique_candidates(selected)["status"]=="PASS"
    with pytest.raises(SystemExit,match="duplicate snapshots"):
        validate_unique_candidates(selected+[dict(selected[0],id="new-wrapper")])


def test_mixed_snapshot_source_policy_requires_exact_record_declarations():
    metadata={"snapshot_source_policy_hashes":["old","current"]};rows=[{"id":"a","snapshot_source_policy_hash":"old"},{"id":"b","snapshot_source_policy_hash":"current"}]
    assert validate_snapshot_source_records(rows,metadata)==("current","old")
    assert verify_source_policy_paths(("old","current"),("old-path","new-path"),lambda p:{"old-path":"old","new-path":"current"}[p])==("current","old")


def test_roll_controllability_requires_effect_benefit_and_bounded_pitch():
    cfg=SimpleNamespace(roll_controllability_min_roll_sensitivity=.01,roll_controllability_min_roll_rate_sensitivity=.05,
        roll_controllability_min_beneficial_fraction=.25,roll_controllability_max_pitch_side_effect=.10)
    good=[{"roll_sensitivity":.02,"roll_rate_sensitivity":.08,"pitch_side_effect":.02,"beneficial":True,"perturbation_immediate_failure":False} for _ in range(4)]
    weak=[{**row,"roll_sensitivity":.001,"roll_rate_sensitivity":.001} for row in good]
    result=audit_decision({"steering":good,"wheel_drive":weak,"hip":weak,"knee":weak},cfg)
    assert result["roll_controllable"] and result["controllable_channels"]==["steering"]
