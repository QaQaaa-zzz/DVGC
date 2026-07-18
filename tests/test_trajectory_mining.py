from types import SimpleNamespace

from dvgc.roll_controllability import audit_decision
from dvgc.snapshot_provenance import validate_snapshot_source_records,verify_source_policy_paths
from dvgc.trajectory_mining import layer_targets,select_parent_balanced,select_trace_records,trajectory_parent_id


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
        for child in range(4):rows.append({"id":f"{parent}-{child}","trajectory_parent_id":f"t{parent}",
            "original_candidate_parent":"same","descent_layer":("middle","late","early")[parent%3],"mining_rank_score":100-child})
    chosen=select_parent_balanced(rows,target=20,masses={"middle":.4,"late":.4,"early":.2},parent_cap=4)
    assert len(chosen)==20 and len({x["trajectory_parent_id"] for x in chosen})==10
    assert max(sum(x["trajectory_parent_id"]==parent for x in chosen) for parent in {x["trajectory_parent_id"] for x in chosen})<=4
    assert layer_targets(20,{"middle":.4,"late":.4,"early":.2})=={"middle":8,"late":8,"early":4}


def test_layer_targets_do_not_block_validated_surplus_support():
    rows=[{"id":str(i),"trajectory_parent_id":f"t{i}","descent_layer":"middle","mining_rank_score":10-i} for i in range(8)]
    selected=select_parent_balanced(rows,target=8,masses={"middle":.4,"late":.4,"early":.2},parent_cap=4)
    assert len(selected)==8


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
