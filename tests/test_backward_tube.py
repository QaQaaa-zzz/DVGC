import dataclasses

import pytest

from dvgc.backward_tube import (
    ARTIFACT_ROLE, BackwardTubeNode, balanced_rsi_weights, deduplicate_nodes,
    p0_decision, p1_decision, tube_gate, validate_parent_lineage,
)


def _node(identifier, *, phase="descent", layer=1, parent="root", candidate="c", region="late", p0=True, p1=True):
    return BackwardTubeNode(identifier,phase,layer,region,candidate,identifier,{"qpos":[0.]},[0.],parent,"C_L","sequence","a",1,{"id":parent},p0,p0,p1,({"final_recovery":p1},),.1,{"xml":"x"})


def test_survival_without_downstream_final_is_never_safe():
    rows=[{"active_prefix_exact":True,"downstream_entry":True,"final_recovery":False,"termination_tick":24,"downstream_entry_tick":2,"end_code":0}]*2
    assert not p0_decision(rows)["pass"]


def test_p0_requires_repeat_event_identity_and_legal_physics():
    row={"active_prefix_exact":True,"downstream_entry":True,"final_recovery":True,"termination_tick":8,"downstream_entry_tick":2,"end_code":4}
    assert p0_decision([row,row])["pass"]
    assert not p0_decision([row,row|{"termination_tick":9}])["pass"]
    assert not p0_decision([row,row|{"penetration":True}])["pass"]


def test_p1_requires_three_full_chain_branches_and_no_new_failure():
    success={"downstream_entry":True,"final_recovery":True}
    fail={"downstream_entry":False,"final_recovery":False,"failure_type":"pitch_limit"}
    assert p1_decision({"pass":True},[success,success,success,fail],"pitch_limit")["pass"]
    assert not p1_decision({"pass":True},[success,success,fail,fail],"pitch_limit")["pass"]
    assert not p1_decision({"pass":True},[success,success,success,fail|{"failure_type":"roll_limit"}],"pitch_limit")["pass"]


def test_nodes_cannot_claim_certified_role_or_orphan_lineage():
    with pytest.raises(ValueError): dataclasses.replace(_node("x"),artifact_role="certified_tube").validate()
    with pytest.raises(ValueError): dataclasses.replace(_node("x"),parent_node_id=None).validate()


def test_parent_lineage_reaches_root_and_rejects_cycles():
    root=_node("root",phase="landing",layer=0,parent=None,region="root")
    child=_node("child",parent="root")
    assert validate_parent_lineage([root,child],{"root"})["valid"]
    broken=dataclasses.replace(child,parent_node_id="missing")
    assert not validate_parent_lineage([root,broken],{"root"})["valid"]


def test_dedup_prefers_p1_over_nearby_p0():
    p1=_node("p1"); p0=_node("p0",p1=False)
    kept,rejected=deduplicate_nodes([p0,p1],{"p0":[0.01,0.],"p1":[0.,0.]},[1.,1.],.1)
    assert [x.node_id for x in kept]==["p1"] and rejected[0]["node_id"]=="p0"


def test_balanced_rsi_weights_are_normalized_and_do_not_count_duplicates():
    nodes=[_node("a",candidate="a",layer=1),_node("b",candidate="b",layer=2,region="middle")]
    weights=balanced_rsi_weights(nodes)
    assert sum(weights.values())==pytest.approx(1.) and weights["b"]>weights["a"]


def test_start_gate_requires_real_candidate_layer_region_coverage():
    nodes=[]
    for i in range(16):
        nodes.append(_node(str(i),candidate=f"c{i%6}",layer=1+i%3,region=("early","middle","late")[i%3]))
    assert tube_gate(nodes)["status"]=="PASS"
    assert tube_gate(nodes[:15])["status"]=="FAIL"
