from cli.run_backward_descent_rsi_pilot import build_p0_training_records


def test_p0_training_bank_is_balanced_and_never_claims_safe():
    nodes=[]
    for i,(candidate,layer,region,p1) in enumerate((("a",1,"late",True),("a",1,"late",False),("b",2,"middle",False))):
        nodes.append({"node_id":str(i),"p0":True,"p1":p1,"source_state_hash":str(i),"candidate_id":candidate,"layer":layer,"region":region})
    records=build_p0_training_records(nodes,lambda node:{"id":node["node_id"],"final":{"label":"safe"},"policy_version":"old"})
    assert len(records)==3 and sum(row["reset_weight"] for row in records)==1.0
    assert all(row["artifact_role"]=="proposal_support_bank" and "final" not in row and "policy_version" not in row for row in records)
