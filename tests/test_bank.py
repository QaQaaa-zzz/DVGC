import numpy as np
import pytest
from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import branch_evidence


def record(phase="landing"):
    return {"qpos":np.zeros(12,np.float32),"qvel":np.zeros(11,np.float32),"ctrl":np.zeros(4,np.float32),"physical_feature":np.zeros(16,np.float32),"source_phase":phase}


def test_unknown_boundary_separation():
    p=beta_posterior(4,4)
    assert posterior_label(p,8,min_branches=8,safe_threshold=.7,dead_threshold=.3,boundary_max_width=.05)=="unknown"
    assert posterior_label(p,8,min_branches=8,safe_threshold=.7,dead_threshold=.3,boundary_max_width=1.0)=="boundary"


def test_boundary_requires_mean_inside_decision_interval():
    near_safe=beta_posterior(7,2)
    assert near_safe["mean"]>.7 and near_safe["lower"]<.7
    assert posterior_label(near_safe,9,min_branches=8,safe_threshold=.7,dead_threshold=.3,boundary_max_width=1.0)=="unknown"


def test_dual_certification(tmp_path):
    b=SnapshotBank([record()]); r=b.records[0]
    evidence=[branch_evidence(branch_index=i,seed=i,seed_namespace="test",dynamics_variant="nominal",outcome={"chain":1,"final":i<7,"terminated":1,"truncated":0,"steps":10}) for i in range(8)]
    b.update_certification(r["id"],chain_successes=8,chain_failures=0,final_successes=7,final_failures=1,policy_version="p",estimator_version="e",tube_version="t",protocol={"alpha0":1,"beta0":1,"q_low":.05,"q_high":.95,"min_branches":8,"safe_threshold":.5,"dead_threshold":.2,"boundary_max_width":.5},seed_namespace="test",branch_evidence=evidence)
    assert r["chain"]["branches"]==8 and r["final"]["branches"]==8
    assert r["certification_branches"]==evidence
    path=tmp_path/"tube.pkl"; b.save(path); loaded=SnapshotBank.load(path)
    assert loaded.records[0]["certification_branches"]==evidence


def test_certification_rejects_evidence_count_mismatch():
    b=SnapshotBank([record()]); r=b.records[0]
    with pytest.raises(ValueError,match="Expected 8 branch evidence"):
        b.update_certification(r["id"],chain_successes=8,chain_failures=0,final_successes=8,final_failures=0,policy_version="p",estimator_version="e",tube_version="t",protocol={"alpha0":1,"beta0":1,"q_low":.05,"q_high":.95,"min_branches":8,"safe_threshold":.5,"dead_threshold":.2,"boundary_max_width":.5},seed_namespace="test",branch_evidence=[])


def test_certification_provenance_rejects_policy_and_tube_mixtures():
    b=SnapshotBank([record(),record()]); evidence=[branch_evidence(branch_index=i,seed=i,seed_namespace="test",dynamics_variant="nominal",outcome={"chain":1,"final":1,"terminated":1,"truncated":0,"steps":10}) for i in range(8)]
    protocol={"alpha0":1,"beta0":1,"q_low":.05,"q_high":.95,"min_branches":8,"safe_threshold":.5,"dead_threshold":.2,"boundary_max_width":.5}
    for i,row in enumerate(b.records):
        b.update_certification(row["id"],chain_successes=8,chain_failures=0,final_successes=8,final_failures=0,policy_version="p",estimator_version="e",tube_version=f"t{i}",protocol=protocol,seed_namespace="test",branch_evidence=evidence)
    b.metadata.update({"last_policy_version":"p","last_tube_version":"t1"})
    with pytest.raises(ValueError,match="mixes landing Tube versions"):
        b.validate_certification_provenance("landing",policy_version="p",estimator_version="e")
    b.records[1]["tube_version"]="t0"; b.metadata["last_tube_version"]="t0"
    assert b.validate_certification_provenance("landing",policy_version="p",estimator_version="e")=="t0"
    with pytest.raises(ValueError,match="another policy"):
        b.validate_certification_provenance("landing",policy_version="other",estimator_version="e")
