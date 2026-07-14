import numpy as np
from dvgc.bank import SnapshotBank, beta_posterior, posterior_label


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


def test_dual_certification():
    b=SnapshotBank([record()]); r=b.records[0]
    b.update_certification(r["id"],chain_successes=8,chain_failures=0,final_successes=7,final_failures=1,policy_version="p",estimator_version="e",tube_version="t",protocol={"alpha0":1,"beta0":1,"q_low":.05,"q_high":.95,"min_branches":8,"safe_threshold":.5,"dead_threshold":.2,"boundary_max_width":.5},seed_namespace="test")
    assert r["chain"]["branches"]==8 and r["final"]["branches"]==8
