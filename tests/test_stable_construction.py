from types import SimpleNamespace

from dvgc.certification import branch_seed
from dvgc.stable_construction import adaptive_indices, stable_result, stage_b_indices
from cli.merge_stable_descent_stage import merge_stage


CFG = SimpleNamespace(
    beta_alpha0=1.0, beta_beta0=1.0, posterior_q_low=0.05, posterior_q_high=0.95,
    min_branches=8, safe_threshold=0.7, dead_threshold=0.3, boundary_max_width=0.35,
    stable_construction_near_safe_lcb_margin=0.05,
)


def evidence(successes, branches=32, *, seed=1_000_000, index=0, namespace="stage"):
    return [
        {"branch_index": branch, "branch_seed": branch_seed(seed, index, branch),
         "seed_namespace": namespace, "chain_success": branch < successes,
         "final_recovery": branch < successes,
         "terminal_cause": "final_recovery" if branch < successes else "physical_failure",
         "end_reason": "recovery" if branch < successes else "roll_limit"}
        for branch in range(branches)
    ]


def row(index, successes, *, seed, namespace):
    return {"id":str(index), "candidate_index":index,
            "branch_evidence":evidence(successes, seed=seed, index=index, namespace=namespace)}


def test_stable_safe_requires_both_independent_batches_and_combined_lcb():
    a=row(0,28,seed=1_000_000,namespace="a")
    b=row(0,28,seed=2_000_000,namespace="b")
    assert stable_result(a,b,None,CFG)["stable_safe"]
    weak=row(0,22,seed=2_000_000,namespace="b")
    result=stable_result(a,weak,None,CFG)
    assert not result["stable_safe"] and result["label"]!="safe"


def test_dead_stage_a_skips_confirmation_and_disagreement_gets_adaptive():
    dead=row(0,1,seed=1_000_000,namespace="a")
    safe=row(1,28,seed=1_000_000,namespace="a")
    assert stage_b_indices([dead,safe],CFG)==[1]
    weak=row(1,22,seed=2_000_000,namespace="b")
    assert adaptive_indices([safe],[weak],CFG)==[1]


def _shard(indices, start, end, seed=1_000_000):
    common={"status":"PASS","complete":True,"stage":"stage_a","seed":seed,
            "seed_namespace":"stable:stage_a","candidate_bank_sha256":"bank",
            "candidate_source_policy_hash":"source","descent_policy_hash":"policy",
            "candidate_source_policy_hashes":["source"],
            "descent_policy_version":"policy-v","landing_policy_hash":"landing",
            "landing_policy_version":"landing-v","landing_entry_set_sha256":"entry",
            "xml_sha256":"xml","config_hash":"config","runtime_source_fingerprint":"runtime",
            "certification_protocol_version":"stable-descent-cross-seed-v1","construction_seed_epoch":5,
            "protocol":{},"branch_horizon":750,"branches_per_state":32,"total_states":3,
            "selected_states":len(indices)}
    from cli.certify_stable_descent_shard import indices_hash
    common["selected_indices_sha256"]=indices_hash(indices)
    common["selection_start"],common["selection_end"]=start,end
    common["rows"]=[row(i,28,seed=seed,namespace="stable:stage_a") for i in indices[start:end]]
    return common


def test_stable_stage_merge_requires_selected_coverage_and_exact_seeds():
    indices=[0,2]
    merged=merge_stage([_shard(indices,0,1),_shard(indices,1,2)],indices)
    assert merged["candidate_indices"]==indices and merged["states"]==2
    broken=_shard(indices,1,2);broken["rows"][0]["branch_evidence"][0]["branch_seed"]+=1
    try:merge_stage([_shard(indices,0,1),broken],indices)
    except ValueError as exc:assert "seed" in str(exc).lower()
    else:raise AssertionError("stable merge accepted a wrong global branch seed")
