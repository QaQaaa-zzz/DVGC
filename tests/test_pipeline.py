import copy
import json

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.certification import branch_evidence, branch_seed, summarize_branches
from dvgc.certification_merge import merge_certification_parts
from dvgc.pipeline import curriculum_decision, marker_is_current, training_decision, write_marker


def _record(index):
    return {
        "id":f"state-{index}","source_phase":"flight",
        "qpos":np.full(12,index,np.float32),"qvel":np.zeros(11,np.float32),
        "ctrl":np.zeros(4,np.float32),"physical_feature":np.full(16,index,np.float32),
    }


def _certify(bank,index):
    branches=[branch_evidence(branch_index=b,seed=branch_seed(10,index,b),seed_namespace="build:flight",dynamics_variant="nominal",outcome={"chain":1,"final":1,"terminated":1,"truncated":0,"steps":12}) for b in range(8)]
    bank.update_certification(f"state-{index}",chain_successes=8,chain_failures=0,final_successes=8,final_failures=0,policy_version="policy",estimator_version="event_filter_v1",tube_version=f"part-{index}",protocol={"alpha0":1,"beta0":1,"q_low":.05,"q_high":.95,"min_branches":8,"safe_threshold":.5,"dead_threshold":.2,"boundary_max_width":.5},seed_namespace="build:flight",branch_evidence=branches)
    return branches


def _part_report(start,stop,branches):
    return {
        "phase":"flight","policy_version":"policy","estimator_version":"event_filter_v1",
        "seed_namespace":"build:flight","construction_seed":10,"candidate_bank_sha256":"candidate",
        "downstream_bank":"/landing.pkl","downstream_bank_sha256":"landing","total_bank_states":2,
        "state_index_start":start,"state_index_end_exclusive":stop,
        "results":[{"id":f"state-{i}","state_index":i} for i in range(start,stop)],
        "terminal_summary":summarize_branches(branches),
    }


def test_certification_chunks_merge_with_global_seeds_and_one_tube_version():
    base=SnapshotBank([_record(0),_record(1)])
    first,second=copy.deepcopy(base),copy.deepcopy(base)
    b0=_certify(first,0); b1=_certify(second,1)
    for bank in (first,second): bank.metadata["dynamics_variants"]=[{"id":"nominal"}]
    merged,report=merge_certification_parts([first,second],[_part_report(0,1,b0),_part_report(1,2,b1)])
    rows=merged.records_for_phase("flight",include_training_only=False)
    assert report["terminal_summary"]["branches"]==16
    assert len({b["branch_seed"] for row in rows for b in row["certification_branches"]})==16
    assert {row["tube_version"] for row in rows}=={report["tube_version"]}
    assert merged.validate_certification_provenance("flight",policy_version="policy",estimator_version="event_filter_v1")==report["tube_version"]


def test_marker_detects_changed_input_and_output(tmp_path):
    source=tmp_path/"input"; source.write_text("a")
    output=tmp_path/"output"; output.write_text("b")
    marker=tmp_path/"marker.json"
    write_marker(marker,step="x",tokens=["v1"],inputs=[source],outputs=[output],exit_status=0)
    assert marker_is_current(marker,tokens=["v1"],inputs=[source],outputs=[output])==(True,"current")
    output.write_text("changed")
    assert not marker_is_current(marker,tokens=["v1"],inputs=[source],outputs=[output])[0]


def test_training_gate_rejects_runtime_failure_and_policy_regression():
    analysis={"analysis_status":"COMPLETED_HEALTHY","health":{"oom":False,"nonfinite_count":0}}
    evaluation={"final_recovery_rate":.60,"chain_rate":.70,"physical_failure_rate":.40,"timeout_rate":0.0}
    reference={"final_recovery_rate":.70}
    decision=training_decision(analysis,evaluation,minimum_final=.5,maximum_timeout=.05,reference_evaluation=reference,maximum_final_drop=.05)
    assert decision["status"]=="FAIL"
    assert "regressed" in decision["reasons"][0]


def test_curriculum_gate_uses_chain_lcb_and_landing_retention():
    evaluation={"episodes":160,"chain_rate":.20,"final_recovery_rate":.30}
    landing={"final_recovery_rate":.85}; reference={"final_recovery_rate":.87}
    decision=curriculum_decision(evaluation,landing,reference,minimum_chain_lcb=.10,minimum_final=.25)
    assert decision["status"]=="PASS" and decision["chain_lcb_95"]>.10
    decision=curriculum_decision(evaluation,{"final_recovery_rate":.70},reference,minimum_chain_lcb=.10,minimum_final=.25)
    assert decision["status"]=="FAIL" and "retention" in decision["reasons"][0]
