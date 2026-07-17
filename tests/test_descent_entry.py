from types import SimpleNamespace

import numpy as np
import jax.numpy as jp

from dvgc.descent_entry import DESCENT_ENTRY_FEATURE_NAMES, descent_entry_feature, matcher_audit
from dvgc.bank import SnapshotBank
from dvgc.descent_local import balanced_parent, build_candidate_bootstrap_bank, difficulty_layers
from cli.certify_descent_entries import qualified_descent_success
from cli.build_descent_entries import snapshot_identity
from cli.merge_descent_entry_audits import merge_reports
from cli.merge_descent_construction_shards import validate_and_merge
from cli.audit_descent_matcher import calibration_metrics
from dvgc.certification import detailed_terminal_summary, summarize_branches
from dvgc.audit_manifest import completed_manifest, invalid_manifest
from dvgc.config import load_config
from dvgc.rewards import compute_descent_local_reward


def test_descent_entry_feature_is_task_relative_without_changing_dynamics_fields():
    physical=np.arange(16,dtype=np.float32); cfg=SimpleNamespace(step_front_x=3.6,step_top_z=.45)
    feature=descent_entry_feature(physical,cfg)
    assert len(feature)==len(DESCENT_ENTRY_FEATURE_NAMES)==16
    assert np.isclose(feature[0],physical[0]-3.6) and np.isclose(feature[2],physical[2]-.45)
    np.testing.assert_array_equal(feature[3:],physical[3:])


def test_matcher_audit_reports_precision_recall_and_coverage():
    safe=[{"entry_feature":np.zeros(16)}]
    rows=[{"entry_feature":np.zeros(16)},{"entry_feature":np.ones(16)*10}]
    matcher={"center":[0.]*16,"scale":[1.]*16,"radius":1.}
    report=matcher_audit(rows,safe,matcher,[True,False])
    assert report["precision"]==report["recall"]==1.0
    assert report["coverage"]==.5


def test_descent_entry_success_requires_same_branch_chain_and_final():
    assert qualified_descent_success({"chain":True,"final":True})
    assert not qualified_descent_success({"chain":False,"final":True})
    assert not qualified_descent_success({"chain":True,"final":False})


def test_handoff_missed_final_is_not_a_c_d_success_or_physical_failure():
    row={"chain_success":False,"final_recovery":False,"terminal_cause":"handoff_missed_final"}
    report=summarize_branches([row])
    assert report["handoff_missed_finals"]==1
    assert report["final_recoveries"]==report["physical_failures"]==0


def _audit_shard(index,seed):
    evidence={"seed_namespace":"audit","branch_seed":seed,"chain_success":False,"final_recovery":False,"terminal_cause":"physical_failure"}
    return {"seed":7,"seed_namespace":"audit","candidate_bank_sha256":"bank","landing_entry_set_sha256":"entry","descent_policy_hash":"descent","landing_policy_hash":"landing","total_states":2,"start_index":index,"end_index":index+1,"rows":[{"id":str(index),"candidate_index":index,"branch_evidence":[evidence]}]}


def test_descent_audit_shards_require_complete_indices_and_unique_seeds():
    merged=merge_reports([_audit_shard(0,10),_audit_shard(1,11)])
    assert merged["states"]==2 and merged["terminal_summary"]["physical_failures"]==2
    duplicate=_audit_shard(1,10)
    try: merge_reports([_audit_shard(0,10),duplicate])
    except ValueError as exc: assert "globally unique" in str(exc)
    else: raise AssertionError("duplicate global audit seed was accepted")


def test_future_audit_preserves_detailed_physical_end_reasons():
    rows = [
        {"chain_success":False,"final_recovery":False,"terminal_cause":"physical_failure","end_reason":"roll_limit"},
        {"chain_success":False,"final_recovery":False,"terminal_cause":"physical_failure","end_reason":"pitch_limit"},
        {"chain_success":False,"final_recovery":False,"terminal_cause":"physical_failure","end_reason":"platform_back_edge_exit"},
        {"chain_success":False,"final_recovery":False,"terminal_cause":"physical_failure","end_reason":"nonfinite"},
        {"chain_success":False,"final_recovery":False,"terminal_cause":"timeout","end_reason":"stage_timeout"},
        {"chain_success":False,"final_recovery":False,"terminal_cause":"horizon_exhausted","end_reason":None},
    ]
    summary = detailed_terminal_summary(rows)
    assert summary["physical_end_reasons"] == {
        "roll":1,"pitch":1,"platform_back_edge_exit":1,"nonfinite":1,
        "other_physical":0,"timeout":1,"horizon":1,
    }


def test_pointwise_manifest_has_terminal_states_and_strict_provenance():
    launch={"status":"ACTIVE","states":2,"policy_hash":"p","candidate_bank_sha256":"c",
            "landing_entry_set_sha256":"e","landing_policy_hash":"l"}
    merged={"states":2,"descent_policy_hash":"p","candidate_bank_sha256":"c",
            "landing_entry_set_sha256":"e","landing_policy_hash":"l",
            "terminal_summary":{"branches":64}}
    assert completed_manifest(launch,{"status":"PASS"},merged)["status"]=="COMPLETED_PASS"
    assert completed_manifest(launch,{"status":"FAIL"},merged)["status"]=="COMPLETED_FAIL"
    assert invalid_manifest(launch,"seed conflict")["status"]=="INVALID"
    merged["descent_policy_hash"]="wrong"
    try: completed_manifest(launch,{"status":"FAIL"},merged)
    except ValueError as exc: assert "provenance" in str(exc)
    else: raise AssertionError("terminal manifest accepted mismatched policy")


def _construction_shard(index, seed, *, duplicate_seed=False):
    branch_seed_value = seed if duplicate_seed else 7 + index * 10_000
    evidence = [{"branch_index": b, "branch_seed": branch_seed_value + b,
                 "seed_namespace": "construction", "chain_success": False,
                 "final_recovery": False, "terminal_cause": "physical_failure"}
                for b in range(8)]
    common = {"status":"PASS", "complete":True, "seed":7,
              "seed_namespace":"construction", "candidate_bank_sha256":"bank",
              "candidate_source_policy_hash":"source", "landing_entry_set_sha256":"entry",
              "descent_policy_hash":"descent", "descent_policy_version":"descent-v1",
              "descent_estimator_version":"event-v1", "landing_policy_hash":"landing",
              "landing_policy_version":"landing-v1", "xml_sha256":"xml", "config_hash":"config",
              "runtime_source_fingerprint":"runtime", "protocol":{}, "min_branches":8,
              "max_branches":32, "branch_horizon":64, "total_states":2,
              "confirm_safe_to_max":True, "start_index":index, "end_index":index+1}
    common["rows"] = [{"id":str(index), "candidate_index":index, "branches":8,
                       "chain":0, "final":0, "branch_evidence":evidence}]
    return common


def test_descent_construction_shards_require_global_indices_and_stable_branch_seeds():
    source = SnapshotBank([{"id":str(i), "source_phase":"flight", "qpos":np.zeros(2),
                            "qvel":np.zeros(2), "ctrl":np.zeros(1),
                            "physical_feature":np.zeros(16)} for i in range(2)])
    cfg = SimpleNamespace(beta_alpha0=1., beta_beta0=1., posterior_q_low=.05,
                          posterior_q_high=.95, min_branches=8, safe_threshold=.7,
                          dead_threshold=.3, boundary_max_width=.4)
    rows = validate_and_merge([_construction_shard(0, 7), _construction_shard(1, 10007)], source, cfg)
    assert [row["candidate_index"] for row in rows] == [0, 1]
    bad = _construction_shard(1, 7, duplicate_seed=True)
    try: validate_and_merge([_construction_shard(0, 7), bad], source, cfg)
    except ValueError as exc: assert "seed" in str(exc).lower()
    else: raise AssertionError("unstable construction branch seed was accepted")


def test_descent_matcher_calibration_metrics_use_independent_rates():
    result = calibration_metrics([0.1, 0.9], [0.0, 1.0], bins=2)
    assert np.isclose(result["brier"], 0.01)
    assert np.isclose(result["ece"], 0.1)


def test_descent_proposal_identity_covers_full_policy_snapshot():
    row={key:np.zeros(2,np.float32) for key in ("qpos","qvel","ctrl","qacc_warmstart")}; row.update({"oracle_phase":2,"had_airborne":1,"had_valid_landing":0,"contact_age":0,"airborne_count":2,"recovery_count":0,"policy_state":{"obs_history":np.zeros((2,3),np.float32),"last_action":np.zeros(2,np.float32)}})
    same={**row,"qpos":row["qpos"].copy(),"policy_state":{k:v.copy() for k,v in row["policy_state"].items()}}
    changed={**same,"policy_state":{**same["policy_state"],"last_action":np.ones(2,np.float32)}}
    assert snapshot_identity(row)==snapshot_identity(same)
    assert snapshot_identity(row)!=snapshot_identity(changed)


def test_descent_layers_and_parent_sampling_are_deterministic_and_balanced():
    rows=[]
    for i in range(6): rows.append({"id":str(i),"entry_source_id":str(i//2),"proposal_step":i%2,"physical_feature":np.full(16,float(i),np.float32)})
    layers=difficulty_layers(rows,{str(i):float(i) for i in range(6)})
    assert list(layers.values()).count("late")==2 and list(layers.values()).count("middle")==2 and list(layers.values()).count("early")==2
    children={row["id"]:0 for row in rows}; children["0"]=children["1"]=2
    selected=balanced_parent(rows,children,np.random.default_rng(7),4)
    assert selected["entry_source_id"]!="0"


def test_candidate_bootstrap_weights_groups_then_parents(tmp_path):
    rows=[]
    for group,count in (("provisional_safe",3),("boundary",4),("successful_anchor",2)):
        for i in range(count):
            parent=f"{group}-p{i//2}"
            rows.append({"id":f"{group}-{i}","source_phase":"flight","qpos":np.zeros(2),"qvel":np.zeros(2),"ctrl":np.zeros(1),"physical_feature":np.zeros(16),"oracle_phase":2,"local_bootstrap_eligible":True,"bootstrap_group":group,"descent_layer":("late","middle","early")[i%3],"parent_candidate_id":parent})
    source=SnapshotBank(rows); path=tmp_path/"pool.pkl"; source.save(path)
    cfg=SimpleNamespace(descent_local_reset_safe_mass=.35,descent_local_reset_boundary_mass=.45,descent_local_reset_anchor_mass=.20)
    training,report=build_candidate_bootstrap_bank(source,str(path),cfg)
    assert np.isclose(sum(row["reset_weight"] for row in training.records),1.0)
    assert report["expected_group_reset_ratio"]=={"provisional_safe":.35,"boundary":.45,"successful_anchor":.2}
    grouped={}
    for row in training.records: grouped.setdefault((row["bootstrap_group"],row["reset_parent_id"]),0.0); grouped[(row["bootstrap_group"],row["reset_parent_id"])]+=row["reset_weight"]
    assert np.isclose(grouped[("boundary","boundary-p0")],grouped[("boundary","boundary-p1")])
    assert all(row["reset_source"]=="flight_curriculum" and row["original_bank_sha256"]==report["source_bank_sha256"] for row in training.records)


def _local_reward(**overrides):
    cfg=load_config("configs/default.json")
    values={"pitch":cfg.descent_local_pitch_center,"pitch_rate":cfg.descent_local_pitch_rate_center,"roll":cfg.descent_local_roll_center,"roll_rate":cfg.descent_local_roll_rate_center,"vx":cfg.descent_local_vx_center,"previous_distance":2.0,"current_distance":2.0,"action":jp.zeros(4),"previous_action":jp.zeros(4),"chain":jp.asarray(False),"hard_failure":jp.asarray(False)}
    values.update(overrides); return compute_descent_local_reward(cfg=cfg,**values)


def test_descent_local_reward_has_bounded_potential_and_stability_semantics():
    centered=_local_reward(); bad_pitch=_local_reward(pitch=0.5); bad_rate=_local_reward(pitch_rate=4.0)
    toward=_local_reward(current_distance=1.5); away=_local_reward(current_distance=2.5)
    jump=_local_reward(action=jp.ones(4),previous_action=-jp.ones(4))
    assert float(centered["reward"])>float(bad_pitch["reward"])
    assert float(centered["reward"])>float(bad_rate["reward"])
    assert float(toward["progress"])>0>float(away["progress"])
    assert float(centered["progress"])==0 and float(jump["reward"])<float(centered["reward"])
    assert -.35<=float(bad_rate["shaping"])<=.25


def test_descent_chain_dominates_survival_and_failure():
    chain=_local_reward(chain=jp.asarray(True)); failure=_local_reward(hard_failure=jp.asarray(True))
    assert float(chain["chain"])==8.0
    assert float(chain["reward"])>float(failure["reward"])
    assert 64*float(chain["survival"])<4.0
