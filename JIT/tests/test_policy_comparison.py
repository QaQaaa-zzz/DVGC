from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from jit_dvgc import policy_comparison as workflow
from jit_dvgc.jump_evidence_validation import write, read, file_sha
from jit_dvgc.evidence_integrity import canonical_sha256


def test_failed_attempts_are_charged_conservatively():
    assert workflow.attempt_cost({}, 400) == (400, False)
    assert workflow.attempt_cost({"status":"engineering_error", "environment_interactions":7},400) == (400,False)
    assert workflow.attempt_cost({"status":"completed_shard", "environment_interactions":7},400) == (7,True)
    assert workflow.attempt_cost({"status":"completed_shard", "environment_interactions":401},400) == (400,False)


def test_ledger_counts_each_attempt_once_without_counting_merge(tmp_path):
    plan = {"plan_sha256":"p", "request":{"interaction_budget":1000}, "jobs":[{"shard_maximum_interactions":[400]}]}
    for i in range(2):
        root = tmp_path / f"jobs/train_pi_0/shard_000/attempt_{i:04d}"
        write(root / "reservation.json", {"plan_sha256":"p","job_index":0,"shard_index":0,"maximum_interactions":400})
    write(root / "result/summary.json", {"status":"completed_shard","environment_interactions":15})
    write(tmp_path / "jobs/train_pi_0/merged/summary.json", {"environment_interactions":15})
    ledger = workflow.ledger(tmp_path, plan)
    assert ledger["charged_new_label_interactions"] == 415
    assert ledger["known_new_label_interactions"] == 15
    assert not ledger["all_attempt_costs_measured"]


def test_sources_and_inputs_cannot_change_on_resume(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("# first\n")
    asset = tmp_path / "asset.json"
    write(asset, {"x":1})
    plan = {"repo":str(tmp_path), "input_files":{str(asset):file_sha(asset)}, "sources":{"source.py":file_sha(source)}}
    plan["plan_sha256"] = canonical_sha256(plan)
    write(tmp_path / "plan.json",plan)
    assert workflow.verify_plan(tmp_path / "plan.json") == plan
    source.write_text("# second\n")
    with pytest.raises(ValueError,match="source changed"): workflow.verify_plan(tmp_path / "plan.json")


def test_compact_bundle_keeps_review_data_and_full_option_keeps_vectors(tmp_path):
    for name in ("a.png","a.svg","a.pdf","a.csv","a.json","a.pkl","a.npz"):
        (tmp_path / name).write_text("data")
    with zipfile.ZipFile(workflow.bundle(tmp_path)) as z:
        assert {"a.png","a.csv","a.json"} <= set(z.namelist())
        assert "a.svg" not in z.namelist() and "a.pdf" not in z.namelist()
        assert "a.pkl" not in z.namelist() and "a.npz" not in z.namelist()

    with zipfile.ZipFile(workflow.bundle(tmp_path, full=True)) as z:
        assert {"a.svg", "a.pdf"} <= set(z.namelist())


def test_supervisor_resumes_completed_shards_and_skips_old_caches(tmp_path, monkeypatch):
    calls=[]
    class Process:
        def __init__(self, command, **kwargs):
            value=lambda flag: command[command.index(flag)+1]
            kind=value("--worker")
            calls.append((kind,kwargs["env"]))
            destination=Path(value("--worker-output"))
            if kind=="prepare" and not (tmp_path / "plan.json").exists():
                request=read(tmp_path / "request.json")
                plan={"repo":str(tmp_path),"request":request,"input_files":{},"sources":{},
                      "members":[{"policy":{"name":"pi_0"}}],"first_attempt_maximum_interactions":10,
                      "jobs":[{"job_id":"train_pi_0","role":"train","evaluator":"pi_0","reuse_directory":None,"shard_maximum_interactions":[10]},
                              {"job_id":"train_pi_1","role":"train","evaluator":"pi_1","reuse_directory":"cache","shard_maximum_interactions":[]}]}
                plan["plan_sha256"]=canonical_sha256(plan)
                write(tmp_path / "plan.json",plan)
            if kind=="label": write(destination / "result/summary.json",{"status":"completed_shard","environment_interactions":3})
        def wait(self,timeout=None): return 0
        def poll(self): return 0
    monkeypatch.setattr(workflow.subprocess,"Popen",Process)
    assert workflow.run_comparison(tmp_path,tmp_path)["status"]=="completed"
    assert workflow.run_comparison(tmp_path,tmp_path)["charged_new_label_interactions"]==3
    assert sum(k=="label" for k,_ in calls)==1
    assert all(env["JAX_PLATFORMS"]==("cuda" if kind=="label" else "cpu") for kind,env in calls)


def test_public_failure_still_packages_report(tmp_path, monkeypatch):
    class Process:
        def __init__(self,*a,**k): pass
        def wait(self,timeout=None): return 1
        def poll(self): return 1
    monkeypatch.setattr(workflow.subprocess,"Popen",Process)
    assert workflow.run_comparison(tmp_path,tmp_path)["status"]=="engineering_error"
    assert (tmp_path / "results_to_send.zip").exists()


@pytest.fixture
def prepare_fixture(tmp_path, monkeypatch):
    """Real preflight orchestration/contracts; fake external checkpoint and physics assets."""
    from jit_dvgc import policy_comparison_runtime as runtime
    from jit_dvgc import unified_policy_freeze, unified_envelope_snapshot, unified_continuation_labels, config, model
    from jit_dvgc.analysis import capability_tube, nominal_jump_centerline
    from jit_dvgc.acquisition.causal_jump import ACQUISITION_MODE
    from jit_dvgc.jump_evidence_validation import FIXED_XML_SHA256
    repo, output = tmp_path / "repo", tmp_path / "output"
    scan = repo / "scan"
    paths=[]
    records=[]
    for i in range(4):
        config_path=repo / f"config{i}.json"
        write(config_path,{"ppo":{"episode_horizon":10},"inputs":{"up_config_path":"up.json"}})
        record={"name":f"pi_{i}","iteration":i,"actor_sha256":str(i)*64,"payload_sha256":str(i+4)*64,
                "formal_config":str(config_path),"formal_config_sha256":"c"*64,"xml_sha256":FIXED_XML_SHA256}
        path=repo / f"JIT/runs/frozen_unified/pi_{i}/frozen_unified_policy.json"
        write(path,{"policy":record})
        paths.append(str(path)); records.append(record)
    write(repo / "up.json",{})
    center={"points":[],"centerline_sha256":"b"*64}
    write(repo / "centerline.json",center)
    source={"jump_tube_contract":{"continuation_success_criterion":"first_valid_landing_before_physical_failure",
        "continuation_frozen_policies":paths[:3],"proposal_frozen_policy":paths[0],
        "nominal_centerline":"centerline.json","nominal_centerline_sha256":"b"*64,"jump_start_state_sha256":"a"*64},
        "fixed_probe_panel":{"max_label_ticks":10},"anchors":[{"role":r,"parent_group_id":r} for r in runtime.ROLES],
        "seeds":{r:{"labeling":100+i} for i,r in enumerate(runtime.ROLES)}}
    source["plan_sha256"]=canonical_sha256(source)
    write(scan / "frontier_plan_causal_expanded.json",source)
    snapshots={}
    for role in runtime.ROLES:
        root=scan / f"frontier_{role}/acquisition"
        protocol={"role":role}
        protocol["protocol_sha256"]=canonical_sha256(protocol)
        write(root / "protocol.json",protocol)
        rows=[]
        for i,phase in enumerate(("upstream","downstream")):
            provenance={"schema":"jit_jump_start_reachability_provenance_v1","jump_start_connected":True,
                "natural_start_connected":False,"generated_by_env_step_only":True,"proposal_anchor_used_as_reset":False,
                "rsi_used_to_establish_reachability":False,"qpos_qvel_injection_used":False,
                "jump_start_state_sha256":"a"*64,"perturbation_start_state_sha256":"d"*64,
                "environment_transitions_from_jump_start":i+1,"proposal_parent_group_id":role}
            provenance["reachability_sha256"]=canonical_sha256(provenance)
            row={"candidate_id":f"{role}_{i}","candidate_kind":"reachable_unified_frontier_probe","phase":phase,
                 "phase_index":i,"split":"train","state_sha256":str(i+1)*64,"parent_group_id":role,
                 "parent_state_sha256":"d"*64,"source_bank":"bank","snapshot":str(i),"policy_iteration":0,
                 "policy_actor_sha256":records[0]["actor_sha256"],"policy_payload_sha256":records[0]["payload_sha256"],
                 "protocol_sha256":protocol["protocol_sha256"],"jump_start_reachability":provenance}
            rows.append(row)
            snapshots[str(root / "bank" / str(i))]=SimpleNamespace(qpos=i,qvel=i,down_events={"valid_contact_seen":False},up_events={"apex_seen":bool(i)})
        catalog={"schema":"jit_unified_boundary_catalog_v1","status":"completed","artifact_role":"unlabeled_policy_conditioned_frontier_candidates",
             "split":"train","training_transitions":0,"expert_switching_used":False,"test_data_used":False,
             "validation_data_used":False,"final_evaluation_data_used":False,"frozen_unified_manifest_sha256":file_sha(paths[0]),
             "iteration":0,"policy_name":"pi_0","policy_actor_sha256":records[0]["actor_sha256"],
             "policy_payload_sha256":records[0]["payload_sha256"],"candidate_count":2,"entries":rows,
             "claim_boundary":{"unlabeled_acquisition_only":True,"tube_expansion_claim":False,"jce_jel_claim":False,"certified_safe_set_claim":False},
             "acquisition_mode":ACQUISITION_MODE,"jump_start_reachability_proven":True,"rsi_used_for_reachability":False,
             "protocol_sha256":protocol["protocol_sha256"]}
        write(root / "catalog.json",catalog)
        write(root / "summary.json",{"status":"completed","candidate_count":2,"protocol_sha256":protocol["protocol_sha256"],"environment_interactions":5})
    write(output / "request.json",{"repo":str(repo),"output":str(output),"scan_root":str(scan),
          "shard_size":1,"interaction_budget":1000,"frozen_policies":[]})
    monkeypatch.setattr(unified_policy_freeze,"load_frozen_unified_manifest",lambda p:read(p))
    monkeypatch.setattr(unified_envelope_snapshot,"load_unified_envelope_snapshot",lambda p:snapshots[str(p)])
    monkeypatch.setattr(unified_envelope_snapshot,"snapshot_context_sha256",lambda s:str(s.qpos)*64)
    monkeypatch.setattr(unified_continuation_labels,"validate_candidate_snapshot",lambda snapshot,row,*,policy_record:None)
    monkeypatch.setattr(config,"load_config",lambda p:object())
    monkeypatch.setattr(model,"load_host_model",lambda c:SimpleNamespace(xml_sha256=FIXED_XML_SHA256))
    monkeypatch.setattr(nominal_jump_centerline,"load_nominal_jump_centerline",lambda p:read(p))
    def coords(qpos,qvel,*,bundle):
        c=dict.fromkeys(capability_tube.FULL_PHYSICAL_FIELDS,0.)
        c.update(root_x_m=2.5,root_z_m=.2,root_vz_mps=1. if qpos==0 else -1.)
        return c
    monkeypatch.setattr(capability_tube,"physical_coordinates_from_arrays",coords)
    return runtime,repo,scan,output


def test_preflight_preserves_role_seeds_and_builds_all_policy_jobs(prepare_fixture):
    runtime,repo,scan,output=prepare_fixture
    result=runtime.prepare(output)
    assert result["first_attempt_maximum_interactions"]==240
    plan=workflow.verify_plan(output / "plan.json")
    assert len(plan["jobs"])==12
    assert {j["seed"] for j in plan["jobs"] if j["role"]=="calibration"}=={101}
    assert plan["snapshot_replay_equivalence_verified"] is False
    assert plan["legacy_family"]==["pi_0","pi_1","pi_2"]
    assert all(j["shard_maximum_interactions"]==[10,10] for j in plan["jobs"])


def test_preflight_refuses_candidate_from_another_role(prepare_fixture):
    runtime,repo,scan,output=prepare_fixture
    path=scan / "frontier_train/acquisition/catalog.json"
    catalog=read(path)
    catalog["entries"][0]["parent_group_id"]="calibration"
    write(path,catalog)
    with pytest.raises(ValueError,match="logical role"): runtime.prepare(output)


def test_preflight_locks_reused_label_files_and_reserves_only_missing_jobs(prepare_fixture,monkeypatch):
    runtime,repo,scan,output=prepare_fixture
    cache=scan / "frontier_calibration/labels/per_policy/pi_0"
    for name,value in (("summary.json",{"status":"completed","environment_interactions":4}),("labels.json",[]),("protocol.json",{})):
        write(cache/name,value)
    monkeypatch.setattr(runtime,"complete_output",lambda directory,*args: ({"status":"completed"},[]) if Path(directory)==cache else None)
    assert runtime.prepare(output)["first_attempt_maximum_interactions"]==220
    plan=read(output / "plan.json")
    job=next(j for j in plan["jobs"] if j["job_id"]=="calibration_pi_0")
    assert job["reuse_directory"]==str(cache) and job["shard_maximum_interactions"]==[]
    write(cache/"labels.json",[{"tampered":True}])
    with pytest.raises(ValueError,match="input changed"): workflow.verify_plan(output / "plan.json")
