from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from jit_dvgc import jump_evidence_validation as audit


def test_sampling_covers_apex_neighbors_without_postlanding_or_ascending_downstream():
    frames = [{"phase_index": 0, "done": False, "valid_contact_seen": False, "vz": 1.0} for _ in range(5)]
    frames += [{"phase_index": 1, "done": False, "valid_contact_seen": False, "vz": vz} for vz in (0.1, -0.1, -0.2, -0.3)]
    frames.append({"phase_index": 1, "done": False, "valid_contact_seen": True, "vz": -1.0})
    indices, counts = audit.select_sample_indices(frames, 3)
    assert indices == [0, 2, 4, 6, 7, 8]
    assert counts == {"0": 5, "1": 3}
    assert audit.select_sample_indices(frames[:1], 3)[0] == [0]
    assert audit.select_sample_indices([], 3)[0] == []


def test_comparator_reports_first_divergence_and_rejects_nan_missing_and_horizon_drift():
    a = [{"x": [0.0], "done": False}, {"x": [1.0], "done": True}]
    b = [{"x": [1e-8], "done": False}, {"x": [1.1], "done": True}]
    report = audit.compare_traces(a, b, atol=1e-6, rtol=1e-5, fields=("x", "done"))
    assert report["passed"] is False
    assert report["fields"]["x"]["first_different_frame"] == 1
    for bad in ([], [{"x": [float("nan")], "done": False}], [{"done": False}], a[:1]):
        assert not audit.compare_traces(a, bad, atol=1e-6, rtol=1e-5, fields=("x", "done"))["passed"]
    assert not audit.compare_traces([{"done": True}], [{"done": False}], atol=1, rtol=1, fields=("done",))["passed"]


@pytest.fixture
def plan_fixture(tmp_path):
    repo = tmp_path / "repo"
    config_path = repo / "JIT/configs/pi_unified_round1_natural10.json"
    config = {"ppo": {"episode_horizon": 400}}
    audit.write(config_path, config)
    checkpoint = repo / "JIT/runs/pi_unified/run/checkpoints/transition_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "payload.pkl").write_bytes(b"fixture not a production checkpoint")
    audit.write(checkpoint / "identity.json", {})
    audit.write(checkpoint.parent.parent / "formal_report.json", {})
    policy = {"name": "pi_0", "iteration": 0, "xml_sha256": audit.FIXED_XML_SHA256,
              "formal_config": str(config_path), "formal_config_sha256": audit.canonical_sha256(config),
              "checkpoint": str(checkpoint), "actor_sha256": "a" * 64, "payload_sha256": "b" * 64}
    frozen = {"status": "frozen", "policy": policy}
    frozen["freeze_protocol_sha256"] = audit.canonical_sha256(frozen)
    frozen_path = repo / "JIT/runs/frozen_unified/round1/frozen_unified_policy.json"
    audit.write(frozen_path, frozen)
    source = repo / "JIT/src/jit_dvgc/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("# locked source\n")
    return repo, frozen_path, tmp_path / "output"


def test_plan_uses_canonical_config_identity_and_enforces_budget(plan_fixture):
    repo, frozen, output = plan_fixture
    plan = audit.build_plan(repo, output)
    assert plan["proposer"]["path"] == str(frozen)
    assert plan["maximum_environment_interactions"] == 17200
    assert plan["formal_envelope_claim_authorized"] is False
    audit.write(output / "plan.json", plan)
    assert audit.load_plan(output / "plan.json") == plan
    with pytest.raises(ValueError, match="exceeds"):
        audit.build_plan(repo, output, interaction_budget=17199)
    (repo / "JIT/src/jit_dvgc/fixture.py").write_text("# modified\n")
    with pytest.raises(ValueError, match="source changed"):
        audit.load_plan(output / "plan.json")


def test_ambiguous_pi0_and_missing_checkpoint_fail_before_rollout(plan_fixture):
    repo, frozen, output = plan_fixture
    audit.write(frozen.parent.parent / "copy/frozen_unified_policy.json", audit.read(frozen))
    with pytest.raises(ValueError, match="uniquely"):
        audit.build_plan(repo, output)
    checkpoint = Path(audit.read(frozen)["policy"]["checkpoint"])
    (checkpoint / "payload.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="missing production artifacts"):
        audit.build_plan(repo, output, frozen_policy=frozen)


def test_failed_preflight_still_returns_a_small_result_bundle(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "output"
    report = audit.run_validation(repo, output)
    assert report["status"] == "engineering_error"
    assert report["production_validation_completed"] is False
    assert report["known_environment_interactions"] == 0
    with zipfile.ZipFile(output / "results_to_send.zip") as archive:
        assert {"summary.json", "failure.json"} <= set(archive.namelist())
    with pytest.raises(FileExistsError):
        audit.run_validation(repo, output)


def test_packaging_never_includes_binary_checkpoints_or_snapshots(tmp_path):
    audit.write(tmp_path / "summary.json", {"status": "fixture"})
    (tmp_path / "snapshot.pkl").write_bytes(b"not for upload")
    (tmp_path / "checkpoint.npz").write_bytes(b"not for upload")
    (tmp_path / "process.log").write_text("diagnostic\n")
    with zipfile.ZipFile(audit.package_results(tmp_path)) as archive:
        assert set(archive.namelist()) == {"summary.json", "process.log"}


def label_fixture(tmp_path):
    catalog = {"candidate_count": 1, "entries": [{"candidate_id": "c0", "state_sha256": "s", "snapshot_context_sha256": "h"}]}
    catalog_path = tmp_path / "catalog.json"
    audit.write(catalog_path, catalog)
    common = {field: "identity" for field in ("policy_actor_sha256", "policy_payload_sha256", "frozen_unified_manifest_sha256",
        "candidate_catalog_protocol_sha256", "acquisition_policy_actor_sha256", "acquisition_policy_payload_sha256")}
    common.update(candidate_catalog_file_sha256=audit.file_sha(catalog_path), candidate_count=1,
                  protocol_seed=7, max_ticks_per_candidate=400, success_criterion="first_valid_landing")
    row = {field: False for field in audit.SEMANTIC_LABEL_FIELDS}
    row.update(catalog["entries"][0], actor_observation=[0.0] * 76, label=1,
               candidate_index=0, policy_key_candidate_index=0)
    for name in ("serial", "merged"):
        protocol = {**common, "execution_mode": name}
        protocol["protocol_sha256"] = audit.canonical_sha256(protocol)
        directory = tmp_path / name
        audit.write(directory / "protocol.json", protocol)
        audit.write(directory / "labels.json", [{**row, "label_protocol_sha256": protocol["protocol_sha256"]}])
        audit.write(directory / "summary.json", {"status": "completed", "protocol_sha256": protocol["protocol_sha256"],
                    "labels_file_sha256": audit.file_sha(directory / "labels.json")})
    return catalog_path


def test_serial_shard_comparison_exempts_only_execution_metadata(tmp_path):
    catalog = label_fixture(tmp_path)
    compare = lambda: audit.compare_label_outputs(tmp_path / "serial", tmp_path / "merged", catalog, atol=1e-6, rtol=1e-5)
    assert compare()["passed"]
    path = tmp_path / "merged/labels.json"
    rows = audit.read(path)
    rows[0]["label"] = 0
    audit.write(path, rows)
    with pytest.raises(ValueError, match="file hash drift"):
        compare()
    summary = audit.read(tmp_path / "merged/summary.json")
    summary["labels_file_sha256"] = audit.file_sha(path)
    audit.write(tmp_path / "merged/summary.json", summary)
    result = compare()
    assert not result["passed"]
    assert "label" in result["mismatches"][0]["fields"]


def test_supervisor_runs_fresh_stages_and_does_not_double_count_merge(plan_fixture, monkeypatch):
    repo, frozen, output = plan_fixture
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            value = lambda flag: command[command.index(flag) + 1]
            directory, kind = Path(value("--worker-output")), value("--worker")
            report = {"environment_interactions": 0 if kind == "merge" else 5}
            if kind == "capture":
                report.update(candidate_count=6, coverage_complete=True, prefix_valid_landing_observed=True)
            if kind == "replay":
                report.update({field: {"passed": True} for field in ("prefix_replay", "preserved_restore", "fresh_restore", "counter_effect")})
            audit.write(directory / "report.json", report)
        def wait(self, timeout=None): return 0
        def poll(self): return 0
    monkeypatch.setattr(audit.subprocess, "Popen", Process)
    monkeypatch.setattr(audit, "compare_label_outputs", lambda *a, **k: {"passed": True})
    report = audit.run_validation(repo, output)
    assert len(calls) == 12  # capture + six replays + serial + three shards + merge
    assert report["status"] == "passed_on_sampled_states"
    assert report["known_environment_interactions"] == 55
    assert report["formal_envelope_claim_authorized"] is False
    assert all(kwargs["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" for _, kwargs in calls)


def test_fresh_counter_control_preserves_physics_and_policy_history():
    import jax.numpy as jp
    from jit_dvgc.unified_continuation_labels import fresh_unified_continuation_state
    class State(SimpleNamespace):
        def replace(self, **changes): return State(**{**vars(self), **changes})
    class Events(SimpleNamespace):
        def replace(self, **changes): return Events(**{**vars(self), **changes})
    data, obs, history = object(), object(), object()
    state = State(done=jp.array(0.0), data=data, obs=obs, reward=jp.array(1.0), metrics={},
        info={"active_phase": jp.array(1), "start_phase": jp.array(0), "history": history,
              "up_events": Events(episode_step=jp.array(17), apex_seen=True), "episode_step": jp.array(18),
              "phase_episode_step": jp.array(1), "phase_transitioned": True, "episode_return": 2.0})
    changed = fresh_unified_continuation_state(state)
    assert changed.data is data and changed.obs is obs and changed.info["history"] is history
    assert int(changed.info["episode_step"]) == int(changed.info["phase_episode_step"]) == 0
    assert int(changed.info["up_events"].episode_step) == 0
    assert int(state.info["episode_step"]) == 18  # Original reference arm remains unchanged.
    assert changed.info["up_events"].apex_seen is True


def test_runtime_cpu_is_never_reported_as_gpu_pass(tmp_path, monkeypatch):
    import jit_dvgc.jump_evidence_runtime as runtime
    monkeypatch.setattr(runtime, "load_plan", lambda _: {"horizon": 400})
    monkeypatch.setattr(runtime.jax, "default_backend", lambda: "cpu")
    assert runtime.run_worker(tmp_path / "plan.json", tmp_path, kind="capture") == 1
    report = audit.read(tmp_path / "report.json")
    assert report["status"] == "engineering_error"
    assert report["environment_interactions"] == 0


@pytest.mark.parametrize("ending,expected", [("landing", 1), ("failure", 0), ("timeout", 0), ("horizon", 0)])
def test_replay_rollout_stops_at_declared_endpoint_or_ceiling(monkeypatch, ending, expected):
    import jax.numpy as jp
    import jit_dvgc.jump_evidence_runtime as runtime
    def state(tick):
        return SimpleNamespace(tick=tick, obs={"state": jp.zeros(76)})
    def view(value):
        finished = value.tick == 2 and ending != "horizon"
        return {"done": finished and ending != "landing", "down/valid_contact_seen": finished and ending == "landing",
                "physical_failure": finished and ending == "failure", "timeout": finished and ending == "timeout",
                "end_code": 1 if finished else 0, "action": [0.0] * 4}
    monkeypatch.setattr(runtime, "view", view)
    calls = []
    def step(value, action):
        calls.append(value.tick)
        return state(value.tick + 1)
    trace, outcome = runtime.rollout(state(0), lambda obs, key: jp.zeros(4), step, seed=7, index=0, horizon=3)
    assert len(calls) == (3 if ending == "horizon" else 2)
    assert outcome["label"] == expected
    assert outcome["ticks"] == len(calls)
    assert len(trace) == len(calls) + 1
