from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from jit_dvgc import probe_bank as probes
from jit_dvgc.evidence_integrity import canonical_sha256


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def bank_fixture(tmp_path, monkeypatch):
    import jit_dvgc.unified_policy_freeze as frozen
    members = []
    for index in range(4):
        name = f"probe_{index}"
        record = {"name": name, "iteration": index, "actor_sha256": f"{index+1:064x}",
                  "payload_sha256": f"{index+10:064x}", "formal_config_sha256": "c" * 64,
                  "xml_sha256": "a" * 64}
        path = tmp_path / f"{name}.json"
        write(path, {"policy": record})
        members.append({"frozen_policy": str(path), "roles": ["proposer", "evaluator"]})
    monkeypatch.setattr(frozen, "load_frozen_unified_manifest", lambda path: json.loads(path.read_text()))
    spec = {"version": "pilot-1", "task": {"xml_sha256": "a" * 64,
            "start_contract_sha256": "b" * 64, "centerline_sha256": "c" * 64,
            "resolution_sha256": "d" * 64, "success_criterion": "first_valid_landing",
            "continuation_start_semantics": "fresh_continuation_v1"},
            "max_ticks": 400, "label_interaction_budget": 10000,
            "max_candidates_per_process": 1, "members": members}
    bank_path = tmp_path / "bank.json"
    bank = probes.lock_probe_bank(spec, bank_path)
    return bank_path, bank, spec


def catalog_fixture(tmp_path, bank, index=0):
    proposer = bank["members"][index]["policy"]
    protocol = {"policy_name": proposer["name"], "policy_actor_sha256": proposer["actor_sha256"],
                "policy_payload_sha256": proposer["payload_sha256"],
                "evidence_mode": "probe_bank_arrivals_v1", "probe_bank_sha256": bank["bank_sha256"],
                "logical_role": "train", "start_contract_sha256": bank["task"]["start_contract_sha256"],
                "nominal_centerline_sha256": bank["task"]["centerline_sha256"]}
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    entries = [{"candidate_id": f"c{i}", "state_sha256": str(i), "phase": "upstream", "phase_index": 0,
                "snapshot_context_sha256": str(i), "parent_group_id": f"g{i}", "parent_state_sha256": "parent",
                "source_bank": "bank", "snapshot": f"s{i}"} for i in range(2)]
    path = tmp_path / f"catalog{index}" / "catalog.json"
    write(path.parent / "protocol.json", protocol)
    write(path, {"status": "completed", "candidate_count": 2, "protocol_sha256": protocol["protocol_sha256"], "entries": entries})
    return path


def test_bank_accepts_four_complementary_probes_without_actor_retention(bank_fixture):
    path, bank, _ = bank_fixture
    assert len(probes.load_probe_bank(path)["members"]) == 4
    assert bank["whole_tube_actor_retention_required"] is False
    assert bank["formal_envelope_claim_authorized"] is False


def test_bank_rejects_duplicate_names_and_changed_frozen_file(bank_fixture, tmp_path):
    path, bank, spec = bank_fixture
    spec["members"].append(spec["members"][0])
    with pytest.raises(ValueError, match="duplicate"):
        probes.lock_probe_bank(spec, tmp_path / "duplicate.json")
    frozen = Path(bank["members"][0]["frozen_policy"])
    frozen.write_text(frozen.read_text() + " ")
    with pytest.raises(ValueError, match="file identity"):
        probes.load_probe_bank(path)


def test_two_proposers_four_evaluators_get_distinct_bounded_jobs(bank_fixture, tmp_path, monkeypatch):
    path, bank, _ = bank_fixture
    monkeypatch.setattr(probes, "validate_probe_arrivals", lambda *args: None)
    catalogs = [catalog_fixture(tmp_path, bank, index) for index in (0, 1)]
    plan = probes.prepare_probe_labels(path, catalogs, tmp_path / "plan.json", role="train", seed=7)
    assert len(plan["jobs"]) == 16
    assert len({j["job_id"] for j in plan["jobs"]}) == 16
    assert plan["maximum_label_interactions"] == 6400
    assert all(j["maximum_interactions"] == 400 for j in plan["jobs"])
    with pytest.raises(ValueError, match="duplicate proposer catalog"):
        probes.prepare_probe_labels(path, [catalogs[0]] * 2, tmp_path / "bad.json", role="train", seed=7)
    with pytest.raises(ValueError, match="role/start"):
        probes.prepare_probe_labels(path, catalogs, tmp_path / "bad-role.json", role="acceptance", seed=7)


def test_legacy_catalog_cannot_be_relabelled_as_new_bank(bank_fixture, tmp_path):
    path, bank, _ = bank_fixture
    catalog = catalog_fixture(tmp_path, bank)
    protocol_path = catalog.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.pop("probe_bank_sha256")
    protocol.pop("protocol_sha256")
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    write(protocol_path, protocol)
    content = json.loads(catalog.read_text())
    content["protocol_sha256"] = protocol["protocol_sha256"]
    write(catalog, content)
    with pytest.raises(ValueError, match="legacy/different-bank"):
        probes.prepare_probe_labels(path, [catalog], tmp_path / "bad.json", role="train", seed=7)


def test_process_supervisor_merges_real_shards_and_attributes_witnesses(bank_fixture, tmp_path, monkeypatch):
    from jit_dvgc import policy_family_landing as landing
    from jit_dvgc.unified_continuation_shards import build_logical_label_protocol, POLICY_KEY_SCHEME
    path, bank, _ = bank_fixture
    monkeypatch.setattr(probes, "validate_probe_arrivals", lambda *args: None)
    monkeypatch.setattr(landing, "_load_frozen", lambda p: (json.loads(Path(p).read_text())["policy"], probes._file_sha(p)))
    catalog = catalog_fixture(tmp_path, bank)
    plan_path = tmp_path / "plan.json"
    probes.prepare_probe_labels(path, [catalog], plan_path, role="train", seed=7)
    calls = []
    def fake_gpu_process(command, **kwargs):
        calls.append(command)
        def arg(flag): return command[command.index(flag) + 1]
        output = Path(arg("--output-dir"))
        source = json.loads(catalog.read_text())
        evaluator = json.loads(Path(arg("--evaluator-frozen-policy")).read_text())["policy"]
        proposer = bank["members"][0]["policy"]
        index = int(arg("--shard-index"))
        protocol = build_logical_label_protocol(catalog_path=catalog, catalog=source,
            policy_record=evaluator, frozen_manifest_sha256=probes._file_sha(arg("--evaluator-frozen-policy")),
            max_ticks=400, protocol_seed=7, acquisition_policy_record=proposer,
            acquisition_frozen_manifest_sha256=bank["members"][0]["frozen_file_sha256"],
            success_criterion="first_valid_landing")
        protocol["protocol_sha256"] = canonical_sha256(protocol)
        success = int(evaluator["name"] == f"probe_{index}")
        row = {**source["entries"][index], "candidate_index": index, "policy_key_candidate_index": index,
               "policy_key_scheme": POLICY_KEY_SCHEME, "label": success, "continuation_success": bool(success),
               "outcome_class": "first_valid_landing" if success else "airborne_physical_failure",
               "environment_interactions": 5, "evaluator_policy_name": evaluator["name"],
               "evaluator_actor_sha256": evaluator["actor_sha256"], "evaluator_payload_sha256": evaluator["payload_sha256"],
               "success_criterion": "first_valid_landing", "label_protocol_sha256": protocol["protocol_sha256"],
               "acquisition_protocol_sha256": source["protocol_sha256"]}
        write(output / "protocol.json", protocol)
        write(output / "labels.json", [row])
        write(output / "execution.json", {"status": "completed", "logical_protocol_sha256": protocol["protocol_sha256"],
            "shard_index": index, "shard_count": 2, "candidate_start_index": index, "candidate_stop_index_exclusive": index + 1})
        write(output / "summary.json", {"status": "completed_shard", "logical_protocol_sha256": protocol["protocol_sha256"],
            "candidate_count": 1, "environment_interactions": 5})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(probes.subprocess, "run", fake_gpu_process)
    output = tmp_path / "run"
    report = probes.run_probe_labels(plan_path, output)
    assert len(calls) == 8
    assert report["observed_exact_context_count"] == 2
    assert report["charged_label_interactions_including_retries"] == 40
    assert report["evaluator_unique_context_contributions"]["probe_0"] == 1
    assert report["evaluator_unique_context_contributions"]["probe_1"] == 1
    assert probes.run_probe_labels(plan_path, output) == report
    assert len(calls) == 8  # Completed attempts validate and reuse without another rollout.


def test_unknown_failed_attempts_consume_budget(bank_fixture, tmp_path, monkeypatch):
    path, bank, _ = bank_fixture
    monkeypatch.setattr(probes, "validate_probe_arrivals", lambda *args: None)
    catalog = catalog_fixture(tmp_path, bank)
    plan_path = tmp_path / "plan.json"
    plan = probes.prepare_probe_labels(path, [catalog], plan_path, role="train", seed=7)
    output = tmp_path / "run"
    write(output / "plan_identity.json", {"plan_sha256": plan["plan_sha256"]})
    job = plan["jobs"][0]
    for index in range(25):
        write(output / job["job_id"] / f"attempt_{index:04d}" / "reservation.json",
              {"plan_sha256": plan["plan_sha256"], "job": job})
    def forbidden(*args, **kwargs): raise AssertionError("budget-exhausted run launched a process")
    monkeypatch.setattr(probes.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        probes.run_probe_labels(plan_path, output)


def test_context_hash_separates_fifo_clock_and_event_state():
    from jit_dvgc.unified_envelope_snapshot import UnifiedEnvelopeSnapshot, snapshot_context_sha256
    from dataclasses import fields
    # Representation-level identity test; does not construct a simulator or claim replay equivalence.
    snapshot = SimpleNamespace(**{field.name: 0 for field in fields(UnifiedEnvelopeSnapshot)})
    snapshot.qpos = np.zeros(7)
    snapshot.qvel = np.zeros(6)
    snapshot.observation_fifo = np.zeros((3, 4))
    first = snapshot_context_sha256(snapshot)
    snapshot.observation_fifo[0, 0] = 1
    assert snapshot_context_sha256(snapshot) != first
    second = snapshot_context_sha256(snapshot)
    snapshot.episode_step = 7
    assert snapshot_context_sha256(snapshot) != second
    third = snapshot_context_sha256(snapshot)
    snapshot.down_events = {"valid_contact_seen": np.array(True)}
    assert snapshot_context_sha256(snapshot) != third
