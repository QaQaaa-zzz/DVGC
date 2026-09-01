from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_cli_module():
    path = Path("JIT/cli/run_expansion_validation.py")
    spec = importlib.util.spec_from_file_location("jit_run_expansion_validation_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_machine_json_writer_is_not_stdout_dependent(tmp_path: Path) -> None:
    cli = _load_cli_module()
    target = tmp_path / "audit.json"
    payload = {"status": "fresh_validation_preflight_ready", "attempt_count": 304}
    cli._write_machine_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_declared_anchor_guard_allows_near_observations_but_not_duplicate_states(
    monkeypatch,
) -> None:
    cli = _load_cli_module()

    monkeypatch.setattr(
        cli.fresh_shared,
        "load_frozen_upstream_checkpoint_train_evidence",
        lambda path: ({}, []),
    )
    monkeypatch.setattr(
        cli.fresh_shared,
        "load_frozen_iteration_train_evidence",
        lambda path: ({}, []),
    )
    monkeypatch.setattr(cli.fresh_shared, "_rows_from_catalog", lambda path: tuple())

    protocol = {
        "upstream_train_evidence": "unused-up",
        "downstream_train_evidence": "unused-down",
        "consumed_validation_identity_catalog": "unused-old",
        "near_duplicate_audit": {"actor_observation_atol": 0.01},
    }
    obs = np.zeros(76, dtype=np.float32).tolist()
    entries = [
        {
            "phase": "upstream",
            "parent_group_id": "transition_4988928__1000007",
            "state_sha256": "1" * 64,
            "actor_observation": obs,
        },
        {
            "phase": "upstream",
            "parent_group_id": "transition_7987200__1000007",
            "state_sha256": "2" * 64,
            "actor_observation": obs,
        },
    ]

    report = cli._audit_fresh_anchor_independence_declared(
        protocol, source_report={"entries": entries}
    )
    stats = report["phase_stats"]["upstream"]
    assert report["status"] == "independent"
    assert stats["fresh_anchor_exact_duplicate_count"] == 0
    assert stats["fresh_anchor_near_duplicate_observation_pair_count"] == 1
    assert stats["fresh_anchor_near_duplicate_observation_is_exclusion_rule"] is False

    duplicate = [dict(entries[0]), dict(entries[1])]
    duplicate[1]["state_sha256"] = duplicate[0]["state_sha256"]
    with pytest.raises(ValueError, match="repeats source anchor state"):
        cli._audit_fresh_anchor_independence_declared(
            protocol, source_report={"entries": duplicate}
        )


def test_consumed_identity_overlaps_are_excluded_without_replacement(tmp_path: Path) -> None:
    cli = _load_cli_module()
    catalog = {
        "candidate_count": 3,
        "exclusion_counts": {"train_near_duplicate_observation": 2},
        "entries": [
            {"candidate_id": "a", "phase": "upstream"},
            {"candidate_id": "b", "phase": "upstream"},
            {"candidate_id": "c", "phase": "downstream"},
        ],
    }
    audit = {
        "status": "exclusions_required",
        "exact_overlap_candidate_ids": ["a"],
        "near_overlap_candidate_ids": ["b"],
        "exact_state_overlap_count": 1,
        "near_duplicate_observation_overlap_count": 1,
        "consumed_validation_outcomes_read": False,
        "consumed_validation_predictions_read": False,
    }

    closed = cli._apply_consumed_identity_exclusions(
        catalog, audit, output=tmp_path
    )

    assert catalog["prefilter_candidate_count"] == 3
    assert catalog["candidate_count"] == 1
    assert [row["candidate_id"] for row in catalog["entries"]] == ["c"]
    assert catalog["exclusion_counts"]["train_near_duplicate_observation"] == 2
    assert catalog["exclusion_counts"]["consumed_validation_exact_state"] == 1
    assert (
        catalog["exclusion_counts"]["consumed_validation_near_duplicate_observation"]
        == 1
    )
    assert catalog["no_replacement_after_consumed_validation_exclusion"] is True
    assert closed["status"] == "independent_after_exclusion"
    assert closed["excluded_exact_state_count"] == 1
    assert closed["excluded_near_duplicate_observation_count"] == 1
    assert closed["exact_state_overlap_count"] == 0
    assert closed["near_duplicate_observation_overlap_count"] == 0
    assert (tmp_path / "candidate_catalog_independent.json").exists()


def test_consumed_identity_audit_runs_and_filters_before_any_fresh_label(
    monkeypatch, tmp_path: Path
) -> None:
    cli = _load_cli_module()
    events: list[str] = []

    original_anchor_audit = cli.fresh_shared._audit_fresh_anchor_independence
    original_runtime_protocol = cli.fresh_shared._runtime_protocol

    def original_label_candidates(*args, **kwargs):
        events.append("label")
        assert kwargs["catalog"]["candidate_count"] == 1
        assert kwargs["catalog"]["entries"][0]["candidate_id"] == "keep"
        return {"status": "labels_completed"}

    def fake_execute(config, *, resume):
        events.append("execute")
        assert (
            cli.fresh_shared._audit_fresh_anchor_independence
            is cli._audit_fresh_anchor_independence_declared
        )
        catalog = {
            "candidate_count": 2,
            "exclusion_counts": {},
            "entries": [
                {"candidate_id": "drop"},
                {"candidate_id": "keep"},
            ],
        }
        cli.fresh_shared._label_candidates(output=tmp_path, catalog=catalog)
        events.append("execute_return")
        return {"status": "completed", "candidate_count": catalog["candidate_count"]}

    def fake_audit(config, output):
        events.append("identity_audit")
        assert Path(output) == tmp_path
        return {
            "status": "exclusions_required",
            "exact_overlap_candidate_ids": [],
            "near_overlap_candidate_ids": ["drop"],
            "exact_state_overlap_count": 0,
            "near_duplicate_observation_overlap_count": 1,
            "consumed_validation_outcomes_read": False,
            "consumed_validation_predictions_read": False,
        }

    monkeypatch.setattr(cli.fresh_shared, "_label_candidates", original_label_candidates)
    monkeypatch.setattr(cli.fresh_shared, "execute_fresh_shared_validation", fake_execute)
    monkeypatch.setattr(cli, "audit_fresh_candidate_vs_consumed_validation", fake_audit)

    report, audit = cli._execute_fresh_with_identity_guard(
        tmp_path / "config.json", resume=False
    )

    assert report == {"status": "completed", "candidate_count": 1}
    assert audit["status"] == "independent_after_exclusion"
    assert audit["excluded_near_duplicate_observation_count"] == 1
    assert events == ["execute", "identity_audit", "label", "execute_return"]
    assert cli.fresh_shared._label_candidates is original_label_candidates
    assert cli.fresh_shared._audit_fresh_anchor_independence is original_anchor_audit
    assert cli.fresh_shared._runtime_protocol is original_runtime_protocol


def _runtime_record(cli, *, head: str) -> dict:
    base = {
        "schema": "runtime-test",
        "repository_head": head,
        "scientific_protocol_sha256": "sci",
        "attempt_schedule_sha256": "schedule",
        "policy_actor_sha256": "actor",
        "policy_payload_sha256": "payload",
    }
    return {**base, "protocol_sha256": cli.fresh_shared.canonical_sha256(base)}


def test_prelabel_engineering_resume_reuses_existing_runtime_protocol(tmp_path: Path) -> None:
    cli = _load_cli_module()
    existing = _runtime_record(cli, head="old-head")
    proposed = _runtime_record(cli, head="repair-head")
    (tmp_path / "runtime_protocol.json").write_text(
        json.dumps(existing), encoding="utf-8"
    )

    def original(config_path, config, scientific):
        return proposed

    result = cli._runtime_protocol_with_engineering_resume(
        original,
        tmp_path / "config.json",
        {"output_dir": str(tmp_path)},
        {},
        resume=True,
    )

    assert result == existing
    repair = json.loads(
        (tmp_path / "engineering_resume_repair.json").read_text(encoding="utf-8")
    )
    assert repair["status"] == "authorized_prelabel_guard_repair"
    assert repair["original_repository_head"] == "old-head"
    assert repair["repair_repository_head"] == "repair-head"
    assert repair["scientific_protocol_unchanged"] is True
    assert repair["validation_labels_existed_before_repair"] is False


def test_engineering_resume_refuses_cross_head_after_label_progress(tmp_path: Path) -> None:
    cli = _load_cli_module()
    existing = _runtime_record(cli, head="old-head")
    proposed = _runtime_record(cli, head="repair-head")
    (tmp_path / "runtime_protocol.json").write_text(
        json.dumps(existing), encoding="utf-8"
    )
    (tmp_path / "label_progress.json").write_text("{}", encoding="utf-8")

    def original(config_path, config, scientific):
        return proposed

    with pytest.raises(ValueError, match="forbidden after validation labeling"):
        cli._runtime_protocol_with_engineering_resume(
            original,
            tmp_path / "config.json",
            {"output_dir": str(tmp_path)},
            {},
            resume=True,
        )
