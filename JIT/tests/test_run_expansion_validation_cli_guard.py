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


def test_consumed_identity_audit_runs_before_any_fresh_label(monkeypatch, tmp_path: Path) -> None:
    cli = _load_cli_module()
    events: list[str] = []

    original_anchor_audit = cli.fresh_shared._audit_fresh_anchor_independence

    def original_label_candidates(*args, **kwargs):
        events.append("label")
        return {"status": "labels_completed"}

    def fake_execute(config, *, resume):
        events.append("execute")
        assert (
            cli.fresh_shared._audit_fresh_anchor_independence
            is cli._audit_fresh_anchor_independence_declared
        )
        cli.fresh_shared._label_candidates(output=tmp_path)
        events.append("execute_return")
        return {"status": "completed"}

    def fake_audit(config, output):
        events.append("identity_audit")
        assert Path(output) == tmp_path
        return {
            "status": "independent",
            "exact_state_overlap_count": 0,
            "near_duplicate_observation_overlap_count": 0,
            "consumed_validation_outcomes_read": False,
            "consumed_validation_predictions_read": False,
        }

    monkeypatch.setattr(cli.fresh_shared, "_label_candidates", original_label_candidates)
    monkeypatch.setattr(cli.fresh_shared, "execute_fresh_shared_validation", fake_execute)
    monkeypatch.setattr(cli, "audit_fresh_candidate_vs_consumed_validation", fake_audit)

    report, audit = cli._execute_fresh_with_identity_guard(
        tmp_path / "config.json", resume=False
    )

    assert report == {"status": "completed"}
    assert audit["status"] == "independent"
    assert events == ["execute", "identity_audit", "label", "execute_return"]
    assert cli.fresh_shared._label_candidates is original_label_candidates
    assert cli.fresh_shared._audit_fresh_anchor_independence is original_anchor_audit
