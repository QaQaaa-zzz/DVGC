from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


CLI = Path(__file__).parents[1] / "cli" / "select_iteration_policy.py"
spec = importlib.util.spec_from_file_location("select_iteration_policy", CLI)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _record() -> dict:
    return {
        "iteration": 2,
        "name": "pi_2",
        "actor_sha256": "a" * 64,
        "critic_sha256": "c" * 64,
        "normalizer_sha256": "n" * 64,
        "payload_sha256": "b" * 64,
        "xml_sha256": "x" * 64,
        "formal_config": "config.json",
        "formal_config_sha256": "f" * 64,
        "source_training_transitions": 10_009_600,
    }


def _gate() -> dict:
    return {
        "schema": "jit_paired_policy_gate_report_v1",
        "status": "completed",
        "source_iteration": 1,
        "candidate_iteration": 2,
        "candidate_policy_name": "pi_2",
        "candidate_actor_sha256": "a" * 64,
        "candidate_payload_sha256": "b" * 64,
        "iteration_accepted": False,
        "core_source": {"baseline_success_criterion": "first_valid_landing", "candidate_success_criterion": "first_valid_landing"},
        "boundary_source": {"success_criterion": "first_valid_landing"},
        "core_gate": {
            "state_count": 100,
            "baseline_success_count": 98,
            "candidate_success_count": 96,
            "regression_count": 3,
            "improvement_count": 1,
            "passed": False,
        },
        "boundary_gate": {
            "state_count": 10,
            "candidate_success_count": 4,
            "candidate_success_parent_group_count": 2,
            "minimum_candidate_success_parent_groups": 2,
            "baseline_reproduction_failure_count": 0,
        },
    }


def _decision(*, gate_sha: str, retrospective: bool) -> dict:
    payload = {
        "schema": "jit_capability_progression_decision_v1",
        "status": "completed",
        "decision": "envelope_progressed_and_candidate_authority_eligible",
        "source_iteration": 1,
        "candidate_iteration": 2,
        "candidate_policy_name": "pi_2",
        "candidate_actor_sha256": "a" * 64,
        "candidate_payload_sha256": "b" * 64,
        "source_gate_file_sha256": gate_sha,
        "retrospective_analysis": retrospective,
        "formal_prospective_selection_claim": not retrospective,
        "empirical_envelope_expansion_observed": True,
        "candidate_policy_authority_eligible": True,
        "policy_realization": {"passed": True},
        "frontier_progression": {"passed": True},
    }
    payload["decision_sha256"] = module._canonical(payload)
    return payload


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, retrospective: bool):
    frozen = tmp_path / "frozen.json"
    gate_path = tmp_path / "gate.json"
    decision_path = tmp_path / "decision.json"
    frozen.write_text("{}\n", encoding="utf-8")
    gate_path.write_text(json.dumps(_gate()), encoding="utf-8")

    def fake_sha(path: Path) -> str:
        name = Path(path).name
        if name == "gate.json":
            return "g" * 64
        if name == "decision.json":
            return "d" * 64
        if name == "frozen.json":
            return "z" * 64
        return "q" * 64

    decision_path.write_text(
        json.dumps(_decision(gate_sha="g" * 64, retrospective=retrospective)),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "file_sha256", fake_sha)
    monkeypatch.setattr(
        module,
        "load_frozen_unified_manifest",
        lambda _path: {"policy": _record()},
    )
    return frozen, gate_path, decision_path


def test_prospective_capability_selection_can_accept_nonzero_strict_regressions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, gate_path, decision_path = _prepare(
        tmp_path, monkeypatch, retrospective=False
    )
    output = tmp_path / "selected"
    result = module.select(
        frozen_policy=frozen,
        gate_summary=gate_path,
        output_dir=output,
        allow_baseline_reproduction_mismatch=False,
        capability_decision=decision_path,
    )
    assert result["selection_semantics"] == "prospective_capability_progression_v1"
    assert result["core_regression_count"] == 3
    assert result["strict_zero_regression_diagnostic_passed"] is False
    assert result["strict_historical_gate_accepted"] is False
    assert result["prospective_capability_selection_claim"] is True
    assert result["formal_acceptance_claim"] is True
    assert result["claim_boundary"]["historical_strict_gate_pass_claim"] is False
    assert (
        result["claim_boundary"]["prospective_capability_progression_selection_claim"]
        is True
    )


def test_retrospective_capability_decision_cannot_select_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, gate_path, decision_path = _prepare(
        tmp_path, monkeypatch, retrospective=True
    )
    with pytest.raises(ValueError, match="retrospective capability analysis"):
        module.select(
            frozen_policy=frozen,
            gate_summary=gate_path,
            output_dir=tmp_path / "selected",
            allow_baseline_reproduction_mismatch=False,
            capability_decision=decision_path,
        )


def test_historical_selection_path_still_requires_zero_regressions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, gate_path, _decision_path = _prepare(
        tmp_path, monkeypatch, retrospective=False
    )
    with pytest.raises(ValueError, match="historical strict selection"):
        module.select(
            frozen_policy=frozen,
            gate_summary=gate_path,
            output_dir=tmp_path / "selected",
            allow_baseline_reproduction_mismatch=False,
            capability_decision=None,
        )
