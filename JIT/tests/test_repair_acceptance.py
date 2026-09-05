from __future__ import annotations

import json
from pathlib import Path

import pytest

from jit_dvgc.repair_acceptance import (
    _gate_declaration,
    consumed_gate_exclusions,
)


ACTOR = "a" * 64
PAYLOAD = "b" * 64


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _gate(root: Path, *, schema: str, suffix: str, phase: str) -> dict:
    state = (suffix * 64)[:64]
    bank = {
        "schema": schema,
        "status": "locked_before_policy_rollout",
        "source_iteration": 2,
        "candidate_iteration": 3,
        "core_count": 1,
        "boundary_count": 1,
        "core": [],
        "boundary": [
            {
                "phase": phase,
                "state_sha256": state,
                "parent_group_id": f"{phase}_{suffix}",
            }
        ],
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "bank_sha256": f"bank-{suffix}",
    }
    summary = {
        "status": "completed",
        "source_iteration": 2,
        "candidate_iteration": 3,
        "baseline_actor_sha256": ACTOR,
        "baseline_payload_sha256": PAYLOAD,
        "protocol_sha256": f"protocol-{suffix}",
        "bank_sha256": bank["bank_sha256"],
    }
    _write(root / "summary.json", summary)
    _write(root / "bank.json", bank)
    _write(root / "records.json", {"records": []})
    declaration, _, _ = _gate_declaration(root)
    return declaration


def test_consumed_gate_exclusions_union_v1_v2(tmp_path: Path) -> None:
    first = _gate(
        tmp_path / "gate1",
        schema="jit_paired_policy_gate_bank_v1",
        suffix="c",
        phase="upstream",
    )
    second = _gate(
        tmp_path / "gate2",
        schema="jit_paired_policy_gate_bank_v2",
        suffix="d",
        phase="downstream",
    )
    protocol = {
        "source_iteration": 2,
        "candidate_iteration": 3,
        "baseline_actor_sha256": ACTOR,
        "baseline_payload_sha256": PAYLOAD,
        "consumed_gates": [first, second],
    }

    states, groups, audit = consumed_gate_exclusions(protocol)

    assert len(states) == 2
    assert groups == {
        "upstream": {"upstream_c"},
        "downstream": {"downstream_d"},
    }
    assert audit["gate_count"] == 2
    assert audit["union_boundary_state_count"] == 2


def test_consumed_gate_exclusions_reject_duplicate_roots(tmp_path: Path) -> None:
    declaration = _gate(
        tmp_path / "gate",
        schema="jit_paired_policy_gate_bank_v2",
        suffix="e",
        phase="upstream",
    )
    protocol = {
        "source_iteration": 2,
        "candidate_iteration": 3,
        "baseline_actor_sha256": ACTOR,
        "baseline_payload_sha256": PAYLOAD,
        "consumed_gates": [declaration, declaration],
    }
    with pytest.raises(ValueError, match="unique"):
        consumed_gate_exclusions(protocol)
