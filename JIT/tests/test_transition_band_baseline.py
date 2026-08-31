from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jit_dvgc.transition_band_baseline import (
    build_transition_band_baseline,
    load_transition_band_baseline_config,
)


def _sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _label(*, state: str, label: int, phase: str, actor: str, payload: str, lp: str, ap: str):
    return {
        "candidate_id": state[:8],
        "state_sha256": state,
        "phase": phase,
        "label": label,
        "parent_group_id": f"group_{phase}",
        "split": "train",
        "policy_actor_sha256": actor,
        "policy_payload_sha256": payload,
        "label_protocol_sha256": lp,
        "acquisition_protocol_sha256": ap,
    }


def _write_source(root: Path, rows, *, actor: str, payload: str, lp: str, ap: str):
    root.mkdir(parents=True)
    (root / "labels.json").write_text(json.dumps(rows))
    positive = sum(int(row["label"]) for row in rows)
    summary = {
        "schema": "jit_unified_continuation_labels_v1",
        "status": "completed",
        "artifact_role": "pi_k_conditioned_expansion_train_labels",
        "split": "train",
        "candidate_count": len(rows),
        "label_count": len(rows),
        "positive_count": positive,
        "negative_count": len(rows) - positive,
        "protocol_sha256": lp,
        "candidate_catalog_protocol_sha256": ap,
        "policy_actor_sha256": actor,
        "policy_payload_sha256": payload,
        "environment_interactions": 10,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    (root / "summary.json").write_text(json.dumps(summary))


def test_real_iter0_baseline_declaration_is_bound_to_completed_manual_shell(jit_root):
    baseline = load_transition_band_baseline_config(
        jit_root / "configs/envelope_iter0_completed_baseline.json"
    )
    assert baseline["expected_protocol_sha256"] == (
        "dab211deb8c60a3f7b1db2a8fafc06bc7c3ad3e85f7a1c51ecfe2cc0fbdb937a"
    )
    assert baseline["expected_output"] == {
        "candidate_count": 896,
        "positive_count": 883,
        "negative_count": 13,
        "maximum_duration": 8,
    }
    assert [source["name"] for source in baseline["protocol"]["sources"]] == [
        "inner_shell",
        "outer_shell1",
    ]
    assert baseline["protocol"]["sources"][1]["expected_label_protocol_sha256"] == (
        "7974bbf6cefc9c1a8bed2a156e23d1ead6f9d9bb497f7a48535c7d60851afa58"
    )


def test_baseline_builder_deduplicates_same_physical_state_and_preserves_provenance(tmp_path):
    actor = "1" * 64
    payload = "2" * 64
    lp1, ap1 = "3" * 64, "4" * 64
    lp2, ap2 = "5" * 64, "6" * 64
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    source1 = tmp_path / "s1"
    source2 = tmp_path / "s2"
    _write_source(
        source1,
        [
            _label(state=a, label=1, phase="upstream", actor=actor, payload=payload, lp=lp1, ap=ap1),
            _label(state=b, label=0, phase="upstream", actor=actor, payload=payload, lp=lp1, ap=ap1),
        ],
        actor=actor,
        payload=payload,
        lp=lp1,
        ap=ap1,
    )
    _write_source(
        source2,
        [
            _label(state=b, label=0, phase="upstream", actor=actor, payload=payload, lp=lp2, ap=ap2),
            _label(state=c, label=1, phase="downstream", actor=actor, payload=payload, lp=lp2, ap=ap2),
        ],
        actor=actor,
        payload=payload,
        lp=lp2,
        ap=ap2,
    )
    protocol = {
        "schema": "jit_unified_transition_band_baseline_protocol_v1",
        "status": "predeclared",
        "iteration": 0,
        "policy_actor_sha256": actor,
        "policy_payload_sha256": payload,
        "sources": [
            {
                "name": "s1",
                "labels_path": str(source1 / "labels.json"),
                "expected_label_protocol_sha256": lp1,
                "expected_acquisition_protocol_sha256": ap1,
                "expected_candidate_count": 2,
                "expected_positive_count": 1,
                "expected_negative_count": 1,
                "maximum_duration": 2,
            },
            {
                "name": "s2",
                "labels_path": str(source2 / "labels.json"),
                "expected_label_protocol_sha256": lp2,
                "expected_acquisition_protocol_sha256": ap2,
                "expected_candidate_count": 2,
                "expected_positive_count": 1,
                "expected_negative_count": 1,
                "maximum_duration": 8,
            },
        ],
        "deduplicate_by": "state_sha256",
        "conflict_policy": "reject",
        "claim_boundary": {
            "training_only_baseline": True,
            "continuation_field_trained": False,
            "tube_1_constructed": False,
            "pi_1_trained": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    acquisition_sources = {
        "schema": "jit_unified_transition_band_baseline_acquisition_sources_v1",
        "source_acquisition_protocol_sha256": [ap1, ap2],
    }
    config = {
        "schema": "jit_unified_transition_band_baseline_build_config_v1",
        "output_dir": str(tmp_path / "out"),
        "expected_protocol_sha256": _sha(protocol),
        "expected_composite_acquisition_protocol_sha256": _sha(acquisition_sources),
        "expected_output": {
            "candidate_count": 3,
            "positive_count": 2,
            "negative_count": 1,
            "maximum_duration": 8,
        },
        "protocol": protocol,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    summary = build_transition_band_baseline(config_path)
    assert summary["candidate_count"] == 3
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 1
    assert summary["duplicate_state_count"] == 1
    labels = json.loads((tmp_path / "out" / "labels.json").read_text())
    assert len({row["state_sha256"] for row in labels}) == 3
    assert {row["baseline_source"] for row in labels} == {"s1", "s2"}
