from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import jax
import jax.numpy as jp
import numpy as np
import pytest

from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, config_hash, file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.reference import ReferenceTrajectory
from dvgc.two_phase_guideline import (
    GuidelinePerturbation,
    build_guideline_banks,
    select_guideline_indices,
)
from dvgc.two_phase_roundtrip import (
    RoundtripTolerances,
    compare_two_phase_payloads,
    compare_two_phase_roundtrip,
    select_roundtrip_representatives,
)
from dvgc.two_phase_runtime import TwoPhaseThresholds, build_two_phase_geometry
from dvgc.two_phase_semantics import ApexBandThresholds, RecoveryThresholds


def _provenance(cfg):
    digest = hashlib.sha256
    return {
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_sha256": config_hash(cfg),
        "action_mapping_version": ACTION_MAPPING_VERSION,
        "policy_params_sha256": digest(b"guideline_open_loop_no_policy").hexdigest(),
        "policy_config_sha256": config_hash(cfg),
        "policy_manifest_sha256": digest(b"guideline_controller_provenance").hexdigest(),
        "normalizer_sha256": digest(b"no_policy_normalizer").hexdigest(),
        "source_fingerprint": digest(b"gate_b_roundtrip_test").hexdigest(),
    }


def _thresholds():
    return TwoPhaseThresholds(
        apex=ApexBandThresholds(
            max_abs_com_vz=0.2,
            min_clearance=0.05,
            max_abs_roll=0.5,
            max_abs_pitch=0.5,
            max_angular_speed=5.0,
            min_forward_velocity=0.1,
            relative_x_min=-1.0,
            relative_x_max=1.0,
        ),
        recovery=RecoveryThresholds(
            max_abs_roll=0.5,
            max_abs_pitch=0.5,
            max_angular_speed=5.0,
            min_forward_velocity=0.1,
            required_hold_ticks=3,
        ),
    )


def _payload():
    return {
        "continuous": {
            "qpos": np.asarray([1.0, 2.0]),
            "qvel": np.asarray([3.0]),
            "ctrl": np.asarray([0.1]),
            "last_action": np.asarray([0.2]),
            "actor_observation": np.asarray([0.3]),
            "privileged_observation": np.asarray([0.4]),
            "obs_history": np.asarray([[0.5]]),
            "actor_packet_fifo": np.asarray([[0.6], [0.7], [0.8]]),
            "two_phase_signals": np.asarray([0.9]),
            "estimator_continuous": np.asarray([0.1, 0.2]),
        },
        "discrete": {
            "terminated": np.asarray(0),
            "truncated": np.asarray(0),
            "jump_signal_latched": np.asarray(True),
            "two_phase_event_latches": np.asarray([True, False]),
            "apex_signal_flags": np.asarray([True, False, False]),
            "recovery_signal_flags": np.asarray([True, True, True, False]),
        },
    }


@pytest.mark.parametrize(
    "field",
    [
        "qpos",
        "qvel",
        "ctrl",
        "last_action",
        "actor_observation",
        "privileged_observation",
        "obs_history",
        "actor_packet_fifo",
        "two_phase_signals",
        "estimator_continuous",
    ],
)
def test_payload_comparison_names_each_continuous_mismatch(field):
    left = _payload()
    right = _payload()
    right["continuous"][field] = right["continuous"][field] + 1.0

    result = compare_two_phase_payloads(left, right, RoundtripTolerances())

    assert result["valid"] is False
    assert field in result["failed_fields"]


@pytest.mark.parametrize(
    "field",
    [
        "terminated",
        "truncated",
        "jump_signal_latched",
        "two_phase_event_latches",
        "apex_signal_flags",
        "recovery_signal_flags",
    ],
)
def test_payload_comparison_requires_exact_discrete_state(field):
    left = _payload()
    right = _payload()
    right["discrete"][field] = np.logical_not(right["discrete"][field])

    result = compare_two_phase_payloads(left, right, RoundtripTolerances())

    assert result["valid"] is False
    assert field in result["failed_fields"]


def test_representative_selection_requires_apex_thickness_and_boundaries():
    up = SnapshotBank(
        [
            {"qpos": [0], "qvel": [0], "ctrl": [0], "physical_feature": [0], "source_phase": "takeoff", "guideline_selection": name, "two_phase_context": {"event_position": "event"}}
            for name in ("front", "middle", "back")
        ]
    )
    down = SnapshotBank(
        [
            {"qpos": [0], "qvel": [0], "ctrl": [0], "physical_feature": [0], "source_phase": "flight", "guideline_selection": name, "two_phase_context": {"event_position": name if name in ("pre", "nearest", "post") else "event"}}
            for name in ("pre", "nearest", "post", "early_descent")
        ]
    )

    selected = select_roundtrip_representatives(up, down)

    assert set(selected) == {"up_boundary_front", "up_boundary_back", "down_pre", "down_nearest", "down_post", "down_boundary"}


def test_real_roundtrip_uses_only_explicit_restore_and_compares_full_state(monkeypatch):
    import dvgc.rollout

    monkeypatch.setattr(
        dvgc.rollout,
        "restore_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("compatibility fallback called")),
    )
    cfg = load_config(
        "configs/default.json",
        {"domain_randomization": False, "obs_noise_enable": False, "use_bank_resets": False},
    )
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    reference = ReferenceTrajectory.load("data/reference_jump.csv")
    selection = select_guideline_indices(reference, reference.anchors())
    built = build_guideline_banks(
        env,
        reference,
        selection,
        provenance=_provenance(cfg),
        seed=5100,
        perturbations=(GuidelinePerturbation("nominal", 0.0, 0.0),),
    )
    up = built.phase_up
    record = next(row for row in up.records if row["guideline_selection"] == "middle")
    actions = [
        jp.asarray([0.0, 1.0, -0.1, 0.1], jp.float32),
        jp.asarray([0.1, 0.8, -0.2, 0.2], jp.float32),
        jp.asarray([-0.1, 0.6, -0.1, 0.0], jp.float32),
    ]

    report = compare_two_phase_roundtrip(
        env,
        record,
        built.original_states[record["id"]],
        build_two_phase_geometry(env.mj_model, cfg),
        _thresholds(),
        seed=5200,
        actions=actions,
    )

    assert report["status"] == "pass", report["failed_fields"]
    assert report["snapshot_id"] == record["id"]
    assert report["parent_trajectory_id"] == record["two_phase_context"]["parent_trajectory_id"]
    assert report["control_ticks"] == 3
    assert len(report["ticks"]) == 4
    assert all(row["valid"] for row in report["ticks"])
    assert report["restore_mode"] == "timing_explicit_independent_reconstruction"
    assert report["comparison"] == "capture_time_original_vs_independent_restore"

    corrupted = copy.deepcopy(record)
    corrupted["physical_state_t"]["qpos"] = (
        np.asarray(corrupted["physical_state_t"]["qpos"]) + 0.01
    )
    failed = compare_two_phase_roundtrip(
        env,
        corrupted,
        built.original_states[record["id"]],
        build_two_phase_geometry(env.mj_model, cfg),
        _thresholds(),
        seed=5200,
        actions=actions[:1],
    )
    assert failed["status"] == "gate_pause"
    assert "qpos" in failed["failed_fields"]
