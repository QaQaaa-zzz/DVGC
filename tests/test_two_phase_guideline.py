from __future__ import annotations

from dataclasses import asdict

import mujoco
import numpy as np
import pandas as pd
import pytest

from dvgc.config import load_config
from dvgc.reference import ReferenceAnchors, ReferenceTrajectory
from dvgc.two_phase_guideline import (
    GuidelineMargins,
    audit_geometry_clearance,
    build_threshold_manifest,
    canonical_manifest_hash,
    select_guideline_indices,
    validate_guideline_event_order,
)
from dvgc.two_phase_runtime import build_two_phase_geometry


XML = "assets/orange_bike_4kg_horizontal.xml"
SOURCE_HASHES = {"xml": "a" * 64, "reference": "b" * 64, "config": "c" * 64, "code": "d" * 64, "geometry_manifest": "e" * 64}
SOURCE_PATHS = {"xml": XML, "reference": "data/reference_jump.csv", "config": "configs/default.json", "code": "dvgc/two_phase_guideline.py"}


def _reference() -> tuple[ReferenceTrajectory, ReferenceAnchors]:
    count = 24
    frame = pd.DataFrame(
        {
            "time": np.arange(count) * 0.01,
            "pos_x": np.linspace(2.0, 6.0, count),
            "pos_z": [0.25] * 6 + list(np.linspace(0.3, 0.8, 7)) + list(np.linspace(0.79, 0.41, 7)) + [0.41] * 4,
            "vel_x": np.linspace(1.0, 2.0, count),
            "vel_z": [-0.1] * 4 + [0.3, 0.6, 1.0, 0.8, 0.5, 0.2, 0.08, 0.01, -0.02, -0.15, -0.3, -0.4, -0.3, -0.2, -0.1, -0.02, 0.0, 0.0, 0.0, 0.0],
            "roll_angle": np.linspace(-2.0, 2.0, count),
            "pitch_angle": np.linspace(-3.0, 3.0, count),
            "yaw_angle": np.zeros(count),
            "hip_position": np.zeros(count),
            "knee_position": np.zeros(count),
            "hip_velocity": np.zeros(count),
            "knee_velocity": np.zeros(count),
            "action_hip": np.zeros(count),
            "action_knee": np.zeros(count),
            "action_steering": np.zeros(count),
            "action_rearwheel": np.ones(count),
        }
    )
    anchors = ReferenceAnchors(
        approach_end=4,
        takeoff_end=8,
        apex=11,
        landing_start=18,
        recovery_start=20,
        recovery_end=23,
    )
    return ReferenceTrajectory(frame), anchors


def _margins() -> GuidelineMargins:
    return GuidelineMargins(
        apex_abs_vz=0.02,
        apex_clearance=0.01,
        apex_abs_roll=0.03,
        apex_abs_pitch=0.04,
        apex_angular_speed=0.05,
        apex_forward_velocity=0.10,
        apex_relative_x=0.20,
        recovery_abs_roll=0.02,
        recovery_abs_pitch=0.03,
        recovery_angular_speed=0.04,
        recovery_forward_velocity=0.10,
    )


def _apex_samples():
    return [
        {
            "com_vz": 0.04,
            "clearance": 0.18,
            "roll": -0.10,
            "pitch": 0.12,
            "angular_speed": 0.20,
            "forward_velocity": 1.20,
            "obstacle_relative_x": 0.15,
        },
        {
            "com_vz": -0.06,
            "clearance": 0.16,
            "roll": 0.08,
            "pitch": -0.14,
            "angular_speed": 0.25,
            "forward_velocity": 1.00,
            "obstacle_relative_x": -0.10,
        },
    ]


def _recovery_samples():
    return [
        {"roll": -0.08, "pitch": 0.10, "angular_speed": 0.30, "forward_velocity": 0.90},
        {"roll": 0.12, "pitch": -0.07, "angular_speed": 0.20, "forward_velocity": 1.10},
    ]


def test_fixed_guideline_selection_uses_launch_apex_descent_and_recovery_slices():
    reference, anchors = _reference()

    selection = select_guideline_indices(reference, anchors)

    assert selection.launch == {"front": 4, "middle": 6, "back": 8}
    assert selection.apex == {"pre": 10, "nearest": 11, "post": 12}
    assert selection.early_descent == 13
    assert selection.recovery == (20, 21, 23)


def test_apex_selection_does_not_drift_to_a_later_low_vertical_speed():
    reference, anchors = _reference()
    reference.df.loc[17, "vel_z"] = 0.0

    assert select_guideline_indices(reference, anchors).apex["nearest"] == 11


def test_thresholds_are_literal_extrema_plus_named_margins():
    reference, anchors = _reference()
    selection = select_guideline_indices(reference, anchors)

    manifest = build_threshold_manifest(
        selection=selection,
        apex_samples=_apex_samples(),
        recovery_samples=_recovery_samples(),
        margins=_margins(),
        required_recovery_hold_ticks=5,
        source_hashes=SOURCE_HASHES,
        source_paths=SOURCE_PATHS,
        reference_anchors=anchors,
        extraction_code_version="two_phase_guideline_v1",
        controller_provenance="guideline open-loop action sequence",
        creation_seed=17,
    )

    assert manifest["selected_thresholds"]["apex"] == {
        "max_abs_com_vz": pytest.approx(0.08),
        "min_clearance": pytest.approx(0.15),
        "max_abs_roll": pytest.approx(0.13),
        "max_abs_pitch": pytest.approx(0.18),
        "max_angular_speed": pytest.approx(0.30),
        "min_forward_velocity": pytest.approx(0.90),
        "relative_x_min": pytest.approx(-0.30),
        "relative_x_max": pytest.approx(0.35),
    }
    assert manifest["selected_thresholds"]["recovery"] == {
        "max_abs_roll": pytest.approx(0.14),
        "max_abs_pitch": pytest.approx(0.13),
        "max_angular_speed": pytest.approx(0.34),
        "min_forward_velocity": pytest.approx(0.80),
        "required_hold_ticks": 5,
    }
    assert manifest["engineering_margins"] == asdict(_margins())
    assert manifest["source_category"] == "guideline_physical_envelope"


@pytest.mark.parametrize("forbidden", ["reward", "success", "continuation_label", "source_policy_hash"])
def test_threshold_builder_rejects_nonphysical_result_or_metadata_fields(forbidden):
    sample = _apex_samples()[0] | {forbidden: 1}

    with pytest.raises(ValueError, match="unregistered fields"):
        build_threshold_manifest(
            selection=select_guideline_indices(*_reference()),
            apex_samples=[sample],
            recovery_samples=_recovery_samples(),
            margins=_margins(),
            required_recovery_hold_ticks=5,
            source_hashes=SOURCE_HASHES,
            source_paths=SOURCE_PATHS,
            reference_anchors=_reference()[1],
            extraction_code_version="two_phase_guideline_v1",
            controller_provenance="guideline open-loop action sequence",
            creation_seed=17,
        )


def test_manifest_hash_is_canonical_stable_and_binds_every_source():
    kwargs = dict(
        selection=select_guideline_indices(*_reference()),
        apex_samples=_apex_samples(),
        recovery_samples=_recovery_samples(),
        margins=_margins(),
        required_recovery_hold_ticks=5,
        source_hashes=SOURCE_HASHES,
        source_paths=SOURCE_PATHS,
        reference_anchors=_reference()[1],
        extraction_code_version="two_phase_guideline_v1",
        controller_provenance="guideline open-loop action sequence",
        creation_seed=17,
    )
    first = build_threshold_manifest(**kwargs)
    second = build_threshold_manifest(**kwargs)

    assert first == second
    assert first["canonical_manifest_hash"] == canonical_manifest_hash(first)
    assert first["feature_definitions"]["obstacle_relative_x"]["unit"] == "m"
    assert "controller_provenance" in first
    assert first["source_paths"] == SOURCE_PATHS
    assert first["reference_anchors"] == _reference()[1].as_dict()
    assert first["extraction_code_version"] == "two_phase_guideline_v1"
    for source in kwargs["source_hashes"]:
        changed = dict(kwargs)
        changed["source_hashes"] = kwargs["source_hashes"] | {source: "f" * 64}
        assert build_threshold_manifest(**changed)["canonical_manifest_hash"] != first["canonical_manifest_hash"]


@pytest.mark.parametrize("name", ["expert", "pi_up", "pi_down", "trained policy"])
def test_controller_provenance_rejects_expert_or_trained_policy_claims(name):
    with pytest.raises(ValueError, match="controller provenance"):
        build_threshold_manifest(
            selection=select_guideline_indices(*_reference()),
            apex_samples=_apex_samples(),
            recovery_samples=_recovery_samples(),
            margins=_margins(),
            required_recovery_hold_ticks=5,
            source_hashes=SOURCE_HASHES,
            source_paths=SOURCE_PATHS,
            reference_anchors=_reference()[1],
            extraction_code_version="two_phase_guideline_v1",
            controller_provenance=name,
            creation_seed=17,
        )


def test_event_order_report_closes_order_and_width_contracts():
    valid_ticks = {name: index for index, name in enumerate((
        "jump_window_entered", "liftoff_seen", "stable_airborne", "ascending",
        "apex_band_entered", "descending", "pre_landing", "first_valid_contact",
        "impact_absorbing", "stable_recovery",
    ))}

    passed = validate_guideline_event_order(valid_ticks, apex_band_width=3, recovery_hold_ticks=5, required_apex_width=2, required_recovery_hold=5)
    inverted = validate_guideline_event_order(valid_ticks | {"descending": 3}, apex_band_width=3, recovery_hold_ticks=5, required_apex_width=2, required_recovery_hold=5)

    assert passed["status"] == "pass"
    assert inverted["status"] == "gate_pause"
    assert "event_order" in inverted["failed"]


def test_host_geometry_audit_reports_reference_distance_and_nearest_pair():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geometry = build_two_phase_geometry(model, load_config("configs/default.json"))

    row = audit_geometry_clearance(model, data, geometry, representative="initial")

    assert row["representative"] == "initial"
    assert np.isfinite(row["jax_clearance"])
    assert np.isfinite(row["mujoco_reference_distance"])
    assert row["absolute_difference"] == pytest.approx(abs(row["jax_clearance"] - row["mujoco_reference_distance"]))
    assert isinstance(row["sign_agreement"], bool)
    assert set(row["nearest_geom_pair"]) == {"robot", "obstacle"}
