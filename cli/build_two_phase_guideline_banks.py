"""Build reproducible Gate B two-phase guideline thresholds and v4 banks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco

from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, config_hash, file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.reference import ReferenceTrajectory
from dvgc.two_phase_guideline import (
    DEFAULT_GUIDELINE_PERTURBATIONS,
    GuidelineMargins,
    audit_geometry_clearance,
    build_guideline_banks,
    build_threshold_manifest,
    canonical_manifest_hash,
    extract_guideline_threshold_samples,
    reconstruct_guideline_state,
    select_guideline_indices,
)
from dvgc.two_phase_roundtrip import (
    compare_two_phase_roundtrip,
    select_roundtrip_representatives,
)
from dvgc.two_phase_runtime import (
    TwoPhaseThresholds,
    build_two_phase_geometry,
    geometry_manifest,
    validate_geometry_manifest,
)
from dvgc.two_phase_semantics import ApexBandThresholds, RecoveryThresholds


DEFAULT_MARGINS = GuidelineMargins(
    apex_abs_vz=0.05,
    apex_clearance=0.02,
    apex_abs_roll=0.035,
    apex_abs_pitch=0.035,
    apex_angular_speed=0.20,
    apex_forward_velocity=0.20,
    apex_relative_x=0.10,
    recovery_abs_roll=0.035,
    recovery_abs_pitch=0.035,
    recovery_angular_speed=0.20,
    recovery_forward_velocity=0.20,
)


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_output(output: Path, run_manifest: dict[str, Any]) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "run_manifest.json", run_manifest)


def _guideline_provenance(cfg: Any, source_fingerprint: str) -> dict[str, str]:
    no_policy = hashlib.sha256(b"guideline_open_loop_no_policy").hexdigest()
    return {
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_sha256": config_hash(cfg),
        "action_mapping_version": ACTION_MAPPING_VERSION,
        "policy_params_sha256": no_policy,
        "policy_config_sha256": config_hash(cfg),
        "policy_manifest_sha256": hashlib.sha256(
            b"guideline_controller_provenance"
        ).hexdigest(),
        "normalizer_sha256": hashlib.sha256(b"no_policy_normalizer").hexdigest(),
        "source_fingerprint": source_fingerprint,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    perturbations = (
        DEFAULT_GUIDELINE_PERTURBATIONS[:1]
        if args.perturbations == "nominal"
        else DEFAULT_GUIDELINE_PERTURBATIONS
    )
    max_transitions = 7 * len(perturbations) * 3
    max_roundtrip_transitions = 26
    run_manifest = {
        "contract_version": 1,
        "purpose": "Gate B deterministic two-phase guideline threshold and initial-bank construction",
        "inputs": {
            "config": str(args.config),
            "reference": str(args.reference),
            "seed": int(args.seed),
            "perturbations": [asdict(item) for item in perturbations],
        },
        "interaction_cost": {
            "maximum_construction_environment_transitions": max_transitions,
            "maximum_roundtrip_environment_transitions": max_roundtrip_transitions,
            "formal_training_transitions": 0,
        },
        "stopping_condition": "stop on first geometry, threshold, snapshot, or output-contract failure",
        "output_directory": str(output),
    }
    _prepare_output(output, run_manifest)

    cfg = load_config(
        args.config,
        {
            "domain_randomization": False,
            "obs_noise_enable": False,
            "use_bank_resets": False,
        },
    )
    reference = ReferenceTrajectory.load(args.reference)
    anchors = reference.anchors(
        step_front_x=float(cfg.step_front_x),
        step_top_z=float(cfg.step_top_z),
        base_ground_z=float(reference.df["pos_z"].iloc[0]),
    )
    selection = select_guideline_indices(reference, anchors)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    geometry = build_two_phase_geometry(model, cfg)
    geom_manifest = geometry_manifest(model, geometry)
    geom_validation = validate_geometry_manifest(
        geom_manifest, model=model, geometry=geometry
    )
    if not geom_validation["valid"]:
        raise ValueError(f"Geometry manifest failed: {geom_validation['failed']}")
    _save_json(output / "geometry_manifest.json", geom_manifest)

    apex_samples, recovery_samples = extract_guideline_threshold_samples(
        model,
        reference,
        selection,
        geometry,
        wheel_roll_radius=float(cfg.wheel_roll_radius),
    )
    source_paths = {
        "xml": str(cfg.xml_path),
        "reference": str(args.reference),
        "config": str(args.config),
        "code": "dvgc/two_phase_runtime.py",
    }
    source_hashes = {
        name: file_sha256(path) for name, path in source_paths.items()
    }
    source_hashes["geometry_manifest"] = canonical_manifest_hash(geom_manifest)
    threshold_manifest = build_threshold_manifest(
        selection=selection,
        apex_samples=apex_samples,
        recovery_samples=recovery_samples,
        margins=DEFAULT_MARGINS,
        required_recovery_hold_ticks=int(cfg.recovery_hold_steps),
        source_hashes=source_hashes,
        source_paths=source_paths,
        geometry_manifest=geom_manifest,
        reference_anchors=anchors,
        extraction_code_version="two_phase_runtime_guideline_v1",
        controller_provenance="guideline open-loop action sequence",
        creation_seed=int(args.seed),
    )
    threshold_manifest["resolved_config_sha256"] = config_hash(cfg)
    threshold_manifest["canonical_manifest_hash"] = canonical_manifest_hash(
        threshold_manifest
    )
    _save_json(output / "threshold_manifest.json", threshold_manifest)

    host_data = mujoco.MjData(model)
    root = int(model.jnt_qposadr[int(model.joint("floating_base_joint").id)])
    audits = []
    for name, index in selection.apex.items():
        proposal = reconstruct_guideline_state(
            model,
            reference,
            index,
            wheel_roll_radius=float(cfg.wheel_roll_radius),
        )
        host_data.qpos[:] = proposal.qpos
        host_data.qpos[root] = geometry.obstacle_front_x + 0.5
        host_data.qvel[:] = proposal.qvel
        mujoco.mj_forward(model, host_data)
        row = audit_geometry_clearance(
            model,
            host_data,
            geometry,
            representative=f"apex_{name}_horizontally_aligned_audit",
            tolerance=float(args.geometry_tolerance),
        )
        row["audit_only_root_x_translation_m"] = float(
            host_data.qpos[root] - proposal.qpos[root]
        )
        audits.append(row)
    geometry_report = {
        "status": "pass" if all(row["status"] == "pass" for row in audits) else "gate_pause",
        "formal_runtime_path": "pure_jax_adapter",
        "host_mujoco_usage": "representative_cross_audit_only",
        "rows": audits,
    }
    _save_json(output / "geometry_cross_audit.json", geometry_report)
    if geometry_report["status"] != "pass":
        raise ValueError("Representative geometry cross-audit entered gate_pause")

    source_fingerprint = hashlib.sha256(
        Path("dvgc/two_phase_runtime.py").read_bytes()
        + Path("dvgc/two_phase_guideline.py").read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    bank_build = build_guideline_banks(
        env,
        reference,
        selection,
        provenance=_guideline_provenance(cfg, source_fingerprint),
        seed=int(args.seed),
        perturbations=perturbations,
    )
    up = bank_build.phase_up
    down = bank_build.phase_down
    construction_report = bank_build.report
    up_path = output / "phase_up_guideline_bank.pkl"
    down_path = output / "phase_down_guideline_bank.pkl"
    up.save(up_path)
    down.save(down_path)

    selected_thresholds = threshold_manifest["selected_thresholds"]
    runtime_thresholds = TwoPhaseThresholds(
        apex=ApexBandThresholds(**selected_thresholds["apex"]),
        recovery=RecoveryThresholds(**selected_thresholds["recovery"]),
    )
    representatives = select_roundtrip_representatives(up, down)
    tick_counts = {
        "up_boundary_front": 1,
        "up_boundary_back": 3,
        "down_pre": 1,
        "down_nearest": 2,
        "down_post": 3,
        "down_boundary": 3,
    }
    roundtrip_rows = []
    for offset, (label, record) in enumerate(representatives.items()):
        action = record["policy_action_t"]
        row = compare_two_phase_roundtrip(
            env,
            record,
            bank_build.original_states[record["id"]],
            geometry,
            runtime_thresholds,
            seed=int(args.seed) + 50_000 + offset,
            actions=[action] * tick_counts[label],
        )
        row["representative"] = label
        roundtrip_rows.append(row)
    roundtrip_report = {
        "status": (
            "pass"
            if all(row["status"] == "pass" for row in roundtrip_rows)
            else "gate_pause"
        ),
        "restore_mode": "timing_explicit_independent_reconstruction",
        "same_snapshot_seed_action": True,
        "formal_training_transitions": 0,
        "environment_transitions": sum(
            2 * row["control_ticks"] for row in roundtrip_rows
        ),
        "rows": roundtrip_rows,
    }
    _save_json(output / "snapshot_roundtrip_report.json", roundtrip_report)
    if roundtrip_report["status"] != "pass":
        raise ValueError("Snapshot round-trip entered gate_pause")

    build_report = {
        "status": (
            "build_and_roundtrip_complete_pending_event_gate"
            if geometry_report["status"] == "pass"
            else "gate_pause"
        ),
        **construction_report,
        "threshold_manifest_hash": threshold_manifest["canonical_manifest_hash"],
        "geometry_manifest_hash": canonical_manifest_hash(geom_manifest),
        "phase_up_bank_sha256": file_sha256(up_path),
        "phase_down_bank_sha256": file_sha256(down_path),
        "guideline_event_validation": "pending_task_5_dynamic_trace",
        "snapshot_roundtrip_validation": roundtrip_report["status"],
        "roundtrip_environment_transitions": roundtrip_report[
            "environment_transitions"
        ],
    }
    _save_json(output / "gate_b_build_report.json", build_report)
    return build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", default="data/reference_jump.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20_260_803)
    parser.add_argument("--perturbations", choices=("nominal", "all"), default="all")
    parser.add_argument("--geometry-tolerance", type=float, default=2e-4)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
