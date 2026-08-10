from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

import pytest

from dvgc.config import ACTION_MAPPING_VERSION
from dvgc.reference import ReferenceAnchors
from dvgc.two_phase_guideline import (
    GuidelineMargins,
    GuidelineSelection,
    build_threshold_manifest,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _module():
    return importlib.import_module("dvgc.phase_expert_training")


def _run_output(tmp_path: Path, name: str = "phase-u-smoke-17") -> str:
    return str(Path("runs/two_phase/phase_experts") / f"{tmp_path.name}-{name}")


def _threshold_manifest(tmp_path: Path) -> Path:
    source_paths = {
        "xml": "assets/orange_bike_4kg_horizontal.xml",
        "reference": "data/reference_jump.csv",
        "config": "configs/default.json",
        "code": "dvgc/two_phase_guideline.py",
    }
    geometry_manifest = {"contract_version": 1, "geoms": [{"geom_id": 0}]}
    source_hashes = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    source_hashes["geometry_manifest"] = hashlib.sha256(
        json.dumps(geometry_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = build_threshold_manifest(
        selection=GuidelineSelection(
            launch={"front": 1, "middle": 2, "back": 3},
            apex={"pre": 4, "nearest": 5, "post": 6},
            early_descent=7,
            recovery=(8, 9, 10),
        ),
        apex_samples=[
            {
                "com_vz": 0.01,
                "clearance": 0.2,
                "roll": 0.02,
                "pitch": 0.03,
                "angular_speed": 0.2,
                "forward_velocity": 2.0,
                "obstacle_relative_x": 0.1,
            }
        ],
        recovery_samples=[
            {"roll": 0.02, "pitch": 0.03, "angular_speed": 0.2, "forward_velocity": 2.0}
        ],
        margins=GuidelineMargins(
            apex_abs_vz=0.01,
            apex_clearance=0.01,
            apex_abs_roll=0.01,
            apex_abs_pitch=0.01,
            apex_angular_speed=0.01,
            apex_forward_velocity=0.01,
            apex_relative_x=0.01,
            recovery_abs_roll=0.01,
            recovery_abs_pitch=0.01,
            recovery_angular_speed=0.01,
            recovery_forward_velocity=0.01,
        ),
        required_recovery_hold_ticks=3,
        source_hashes=source_hashes,
        source_paths=source_paths,
        geometry_manifest=geometry_manifest,
        reference_anchors=ReferenceAnchors(1, 3, 5, 7, 8, 10),
        extraction_code_version="test",
        controller_provenance="kinematic guideline envelope",
        creation_seed=17,
    )
    _write_json(tmp_path / "geometry_manifest.json", geometry_manifest)
    return _write_json(tmp_path / "threshold_manifest.json", manifest)


def _spec(module, tmp_path: Path, **overrides):
    values = {
        "phase": module.PHASE_PROPULSION_ASCENT,
        "experiment_level": "smoke",
        "requested_total_transitions": 1_600,
        "seed": 17,
        "config_path": "configs/default.json",
        "training_config_path": "configs/phase_expert_smoke.json",
        "threshold_manifest_path": str(_threshold_manifest(tmp_path)),
        "authorization_manifest_path": None,
        "output_dir": _run_output(tmp_path),
        "descent_seed_bank": None,
        "descent_seed_manifest": None,
        "resume_run": None,
        "restore_checkpoint": None,
    }
    return module.PhaseExpertRunSpec(**(values | overrides))


def _authorization(module, spec, config: dict) -> dict:
    threshold = module.load_phase_expert_threshold_manifest(spec.threshold_manifest_path)
    interaction = module.build_phase_expert_interaction_budget(spec, config)
    report = interaction.training
    return {
        "decision": "authorize",
        "run_id": Path(spec.output_dir).name,
        "phase": spec.phase,
        "experiment_level": spec.experiment_level,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_tree_sha256": module.phase_expert_source_tree_sha256(),
        "seed": spec.seed,
        "output_directory": str(Path(spec.output_dir).resolve()),
        "xml_sha256": hashlib.sha256(
            Path("assets/orange_bike_4kg_horizontal.xml").read_bytes()
        ).hexdigest(),
        "threshold_manifest_canonical_hash": threshold.canonical_manifest_hash,
        "training_config_sha256": hashlib.sha256(
            Path(spec.training_config_path).read_bytes()
        ).hexdigest(),
        "requested_training_transition_ceiling": report.requested_total_transitions,
        "effective_training_transition_ceiling": report.effective_total_transitions,
        "brax_evaluation_transition_ceiling": interaction.brax_evaluation_transition_ceiling,
        "fixed_evaluation_transition_ceiling": interaction.fixed_evaluation_transition_ceiling,
        "combined_interaction_transition_ceiling": interaction.combined_transition_ceiling,
        "issuer": "gate-c1-test",
        "issued_at": "2026-08-10T00:00:00Z",
    }


def test_run_spec_uses_only_public_phases_and_preflight_freezes_threshold_inputs(tmp_path):
    """A phase typo or mutable threshold input would change the reviewed run contract."""
    module = _module()
    spec = _spec(module, tmp_path)

    validated = module.validate_phase_expert_run_spec(spec, preflight_only=True)

    assert module.PHASE_PROPULSION_ASCENT == "propulsion_ascent"
    assert module.PHASE_DESCENT_RECOVERY == "descent_recovery"
    assert validated.thresholds.action_mapping_version == ACTION_MAPPING_VERSION
    with pytest.raises(TypeError):
        validated.thresholds.manifest["canonical_manifest_hash"] = "changed"
    with pytest.raises(ValueError, match="phase"):
        module.validate_phase_expert_run_spec(replace(spec, phase="apex"), preflight_only=True)


def test_normal_execution_requires_one_run_bound_authorization_but_preflight_requires_none(tmp_path):
    """A reusable or missing authorization must not permit environment interaction."""
    module = _module()
    config = json.loads(Path("configs/phase_expert_smoke.json").read_text(encoding="utf-8"))
    spec = _spec(module, tmp_path)

    with pytest.raises(ValueError, match="authorization"):
        module.validate_phase_expert_run_spec(spec, preflight_only=False)
    assert module.validate_phase_expert_run_spec(spec, preflight_only=True).spec == spec

    authorization_path = _write_json(
        tmp_path / "authorization.json", _authorization(module, spec, config)
    )
    authorized = replace(spec, authorization_manifest_path=str(authorization_path))
    assert module.validate_phase_expert_run_spec(authorized, preflight_only=False).authorization["decision"] == "authorize"

    reused = _write_json(
        tmp_path / "reused.json",
        _authorization(module, spec, config) | {"run_id": "other-run"},
    )
    with pytest.raises(ValueError, match="run id"):
        module.validate_phase_expert_run_spec(
            replace(spec, authorization_manifest_path=str(reused)), preflight_only=False
        )

    for field, replacement, message in (
        ("seed", spec.seed + 1, "seed"),
        ("output_directory", str(tmp_path / "elsewhere" / Path(spec.output_dir).name), "output directory"),
        ("source_tree_sha256", "0" * 64, "source tree"),
    ):
        drifted = _write_json(
            tmp_path / f"drift-{field}.json",
            _authorization(module, spec, config) | {field: replacement},
        )
        with pytest.raises(ValueError, match=message):
            module.validate_phase_expert_run_spec(
                replace(spec, authorization_manifest_path=str(drifted)),
                preflight_only=False,
            )


def test_run_spec_rejects_independent_timesteps_promoted_levels_existing_output_and_incomplete_hashes(tmp_path):
    """Alternate budgets, promotion, overwrites, or missing provenance bypass Gate C1 controls."""
    module = _module()
    spec = _spec(module, tmp_path)

    with pytest.raises(TypeError, match="requested_timesteps"):
        module.PhaseExpertRunSpec(**(spec.__dict__ | {"requested_timesteps": 1_600}))
    for level in ("learnability_pilot", "formal_expert"):
        with pytest.raises(ValueError, match="smoke"):
            module.validate_phase_expert_run_spec(replace(spec, experiment_level=level), preflight_only=True)
    output_path = Path(spec.output_dir)
    output_path.mkdir(parents=True)
    try:
        with pytest.raises(ValueError, match="output directory"):
            module.validate_phase_expert_run_spec(spec, preflight_only=True)
    finally:
        output_path.rmdir()

    with pytest.raises(ValueError, match="project config"):
        module.validate_phase_expert_run_spec(
            replace(
                spec,
                output_dir=_run_output(tmp_path, "config-drift"),
                config_path="configs/phase_expert_smoke.json",
            ),
            preflight_only=True,
        )

    for resume_fields in (
        {"resume_run": str(tmp_path / "parent")},
        {"restore_checkpoint": str(tmp_path / "checkpoint")},
        {
            "resume_run": str(tmp_path / "parent"),
            "restore_checkpoint": str(tmp_path / "checkpoint"),
        },
    ):
        with pytest.raises(ValueError, match="resume"):
            module.validate_phase_expert_run_spec(
                replace(spec, output_dir=_run_output(tmp_path, "resume-run"), **resume_fields),
                preflight_only=True,
            )

    manifest = json.loads(Path(spec.threshold_manifest_path).read_text(encoding="utf-8"))
    manifest["source_hashes"].pop("code")
    incomplete = _write_json(tmp_path / "incomplete.json", manifest)
    with pytest.raises(ValueError, match="canonical|source hashes"):
        module.validate_phase_expert_run_spec(
            replace(spec, output_dir=_run_output(tmp_path, "new-run"), threshold_manifest_path=str(incomplete)),
            preflight_only=True,
        )

    with pytest.raises(ValueError, match="runs/two_phase/phase_experts"):
        module.validate_phase_expert_run_spec(
            replace(spec, output_dir=str(tmp_path / "outside-runs")),
            preflight_only=True,
        )


def test_phase_d_is_preflight_only_and_never_falls_back_to_natural_reset(tmp_path):
    """Phase D cannot use a natural reset or execute before Gate C2."""
    module = _module()
    base = _spec(module, tmp_path, phase=module.PHASE_DESCENT_RECOVERY)
    bank = tmp_path / "descent-seeds.jsonl"
    bank.write_text("{}\n", encoding="utf-8")
    natural = _write_json(
        tmp_path / "natural-seed.json",
        {
            "seed_tier": "physically_validated_descent_seed",
            "reset_mode": "natural_start",
            "source_hash": hashlib.sha256(bank.read_bytes()).hexdigest(),
        },
    )
    with pytest.raises(ValueError, match="natural reset"):
        module.validate_phase_expert_run_spec(
            replace(base, descent_seed_bank=str(bank), descent_seed_manifest=str(natural)),
            preflight_only=True,
        )

    valid = _write_json(
        tmp_path / "validated-seed.json",
        {
            "seed_tier": "physically_validated_descent_seed",
            "reset_mode": "bank",
            "source_hash": hashlib.sha256(bank.read_bytes()).hexdigest(),
        },
    )
    phase_d = replace(base, descent_seed_bank=str(bank), descent_seed_manifest=str(valid))
    assert module.validate_phase_expert_run_spec(phase_d, preflight_only=True).spec == phase_d
    with pytest.raises(ValueError, match="preflight"):
        module.validate_phase_expert_run_spec(phase_d, preflight_only=False)


def test_phase_budget_has_zero_alignment_one_to_four_blocks_and_separate_combined_ceiling(tmp_path):
    """Rounded PPO budgets or hidden fixed evaluation would exceed the approved interaction cost."""
    module = _module()
    config = json.loads(Path("configs/phase_expert_smoke.json").read_text(encoding="utf-8"))
    spec = _spec(module, tmp_path)

    report = module.build_phase_expert_budget(spec, config["ppo_layout"])
    interaction = module.build_phase_expert_interaction_budget(spec, config)

    assert report.requested_timesteps == report.requested_total_transitions == 1_600
    assert report.effective_timesteps == report.effective_total_transitions == 1_600
    assert report.alignment_overhead == 0
    assert report.ppo_rollout_blocks == 1
    assert interaction.brax_evaluation_transition_ceiling == 1_600
    assert interaction.fixed_evaluation_transition_ceiling == 1_600
    assert interaction.combined_transition_ceiling == 4_800
    with pytest.raises(ValueError, match="aligned"):
        module.build_phase_expert_budget(replace(spec, requested_total_transitions=1_601), config["ppo_layout"])
    with pytest.raises(ValueError, match="one through four"):
        module.build_phase_expert_budget(replace(spec, requested_total_transitions=8_000), config["ppo_layout"])
    with pytest.raises(ValueError, match="combined"):
        module.build_phase_expert_interaction_budget(
            replace(spec, requested_total_transitions=6_400),
            config | {"maximum_interaction_cost": config["maximum_interaction_cost"] | {"combined_transitions": 7_999}},
        )
    with pytest.raises(ValueError, match="fixed evaluation"):
        module.build_phase_expert_interaction_budget(
            spec,
            config | {"evaluation": config["evaluation"] | {"environment_count": 7}},
        )
    with pytest.raises(ValueError, match="combined"):
        module.build_phase_expert_interaction_budget(
            spec,
            config
            | {
                "maximum_interaction_cost": config["maximum_interaction_cost"]
                | {"combined_transitions": 8_001}
            },
        )


def test_smoke_config_locks_base_mode_cost_stops_rewards_and_disjoint_deterministic_seeds(tmp_path):
    """A legacy base mode or colliding evaluation seed would invalidate smoke evidence."""
    module = _module()
    config = json.loads(Path("configs/phase_expert_smoke.json").read_text(encoding="utf-8"))
    spec = _spec(module, tmp_path)

    base = module.resolve_gate_c1_base_mode(config)
    assert base == {
        "training_stage": "full",
        "use_bank_resets": False,
        "expert_chain_termination": False,
        "stage_reachability_objective": "",
        "domain_randomization": False,
    }
    assert config["maximum_interaction_cost"] == {
        "training_transitions": 6_400,
        "brax_evaluation_transitions": 1_600,
        "fixed_evaluation_transitions": 1_600,
        "combined_transitions": 9_600,
    }
    assert config["stopping_conditions"] and config["reward_bounds"]["total_min"] < config["reward_bounds"]["total_max"]
    assert config["checkpoint_cadence_transitions"] == 1_600
    assert config["adapter_ownership"] == {
        "reward": True,
        "done": True,
        "timeout": True,
    }
    assert "stop_after_brax_evaluation_ceiling" in config["stopping_conditions"]
    assert "stop_after_combined_interaction_ceiling" in config["stopping_conditions"]
    with pytest.raises(ValueError, match="adapter ownership"):
        module.resolve_gate_c1_base_mode(
            config | {"adapter_ownership": config["adapter_ownership"] | {"done": False}}
        )
    namespaces = module.validate_phase_expert_seed_namespaces(spec, config)
    assert len(namespaces.evaluation_seeds) == config["evaluation"]["environment_count"]
    assert set(namespaces.training_seeds).isdisjoint(namespaces.evaluation_seeds)
    assert namespaces == module.validate_phase_expert_seed_namespaces(spec, config)
    collision = config | {"evaluation": config["evaluation"] | {"seeds": [namespaces.training_seeds[0]]}}
    with pytest.raises(ValueError, match="disjoint"):
        module.validate_phase_expert_seed_namespaces(spec, collision)


def test_threshold_loader_rejects_tampering_and_obsolete_action_or_guideline_provenance(tmp_path):
    """A threshold contract must bind its canonical content, sources, action mapping, and geometry."""
    module = _module()
    manifest_path = _threshold_manifest(tmp_path)
    loaded = module.load_phase_expert_threshold_manifest(manifest_path)
    assert loaded.canonical_manifest_hash == loaded.manifest["canonical_manifest_hash"]
    assert loaded.reference_rollout_source == "kinematic_guideline_envelope"

    for field, value, message in (
        ("action_mapping_version", "legacy", "action mapping"),
        ("reference_rollout_source", "guideline_open_loop_actions", "reference rollout"),
        ("controller_provenance", "guideline open-loop action sequence", "controller provenance"),
        ("source_category", "dynamic_controller_rollout", "source category"),
    ):
        tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
        tampered[field] = value
        tampered["canonical_manifest_hash"] = module.canonical_manifest_hash(tampered)
        path = _write_json(tmp_path / f"{field}.json", tampered)
        with pytest.raises(ValueError, match=message):
            module.load_phase_expert_threshold_manifest(path)

    copied_xml = tmp_path / "copied-model.xml"
    copied_xml.write_bytes(Path("assets/orange_bike_4kg_horizontal.xml").read_bytes())
    copied_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied_manifest["source_paths"]["xml"] = str(copied_xml)
    copied_manifest["canonical_manifest_hash"] = module.canonical_manifest_hash(
        copied_manifest
    )
    copied_path = _write_json(tmp_path / "copied-xml-manifest.json", copied_manifest)
    with pytest.raises(ValueError, match="authoritative XML path"):
        module.load_phase_expert_threshold_manifest(copied_path)

    geometry = json.loads((tmp_path / "geometry_manifest.json").read_text(encoding="utf-8"))
    geometry["geoms"].append({"geom_id": 1})
    _write_json(tmp_path / "geometry_manifest.json", geometry)
    with pytest.raises(ValueError, match="geometry"):
        module.load_phase_expert_threshold_manifest(manifest_path)
