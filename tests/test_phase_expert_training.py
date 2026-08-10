from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import NamedTuple

import jax
from jax import numpy as jp
import pytest
from mujoco_playground._src import mjx_env

from dvgc.config import ACTION_MAPPING_VERSION
from dvgc.reference import ReferenceAnchors
from dvgc.two_phase_guideline import (
    GuidelineMargins,
    GuidelineSelection,
    build_threshold_manifest,
)
from dvgc.two_phase_runtime import TwoPhaseThresholds
from dvgc.two_phase_semantics import ApexBandSignals, ApexBandThresholds, RecoverySignals, RecoveryThresholds


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
            "records": [{
                "seed_role": "physically_validated_descent_seed",
                "mujoco_forward_valid": True,
                "finite_state_valid": True,
                "no_penetration": True,
                "legal_geometry": True,
                "short_horizon_dynamic_valid": True,
                "real_three_frame_fifo_valid": True,
                "timing_explicit_snapshot_valid": True,
            }],
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


class _FakeData(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array


def _adapter_thresholds() -> TwoPhaseThresholds:
    return TwoPhaseThresholds(
        apex=ApexBandThresholds(
            max_abs_com_vz=0.2,
            min_clearance=0.1,
            max_abs_roll=0.5,
            max_abs_pitch=0.5,
            max_angular_speed=1.0,
            min_forward_velocity=1.0,
            relative_x_min=-0.2,
            relative_x_max=0.2,
        ),
        recovery=RecoveryThresholds(
            max_abs_roll=0.5,
            max_abs_pitch=0.5,
            max_angular_speed=1.0,
            min_forward_velocity=0.5,
            required_hold_ticks=3,
        ),
    )


def _fake_signals(state, _geometry, previous_hold_count):
    info = state.info
    apex = ApexBandSignals(
        stable_airborne=info["fake/stable_airborne"],
        com_vz=info["fake/com_vz"],
        clearance=info["fake/clearance"],
        roll=info["fake/roll"],
        pitch=info["fake/pitch"],
        angular_speed=info["fake/angular_speed"],
        forward_velocity=info["fake/forward_velocity"],
        obstacle_relative_x=info["fake/relative_x"],
        illegal_contact=info["fake/illegal_contact"],
        physical_failure=info["fake/physical_failure"],
    )
    recovery = RecoverySignals(
        stable_wheel_support=info["fake/support"],
        landing_region_valid=jp.asarray(False),
        no_body_contact=info["fake/no_body_contact"],
        roll=info["fake/roll"],
        pitch=info["fake/pitch"],
        angular_speed=info["fake/angular_speed"],
        forward_velocity=info["fake/forward_velocity"],
        previous_recovery_hold_count=previous_hold_count,
        physical_failure=info["fake/physical_failure"],
    )
    return apex, recovery


class _FakeBaseEnv:
    action_size = 8
    _neutral_action = jp.zeros((8,), jp.float32)
    _config = SimpleNamespace(max_roll_deg=35.0, max_pitch_deg=75.0)

    @staticmethod
    def _info():
        return {
            "last_action": jp.zeros((8,), jp.float32),
            "obs_history": jp.zeros((3, 4), jp.float32),
            "actor_packet_fifo_valid": jp.asarray(1, jp.int32),
            "jump_signal_latched": jp.asarray(False),
            "jump_window_active": jp.asarray(False),
            "end_code": jp.asarray(0, jp.int32),
            "fake/stable_airborne": jp.asarray(False),
            "fake/com_vz": jp.asarray(0.0, jp.float32),
            "fake/clearance": jp.asarray(0.2, jp.float32),
            "fake/roll": jp.asarray(0.0, jp.float32),
            "fake/pitch": jp.asarray(0.0, jp.float32),
            "fake/angular_speed": jp.asarray(0.0, jp.float32),
            "fake/forward_velocity": jp.asarray(2.0, jp.float32),
            "fake/relative_x": jp.asarray(0.0, jp.float32),
            "fake/illegal_contact": jp.asarray(False),
            "fake/physical_failure": jp.asarray(False),
            "fake/support": jp.asarray(True),
            "fake/no_body_contact": jp.asarray(True),
        }

    def reset(self, rng):
        del rng
        return mjx_env.State(
            data=_FakeData(jp.zeros((7,)), jp.zeros((6,)), jp.zeros((8,))),
            obs={"state": jp.zeros((12,)), "privileged_state": jp.zeros((6,))},
            reward=jp.asarray(0.0),
            done=jp.asarray(0.0),
            metrics={"reward": jp.asarray(0.0)},
            info=self._info(),
        )

    def step(self, state, action):
        end_code = action[6].astype(jp.int32)
        info = state.info | {
            "last_action": action,
            "jump_signal_latched": action[0] > 0.0,
            "jump_window_active": action[1] > 0.0,
            "end_code": end_code,
            "fake/stable_airborne": action[2] > 0.0,
            "fake/com_vz": action[3],
            "fake/clearance": action[4],
            "fake/relative_x": action[5],
            "fake/support": action[7] > 0.0,
            "fake/physical_failure": jp.isin(end_code, jp.asarray([2, 3, 4, 5, 6, 7, 15])),
        }
        return state.replace(
            data=state.data._replace(qvel=state.data.qvel.at[0].set(2.0).at[2].set(action[3])),
            reward=jp.asarray(999.0),
            done=jp.asarray(1.0),
            metrics={"reward": jp.asarray(999.0)},
            info=info,
        )


def _adapter(module):
    return module.PhaseExpertEnvAdapter(
        _FakeBaseEnv(),
        geometry=None,
        thresholds=_adapter_thresholds(),
        reward_config=module.PhaseURewardConfig(),
        episode_horizon=20,
        signal_extractor=_fake_signals,
    )


def test_phase_u_adapter_reset_is_natural_audited_and_jittable_without_reference_inputs():
    module = _module()
    adapter = _adapter(module)

    state = jax.jit(adapter.reset)(jax.random.PRNGKey(1))

    assert adapter.phase == module.PHASE_PROPULSION_ASCENT
    assert bool(state.info["phase_expert/reset_valid"])
    assert int(state.info["phase_expert/source_phase_id"]) == 0
    assert not bool(state.done)
    assert state.done.dtype == _FakeBaseEnv().reset(jax.random.PRNGKey(0)).done.dtype
    assert jp.all(state.info["last_action"] == 0.0)
    assert state.info["obs_history"].shape == (3, 4)
    batched = jax.vmap(adapter.reset)(jax.random.split(jax.random.PRNGKey(2), 2))
    assert batched.done.shape == (2,)


def test_early_airborne_cannot_succeed_but_later_ordered_apex_entry_does_and_latch_is_monotonic():
    module = _module()
    adapter = _adapter(module)
    state = adapter.reset(jax.random.PRNGKey(3))

    early_apex = jp.asarray([0, 0, 1, 0, 0.2, 0.0, 0, 0], jp.float32)
    state = adapter.step(state, early_apex)
    assert not bool(state.info["phase_expert/success"])
    assert not bool(state.done)

    sequence = (
        jp.asarray([1, 1, 0, 0.5, 0.2, 0.0, 0, 1], jp.float32),
        jp.asarray([0, 1, 0, 0.5, 0.2, 0.0, 0, 0], jp.float32),
        jp.asarray([0, 1, 1, 0.5, 0.2, 0.0, 0, 0], jp.float32),
        jp.asarray([0, 1, 1, 0.5, 0.2, 0.0, 0, 0], jp.float32),
        jp.asarray([0, 0, 1, 0.0, 0.2, 0.0, 0, 0], jp.float32),
    )
    for action in sequence:
        state = jax.jit(adapter.step)(state, action)
    assert bool(state.info["phase_expert/event/jump_window_entered"])
    assert bool(state.info["phase_expert/event/apex_band_entered"])
    assert bool(state.info["phase_expert/success"])
    assert bool(state.done)


def test_pre_window_takeoff_and_apex_progress_rewards_are_zero_even_when_airborne_and_rising():
    module = _module()
    signals, _ = _fake_signals(
        _FakeBaseEnv().reset(jax.random.PRNGKey(0)).replace(
            info=_FakeBaseEnv._info()
            | {"fake/stable_airborne": jp.asarray(True), "fake/com_vz": jp.asarray(1.0)}
        ),
        None,
        jp.asarray(0),
    )
    components = jax.jit(module.phase_u_reward_components)(
        signals,
        _adapter_thresholds().apex,
        jp.asarray(False),
        jp.asarray(False),
        jp.zeros((8,)),
        jp.zeros((8,)),
        module.PhaseURewardConfig(),
    )
    assert float(components["takeoff_ascent_progress"]) == 0.0
    assert float(components["apex_progress"]) == 0.0
    assert float(components["forward_propulsion"]) > 0.0


def test_adapter_reconstructs_runtime_window_from_latch_root_x_and_end_x():
    module = _module()
    adapter = module.PhaseExpertEnvAdapter(
        _FakeBaseEnv(),
        geometry=SimpleNamespace(root_qpos_adr=0),
        thresholds=_adapter_thresholds(),
        reward_config=module.PhaseURewardConfig(),
        episode_horizon=20,
        signal_extractor=_fake_signals,
    )
    base = _FakeBaseEnv().reset(jax.random.PRNGKey(0))
    info = dict(base.info)
    del info["jump_window_active"]
    info |= {"jump_signal_latched": jp.asarray(True), "jump_window_end_x": jp.asarray(0.5)}
    inside = base.replace(data=base.data._replace(qpos=base.data.qpos.at[0].set(0.4)), info=info)
    outside = inside.replace(data=inside.data._replace(qpos=inside.data.qpos.at[0].set(0.6)))
    assert bool(jax.jit(adapter._window_active)(inside))
    assert not bool(jax.jit(adapter._window_active)(outside))


@pytest.mark.parametrize("end_code", [2, 3, 4, 5, 6, 7, 15])
def test_phase_u_adapter_preserves_physical_failure_end_codes(end_code):
    module = _module()
    adapter = _adapter(module)
    action = jp.asarray([0, 0, 0, 0, 0.2, 0, end_code, 1], jp.float32)
    state = jax.jit(adapter.step)(adapter.reset(jax.random.PRNGKey(4)), action)
    assert bool(state.done)
    assert bool(state.info["phase_expert/physical_failure"])


@pytest.mark.parametrize("end_code", [1, 8, 14, 16])
def test_phase_u_adapter_ignores_legacy_success_timeout_and_chain_end_codes(end_code):
    module = _module()
    adapter = _adapter(module)
    action = jp.asarray([0, 0, 0, 0, 0.2, 0, end_code, 1], jp.float32)
    state = jax.jit(adapter.step)(adapter.reset(jax.random.PRNGKey(5)), action)
    assert not bool(state.done)


@pytest.mark.parametrize("end_code", [10, 11, 12, 13])
def test_takeoff_task_failures_activate_only_after_legal_jump_latch(end_code):
    module = _module()
    adapter = _adapter(module)
    failure = jp.asarray([0, 0, 0, 0, 0.2, 0, end_code, 1], jp.float32)
    state = adapter.step(adapter.reset(jax.random.PRNGKey(6)), failure)
    assert not bool(state.done)
    latched = adapter.step(
        adapter.reset(jax.random.PRNGKey(7)),
        jp.asarray([1, 1, 0, 0, 0.2, 0, 0, 1], jp.float32),
    )
    state = adapter.step(latched, failure)
    assert bool(state.done)
    assert bool(state.info["phase_expert/task_failure"])


def test_evaluation_outcomes_are_closed_and_mutually_exclusive():
    module = _module()
    rows = [
        {"success": True, "physical_failure": False, "timeout": False, "end_code": 0},
        {"success": False, "physical_failure": True, "timeout": False, "end_code": 4},
        {"success": False, "physical_failure": False, "timeout": True, "end_code": 0},
        {"success": False, "physical_failure": False, "timeout": False, "end_code": 11},
    ]
    report = module.summarize_phase_expert_evaluation(rows)
    assert report["outcome_counts"] == {
        "success": 1,
        "physical_failure": 1,
        "timeout": 1,
        "other_failure": 1,
    }
    assert report["termination_reason_counts"] == {"none": 2, "roll_limit": 1, "takeoff_wheelie_failure": 1}
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.summarize_phase_expert_evaluation(
            [{"success": True, "physical_failure": True, "timeout": False, "end_code": 4}]
        )


def test_run_artifacts_are_immutable_atomic_and_metrics_are_finite(tmp_path):
    module = _module()
    run = tmp_path / "run"
    manifest = {"run_id": "run", "phase": "propulsion_ascent", "source_hash": "a" * 64}
    module.initialize_phase_expert_artifacts(run, manifest, {"status": "initialized"})
    assert json.loads((run / "run_manifest.json").read_text()) == manifest
    module.update_phase_expert_status(run, {"status": "running", "transitions": 0})
    assert json.loads((run / "status.json").read_text())["status"] == "running"
    module.append_phase_expert_metrics(run, {"step": 0, "loss": 1.25})
    module.append_phase_expert_metrics(run, {"step": 1, "loss": 1.0})
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="finite"):
        module.append_phase_expert_metrics(run, {"loss": float("nan")})
    with pytest.raises(FileExistsError):
        module.initialize_phase_expert_artifacts(run, manifest, {"status": "initialized"})


def test_checkpoint_sidecar_binds_recursive_identity_and_rejects_drift(tmp_path):
    module = _module()
    checkpoint = tmp_path / "orbax" / "1600"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state").write_bytes(b"full training state")
    contract = {
        "phase": "propulsion_ascent",
        "cumulative_training_transitions": 1600,
        "full_training_state": True,
        "prng_lineage": "phase_u_train_v1:7",
        "reset_contract_hash": "a" * 64,
        "reward_contract_hash": "b" * 64,
        "evaluation_contract_hash": "c" * 64,
        "xml_sha256": "d" * 64,
        "action_schema_hash": "e" * 64,
        "observation_schema_hash": "f" * 64,
        "history_schema_hash": "1" * 64,
        "parent_checkpoint": None,
    }
    sidecar = module.write_phase_expert_checkpoint_sidecar(checkpoint, contract)
    assert sidecar["recursive_checkpoint_sha256"] == module.recursive_path_sha256(checkpoint)
    module.validate_phase_expert_checkpoint_sidecar(checkpoint, contract)
    (checkpoint / "state").write_bytes(b"drift")
    with pytest.raises(ValueError, match="identity"):
        module.validate_phase_expert_checkpoint_sidecar(checkpoint, contract)


def test_descent_seed_manifest_requires_physical_evidence_and_rejects_claims(tmp_path):
    module = _module()
    bank = tmp_path / "seed.bank"
    bank.write_bytes(b"seed")
    manifest = {
        "reset_mode": "bank",
        "seed_tier": "physically_validated_descent_seed",
        "source_hash": hashlib.sha256(b"seed").hexdigest(),
        "records": [{
            "seed_role": "physically_validated_descent_seed",
            "mujoco_forward_valid": True,
            "finite_state_valid": True,
            "no_penetration": True,
            "legal_geometry": True,
            "short_horizon_dynamic_valid": True,
            "real_three_frame_fifo_valid": True,
            "timing_explicit_snapshot_valid": True,
        }],
    }
    path = _write_json(tmp_path / "seed.json", manifest)
    module.validate_descent_seed_manifest(bank, path)
    for forbidden in ("reachable", "expert_snapshot", "Tube", "safe"):
        bad = json.loads(json.dumps(manifest))
        bad["records"][0][forbidden] = True
        _write_json(path, bad)
        with pytest.raises(ValueError, match="forbidden claim"):
            module.validate_descent_seed_manifest(bank, path)
