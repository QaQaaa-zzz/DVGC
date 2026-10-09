"""Phase-specific frontier acquisition and labeling for the v3 JIT revision.

This adapter deliberately leaves the historical single-panel collector unchanged.
A v3 plan may assign a different predeclared probe panel to upstream and
downstream while preserving the same frozen policy, Tube, parent-role split,
continuation definition, and data-role isolation.

The adapter also carries forward the historical continuation-label memory lesson:
large logical banks are not labeled monolithically.  They are stopped after
acquisition and resumed through independent-process contiguous shards, whose
logical protocol and PRNG indexing are identical to a serial labeling pass.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping

import jax

from .checkpoint import load_checkpoint
from .config import file_sha256
from .iterative_frontier_protocol import (
    ROLE_SCHEMA,
    ROLES,
    _acquisition_phase_support,
    _anchors_from_plan,
    _completed_acquisition,
    _completed_labeling,
    _load_plan,
    _logical_rows,
    _phase_counts,
    _selected_policy,
    _source_tube,
    _verify_self_hash,
    _write,
    canonical_sha256,
)
from .ppo import make_checkpoint_policy
from .unified_boundary import collect_unified_boundary_candidates
from .unified_continuation_labels import (
    label_unified_continuations,
    validate_unified_boundary_catalog,
)
from .unified_continuation_shards import (
    label_unified_continuation_shard,
    merge_unified_continuation_shards,
)
from .unified_formal import build_unified_formal_environment
from .unified_training import checkpoint_identity


PHASES = ("upstream", "downstream")
V3_REVISION_NAME = "phase_specific_two_axis_v3"
DEFAULT_MAX_CANDIDATES_PER_LABEL_PROCESS = 930


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_phase_panel(panel: Mapping[str, Any], phase: str) -> dict[str, Any]:
    required = {
        "action_names",
        "signs",
        "strengths",
        "durations",
        "active_action_dimensions",
    }
    missing = required.difference(panel)
    if missing:
        raise ValueError(f"{phase} probe panel missing fields: {sorted(missing)}")
    action_names = [str(value) for value in panel["action_names"]]
    signs = [int(value) for value in panel["signs"]]
    strengths = [float(value) for value in panel["strengths"]]
    durations = [int(value) for value in panel["durations"]]
    width = int(panel["active_action_dimensions"])
    if not action_names or len(set(action_names)) != len(action_names):
        raise ValueError(f"{phase} action_names must be nonempty and unique")
    if set(signs) != {-1, 1}:
        raise ValueError(f"{phase} signs must contain exactly -1 and +1")
    if not strengths or any(value <= 0.0 for value in strengths):
        raise ValueError(f"{phase} strengths must be positive")
    if not durations or any(value <= 0 for value in durations):
        raise ValueError(f"{phase} durations must be positive")
    if width <= 0 or width > len(action_names):
        raise ValueError(f"{phase} active_action_dimensions invalid")
    return {
        "action_names": action_names,
        "signs": signs,
        "strengths": strengths,
        "durations": durations,
        "active_action_dimensions": width,
    }


def phase_probe_panels(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    revision = plan.get("protocol_revision")
    if not isinstance(revision, Mapping) or revision.get("name") != V3_REVISION_NAME:
        raise ValueError("phase-specific frontier adapter requires the v3 revision")
    raw = plan.get("phase_probe_panels")
    if not isinstance(raw, Mapping):
        raise ValueError("v3 frontier plan phase_probe_panels missing")
    if set(raw) != set(PHASES):
        raise ValueError("v3 frontier plan must declare exactly upstream/downstream panels")
    return {
        phase: _validate_phase_panel(raw[phase], phase)
        for phase in PHASES
    }


def panel_direction_count(panel: Mapping[str, Any]) -> int:
    normalized = _validate_phase_panel(panel, "panel")
    n = len(normalized["action_names"])
    width = int(normalized["active_action_dimensions"])
    signs = len(normalized["signs"])
    return math.comb(n, width) * (signs ** width)


def panel_variant_count(panel: Mapping[str, Any]) -> int:
    normalized = _validate_phase_panel(panel, "panel")
    return (
        panel_direction_count(normalized)
        * len(normalized["strengths"])
        * len(normalized["durations"])
    )


def required_label_shard_count(
    candidate_count: int,
    *,
    max_candidates_per_process: int = DEFAULT_MAX_CANDIDATES_PER_LABEL_PROCESS,
) -> int:
    candidate_count = int(candidate_count)
    maximum = int(max_candidates_per_process)
    if candidate_count <= 0 or maximum <= 0:
        raise ValueError("candidate_count and max_candidates_per_process must be positive")
    return max(1, math.ceil(candidate_count / maximum))


def _phase_dir(acquisition_dir: Path, phase: str) -> Path:
    return Path(acquisition_dir) / f"phase_{phase}"


def _completed_phase_acquisition(
    *,
    phase_dir: Path,
    phase: str,
    panel: Mapping[str, Any],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
) -> dict[str, Any] | None:
    if not phase_dir.exists():
        return None
    catalog_path = phase_dir / "catalog.json"
    protocol_path = phase_dir / "protocol.json"
    if not catalog_path.is_file() or not protocol_path.is_file():
        raise RuntimeError(
            "phase-specific acquisition directory exists without a completed "
            f"protocol/catalog; preserve the engineering-failure artifact: {phase_dir}"
        )
    catalog = _read_json(catalog_path)
    protocol = _read_json(protocol_path)
    validate_unified_boundary_catalog(
        catalog,
        policy_record=policy_record,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    if protocol.get("protocol_sha256") != catalog.get("protocol_sha256"):
        raise ValueError(f"{phase} acquisition protocol/catalog SHA drift")
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{phase} acquisition contains no candidates")
    if any(str(row.get("phase")) != phase for row in entries):
        raise ValueError(f"{phase} acquisition contains another phase")
    normalized = _validate_phase_panel(panel, phase)
    if [str(value) for value in protocol.get("selected_action_names", [])] != normalized["action_names"]:
        raise ValueError(f"{phase} acquisition action-name panel drift")
    if [int(value) for value in protocol.get("selected_signs", [])] != sorted(normalized["signs"]):
        raise ValueError(f"{phase} acquisition sign panel drift")
    if [float(value) for value in protocol.get("strengths", [])] != normalized["strengths"]:
        raise ValueError(f"{phase} acquisition strength panel drift")
    if [int(value) for value in protocol.get("durations", [])] != normalized["durations"]:
        raise ValueError(f"{phase} acquisition duration panel drift")
    actual_width = int(protocol.get("active_action_dimensions", 1))
    if actual_width != normalized["active_action_dimensions"]:
        raise ValueError(f"{phase} acquisition perturbation dimensionality drift")
    return catalog


def _merged_acquisition_payloads(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    role: str,
    phase_catalogs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    panels = phase_probe_panels(plan)
    if set(phase_catalogs) != set(PHASES):
        raise ValueError("phase-specific merge requires both phase catalogs")
    catalogs = {phase: dict(phase_catalogs[phase]) for phase in PHASES}
    first = catalogs["upstream"]
    for phase in PHASES:
        catalog = catalogs[phase]
        for field in (
            "iteration",
            "policy_name",
            "policy_actor_sha256",
            "policy_payload_sha256",
            "frozen_unified_manifest_sha256",
            "source_tube_manifest_sha256",
        ):
            if catalog.get(field) != first.get(field):
                raise ValueError(f"phase-specific acquisition {field} drift across phases")

    root_seed = int(plan["seeds"][role]["acquisition"])
    phase_protocols = {}
    for phase in PHASES:
        catalog = catalogs[phase]
        phase_protocols[phase] = {
            "directory": f"phase_{phase}",
            "protocol_sha256": str(catalog["protocol_sha256"]),
            "panel": panels[phase],
            "candidate_count": int(catalog["candidate_count"]),
            "environment_interactions": int(catalog["environment_interactions"]),
        }

    protocol = {
        "schema": "jit_unified_boundary_protocol_v1",
        "status": "predeclared",
        "purpose": "phase_specific_policy_conditioned_real_dynamics_frontier_acquisition",
        "split": "train",
        "iteration": int(first["iteration"]),
        "policy_name": str(first["policy_name"]),
        "policy_actor_sha256": str(first["policy_actor_sha256"]),
        "policy_payload_sha256": str(first["policy_payload_sha256"]),
        "frozen_unified_manifest_sha256": str(first["frozen_unified_manifest_sha256"]),
        "source_tube_manifest_sha256": str(first["source_tube_manifest_sha256"]),
        "plan": str(plan_path),
        "plan_sha256": str(plan["plan_sha256"]),
        "logical_role": str(role),
        "protocol_seed": root_seed,
        "phase_protocols": phase_protocols,
        "phase_seed_semantics": (
            "the unchanged role-level acquisition seed is reused as each phase-local "
            "collector root; low-level variant indices are phase-local and the frozen "
            "policy is deterministic"
        ),
        "state_generation": (
            "reset exact predeclared Tube_1 newest-shell parent; apply only the "
            "phase-specific bounded action perturbation declared before v3 outcomes; "
            "advance only through authoritative env.step; reject terminal/nonfinite/"
            "phase-crossing states"
        ),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    protocol_sha = canonical_sha256(protocol)
    protocol_with_sha = {**protocol, "protocol_sha256": protocol_sha}

    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_states: set[str] = set()
    exclusions: Counter[str] = Counter()
    phase_attempted: dict[str, int] = {}
    phase_candidates: dict[str, int] = {}
    phase_exclusions: dict[str, dict[str, int]] = {}
    attempted = 0
    interactions = 0
    maximum_interactions = 0
    anchor_count = 0

    for phase in PHASES:
        catalog = catalogs[phase]
        attempted += int(catalog["attempted_candidate_count"])
        interactions += int(catalog["environment_interactions"])
        maximum_interactions += int(catalog["maximum_environment_interactions"])
        anchor_count += int(catalog["anchor_count"])
        phase_attempted[phase] = int(catalog["phase_attempted_candidate_counts"][phase])
        phase_candidates[phase] = int(catalog["phase_candidate_counts"][phase])
        phase_exclusions[phase] = {
            str(key): int(value)
            for key, value in dict(catalog["phase_exclusion_counts"][phase]).items()
        }
        exclusions.update(
            {str(key): int(value) for key, value in dict(catalog["exclusion_counts"]).items()}
        )
        for source in catalog["entries"]:
            row = dict(source)
            candidate_id = str(row["candidate_id"])
            state_sha = str(row["state_sha256"])
            if candidate_id in seen_ids:
                raise ValueError("duplicate candidate id across phase acquisitions")
            if state_sha in seen_states:
                raise ValueError("duplicate physical state across phase acquisitions")
            seen_ids.add(candidate_id)
            seen_states.add(state_sha)
            row["phase_acquisition_protocol_sha256"] = str(row["protocol_sha256"])
            row["protocol_sha256"] = protocol_sha
            row["source_bank"] = f"phase_{phase}/{row['source_bank']}"
            entries.append(row)

    report = {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": int(first["iteration"]),
        "policy_name": str(first["policy_name"]),
        "policy_actor_sha256": str(first["policy_actor_sha256"]),
        "policy_payload_sha256": str(first["policy_payload_sha256"]),
        "frozen_unified_manifest_sha256": str(first["frozen_unified_manifest_sha256"]),
        "source_tube_manifest_sha256": str(first["source_tube_manifest_sha256"]),
        "protocol_sha256": protocol_sha,
        "frontier_score_ceiling": 1.0,
        "anchor_count": anchor_count,
        "attempted_candidate_count": attempted,
        "candidate_count": len(entries),
        "phase_attempted_candidate_counts": phase_attempted,
        "phase_candidate_counts": phase_candidates,
        "phase_exclusion_counts": phase_exclusions,
        "environment_interactions": interactions,
        "maximum_environment_interactions": maximum_interactions,
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
        "entries": entries,
    }
    return protocol_with_sha, report


def _write_merged_acquisition(
    *,
    acquisition_dir: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    role: str,
    phase_catalogs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    protocol, report = _merged_acquisition_payloads(
        plan_path=plan_path,
        plan=plan,
        role=role,
        phase_catalogs=phase_catalogs,
    )
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    _write(acquisition_dir / "protocol.json", protocol)
    _write(acquisition_dir / "catalog.json", report)
    _write(
        acquisition_dir / "summary.json",
        {key: value for key, value in report.items() if key != "entries"},
    )
    return report


def _runtime(
    *,
    record: Mapping[str, Any],
    artifact: Any,
) -> tuple[Any, Any]:
    runtime_config, runtime_artifact, env = build_unified_formal_environment(
        Path(str(record["formal_config"]))
    )
    if runtime_artifact.manifest["manifest_sha256"] != artifact.manifest["manifest_sha256"]:
        raise ValueError("phase-specific frontier runtime Tube drift")
    if env._bundle.xml_sha256 != record["xml_sha256"]:
        raise ValueError("phase-specific frontier runtime XML drift")
    payload = load_checkpoint(
        Path(str(record["checkpoint"])),
        expected=checkpoint_identity(runtime_config, env),
    )
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))
    return env, policy


def _label_execution_limit(plan: Mapping[str, Any]) -> int:
    raw = plan.get("label_execution")
    if not isinstance(raw, Mapping):
        return DEFAULT_MAX_CANDIDATES_PER_LABEL_PROCESS
    maximum = int(
        raw.get(
            "max_candidates_per_independent_process",
            DEFAULT_MAX_CANDIDATES_PER_LABEL_PROCESS,
        )
    )
    if maximum <= 0:
        raise ValueError("label_execution max candidates per process must be positive")
    return maximum


def _shard_instruction(
    *,
    plan_path: Path,
    output_dir: Path,
    role: str,
    candidate_count: int,
    shard_count: int,
    maximum: int,
) -> str:
    return (
        f"v3 {role} acquisition completed with {candidate_count} candidates, exceeding "
        f"the predeclared {maximum}-candidate monolithic-label limit. Stop before "
        "continuation labeling and use independent-process shards to avoid repeating the "
        "historical 3720-candidate OOM. Run: "
        f"python JIT/cli/run_frontier_label_shards.py run-all --plan {plan_path} "
        f"--role-root {output_dir} --role {role}. "
        f"This will execute {shard_count} contiguous shards and merge them by global "
        "candidate index. Then rerun the iteration workflow."
    )


def run_phase_specific_frontier_role(
    *,
    plan_path: Path,
    role: str,
    output_dir: Path,
) -> dict[str, Any]:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "role_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        _verify_self_hash(existing, "role_manifest_sha256")
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = _load_plan(Path(plan_path))
    panels = phase_probe_panels(plan)
    selected, record, frozen_path = _selected_policy(Path(str(plan["selected_policy"])))
    artifact = _source_tube(selected, record, Path(str(plan["source_tube"])))
    if artifact.manifest["manifest_sha256"] != plan["source_tube_manifest_sha256"]:
        raise ValueError("phase-specific frontier role source Tube drift")
    anchors = _anchors_from_plan(plan, artifact, role)
    frozen_sha = file_sha256(frozen_path)
    acquisition_dir = output_dir / "acquisition"
    labels_dir = output_dir / "labels"
    seeds = plan["seeds"][role]
    panel = plan["fixed_probe_panel"]

    acquisition = None
    if (acquisition_dir / "catalog.json").is_file():
        acquisition = _completed_acquisition(
            acquisition_dir=acquisition_dir,
            policy_record=record,
            frozen_manifest_sha256=frozen_sha,
        )

    runtime_env = None
    runtime_policy = None

    def ensure_runtime() -> tuple[Any, Any]:
        nonlocal runtime_env, runtime_policy
        if runtime_env is None or runtime_policy is None:
            runtime_env, runtime_policy = _runtime(record=record, artifact=artifact)
        return runtime_env, runtime_policy

    if acquisition is None:
        phase_catalogs: dict[str, Mapping[str, Any]] = {}
        for phase in PHASES:
            phase_dir = _phase_dir(acquisition_dir, phase)
            completed = _completed_phase_acquisition(
                phase_dir=phase_dir,
                phase=phase,
                panel=panels[phase],
                policy_record=record,
                frozen_manifest_sha256=frozen_sha,
            )
            if completed is None:
                env, policy = ensure_runtime()
                phase_anchors = tuple(anchor for anchor in anchors if anchor.phase == phase)
                if not phase_anchors:
                    raise ValueError(f"v3 plan has no {role} anchors in {phase}")
                selected_panel = panels[phase]
                completed = collect_unified_boundary_candidates(
                    phase_anchors,
                    phase_dir,
                    env=env,
                    policy=policy,
                    policy_record=record,
                    frozen_manifest_sha256=frozen_sha,
                    protocol_seed=int(seeds["acquisition"]),
                    frontier_score_ceiling=1.0,
                    strengths=tuple(selected_panel["strengths"]),
                    durations=tuple(selected_panel["durations"]),
                    action_names=tuple(selected_panel["action_names"]),
                    signs=tuple(selected_panel["signs"]),
                    active_action_dimensions=int(
                        selected_panel["active_action_dimensions"]
                    ),
                )
            phase_catalogs[phase] = completed
        acquisition = _write_merged_acquisition(
            acquisition_dir=acquisition_dir,
            plan_path=Path(plan_path),
            plan=plan,
            role=role,
            phase_catalogs=phase_catalogs,
        )

    acquisition_phase_support = _acquisition_phase_support(acquisition, role=role)
    labeling = _completed_labeling(labels_dir)

    if labeling is None:
        candidate_count = int(acquisition["candidate_count"])
        maximum = _label_execution_limit(plan)
        shard_count = required_label_shard_count(
            candidate_count,
            max_candidates_per_process=maximum,
        )
        if shard_count > 1:
            shard_plan = {
                "schema": "jit_frontier_label_shard_plan_v1",
                "status": "required_before_labeling",
                "plan": str(plan_path),
                "plan_sha256": str(plan["plan_sha256"]),
                "role": role,
                "candidate_count": candidate_count,
                "max_candidates_per_independent_process": maximum,
                "shard_count": shard_count,
                "execution_only": True,
                "logical_label_protocol_unchanged": True,
                "global_candidate_index_prng_identity_preserved": True,
                "historical_reason": (
                    "a 3720-candidate continuation bank previously OOMed in one process "
                    "and succeeded as four independent 930-candidate shards"
                ),
                "training_transitions": 0,
                "test_data_used": False,
            }
            shard_plan["shard_plan_sha256"] = canonical_sha256(shard_plan)
            shard_plan_path = output_dir / "label_shard_plan.json"
            if shard_plan_path.is_file():
                existing = _read_json(shard_plan_path)
                _verify_self_hash(existing, "shard_plan_sha256")
                if existing != shard_plan:
                    raise ValueError("frontier label shard plan drift")
            else:
                _write(shard_plan_path, shard_plan)
            raise RuntimeError(
                _shard_instruction(
                    plan_path=Path(plan_path),
                    output_dir=output_dir,
                    role=role,
                    candidate_count=candidate_count,
                    shard_count=shard_count,
                    maximum=maximum,
                )
            )

        env, policy = ensure_runtime()
        labeling = label_unified_continuations(
            acquisition_dir / "catalog.json",
            labels_dir,
            env=env,
            policy=policy,
            policy_record=record,
            frozen_manifest_sha256=frozen_sha,
            max_ticks=int(panel["max_label_ticks"]),
            protocol_seed=int(seeds["labeling"]),
        )

    logical_rows = _logical_rows(
        labels_dir=labels_dir,
        output_dir=output_dir,
        role=role,
        labeling=labeling,
    )
    phase_counts = _phase_counts(logical_rows, role=role)

    manifest = {
        "schema": ROLE_SCHEMA,
        "status": "completed",
        "iteration": int(plan["iteration"]),
        "role": role,
        "logical_split": role,
        "legacy_low_level_split_marker": "train",
        "legacy_marker_is_not_logical_data_role": True,
        "plan": str(plan_path),
        "plan_sha256": str(plan["plan_sha256"]),
        "selected_policy_sha256": str(plan["selected_policy_sha256"]),
        "policy_actor_sha256": str(record["actor_sha256"]),
        "policy_payload_sha256": str(record["payload_sha256"]),
        "source_tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "phase_specific_probe_panels": panels,
        "acquisition_phase_support": acquisition_phase_support,
        "logical_labels": str(output_dir / "logical_labels.json"),
        "logical_labels_file_sha256": file_sha256(output_dir / "logical_labels.json"),
        "source_acquisition_catalog": str(acquisition_dir / "catalog.json"),
        "source_acquisition_catalog_sha256": file_sha256(acquisition_dir / "catalog.json"),
        "source_label_summary": str(labels_dir / "summary.json"),
        "source_label_summary_sha256": file_sha256(labels_dir / "summary.json"),
        "phase_counts": phase_counts,
        "environment_interactions": int(acquisition["environment_interactions"])
        + int(labeling["environment_interactions"]),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "role_locked_before_next_policy_training": True,
            "train_rows_may_fit_fields": role == "train",
            "rows_may_calibrate_thresholds": role == "calibration",
            "rows_may_gate_next_policy": role == "acceptance",
            "rows_may_enter_tube": role == "train",
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }
    manifest["role_manifest_sha256"] = canonical_sha256(manifest)
    _write(manifest_path, manifest)
    return manifest


def _frontier_label_context(
    *,
    plan_path: Path,
    role_root: Path,
    role: str,
) -> tuple[dict[str, Any], Mapping[str, Any], Any, Path, str]:
    plan = _load_plan(Path(plan_path))
    phase_probe_panels(plan)
    selected, record, frozen_path = _selected_policy(Path(str(plan["selected_policy"])))
    artifact = _source_tube(selected, record, Path(str(plan["source_tube"])))
    acquisition_dir = Path(role_root) / "acquisition"
    frozen_sha = file_sha256(frozen_path)
    acquisition = _completed_acquisition(
        acquisition_dir=acquisition_dir,
        policy_record=record,
        frozen_manifest_sha256=frozen_sha,
    )
    if acquisition is None:
        raise RuntimeError("frontier shard labeling requires completed merged acquisition")
    _acquisition_phase_support(acquisition, role=role)
    return plan, record, artifact, acquisition_dir, frozen_sha


def run_frontier_label_shard(
    *,
    plan_path: Path,
    role_root: Path,
    role: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    plan, record, artifact, acquisition_dir, frozen_sha = _frontier_label_context(
        plan_path=plan_path,
        role_root=role_root,
        role=role,
    )
    output_dir = (
        Path(role_root)
        / "label_shards"
        / f"shard_{int(shard_index):03d}_of_{int(shard_count):03d}"
    )
    if (output_dir / "summary.json").is_file():
        summary = _read_json(output_dir / "summary.json")
        if summary.get("status") == "completed_shard":
            return summary
        raise RuntimeError(f"existing frontier label shard is not completed: {output_dir}")
    env, policy = _runtime(record=record, artifact=artifact)
    seeds = plan["seeds"][role]
    max_ticks = int(plan["fixed_probe_panel"]["max_label_ticks"])
    return label_unified_continuation_shard(
        acquisition_dir / "catalog.json",
        output_dir,
        env=env,
        policy=policy,
        policy_record=record,
        frozen_manifest_sha256=frozen_sha,
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        max_ticks=max_ticks,
        protocol_seed=int(seeds["labeling"]),
    )


def merge_frontier_label_shards(
    *,
    plan_path: Path,
    role_root: Path,
    role: str,
    shard_count: int,
) -> dict[str, Any]:
    plan, _record, _artifact, acquisition_dir, _frozen_sha = _frontier_label_context(
        plan_path=plan_path,
        role_root=role_root,
        role=role,
    )
    labels_dir = Path(role_root) / "labels"
    if (labels_dir / "summary.json").is_file():
        summary = _completed_labeling(labels_dir)
        if summary is None:
            raise RuntimeError("frontier merged labels unexpectedly unavailable")
        return summary
    shard_dirs = [
        Path(role_root)
        / "label_shards"
        / f"shard_{index:03d}_of_{int(shard_count):03d}"
        for index in range(int(shard_count))
    ]
    return merge_unified_continuation_shards(
        acquisition_dir / "catalog.json",
        shard_dirs,
        labels_dir,
    )
