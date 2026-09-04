"""Iteration-generic frontier data protocol for k >= 1.

Later envelope iterations reuse one frozen continuation architecture and create
three outcome-blind, parent-disjoint frontier roles before the next policy is
trained:

* ``train``       -- the only rows allowed to fit C_up^k/C_down^k;
* ``calibration`` -- threshold calibration only, never Tube entries;
* ``acceptance``  -- locked baseline audit rows for pi_k -> pi_(k+1) selection.

The historical low-level acquisition/label artifacts still carry
``split=train``. This adapter never rewrites them. It creates a self-hashed
logical role artifact consumed by later iteration-generic stages.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import jax

from .checkpoint import load_checkpoint
from .config import file_sha256
from .ppo import make_checkpoint_policy
from .soft_tube import SoftTubeArtifact, load_soft_tube
from .unified_boundary import TubeBoundaryAnchor, collect_unified_boundary_candidates
from .unified_continuation_labels import (
    label_unified_continuations,
    validate_unified_boundary_catalog,
)
from .unified_formal import build_unified_formal_environment, load_unified_formal_config
from .unified_policy_freeze import load_frozen_unified_manifest
from .unified_training import checkpoint_identity


PLAN_SCHEMA = "jit_iterative_frontier_plan_v1"
ROLE_SCHEMA = "jit_iterative_frontier_role_v1"
ROLES = ("train", "calibration", "acceptance")
ROLE_PATTERN = ("train", "train", "train", "calibration", "acceptance")
DEFAULT_STRENGTHS = (0.025, 0.05, 0.10)
DEFAULT_DURATIONS = (4, 8, 16, 32)
DEFAULT_SEEDS = {
    "train": (9_521_101, 9_521_201),
    "calibration": (9_522_101, 9_522_201),
    "acceptance": (9_523_101, 9_523_201),
}


def exact_state_disjoint_role_rows(
    *,
    train: list[Mapping[str, Any]],
    calibration: list[Mapping[str, Any]],
    acceptance: list[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Assign duplicate physical states by TRAIN > CALIBRATION > ACCEPTANCE.

    The assignment depends only on the locked physical-state identity, never on
    a continuation outcome.  This preserves all TRAIN support while producing
    exact-state-disjoint holdout views without new environment interactions.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    exclusions: dict[str, int] = {}
    claimed: set[str] = set()
    for role, rows in (
        ("train", train),
        ("calibration", calibration),
        ("acceptance", acceptance),
    ):
        kept: list[dict[str, Any]] = []
        local: set[str] = set()
        excluded = 0
        for source in rows:
            row = dict(source)
            if row.get("logical_role") != role or row.get("split") != role:
                raise ValueError(f"{role} logical role drift during exact-state partition")
            state = str(row.get("state_sha256", ""))
            if not state:
                raise ValueError(f"{role} state identity missing during exact-state partition")
            if state in local:
                raise ValueError(f"{role} contains duplicate physical states")
            local.add(state)
            if state in claimed:
                excluded += 1
                continue
            kept.append(row)
        if not kept:
            raise ValueError(f"exact-state partition emptied {role}")
        result[role] = kept
        exclusions[role] = excluded
        claimed.update(str(row["state_sha256"]) for row in kept)
    return result, exclusions


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def _selected_policy(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = Path(path)
    selected = _read(path)
    if (
        not isinstance(selected, dict)
        or selected.get("schema") != "jit_selected_iteration_policy_v1"
    ):
        raise ValueError("iterative frontier requires selected-policy artifact")
    if (
        selected.get("status") != "selected"
        or selected.get("engineering_selection") is not True
    ):
        raise ValueError("iterative frontier policy is not selected")
    _verify_self_hash(selected, "selection_sha256")
    frozen_path = Path(str(selected["frozen_policy"]))
    if file_sha256(frozen_path) != selected["frozen_policy_file_sha256"]:
        raise ValueError("selected frozen-policy file drift")
    frozen = load_frozen_unified_manifest(frozen_path)
    record = dict(frozen["policy"])
    for field in (
        "actor_sha256",
        "payload_sha256",
        "xml_sha256",
        "formal_config_sha256",
    ):
        if record.get(field) != selected.get(field):
            raise ValueError(f"selected frozen policy {field} drift")
    if int(record["iteration"]) != int(selected["iteration"]):
        raise ValueError("selected frozen policy iteration drift")
    return selected, record, frozen_path


def _source_tube(
    selected: Mapping[str, Any],
    record: Mapping[str, Any],
    source_tube: Path,
) -> SoftTubeArtifact:
    formal = load_unified_formal_config(Path(str(record["formal_config"])))
    if formal.config_sha256 != record["formal_config_sha256"]:
        raise ValueError("selected formal config drift")
    artifact = load_soft_tube(Path(source_tube))
    if artifact.manifest.get("manifest_sha256") != formal.soft_tube_manifest_sha256:
        raise ValueError("selected policy was not trained from the declared source Tube")
    if int(artifact.manifest.get("iteration", 0)) != int(selected["iteration"]):
        raise ValueError("source Tube iteration must equal selected policy iteration")
    return artifact


def _phase_indexed_rows(
    artifact: SoftTubeArtifact,
) -> list[tuple[int, int, str, Mapping[str, Any]]]:
    counters = Counter()
    result = []
    for global_index, row in enumerate(artifact.entries):
        phase = str(row["phase"])
        if phase not in ("upstream", "downstream"):
            raise ValueError("source Tube phase drift")
        local = int(counters[phase])
        counters[phase] += 1
        result.append((global_index, local, phase, row))
    return result


def _frontier_pool(
    artifact: SoftTubeArtifact,
) -> dict[str, list[tuple[int, int, Mapping[str, Any]]]]:
    indexed = _phase_indexed_rows(artifact)
    core_count = int(artifact.manifest.get("core_retained_count", -1))
    if core_count <= 0 or core_count >= len(artifact.entries):
        raise ValueError(
            "iterative frontier requires a Tube with a nonempty newest expansion shell"
        )
    by_phase: dict[str, list[tuple[int, int, Mapping[str, Any]]]] = {
        "upstream": [],
        "downstream": [],
    }
    for global_index, local_index, phase, row in indexed:
        if global_index >= core_count:
            by_phase[phase].append((global_index, local_index, row))
    for phase in by_phase:
        if not by_phase[phase]:
            raise ValueError(f"newest Tube shell has no {phase} support")
    return by_phase


def prepare_frontier_plan(
    *,
    selected_policy: Path,
    source_tube: Path,
    output: Path,
    max_parent_groups_per_phase: int = 15,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"frontier plan already exists: {output}")
    selected, record, _frozen_path = _selected_policy(Path(selected_policy))
    artifact = _source_tube(selected, record, Path(source_tube))
    pools = _frontier_pool(artifact)

    anchors: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for phase, phase_index in (("upstream", 0), ("downstream", 1)):
        ordered = sorted(
            pools[phase],
            key=lambda item: (
                float(item[2]["value_score"]),
                str(item[2]["parent_group_id"]),
                str(item[2]["state_sha256"]),
            ),
        )
        parent_unique = []
        seen_groups: set[str] = set()
        seen_states: set[str] = set()
        for global_index, local_index, row in ordered:
            group = str(row["parent_group_id"])
            state = str(row["state_sha256"])
            if group in seen_groups or state in seen_states:
                continue
            seen_groups.add(group)
            seen_states.add(state)
            parent_unique.append((global_index, local_index, row))
            if len(parent_unique) >= int(max_parent_groups_per_phase):
                break
        if len(parent_unique) < 5:
            raise ValueError(
                f"automatic iteration needs >=5 newest-shell parent groups in {phase}; "
                f"found {len(parent_unique)}. Stop for a new parent-generation decision."
            )

        phase_counts = Counter()
        for index, (global_index, local_index, row) in enumerate(parent_unique):
            role = ROLE_PATTERN[index % len(ROLE_PATTERN)]
            phase_counts[role] += 1
            anchors.append(
                {
                    "role": role,
                    "phase": phase,
                    "phase_index": phase_index,
                    "entry_index": int(local_index),
                    "global_index": int(global_index),
                    "state_sha256": str(row["state_sha256"]),
                    "parent_group_id": str(row["parent_group_id"]),
                    "value_score": float(row["value_score"]),
                    "sampling_weight": float(row["sampling_weight"]),
                }
            )
        if (
            phase_counts["train"] < 3
            or phase_counts["calibration"] < 1
            or phase_counts["acceptance"] < 1
        ):
            raise ValueError(
                f"automatic frontier split support insufficient in {phase}"
            )
        counts[phase] = {role: int(phase_counts[role]) for role in ROLES}

    plan = {
        "schema": PLAN_SCHEMA,
        "status": "predeclared_before_frontier_outcomes",
        "iteration": int(selected["iteration"]),
        "policy_name": str(selected["policy_name"]),
        "selected_policy": str(selected_policy),
        "selected_policy_sha256": str(selected["selection_sha256"]),
        "policy_actor_sha256": str(selected["actor_sha256"]),
        "policy_payload_sha256": str(selected["payload_sha256"]),
        "source_tube": str(source_tube),
        "source_tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "source_tube_entry_count": len(artifact.entries),
        "source_tube_core_retained_count": int(
            artifact.manifest["core_retained_count"]
        ),
        "frontier_definition": (
            "newest_expansion_shell_only_lowest_score_parent_unique"
        ),
        "role_pattern": list(ROLE_PATTERN),
        "role_semantics": {
            "train": "C^k fitting and candidate Tube expansion only",
            "calibration": "phase-threshold calibration only; never training or Tube entry",
            "acceptance": "pre-next-policy locked baseline audit only",
        },
        "role_parent_group_counts": counts,
        "fixed_probe_panel": {
            "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
            "signs": [-1, 1],
            "strengths": list(DEFAULT_STRENGTHS),
            "durations": list(DEFAULT_DURATIONS),
            "max_label_ticks": 400,
        },
        "seeds": {
            role: {
                "acquisition": DEFAULT_SEEDS[role][0],
                "labeling": DEFAULT_SEEDS[role][1],
            }
            for role in ROLES
        },
        "anchors": anchors,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "outcome_blind_split_predeclared": True,
            "continuation_fields_fitted": False,
            "next_tube_constructed": False,
            "next_policy_trained": False,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    _write(output, plan)
    return plan


def _load_plan(path: Path) -> dict[str, Any]:
    plan = _read(path)
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid iterative frontier plan")
    if plan.get("status") != "predeclared_before_frontier_outcomes":
        raise ValueError("iterative frontier plan status drift")
    _verify_self_hash(plan, "plan_sha256")
    return plan


def _anchors_from_plan(
    plan: Mapping[str, Any],
    artifact: SoftTubeArtifact,
    role: str,
) -> tuple[TubeBoundaryAnchor, ...]:
    if role not in ROLES:
        raise ValueError(f"unsupported frontier role: {role}")
    indexed = _phase_indexed_rows(artifact)
    by_global = {
        global_index: (local, phase, row)
        for global_index, local, phase, row in indexed
    }
    result = []
    for declared in plan["anchors"]:
        if declared["role"] != role:
            continue
        global_index = int(declared["global_index"])
        if global_index not in by_global:
            raise ValueError("frontier plan global index drift")
        local, phase, row = by_global[global_index]
        if (
            phase != declared["phase"]
            or local != int(declared["entry_index"])
        ):
            raise ValueError("frontier plan phase/local index drift")
        for field in ("state_sha256", "parent_group_id"):
            if str(row[field]) != str(declared[field]):
                raise ValueError(f"frontier plan anchor {field} drift")
        if float(row["value_score"]) != float(declared["value_score"]):
            raise ValueError("frontier plan anchor value score drift")
        result.append(
            TubeBoundaryAnchor(
                phase=phase,
                phase_index=int(declared["phase_index"]),
                entry_index=local,
                global_index=global_index,
                row=row,
            )
        )
    if not result:
        raise ValueError(f"frontier plan has no {role} anchors")
    return tuple(result)


def _completed_acquisition(
    *,
    acquisition_dir: Path,
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
) -> dict[str, Any] | None:
    if not acquisition_dir.exists():
        return None
    catalog_path = acquisition_dir / "catalog.json"
    if not catalog_path.is_file():
        raise RuntimeError(
            "frontier acquisition directory exists without a completed catalog; "
            f"preserve this engineering-failure artifact before retrying: {acquisition_dir}"
        )
    catalog = _read(catalog_path)
    if not isinstance(catalog, dict):
        raise ValueError("frontier acquisition catalog must be a JSON object")
    validate_unified_boundary_catalog(
        catalog,
        policy_record=policy_record,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    if int(catalog.get("candidate_count", 0)) <= 0:
        raise ValueError("completed frontier acquisition contains no candidates")
    return catalog


def _acquisition_phase_support(
    acquisition: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, dict[str, int]]:
    """Reject structurally impossible roles before expensive continuation labeling.

    This gate uses only unlabeled acquisition structure. It does not inspect
    continuation outcomes. TRAIN ultimately requires >=20 positive and >=20
    negative labels across >=3 parent groups, so <40 acquired candidates or <3
    groups in either phase can never pass. CALIBRATION/ACCEPTANCE require both
    label classes, so <2 candidates or no parent group can never pass either.
    """
    entries = acquisition.get("entries")
    if not isinstance(entries, list):
        raise ValueError("frontier acquisition entries must be a JSON array")
    minimum_candidates = 40 if role == "train" else 2
    minimum_groups = 3 if role == "train" else 1
    support: dict[str, dict[str, int]] = {}
    for phase in ("upstream", "downstream"):
        rows = [row for row in entries if row.get("phase") == phase]
        groups = {str(row.get("parent_group_id", "")) for row in rows}
        counts = {
            "candidate_count": len(rows),
            "parent_group_count": len(groups),
            "minimum_candidate_count_before_labeling": minimum_candidates,
            "minimum_parent_group_count_before_labeling": minimum_groups,
        }
        support[phase] = counts
        if len(rows) < minimum_candidates or len(groups) < minimum_groups:
            raise ValueError(
                f"{role.upper()} acquisition support not ready in {phase} before continuation labeling: "
                f"{counts}; aggregate_exclusions={acquisition.get('exclusion_counts', {})}. "
                "Stop before expensive labeling and open a new predeclared frontier/parent-generation decision."
            )
    return support


def _completed_labeling(labels_dir: Path) -> dict[str, Any] | None:
    if not labels_dir.exists():
        return None
    summary_path = labels_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            "frontier labeling directory exists without summary.json; "
            f"preserve this engineering-failure artifact before retrying: {labels_dir}"
        )
    summary = _read(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("frontier label summary must be a JSON object")
    status = str(summary.get("status", ""))
    if status == "engineering_error":
        raise RuntimeError(
            "previous frontier labeling attempt ended in engineering_error: "
            f"{summary.get('error', 'unknown error')}; "
            f"completed_candidate_count={summary.get('completed_candidate_count')}; "
            f"environment_interactions={summary.get('environment_interactions')}"
        )
    if status != "completed":
        raise RuntimeError(
            f"frontier labeling has non-completed status {status!r}: {summary_path}"
        )
    labels_path = labels_dir / "labels.json"
    if not labels_path.is_file():
        raise RuntimeError(
            f"completed frontier labeling is missing labels.json: {labels_path}"
        )
    raw_labels = _read(labels_path)
    if not isinstance(raw_labels, list):
        raise ValueError("frontier labels must be a JSON array")
    if len(raw_labels) != int(summary.get("candidate_count", -1)):
        raise ValueError("frontier completed label count drift")
    return summary


def _logical_rows(
    *,
    labels_dir: Path,
    output_dir: Path,
    role: str,
    labeling: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_labels = _read(labels_dir / "labels.json")
    if (
        not isinstance(raw_labels, list)
        or len(raw_labels) != int(labeling["candidate_count"])
    ):
        raise ValueError("frontier role labels count drift")
    logical_rows: list[dict[str, Any]] = []
    for source in raw_labels:
        row = dict(source)
        row["source_legacy_split"] = str(row.get("split", ""))
        row["split"] = role
        row["logical_role"] = role
        logical_rows.append(row)

    logical_payload = {
        "schema": "jit_iterative_frontier_logical_labels_v1",
        "role": role,
        "entries": logical_rows,
    }
    logical_payload["labels_sha256"] = canonical_sha256(logical_payload)
    logical_path = output_dir / "logical_labels.json"
    if logical_path.is_file():
        existing = _read(logical_path)
        _verify_self_hash(existing, "labels_sha256")
        if existing != logical_payload:
            raise ValueError("frontier logical labels drift from completed source labels")
    else:
        _write(logical_path, logical_payload)
    return logical_rows


def _phase_counts(
    logical_rows: list[dict[str, Any]],
    *,
    role: str,
) -> dict[str, dict[str, int]]:
    phase_counts: dict[str, dict[str, int]] = {}
    for phase in ("upstream", "downstream"):
        rows = [row for row in logical_rows if row["phase"] == phase]
        positives = sum(int(row["label"]) for row in rows)
        groups = {str(row["parent_group_id"]) for row in rows}
        phase_counts[phase] = {
            "candidate_count": len(rows),
            "positive_count": positives,
            "negative_count": len(rows) - positives,
            "parent_group_count": len(groups),
        }

    if role == "train":
        for phase, row in phase_counts.items():
            if (
                row["positive_count"] < 20
                or row["negative_count"] < 20
                or row["parent_group_count"] < 3
            ):
                raise ValueError(
                    f"TRAIN continuation support not ready in {phase}: {row}"
                )
    else:
        for phase, row in phase_counts.items():
            if (
                row["positive_count"] <= 0
                or row["negative_count"] <= 0
                or row["parent_group_count"] <= 0
            ):
                raise ValueError(
                    f"{role} continuation support not ready in {phase}: {row}"
                )
    return phase_counts


def run_frontier_role(
    *,
    plan_path: Path,
    role: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run or safely resume one predeclared frontier role.

    Completed acquisition and completed labeling artifacts are reused after
    validation. A previous labeling ``engineering_error`` is surfaced directly
    instead of being hidden behind a generic output-directory error. The
    function never silently reruns a failed labeling attempt.
    """
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "role_manifest.json"
    if manifest_path.is_file():
        existing = _read(manifest_path)
        _verify_self_hash(existing, "role_manifest_sha256")
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = _load_plan(Path(plan_path))
    selected, record, frozen_path = _selected_policy(
        Path(str(plan["selected_policy"]))
    )
    artifact = _source_tube(
        selected,
        record,
        Path(str(plan["source_tube"])),
    )
    if (
        artifact.manifest["manifest_sha256"]
        != plan["source_tube_manifest_sha256"]
    ):
        raise ValueError("frontier role source Tube drift")
    anchors = _anchors_from_plan(plan, artifact, role)

    acquisition_dir = output_dir / "acquisition"
    labels_dir = output_dir / "labels"
    seeds = plan["seeds"][role]
    panel = plan["fixed_probe_panel"]
    frozen_sha = file_sha256(frozen_path)

    acquisition = _completed_acquisition(
        acquisition_dir=acquisition_dir,
        policy_record=record,
        frozen_manifest_sha256=frozen_sha,
    )
    labeling = _completed_labeling(labels_dir)

    runtime_config = None
    runtime_artifact = None
    env = None
    policy = None

    def ensure_runtime() -> tuple[Any, Any]:
        nonlocal runtime_config, runtime_artifact, env, policy
        if env is None or policy is None:
            runtime_config, runtime_artifact, env = (
                build_unified_formal_environment(
                    Path(str(record["formal_config"]))
                )
            )
            if (
                runtime_artifact.manifest["manifest_sha256"]
                != artifact.manifest["manifest_sha256"]
            ):
                raise ValueError("frontier role runtime Tube drift")
            if env._bundle.xml_sha256 != record["xml_sha256"]:
                raise ValueError("frontier role runtime XML drift")
            payload = load_checkpoint(
                Path(str(record["checkpoint"])),
                expected=checkpoint_identity(runtime_config, env),
            )
            policy = jax.jit(
                make_checkpoint_policy(env, payload, deterministic=True)
            )
        return env, policy

    if acquisition is None:
        runtime_env, runtime_policy = ensure_runtime()
        acquisition = collect_unified_boundary_candidates(
            anchors,
            acquisition_dir,
            env=runtime_env,
            policy=runtime_policy,
            policy_record=record,
            frozen_manifest_sha256=frozen_sha,
            protocol_seed=int(seeds["acquisition"]),
            frontier_score_ceiling=1.0,
            strengths=tuple(float(x) for x in panel["strengths"]),
            durations=tuple(int(x) for x in panel["durations"]),
            action_names=tuple(str(x) for x in panel["action_names"]),
            signs=tuple(int(x) for x in panel["signs"]),
        )
        if int(acquisition.get("candidate_count", 0)) <= 0:
            raise ValueError(
                f"frontier {role} acquisition produced no candidates"
            )

    acquisition_phase_support = _acquisition_phase_support(
        acquisition,
        role=role,
    )

    if labeling is None:
        runtime_env, runtime_policy = ensure_runtime()
        labeling = label_unified_continuations(
            acquisition_dir / "catalog.json",
            labels_dir,
            env=runtime_env,
            policy=runtime_policy,
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
        "source_tube_manifest_sha256": str(
            artifact.manifest["manifest_sha256"]
        ),
        "acquisition_phase_support": acquisition_phase_support,
        "logical_labels": str(output_dir / "logical_labels.json"),
        "logical_labels_file_sha256": file_sha256(
            output_dir / "logical_labels.json"
        ),
        "source_acquisition_catalog": str(
            acquisition_dir / "catalog.json"
        ),
        "source_acquisition_catalog_sha256": file_sha256(
            acquisition_dir / "catalog.json"
        ),
        "source_label_summary": str(labels_dir / "summary.json"),
        "source_label_summary_sha256": file_sha256(
            labels_dir / "summary.json"
        ),
        "phase_counts": phase_counts,
        "environment_interactions": int(
            acquisition["environment_interactions"]
        )
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
