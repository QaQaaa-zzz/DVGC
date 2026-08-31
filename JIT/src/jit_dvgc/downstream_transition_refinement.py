"""TRAIN-only downstream transition-band refinement after duration bracketing.

This stage is entered only after the coarse duration-only search has established
that upstream is ready while downstream remains label-degenerate and long
perturbation bursts are being clipped by episode termination.  It refines the
known downstream bracket on the contiguous integer grid 17..32 ticks.

A terminal-causing perturbation is never used directly as a continuation label.
Instead, the last finite nonterminal state before that terminal transition is
saved as a real-dynamics candidate, and the frozen unified policy is restarted
from that exact state for the authoritative continuation label.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from .checkpoint import load_checkpoint
from .config import file_sha256
from .constants import ACTION_ORDER, END_REASONS
from .ppo import make_checkpoint_policy
from .soft_tube import load_soft_tube
from .unified_boundary import TubeBoundaryAnchor, select_tube_boundary_anchors
from .unified_continuation_labels import (
    label_unified_continuations,
    validate_unified_boundary_catalog,
)
from .unified_envelope_snapshot import (
    capture_unified_envelope_snapshot,
    physical_state_sha256,
    save_unified_envelope_snapshot,
)
from .unified_formal import build_unified_formal_environment, load_unified_formal_config
from .unified_policy_freeze import load_frozen_unified_manifest
from .unified_training import checkpoint_identity
from .unified_transition_band_search import (
    phase_transition_band_readiness,
    unique_label_rows,
)
from .upstream_boundary import action_basis_directions, validate_strengths


DOWNSTREAM_REFINEMENT_CONFIG_SCHEMA = "jit_downstream_transition_refinement_config_v1"
DOWNSTREAM_REFINEMENT_PROTOCOL_SCHEMA = "jit_downstream_transition_refinement_protocol_v1"
DOWNSTREAM_REFINEMENT_SUMMARY_SCHEMA = "jit_downstream_transition_refinement_summary_v1"
BOUNDARY_PROTOCOL_SCHEMA = "jit_unified_boundary_protocol_v1"
BOUNDARY_CATALOG_SCHEMA = "jit_unified_boundary_catalog_v1"


def _refinement_protocol_purposes(search_mode: str) -> tuple[str, str]:
    purposes = {
        "contiguous_integer_local_refinement": (
            "downstream_contiguous_terminal_clipped_transition_band_refinement",
            "downstream_terminal_clipped_local_refinement",
        ),
        "fixed_duration_strength_extrapolation": (
            "downstream_fixed_duration_strength_extrapolation",
            "downstream_terminal_clipped_strength_extrapolation",
        ),
        "targeted_hip_positive_boundary_completion": (
            "downstream_targeted_hip_positive_boundary_completion",
            "downstream_terminal_clipped_hip_positive_boundary_completion",
        ),
    }
    try:
        return purposes[str(search_mode)]
    except KeyError as exc:
        raise ValueError("unsupported downstream refinement search_mode") from exc


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _repository_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("downstream refinement requires a Git checkout") from exc


def _truth(value: Any) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _integer(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)))


def _finite_state(state: Any, action: Any | None = None) -> bool:
    arrays = (
        np.asarray(jax.device_get(state.data.qpos)),
        np.asarray(jax.device_get(state.data.qvel)),
        np.asarray(jax.device_get(state.obs["state"])),
    )
    if action is not None:
        arrays = (*arrays, np.asarray(jax.device_get(action)))
    return all(np.isfinite(value).all() for value in arrays)


def load_downstream_refinement_config(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    if raw.get("schema") != DOWNSTREAM_REFINEMENT_CONFIG_SCHEMA:
        raise ValueError("unsupported downstream refinement config schema")
    if int(raw.get("iteration", -1)) != 0:
        raise ValueError("current downstream refinement is locked to iteration 0")
    if not str(raw.get("frozen_policy", "")) or not str(raw.get("output_dir", "")):
        raise ValueError("downstream refinement requires frozen_policy and output_dir")

    prior = raw.get("prior_search")
    if not isinstance(prior, Mapping):
        raise ValueError("downstream refinement requires prior_search")
    for key in (
        "root",
        "expected_status",
        "expected_protocol_sha256",
        "expected_accumulated_unique_label_count",
        "expected_upstream",
        "expected_downstream",
    ):
        if key not in prior:
            raise ValueError(f"prior_search missing {key}")
    if prior["expected_status"] != "search_exhausted":
        raise ValueError("downstream refinement must inherit an exhausted coarse search")
    if prior["expected_upstream"].get("ready") is not True:
        raise ValueError("downstream refinement requires upstream already ready")
    if prior["expected_downstream"].get("ready") is not False:
        raise ValueError("downstream refinement requires downstream unresolved")

    acquisition = raw.get("fixed_acquisition")
    if not isinstance(acquisition, Mapping):
        raise ValueError("downstream refinement requires fixed_acquisition")
    if acquisition.get("phase") != "downstream":
        raise ValueError("downstream refinement may probe downstream only")
    if int(acquisition.get("anchors_per_phase", 0)) <= 0:
        raise ValueError("anchors_per_phase must be positive")
    ceiling = float(acquisition.get("frontier_score_ceiling", np.nan))
    if not np.isfinite(ceiling) or not 0.0 <= ceiling <= 1.0:
        raise ValueError("frontier_score_ceiling must lie in [0, 1]")
    search_mode = str(acquisition.get("search_mode", ""))
    _refinement_protocol_purposes(search_mode)
    action_names = tuple(acquisition.get("action_names", ()))
    signs = tuple(int(x) for x in acquisition.get("signs", ()))
    strengths = tuple(float(x) for x in acquisition.get("strengths", ()))
    grid = tuple(int(x) for x in acquisition.get("duration_grid", ()))
    if search_mode == "contiguous_integer_local_refinement":
        if action_names != ACTION_ORDER:
            raise ValueError("downstream refinement must preserve canonical action order")
        if signs != (-1, 1):
            raise ValueError("downstream refinement signs must remain [-1, +1]")
        if strengths != (0.025, 0.05, 0.10):
            raise ValueError("downstream refinement strengths must remain fixed")
        if grid != tuple(range(17, 33)):
            raise ValueError("downstream refinement duration_grid must be contiguous 17..32")
    elif search_mode == "fixed_duration_strength_extrapolation":
        if action_names != ACTION_ORDER:
            raise ValueError("downstream refinement must preserve canonical action order")
        if signs != (-1, 1):
            raise ValueError("downstream refinement signs must remain [-1, +1]")
        if strengths != (0.15, 0.20, 0.30):
            raise ValueError("downstream strength-extrapolation strengths drift")
        if grid != (30,):
            raise ValueError("downstream strength-extrapolation duration_grid must be [30]")
    elif search_mode == "targeted_hip_positive_boundary_completion":
        if action_names != ("hip",):
            raise ValueError("downstream hip completion action_names must be [hip]")
        if signs != (1,):
            raise ValueError("downstream hip completion signs must be [+1]")
        if strengths != (0.32, 0.35, 0.40, 0.45, 0.50):
            raise ValueError("downstream hip completion strengths drift")
        if grid != (30,):
            raise ValueError("downstream hip completion duration_grid must be [30]")
    clipping = acquisition.get("terminal_clipping", {})
    if clipping != {
        "enabled": True,
        "capture": "last_finite_nonterminal_state_before_terminal",
        "terminal_outcome_is_not_the_continuation_label": True,
    }:
        raise ValueError("downstream refinement terminal-clipping contract drift")
    for key in ("acquisition_protocol_seed_base", "label_protocol_seed_base"):
        if int(acquisition.get(key, -1)) < 0:
            raise ValueError(f"{key} must be nonnegative")

    continuation = raw.get("continuation_labeling")
    if not isinstance(continuation, Mapping):
        raise ValueError("downstream refinement requires continuation_labeling")
    if int(continuation.get("max_ticks", 0)) != 400:
        raise ValueError("downstream refinement must preserve 400-tick continuation horizon")
    if int(continuation.get("branches_per_candidate", 0)) != 1:
        raise ValueError("deterministic downstream refinement requires one branch")

    readiness = raw.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("downstream refinement requires readiness")
    for key in (
        "minimum_positive_candidates",
        "minimum_negative_candidates",
        "minimum_parent_groups_with_positive",
        "minimum_parent_groups_with_negative",
    ):
        if int(readiness.get(key, 0)) <= 0:
            raise ValueError(f"readiness {key} must be positive")

    expected_claims = {
        "training_only_search": True,
        "upstream_transition_band_frozen": True,
        "continuation_field_trained": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }
    if raw.get("claim_boundary") != expected_claims:
        raise ValueError("downstream refinement claim boundary drift")
    return raw


def _validate_prior_search(
    config: Mapping[str, Any], policy_record: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior = config["prior_search"]
    root = Path(prior["root"])
    summary = _read_json(root / "summary.json")
    accumulated = _read_json(root / "accumulated_train_labels.json")
    rows = accumulated.get("entries")
    if not isinstance(rows, list):
        raise ValueError("prior accumulated TRAIN labels must contain entries")
    if summary.get("status") != prior["expected_status"]:
        raise ValueError("prior search status drift")
    if summary.get("protocol_sha256") != prior["expected_protocol_sha256"]:
        raise ValueError("prior search protocol SHA-256 drift")
    if int(summary.get("accumulated_unique_label_count", -1)) != int(
        prior["expected_accumulated_unique_label_count"]
    ):
        raise ValueError("prior accumulated label count drift")
    if len(rows) != int(prior["expected_accumulated_unique_label_count"]):
        raise ValueError("prior accumulated label file count drift")
    if summary.get("policy_actor_sha256") != policy_record["actor_sha256"]:
        raise ValueError("prior search actor identity drift")
    if summary.get("policy_payload_sha256") != policy_record["payload_sha256"]:
        raise ValueError("prior search payload identity drift")
    if summary.get("source_tube_manifest_sha256") != config["source_tube_manifest_sha256"]:
        raise ValueError("prior search source Tube drift")
    for phase, key in (("upstream", "expected_upstream"), ("downstream", "expected_downstream")):
        actual = summary.get("readiness", {}).get(phase)
        expected = prior[key]
        if actual != expected:
            raise ValueError(f"prior search {phase} readiness drift")
    unique = unique_label_rows([rows])
    if len(unique) != len(rows):
        raise ValueError("prior accumulated labels contain duplicate physical states")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("prior search contains non-TRAIN labels")
    identity = {
        "root": str(root),
        "summary_file_sha256": file_sha256(root / "summary.json"),
        "accumulated_labels_file_sha256": file_sha256(root / "accumulated_train_labels.json"),
        "protocol_sha256": summary["protocol_sha256"],
        "accumulated_unique_label_count": len(rows),
        "readiness": summary["readiness"],
    }
    return [dict(row) for row in rows], identity


def _prepare_downstream_refinement(config_path: Path) -> dict[str, Any]:
    """Validate every zero-interaction input shared by audit and execution."""
    config_path = Path(config_path)
    config = load_downstream_refinement_config(config_path)
    frozen_path = Path(config["frozen_policy"])
    frozen = load_frozen_unified_manifest(frozen_path)
    policy_record = frozen["policy"]
    if int(policy_record["iteration"]) != int(config["iteration"]):
        raise ValueError("downstream refinement policy iteration drift")

    prior_labels, prior_identity = _validate_prior_search(config, policy_record)
    formal = load_unified_formal_config(Path(policy_record["formal_config"]))
    if formal.config_sha256 != policy_record["formal_config_sha256"]:
        raise ValueError("downstream refinement frozen policy/formal config drift")
    artifact = load_soft_tube(Path(formal.soft_tube_path))
    if artifact.manifest["manifest_sha256"] != formal.soft_tube_manifest_sha256:
        raise ValueError("downstream refinement formal config/source Tube drift")
    if artifact.manifest["manifest_sha256"] != config["source_tube_manifest_sha256"]:
        raise ValueError("downstream refinement source Tube identity drift")

    checkpoint_path = Path(policy_record["checkpoint"])
    checkpoint_payload_sha256 = file_sha256(checkpoint_path / "payload.pkl")
    if checkpoint_payload_sha256 != policy_record["payload_sha256"]:
        raise ValueError("downstream refinement checkpoint payload drift")

    anchors_all, anchor_audit = select_tube_boundary_anchors(
        artifact,
        max_per_phase=int(config["fixed_acquisition"]["anchors_per_phase"]),
        frontier_score_ceiling=float(
            config["fixed_acquisition"]["frontier_score_ceiling"]
        ),
    )
    anchors = tuple(anchor for anchor in anchors_all if anchor.phase == "downstream")
    if not anchors:
        raise ValueError("downstream refinement selected no downstream anchors")

    return {
        "config_path": config_path,
        "config": config,
        "frozen_path": frozen_path,
        "policy_record": policy_record,
        "prior_labels": prior_labels,
        "prior_identity": prior_identity,
        "formal": formal,
        "artifact": artifact,
        "checkpoint_payload_sha256": checkpoint_payload_sha256,
        "anchors": anchors,
        "anchor_audit": anchor_audit,
    }


def audit_downstream_transition_refinement(config_path: Path) -> dict[str, Any]:
    """Validate declared refinement inputs without MJX or environment interactions."""
    prepared = _prepare_downstream_refinement(config_path)
    config = prepared["config"]
    policy_record = prepared["policy_record"]
    artifact = prepared["artifact"]
    anchors = prepared["anchors"]
    return {
        "schema": DOWNSTREAM_REFINEMENT_PROTOCOL_SCHEMA,
        "status": "artifact_audit_valid",
        "repository_head": _repository_head(),
        "config_path": str(prepared["config_path"]),
        "config_file_sha256": file_sha256(prepared["config_path"]),
        "output_dir": str(config["output_dir"]),
        "iteration": int(config["iteration"]),
        "frozen_policy_path": str(prepared["frozen_path"]),
        "frozen_policy_file_sha256": file_sha256(prepared["frozen_path"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "policy_formal_config_sha256": str(policy_record["formal_config_sha256"]),
        "checkpoint_payload_sha256": prepared["checkpoint_payload_sha256"],
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "prior_search": prepared["prior_identity"],
        "downstream_anchor_count": len(anchors),
        "downstream_anchor_identities": [
            {
                "entry_index": anchor.entry_index,
                "global_index": anchor.global_index,
                "parent_group_id": anchor.parent_group_id,
                "state_sha256": anchor.state_sha256,
                "bootstrap_value_score": anchor.value_score,
            }
            for anchor in anchors
        ],
        "training_transitions": 0,
        "environment_interactions": 0,
        "claim_boundary": config["claim_boundary"],
    }


def _load_validated_duration_acquisition(
    duration_root: Path,
    *,
    duration: int,
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str, int]:
    acquisition_root = Path(duration_root) / "acquisition"
    acquisition_protocol = _read_json(acquisition_root / "protocol.json")
    catalog = _read_json(acquisition_root / "catalog.json")
    acquisition_summary = _read_json(acquisition_root / "summary.json")
    if acquisition_protocol.get("schema") != BOUNDARY_PROTOCOL_SCHEMA:
        raise ValueError(f"downstream duration {duration} acquisition protocol drift")
    if acquisition_protocol.get("status") != "predeclared":
        raise ValueError(f"downstream duration {duration} acquisition protocol status drift")
    if acquisition_protocol.get("split") != "train":
        raise ValueError(f"downstream duration {duration} acquisition split drift")
    if tuple(int(value) for value in acquisition_protocol.get("durations", ())) != (
        int(duration),
    ):
        raise ValueError(f"downstream duration {duration} acquisition duration drift")
    for key, expected in (
        ("iteration", int(policy_record["iteration"])),
        ("policy_name", policy_record["name"]),
        ("policy_actor_sha256", policy_record["actor_sha256"]),
        ("policy_payload_sha256", policy_record["payload_sha256"]),
        ("frozen_unified_manifest_sha256", frozen_manifest_sha256),
    ):
        if acquisition_protocol.get(key) != expected:
            raise ValueError(f"downstream duration {duration} acquisition {key} drift")
    protocol_sha = acquisition_protocol.get("protocol_sha256")
    if not isinstance(protocol_sha, str) or len(protocol_sha) != 64:
        raise ValueError(f"downstream duration {duration} acquisition protocol SHA drift")
    acquisition_protocol_base = {
        key: value
        for key, value in acquisition_protocol.items()
        if key != "protocol_sha256"
    }
    if _canonical_sha256(acquisition_protocol_base) != protocol_sha:
        raise ValueError(f"downstream duration {duration} acquisition protocol SHA drift")
    if catalog.get("protocol_sha256") != protocol_sha:
        raise ValueError(f"downstream duration {duration} catalog protocol drift")
    if acquisition_summary != {key: value for key, value in catalog.items() if key != "entries"}:
        raise ValueError(f"downstream duration {duration} acquisition summary drift")
    entries = catalog.get("entries")
    if not isinstance(entries, list) or len(entries) != int(catalog.get("candidate_count", -1)):
        raise ValueError(f"downstream duration {duration} acquisition count drift")
    for key, expected in (
        ("schema", BOUNDARY_CATALOG_SCHEMA),
        ("status", "completed"),
        ("artifact_role", "unlabeled_policy_conditioned_frontier_candidates"),
        ("split", "train"),
        ("iteration", int(policy_record["iteration"])),
        ("policy_name", policy_record["name"]),
        ("policy_actor_sha256", policy_record["actor_sha256"]),
        ("policy_payload_sha256", policy_record["payload_sha256"]),
        ("frozen_unified_manifest_sha256", frozen_manifest_sha256),
        ("training_transitions", 0),
        ("expert_switching_used", False),
        ("test_data_used", False),
        ("validation_data_used", False),
        ("final_evaluation_data_used", False),
    ):
        if catalog.get(key) != expected:
            raise ValueError(f"downstream duration {duration} acquisition catalog {key} drift")
    if catalog.get("claim_boundary") != {
        "unlabeled_acquisition_only": True,
        "tube_expansion_claim": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError(f"downstream duration {duration} acquisition claim boundary drift")
    acquisition_interactions = int(acquisition_summary.get("environment_interactions", -1))
    if acquisition_interactions < 0:
        raise ValueError(f"downstream duration {duration} acquisition accounting drift")
    return catalog, acquisition_summary, protocol_sha, acquisition_interactions


def _validate_repair_resume_protocol(
    existing_protocol: Mapping[str, Any],
    current_protocol: Mapping[str, Any],
    *,
    expected_source_head: str,
) -> dict[str, Any]:
    """Allow one explicit source repair while keeping every scientific field exact."""
    source_head = str(existing_protocol.get("repository_head", ""))
    repair_head = str(current_protocol.get("repository_head", ""))
    if source_head != str(expected_source_head):
        raise ValueError("repair resume source repository HEAD drift")
    if len(source_head) != 40 or len(repair_head) != 40 or source_head == repair_head:
        raise ValueError("repair resume requires distinct exact repository HEADs")
    ignored = {"repository_head", "protocol_sha256"}
    existing_stable = {
        key: value for key, value in existing_protocol.items() if key not in ignored
    }
    current_stable = {
        key: value for key, value in current_protocol.items() if key not in ignored
    }
    if existing_stable != current_stable:
        raise ValueError("repair resume non-source protocol drift")
    return {
        "schema": "jit_downstream_refinement_repair_resume_v1",
        "status": "predeclared",
        "source_repository_head": source_head,
        "repair_repository_head": repair_head,
        "source_protocol_sha256": existing_protocol.get("protocol_sha256"),
        "repair_protocol_sha256": current_protocol.get("protocol_sha256"),
        "training_transitions": 0,
    }


def _load_retryable_failed_duration(
    duration_root: Path,
    *,
    duration: int,
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one allocation-failed label attempt without accepting partial rows."""
    duration_root = Path(duration_root)
    if (duration_root / "duration_summary.json").exists():
        raise ValueError(f"downstream duration {duration} already has a duration summary")
    catalog, _, acquisition_protocol_sha, _ = _load_validated_duration_acquisition(
        duration_root,
        duration=duration,
        policy_record=policy_record,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    validated_candidates = validate_unified_boundary_catalog(
        catalog,
        policy_record=policy_record,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    labels_root = duration_root / "labels"
    if (labels_root / "labels.json").exists():
        raise ValueError(f"downstream duration {duration} partial labels must not exist")
    label_protocol = _read_json(labels_root / "protocol.json")
    failed_summary_path = labels_root / "summary.json"
    failed_summary = _read_json(failed_summary_path)
    catalog_file_sha256 = file_sha256(duration_root / "acquisition" / "catalog.json")
    for key, expected in (
        ("schema", "jit_unified_continuation_labels_v1"),
        ("iteration", int(policy_record["iteration"])),
        ("policy_actor_sha256", policy_record["actor_sha256"]),
        ("policy_payload_sha256", policy_record["payload_sha256"]),
        ("frozen_unified_manifest_sha256", frozen_manifest_sha256),
        ("candidate_catalog_file_sha256", catalog_file_sha256),
        ("candidate_catalog_protocol_sha256", acquisition_protocol_sha),
        ("candidate_count", len(validated_candidates)),
    ):
        if label_protocol.get(key) != expected:
            raise ValueError(f"downstream duration {duration} failed label protocol {key} drift")
    if label_protocol.get("status") != "predeclared" or label_protocol.get("split") != "train":
        raise ValueError(f"downstream duration {duration} failed label protocol status drift")
    label_protocol_sha = label_protocol.get("protocol_sha256")
    label_protocol_base = {
        key: value for key, value in label_protocol.items() if key != "protocol_sha256"
    }
    if (
        not isinstance(label_protocol_sha, str)
        or len(label_protocol_sha) != 64
        or _canonical_sha256(label_protocol_base) != label_protocol_sha
    ):
        raise ValueError(f"downstream duration {duration} failed label protocol SHA drift")
    for key, expected in (
        ("schema", "jit_unified_continuation_labels_v1"),
        ("status", "engineering_error"),
        ("iteration", int(policy_record["iteration"])),
        ("policy_actor_sha256", policy_record["actor_sha256"]),
        ("policy_payload_sha256", policy_record["payload_sha256"]),
        ("protocol_sha256", label_protocol_sha),
        ("training_transitions", 0),
        ("expert_switching_used", False),
        ("test_data_used", False),
        ("validation_data_used", False),
        ("final_evaluation_data_used", False),
    ):
        if failed_summary.get(key) != expected:
            raise ValueError(f"downstream duration {duration} failed label summary {key} drift")
    completed = int(failed_summary.get("completed_candidate_count", -1))
    interactions = int(failed_summary.get("environment_interactions", -1))
    maximum = int(failed_summary.get("maximum_environment_interactions", -1))
    if not (0 <= completed <= len(validated_candidates)) or not (
        0 <= interactions <= maximum
    ):
        raise ValueError(f"downstream duration {duration} failed label accounting drift")
    error = str(failed_summary.get("error", ""))
    if "Failed to allocate" not in error or "cuda:0" not in error:
        raise ValueError(f"downstream duration {duration} is not the declared allocation failure")
    retry_index = 1
    while (duration_root / f"labels_retry_{retry_index:02d}").exists():
        retry_index += 1
    attempt = {
        "failed_labels_directory": "labels",
        "failed_summary_file_sha256": file_sha256(failed_summary_path),
        "completed_candidate_count": completed,
        "environment_interactions": interactions,
        "maximum_environment_interactions": maximum,
        "error": error,
        "retry_labels_directory": f"labels_retry_{retry_index:02d}",
    }
    return catalog, attempt


def _load_completed_duration(
    duration_root: Path,
    *,
    duration: int,
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one duration-level checkpoint or fail closed on partial evidence."""
    duration_root = Path(duration_root)
    duration_summary_path = duration_root / "duration_summary.json"
    if not duration_summary_path.exists():
        raise ValueError(f"cannot resume incomplete downstream duration {duration}")
    duration_summary = _read_json(duration_summary_path)
    if int(duration_summary.get("duration", -1)) != int(duration):
        raise ValueError(f"downstream duration {duration} summary identity drift")

    catalog, acquisition_summary, protocol_sha, acquisition_interactions = (
        _load_validated_duration_acquisition(
            duration_root,
            duration=duration,
            policy_record=policy_record,
            frozen_manifest_sha256=frozen_manifest_sha256,
        )
    )
    acquisition_root = duration_root / "acquisition"
    entries = catalog["entries"]

    status = duration_summary.get("status")
    if status == "no_candidates":
        if entries or int(duration_summary.get("candidate_count", -1)) != 0:
            raise ValueError(f"downstream duration {duration} zero-candidate drift")
        if (duration_root / "labels").exists():
            raise ValueError(f"downstream duration {duration} has unexpected zero-candidate labels")
        return [], {
            **duration_summary,
            "resumed": True,
            "acquisition_environment_interactions": acquisition_interactions,
            "labeling_environment_interactions": 0,
        }

    if status != "completed":
        raise ValueError(f"cannot resume non-completed downstream duration {duration}")
    labels_directory = str(duration_summary.get("labels_directory", "labels"))
    if Path(labels_directory).name != labels_directory or not labels_directory.startswith(
        "labels"
    ):
        raise ValueError(f"downstream duration {duration} labels directory drift")
    labels_root = duration_root / labels_directory
    labels_path = labels_root / "labels.json"
    label_summary_path = labels_root / "summary.json"
    if not labels_path.exists() or not label_summary_path.exists():
        raise ValueError(f"cannot resume incomplete downstream duration {duration}")
    label_summary = _read_json(label_summary_path)
    if label_summary.get("status") != "completed":
        raise ValueError(f"cannot resume non-completed downstream duration {duration}")
    label_protocol = _read_json(labels_root / "protocol.json")
    rows = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != int(label_summary.get("label_count", -1)):
        raise ValueError(f"downstream duration {duration} label count drift")
    validated_candidates = validate_unified_boundary_catalog(
        catalog,
        policy_record=policy_record,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    catalog_file_sha256 = file_sha256(acquisition_root / "catalog.json")
    if label_protocol.get("schema") != "jit_unified_continuation_labels_v1":
        raise ValueError(f"downstream duration {duration} label protocol schema drift")
    if label_protocol.get("status") != "predeclared" or label_protocol.get("split") != "train":
        raise ValueError(f"downstream duration {duration} label protocol status drift")
    for key, expected in (
        ("iteration", int(policy_record["iteration"])),
        ("policy_name", policy_record["name"]),
        ("policy_actor_sha256", policy_record["actor_sha256"]),
        ("policy_payload_sha256", policy_record["payload_sha256"]),
        ("frozen_unified_manifest_sha256", frozen_manifest_sha256),
        ("candidate_catalog_file_sha256", catalog_file_sha256),
        ("candidate_catalog_protocol_sha256", protocol_sha),
        ("candidate_count", len(validated_candidates)),
    ):
        if label_protocol.get(key) != expected:
            raise ValueError(f"downstream duration {duration} label protocol {key} drift")
        if label_summary.get(key) != expected:
            raise ValueError(f"downstream duration {duration} label summary {key} drift")
    label_protocol_sha = label_protocol.get("protocol_sha256")
    label_protocol_base = {
        key: value for key, value in label_protocol.items() if key != "protocol_sha256"
    }
    if (
        not isinstance(label_protocol_sha, str)
        or len(label_protocol_sha) != 64
        or _canonical_sha256(label_protocol_base) != label_protocol_sha
    ):
        raise ValueError(f"downstream duration {duration} label protocol SHA drift")
    if label_summary.get("protocol_sha256") != label_protocol_sha:
        raise ValueError(f"downstream duration {duration} label protocol SHA drift")
    if int(label_summary.get("candidate_count", -1)) != len(validated_candidates):
        raise ValueError(f"downstream duration {duration} label candidate count drift")
    positive_count = sum(int(row.get("label", -1)) == 1 for row in rows)
    negative_count = sum(int(row.get("label", -1)) == 0 for row in rows)
    if positive_count + negative_count != len(rows):
        raise ValueError(f"downstream duration {duration} label value drift")
    if (
        int(label_summary.get("positive_count", -1)) != positive_count
        or int(label_summary.get("negative_count", -1)) != negative_count
        or int(duration_summary.get("candidate_count", -1)) != len(rows)
        or int(duration_summary.get("positive_count", -1)) != positive_count
        or int(duration_summary.get("negative_count", -1)) != negative_count
    ):
        raise ValueError(f"downstream duration {duration} label summary count drift")
    expected_candidates = {
        str(row["candidate_id"]): (
            str(row["state_sha256"]),
            str(row["parent_group_id"]),
        )
        for row in validated_candidates
    }
    actual_candidates: set[str] = set()
    for row in rows:
        if (
            row.get("split") != "train"
            or row.get("phase") != "downstream"
            or int(row.get("phase_index", -1)) != 1
        ):
            raise ValueError(f"downstream duration {duration} label split/phase drift")
        for key, expected in (
            ("policy_iteration", int(policy_record["iteration"])),
            ("policy_actor_sha256", policy_record["actor_sha256"]),
            ("policy_payload_sha256", policy_record["payload_sha256"]),
            ("acquisition_protocol_sha256", protocol_sha),
            ("label_protocol_sha256", label_protocol_sha),
        ):
            if row.get(key) != expected:
                raise ValueError(f"downstream duration {duration} label row {key} drift")
        candidate_id = str(row["candidate_id"])
        expected_identity = expected_candidates.get(candidate_id)
        actual_identity = (str(row["state_sha256"]), str(row["parent_group_id"]))
        if expected_identity != actual_identity:
            raise ValueError(f"downstream duration {duration} label candidate identity drift")
        if candidate_id in actual_candidates:
            raise ValueError(f"downstream duration {duration} duplicate label identity")
        actual_candidates.add(candidate_id)
    if actual_candidates != set(expected_candidates):
        raise ValueError(f"downstream duration {duration} label candidate coverage drift")
    aborted_attempts = duration_summary.get("aborted_labeling_attempts", [])
    if not isinstance(aborted_attempts, list):
        raise ValueError(f"downstream duration {duration} aborted label attempts drift")
    for attempt in aborted_attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError(f"downstream duration {duration} aborted label attempt drift")
        failed_directory = str(attempt.get("failed_labels_directory", ""))
        if (
            Path(failed_directory).name != failed_directory
            or not failed_directory.startswith("labels")
            or failed_directory == labels_directory
            or attempt.get("retry_labels_directory") != labels_directory
        ):
            raise ValueError(f"downstream duration {duration} aborted label directory drift")
        failed_summary_path = duration_root / failed_directory / "summary.json"
        if file_sha256(failed_summary_path) != attempt.get("failed_summary_file_sha256"):
            raise ValueError(f"downstream duration {duration} failed summary SHA drift")
        failed_summary = _read_json(failed_summary_path)
        for key in (
            "completed_candidate_count",
            "environment_interactions",
            "error",
        ):
            if failed_summary.get(key) != attempt.get(key):
                raise ValueError(
                    f"downstream duration {duration} failed summary {key} drift"
                )
        if failed_summary.get("status") != "engineering_error":
            raise ValueError(f"downstream duration {duration} failed summary status drift")
    aborted_interactions = sum(
        int(attempt.get("environment_interactions", -1)) for attempt in aborted_attempts
    )
    if aborted_interactions < 0 or int(
        duration_summary.get("aborted_labeling_environment_interactions", 0)
    ) != aborted_interactions:
        raise ValueError(f"downstream duration {duration} aborted label accounting drift")
    successful_labeling_interactions = int(label_summary["environment_interactions"])
    return [dict(row) for row in rows], {
        **duration_summary,
        "resumed": True,
        "acquisition_environment_interactions": acquisition_interactions,
        "successful_labeling_environment_interactions": successful_labeling_interactions,
        "labeling_environment_interactions": (
            successful_labeling_interactions + aborted_interactions
        ),
    }


def _collect_duration_candidates(
    anchors: Sequence[TubeBoundaryAnchor],
    output_dir: Path,
    *,
    duration: int,
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
    acquisition_purpose: str,
    protocol_seed: int,
    frontier_score_ceiling: float,
    strengths: Sequence[float],
    action_names: Sequence[str],
    signs: Sequence[int],
    excluded_state_hashes: set[str],
    compiled_reset_fn: Callable[[Any, Any], Any] | None = None,
    compiled_step_fn: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    strengths = validate_strengths(strengths)
    directions = action_basis_directions(action_names=action_names, signs=signs)
    if int(duration) <= 0:
        raise ValueError("downstream refinement duration must be positive")
    if not anchors or any(anchor.phase != "downstream" or anchor.phase_index != 1 for anchor in anchors):
        raise ValueError("downstream refinement requires downstream-only anchors")
    if any(anchor.value_score > float(frontier_score_ceiling) for anchor in anchors):
        raise ValueError("downstream refinement anchor exceeds score ceiling")

    artifact = env.tube_pool.artifact
    support_hashes = {str(row["state_sha256"]) for row in artifact.entries}
    max_interactions = len(anchors) * len(strengths) * len(directions) * int(duration)
    anchor_identities = [
        {
            "phase": anchor.phase,
            "phase_index": anchor.phase_index,
            "entry_index": anchor.entry_index,
            "global_index": anchor.global_index,
            "parent_group_id": anchor.parent_group_id,
            "state_sha256": anchor.state_sha256,
            "bootstrap_value_score": anchor.value_score,
        }
        for anchor in anchors
    ]
    protocol_base = {
        "schema": BOUNDARY_PROTOCOL_SCHEMA,
        "status": "predeclared",
        "purpose": str(acquisition_purpose),
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "policy_formal_config_sha256": policy_record["formal_config_sha256"],
        "frozen_unified_manifest_sha256": frozen_manifest_sha256,
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "source_tube_entry_count": int(artifact.manifest["entry_count"]),
        "anchor_count": len(anchors),
        "anchor_selection": {
            "rule": "bootstrap_score_at_or_below_ceiling_parent_group_unique_state_unique",
            "frontier_score_ceiling": float(frontier_score_ceiling),
            "parent_group_unique_per_phase": True,
            "physical_state_unique": True,
            "anchor_identities": anchor_identities,
        },
        "anchor_semantics": "weak_bootstrap_frontier_probe_not_certified_boundary",
        "protocol_seed": int(protocol_seed),
        "action_order": list(ACTION_ORDER),
        "direction_family": "action_basis",
        "selected_action_names": list(action_names),
        "selected_signs": list(signs),
        "strengths": list(strengths),
        "durations": [int(duration)],
        "terminal_clipping": {
            "enabled": True,
            "capture": "last_finite_nonterminal_state_before_terminal",
            "terminal_outcome_is_not_the_continuation_label": True,
        },
        "maximum_environment_interactions": max_interactions,
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "state_generation": (
            "reset exact downstream TRAIN Tube entry; apply frozen-policy action plus bounded "
            "action-basis perturbation through authoritative env.step; if the perturbation "
            "causes terminal, capture the last finite nonterminal predecessor instead of "
            "treating that terminal outcome as the continuation label"
        ),
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    protocol_sha = _canonical_sha256(protocol_base)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "protocol.json", {**protocol_base, "protocol_sha256": protocol_sha})
    bank = output / "boundary_bank"
    (bank / "snapshots").mkdir(parents=True, exist_ok=False)

    reset = compiled_reset_fn if compiled_reset_fn is not None else jax.jit(env.reset_tube_index)
    step = compiled_step_fn if compiled_step_fn is not None else jax.jit(env.step)
    base_key = jax.random.PRNGKey(int(protocol_seed))
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    exclusions = Counter()
    terminal_probe_outcomes = Counter()
    interactions = 0
    attempted = 0
    variant_index = 0

    for anchor in anchors:
        for strength in strengths:
            for direction in directions:
                attempted += 1
                state = reset(np.int32(anchor.phase_index), np.int32(anchor.entry_index))
                jax.block_until_ready(state)
                if _integer(state.info["active_phase"]) != 1:
                    raise ValueError("downstream refinement reset phase mismatch")
                if _truth(state.info["expert_switching_used"]):
                    raise ValueError("downstream refinement reset used expert switching")
                nominal_actions: list[list[float]] = []
                perturbed_actions: list[list[float]] = []
                effective_deltas: list[list[float]] = []
                terminal_clipped = False
                terminal_meta: dict[str, Any] | None = None
                rejected: str | None = None
                snapshot_nonterminal_steps = 0
                current_variant = variant_index
                variant_index += 1

                for perturb_step in range(int(duration)):
                    previous_state = state
                    variant_key = jax.random.fold_in(base_key, int(current_variant))
                    action_key = jax.random.fold_in(variant_key, int(perturb_step))
                    result = policy(previous_state.obs, action_key)
                    nominal = result[0] if isinstance(result, tuple) else result
                    nominal = np.asarray(jax.device_get(nominal), dtype=np.float32).reshape(-1)
                    if nominal.shape != (len(ACTION_ORDER),) or not np.isfinite(nominal).all():
                        raise ValueError("frozen unified policy returned invalid refinement action")
                    requested = nominal + np.asarray(direction["basis_vector"], dtype=np.float32) * np.float32(strength)
                    perturbed = np.clip(requested, -1.0, 1.0).astype(np.float32)
                    stepped = step(previous_state, perturbed)
                    jax.block_until_ready(stepped)
                    interactions += 1
                    nominal_actions.append(nominal.tolist())
                    perturbed_actions.append(perturbed.tolist())
                    effective_deltas.append((perturbed - nominal).tolist())
                    if _truth(stepped.info["expert_switching_used"]):
                        raise ValueError("downstream refinement used expert switching")
                    if not _finite_state(stepped, perturbed):
                        rejected = "nonfinite"
                        break
                    if _integer(stepped.info["active_phase"]) != 1:
                        rejected = "phase_transition"
                        break
                    if _truth(stepped.done):
                        if not _finite_state(previous_state):
                            rejected = "nonfinite_preterminal"
                            break
                        state = previous_state
                        terminal_clipped = True
                        snapshot_nonterminal_steps = int(perturb_step)
                        end_code = _integer(stepped.info["end_code"])
                        terminal_meta = {
                            "end_code": end_code,
                            "end_reason": END_REASONS.get(end_code, f"unknown_{end_code}"),
                            "terminal_success": _truth(stepped.info["success"]),
                            "physical_failure": _truth(stepped.info["physical_failure"]),
                            "timeout": _truth(stepped.info["timeout"]),
                            "executed_interactions": int(perturb_step + 1),
                        }
                        terminal_probe_outcomes[terminal_meta["end_reason"]] += 1
                        break
                    state = stepped
                    snapshot_nonterminal_steps = int(perturb_step + 1)

                if rejected is not None:
                    exclusions[rejected] += 1
                    continue

                snapshot = capture_unified_envelope_snapshot(
                    state,
                    env=env,
                    parent_trajectory=anchor.parent_group_id,
                    parent_state_sha256=anchor.state_sha256,
                    config_sha256=str(policy_record["formal_config_sha256"]),
                    policy_actor_sha256=str(policy_record["actor_sha256"]),
                    policy_payload_sha256=str(policy_record["payload_sha256"]),
                    policy_iteration=int(policy_record["iteration"]),
                )
                state_hash = physical_state_sha256(snapshot)
                if state_hash in support_hashes:
                    exclusions["existing_support"] += 1
                    continue
                if state_hash in excluded_state_hashes:
                    exclusions["previously_labeled_state"] += 1
                    continue
                if state_hash in seen:
                    exclusions["duplicate"] += 1
                    continue
                seen.add(state_hash)
                excluded_state_hashes.add(state_hash)
                relative = Path("snapshots") / f"candidate_{len(entries):06d}"
                save_unified_envelope_snapshot(bank / relative, snapshot)
                qpos = np.asarray(snapshot.qpos)
                qvel = np.asarray(snapshot.qvel)
                metrics = state.metrics
                entries.append(
                    {
                        "candidate_id": f"pi{policy_record['iteration']}_downstream_d{duration}_{len(entries):06d}",
                        "candidate_kind": "reachable_unified_frontier_probe",
                        "split": "train",
                        "phase": "downstream",
                        "phase_index": 1,
                        "snapshot": str(relative),
                        "source_bank": "boundary_bank",
                        "state_sha256": state_hash,
                        "parent_group_id": anchor.parent_group_id,
                        "parent_state_sha256": anchor.state_sha256,
                        "anchor_entry_index": anchor.entry_index,
                        "anchor_global_index": anchor.global_index,
                        "anchor_value_score": anchor.value_score,
                        "anchor_sampling_weight": float(anchor.row["sampling_weight"]),
                        "anchor_role": str(anchor.row.get("role", "")),
                        "anchor_source_bank": str(anchor.row.get("source_bank", "")),
                        "policy_iteration": int(policy_record["iteration"]),
                        "policy_actor_sha256": str(policy_record["actor_sha256"]),
                        "policy_payload_sha256": str(policy_record["payload_sha256"]),
                        "protocol_sha256": protocol_sha,
                        "perturbation": {
                            **direction,
                            "strength": float(strength),
                            "duration": int(duration),
                            "variant_index": int(current_variant),
                            "nominal_actions": nominal_actions,
                            "perturbed_actions": perturbed_actions,
                            "effective_deltas": effective_deltas,
                            "terminal_clipped": bool(terminal_clipped),
                            "snapshot_nonterminal_steps": int(snapshot_nonterminal_steps),
                            "executed_interactions": len(perturbed_actions),
                            "terminal_probe_outcome": terminal_meta,
                        },
                        "episode_step": snapshot.episode_step,
                        "phase_episode_step": snapshot.phase_episode_step,
                        "x": float(qpos[0]),
                        "z": float(qpos[2]),
                        "vx": float(qvel[0]),
                        "vz": float(qvel[2]),
                        "roll": float(np.asarray(jax.device_get(metrics["signal/roll"]))),
                        "pitch": float(np.asarray(jax.device_get(metrics["signal/pitch"]))),
                        "yaw": float(np.asarray(jax.device_get(metrics["signal/yaw"]))),
                    }
                )

    if interactions > max_interactions:
        raise ValueError("downstream refinement acquisition exceeded interaction ceiling")
    report = {
        "schema": BOUNDARY_CATALOG_SCHEMA,
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "frozen_unified_manifest_sha256": frozen_manifest_sha256,
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "protocol_sha256": protocol_sha,
        "frontier_score_ceiling": float(frontier_score_ceiling),
        "anchor_count": len(anchors),
        "attempted_candidate_count": attempted,
        "candidate_count": len(entries),
        "environment_interactions": interactions,
        "maximum_environment_interactions": max_interactions,
        "terminal_clipped_candidate_count": sum(
            int(row["perturbation"]["terminal_clipped"]) for row in entries
        ),
        "terminal_probe_outcomes": dict(sorted(terminal_probe_outcomes.items())),
        "exclusion_counts": dict(sorted(exclusions.items())),
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
        "entries": entries,
    }
    _write_json(output / "catalog.json", report)
    _write_json(output / "summary.json", {k: v for k, v in report.items() if k != "entries"})
    return report


def search_downstream_transition_refinement(
    config_path: Path,
    *,
    resume: bool = False,
    repair_source_head: str | None = None,
) -> dict[str, Any]:
    prepared = _prepare_downstream_refinement(config_path)
    config_path = prepared["config_path"]
    config = prepared["config"]
    output = Path(config["output_dir"])
    frozen_path = prepared["frozen_path"]
    policy_record = prepared["policy_record"]
    prior_labels = prepared["prior_labels"]
    prior_identity = prepared["prior_identity"]
    formal = prepared["formal"]
    artifact = prepared["artifact"]
    anchors = prepared["anchors"]
    anchor_audit = prepared["anchor_audit"]
    search_purpose, acquisition_purpose = _refinement_protocol_purposes(
        config["fixed_acquisition"]["search_mode"]
    )

    grid = tuple(int(x) for x in config["fixed_acquisition"]["duration_grid"])
    directions = len(config["fixed_acquisition"]["action_names"]) * len(
        config["fixed_acquisition"]["signs"]
    )
    strengths = len(config["fixed_acquisition"]["strengths"])
    maximum_acquisition = len(anchors) * directions * strengths * sum(grid)
    maximum_labeling = (
        len(anchors)
        * directions
        * strengths
        * len(grid)
        * int(config["continuation_labeling"]["max_ticks"])
    )
    protocol_base = {
        "schema": DOWNSTREAM_REFINEMENT_PROTOCOL_SCHEMA,
        "status": "predeclared",
        "purpose": search_purpose,
        "repository_head": _repository_head(),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "iteration": int(config["iteration"]),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "frozen_policy_file_sha256": file_sha256(frozen_path),
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "prior_search": prior_identity,
        "upstream_transition_band_frozen": True,
        "anchor_audit": anchor_audit,
        "downstream_anchor_identities": [
            {
                "entry_index": a.entry_index,
                "global_index": a.global_index,
                "parent_group_id": a.parent_group_id,
                "state_sha256": a.state_sha256,
                "bootstrap_value_score": a.value_score,
            }
            for a in anchors
        ],
        "fixed_acquisition": config["fixed_acquisition"],
        "continuation_labeling": config["continuation_labeling"],
        "readiness": config["readiness"],
        "maximum_acquisition_environment_interactions": maximum_acquisition,
        "maximum_labeling_environment_interactions": maximum_labeling,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": config["claim_boundary"],
    }
    protocol_sha = _canonical_sha256(protocol_base)
    protocol = {**protocol_base, "protocol_sha256": protocol_sha}

    repair_record: dict[str, Any] | None = None
    if output.exists():
        if not resume:
            raise FileExistsError(f"downstream refinement output already exists: {output}")
        existing = _read_json(output / "search_protocol.json")
        if existing != protocol:
            if repair_source_head is None:
                raise ValueError("cannot resume downstream refinement under a different protocol")
            repair_record = _validate_repair_resume_protocol(
                existing,
                protocol,
                expected_source_head=repair_source_head,
            )
            protocol_sha = str(existing["protocol_sha256"])
        elif repair_source_head is not None:
            raise ValueError("repair resume requires a changed repository HEAD")
        if (output / "summary.json").exists():
            done = _read_json(output / "summary.json")
            if done.get("status") in ("transition_band_ready", "search_exhausted"):
                return done
    else:
        if repair_source_head is not None:
            raise ValueError("repair resume requires an existing failed run")
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "search_protocol.json", protocol)

    if jax.default_backend() != "gpu":
        raise RuntimeError("downstream refinement requires the visible JAX GPU")
    runtime_config, runtime_artifact, env = build_unified_formal_environment(
        Path(policy_record["formal_config"])
    )
    if runtime_config.config_sha256 != formal.config_sha256:
        raise ValueError("downstream refinement runtime config drift")
    if runtime_artifact.manifest["manifest_sha256"] != artifact.manifest["manifest_sha256"]:
        raise ValueError("downstream refinement runtime Tube drift")
    if env._bundle.xml_sha256 != policy_record["xml_sha256"]:
        raise ValueError("downstream refinement runtime XML drift")
    payload = load_checkpoint(
        Path(policy_record["checkpoint"]), expected=checkpoint_identity(runtime_config, env)
    )
    if file_sha256(Path(policy_record["checkpoint"]) / "payload.pkl") != policy_record[
        "payload_sha256"
    ]:
        raise ValueError("downstream refinement checkpoint payload drift")
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))
    compiled_reset_fn = jax.jit(env.reset_tube_index)
    compiled_step_fn = jax.jit(env.step)

    label_sets: list[list[dict[str, Any]]] = [prior_labels]
    excluded_state_hashes = {str(row["state_sha256"]) for row in prior_labels}
    duration_reports: list[dict[str, Any]] = []
    acquisition_interactions = 0
    labeling_interactions = 0

    completed_duration_count = 0
    for duration in grid:
        duration_root = output / f"duration_{duration:02d}"
        if duration_root.exists():
            if not (duration_root / "duration_summary.json").exists():
                if repair_record is not None:
                    break
                raise ValueError(f"cannot resume incomplete downstream duration {duration}")
            rows, duration_report = _load_completed_duration(
                duration_root,
                duration=duration,
                policy_record=policy_record,
                frozen_manifest_sha256=file_sha256(frozen_path),
            )
            if rows:
                label_sets.append(rows)
            excluded_state_hashes.update(str(row["state_sha256"]) for row in rows)
            acquisition_interactions += int(
                duration_report["acquisition_environment_interactions"]
            )
            labeling_interactions += int(
                duration_report["labeling_environment_interactions"]
            )
            accumulated = unique_label_rows(label_sets)
            readiness = phase_transition_band_readiness(accumulated, config["readiness"])
            duration_report["downstream_readiness_after_duration"] = readiness["downstream"]
            duration_reports.append(duration_report)
            completed_duration_count += 1
            if readiness["downstream"]["ready"]:
                break
            continue
        break

    accumulated = unique_label_rows(label_sets)
    readiness = phase_transition_band_readiness(accumulated, config["readiness"])

    if not readiness["downstream"]["ready"]:
        for duration in grid[completed_duration_count:]:
            duration_root = output / f"duration_{duration:02d}"
            failed_attempt: dict[str, Any] | None = None
            if duration_root.exists():
                if repair_record is None:
                    raise ValueError(f"unexpected incomplete duration directory: {duration_root}")
                acquisition, failed_attempt = _load_retryable_failed_duration(
                    duration_root,
                    duration=duration,
                    policy_record=policy_record,
                    frozen_manifest_sha256=file_sha256(frozen_path),
                )
                excluded_state_hashes.update(
                    str(row["state_sha256"]) for row in acquisition["entries"]
                )
                repair_evidence = {
                    **repair_record,
                    "failure_duration": duration,
                    "failed_label_attempt": failed_attempt,
                    "acquisition_protocol_sha256": acquisition["protocol_sha256"],
                    "config_file_sha256": file_sha256(config_path),
                }
                repair_evidence_path = output / "resume_repair.json"
                if repair_evidence_path.exists():
                    if _read_json(repair_evidence_path) != repair_evidence:
                        raise ValueError("downstream refinement repair evidence drift")
                else:
                    _write_json(repair_evidence_path, repair_evidence)
            else:
                if repair_record is not None:
                    raise ValueError("repair resume did not find its declared failed duration")
                acquisition = _collect_duration_candidates(
                    anchors,
                    duration_root / "acquisition",
                    duration=duration,
                    env=env,
                    policy=policy,
                    policy_record=policy_record,
                    frozen_manifest_sha256=file_sha256(frozen_path),
                    acquisition_purpose=acquisition_purpose,
                    protocol_seed=int(
                        config["fixed_acquisition"]["acquisition_protocol_seed_base"]
                    )
                    + duration,
                    frontier_score_ceiling=float(
                        config["fixed_acquisition"]["frontier_score_ceiling"]
                    ),
                    strengths=tuple(
                        float(x) for x in config["fixed_acquisition"]["strengths"]
                    ),
                    action_names=tuple(config["fixed_acquisition"]["action_names"]),
                    signs=tuple(int(x) for x in config["fixed_acquisition"]["signs"]),
                    excluded_state_hashes=excluded_state_hashes,
                    compiled_reset_fn=compiled_reset_fn,
                    compiled_step_fn=compiled_step_fn,
                )
            acquisition_interactions += int(acquisition["environment_interactions"])
            if int(acquisition["candidate_count"]) == 0:
                duration_report = {
                    "duration": duration,
                    "status": "no_candidates",
                    "candidate_count": 0,
                    "terminal_clipped_candidate_count": 0,
                    "terminal_probe_outcomes": acquisition.get("terminal_probe_outcomes", {}),
                    "exclusion_counts": acquisition.get("exclusion_counts", {}),
                }
                duration_reports.append(duration_report)
                _write_json(duration_root / "duration_summary.json", duration_report)
                _write_json(
                    output / "progress.json",
                    {
                        "schema": DOWNSTREAM_REFINEMENT_SUMMARY_SCHEMA,
                        "status": "searching",
                        "protocol_sha256": protocol_sha,
                        "completed_duration_count": len(duration_reports),
                        "readiness": readiness,
                        "durations": duration_reports,
                        "training_transitions": 0,
                    },
                )
                continue

            labels_directory = (
                str(failed_attempt["retry_labels_directory"])
                if failed_attempt is not None
                else "labels"
            )
            labels_dir = duration_root / labels_directory
            label_report = label_unified_continuations(
                duration_root / "acquisition" / "catalog.json",
                labels_dir,
                env=env,
                policy=policy,
                policy_record=policy_record,
                frozen_manifest_sha256=file_sha256(frozen_path),
                max_ticks=int(config["continuation_labeling"]["max_ticks"]),
                protocol_seed=int(config["fixed_acquisition"]["label_protocol_seed_base"])
                + duration,
                compiled_step_fn=compiled_step_fn,
            )
            failed_labeling_interactions = (
                int(failed_attempt["environment_interactions"])
                if failed_attempt is not None
                else 0
            )
            successful_labeling_interactions = int(label_report["environment_interactions"])
            labeling_interactions += (
                failed_labeling_interactions + successful_labeling_interactions
            )
            rows = json.loads((labels_dir / "labels.json").read_text(encoding="utf-8"))
            label_sets.append([dict(row) for row in rows])
            accumulated = unique_label_rows(label_sets)
            readiness = phase_transition_band_readiness(accumulated, config["readiness"])
            duration_report = {
                "duration": duration,
                "status": "completed",
                "candidate_count": int(label_report["candidate_count"]),
                "positive_count": int(label_report["positive_count"]),
                "negative_count": int(label_report["negative_count"]),
                "labels_directory": labels_directory,
                "successful_labeling_environment_interactions": (
                    successful_labeling_interactions
                ),
                "aborted_labeling_environment_interactions": (
                    failed_labeling_interactions
                ),
                "aborted_labeling_attempts": (
                    [failed_attempt] if failed_attempt is not None else []
                ),
                "terminal_clipped_candidate_count": int(
                    acquisition.get("terminal_clipped_candidate_count", 0)
                ),
                "terminal_probe_outcomes": acquisition.get("terminal_probe_outcomes", {}),
                "exclusion_counts": acquisition.get("exclusion_counts", {}),
                "downstream_readiness_after_duration": readiness["downstream"],
            }
            duration_reports.append(duration_report)
            _write_json(duration_root / "duration_summary.json", duration_report)
            _write_json(
                output / "progress.json",
                {
                    "schema": DOWNSTREAM_REFINEMENT_SUMMARY_SCHEMA,
                    "status": "searching",
                    "protocol_sha256": protocol_sha,
                    "completed_duration_count": len(duration_reports),
                    "readiness": readiness,
                    "durations": duration_reports,
                    "training_transitions": 0,
                },
            )
            if readiness["downstream"]["ready"]:
                break
            repair_record = None

    accumulated = unique_label_rows(label_sets)
    readiness = phase_transition_band_readiness(accumulated, config["readiness"])
    downstream_ready = bool(readiness["downstream"]["ready"])
    if readiness["upstream"]["ready"] is not True:
        raise ValueError("downstream refinement unexpectedly lost upstream readiness")
    status = "transition_band_ready" if downstream_ready else "search_exhausted"
    _write_json(output / "accumulated_train_labels.json", {"entries": accumulated})
    summary = {
        "schema": DOWNSTREAM_REFINEMENT_SUMMARY_SCHEMA,
        "status": status,
        "protocol_sha256": protocol_sha,
        "iteration": int(config["iteration"]),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "prior_accumulated_unique_label_count": len(prior_labels),
        "accumulated_unique_label_count": len(accumulated),
        "readiness": readiness,
        "durations": duration_reports,
        "acquisition_environment_interactions": acquisition_interactions,
        "labeling_environment_interactions": labeling_interactions,
        "repair_resume": (
            _read_json(output / "resume_repair.json")
            if (output / "resume_repair.json").exists()
            else None
        ),
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "next_scientific_gate": (
            "freeze both phase TRAIN transition bands and design group-disjoint expansion "
            "validation before fitting/calibrating C_up^0 and C_down^0"
            if downstream_ready
            else "stop and make a new explicit downstream acquisition-method decision"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output / "summary.json", summary)
    return summary
