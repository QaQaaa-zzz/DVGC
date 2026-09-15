"""Automated TRAIN-only transition-band search for policy-conditioned envelope iteration.

The search repeatedly pushes the same audited Tube frontier anchors farther from
Tube_0 by changing perturbation duration only.  Each shell is acquired through
real dynamics, labeled under the same frozen deterministic unified policy, and
accumulated until phase-wise transition-band readiness is met.

This capability deliberately stops before continuation-field fitting,
validation design, Tube_1 construction, or PPO training.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from .checkpoint import load_checkpoint
from .config import file_sha256
from .constants import ACTION_ORDER
from .ppo import make_checkpoint_policy
from .soft_tube import load_soft_tube
from .unified_boundary import (
    TubeBoundaryAnchor,
    collect_unified_boundary_candidates,
    select_tube_boundary_anchors,
)
from .unified_continuation_labels import label_unified_continuations
from .unified_formal import build_unified_formal_environment, load_unified_formal_config
from .unified_policy_freeze import load_frozen_unified_manifest
from .unified_training import checkpoint_identity


TRANSITION_BAND_SEARCH_CONFIG_SCHEMA = "jit_unified_transition_band_search_config_v1"
TRANSITION_BAND_SEARCH_PROTOCOL_SCHEMA = "jit_unified_transition_band_search_protocol_v1"
TRANSITION_BAND_SEARCH_SUMMARY_SCHEMA = "jit_unified_transition_band_search_summary_v1"
PHASES = ("upstream", "downstream")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


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


def _repository_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception as exc:  # pragma: no cover - repository runtime guard
        raise RuntimeError("transition-band search requires a Git repository checkout") from exc


def load_transition_band_search_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate one predeclared automated search config."""
    path = Path(path)
    raw = _read_json(path)
    if raw.get("schema") != TRANSITION_BAND_SEARCH_CONFIG_SCHEMA:
        raise ValueError("unsupported unified transition-band search config schema")
    if int(raw.get("iteration", -1)) < 0:
        raise ValueError("transition-band iteration must be nonnegative")
    if not str(raw.get("frozen_policy", "")):
        raise ValueError("transition-band config requires frozen_policy")
    if not str(raw.get("output_dir", "")):
        raise ValueError("transition-band config requires output_dir")

    inner = raw.get("completed_inner_shell")
    if not isinstance(inner, Mapping):
        raise ValueError("transition-band config requires completed_inner_shell")
    for key in (
        "labels_path",
        "expected_label_protocol_sha256",
        "expected_acquisition_protocol_sha256",
        "expected_candidate_count",
        "expected_positive_count",
        "expected_negative_count",
    ):
        if key not in inner:
            raise ValueError(f"completed_inner_shell missing {key}")

    acquisition = raw.get("fixed_acquisition")
    if not isinstance(acquisition, Mapping):
        raise ValueError("transition-band config requires fixed_acquisition")
    if int(acquisition.get("anchors_per_phase", 0)) <= 0:
        raise ValueError("anchors_per_phase must be positive")
    ceiling = float(acquisition.get("frontier_score_ceiling", np.nan))
    if not np.isfinite(ceiling) or not 0.0 <= ceiling <= 1.0:
        raise ValueError("frontier_score_ceiling must lie in [0, 1]")
    action_names = tuple(str(x) for x in acquisition.get("action_names", ()))
    if action_names != ACTION_ORDER:
        raise ValueError("automated search must preserve the full canonical action order")
    signs = tuple(int(x) for x in acquisition.get("signs", ()))
    if signs != (-1, 1):
        raise ValueError("automated search signs must remain [-1, +1]")
    strengths = tuple(float(x) for x in acquisition.get("strengths", ()))
    if strengths != (0.025, 0.05, 0.10):
        raise ValueError("automated search strengths must remain [0.025, 0.05, 0.10]")

    continuation = raw.get("continuation_labeling")
    if not isinstance(continuation, Mapping):
        raise ValueError("transition-band config requires continuation_labeling")
    if int(continuation.get("max_ticks", 0)) != 400:
        raise ValueError("automated continuation search must preserve the 400-tick horizon")
    if int(continuation.get("branches_per_candidate", 0)) != 1:
        raise ValueError("deterministic continuation search requires one branch per candidate")

    criteria = raw.get("readiness")
    if not isinstance(criteria, Mapping):
        raise ValueError("transition-band config requires readiness criteria")
    for key in (
        "minimum_positive_candidates",
        "minimum_negative_candidates",
        "minimum_parent_groups_with_positive",
        "minimum_parent_groups_with_negative",
    ):
        if int(criteria.get(key, 0)) <= 0:
            raise ValueError(f"readiness {key} must be positive")
    if criteria.get("phasewise_stopping") is not True:
        raise ValueError("automated search requires phasewise_stopping=true")

    shells = raw.get("outer_shells")
    if not isinstance(shells, list) or not shells:
        raise ValueError("transition-band config requires at least one outer shell")
    previous_max = int(inner.get("maximum_duration", 0))
    seen_ids: set[str] = set()
    for shell in shells:
        if not isinstance(shell, Mapping):
            raise ValueError("outer shell entries must be JSON objects")
        shell_id = str(shell.get("shell_id", ""))
        if not shell_id or shell_id in seen_ids:
            raise ValueError("outer shell ids must be nonempty and unique")
        seen_ids.add(shell_id)
        durations = tuple(int(x) for x in shell.get("durations", ()))
        if len(durations) != 2 or any(value <= 0 for value in durations):
            raise ValueError("each outer shell requires exactly two positive durations")
        if tuple(sorted(durations)) != durations or durations[0] <= previous_max:
            raise ValueError("outer-shell durations must increase monotonically beyond prior shells")
        if durations[1] != 2 * durations[0]:
            raise ValueError("each outer shell must use a predeclared doubling pair")
        previous_max = durations[1]
        if int(shell.get("acquisition_protocol_seed", -1)) < 0:
            raise ValueError("outer shell acquisition_protocol_seed must be nonnegative")
        if int(shell.get("label_protocol_seed", -1)) < 0:
            raise ValueError("outer shell label_protocol_seed must be nonnegative")

    claims = raw.get("claim_boundary")
    expected_claims = {
        "training_only_search": True,
        "continuation_field_trained": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }
    if claims != expected_claims:
        raise ValueError("transition-band search claim boundary drift")
    return raw


def unique_label_rows(label_sets: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Deduplicate accumulated exact physical states and reject label conflicts."""
    by_state: dict[str, dict[str, Any]] = {}
    for rows in label_sets:
        for source in rows:
            row = dict(source)
            state_sha = str(row["state_sha256"])
            if row.get("split") != "train":
                raise ValueError("transition-band accumulation accepts TRAIN labels only")
            if str(row.get("phase")) not in PHASES:
                raise ValueError("transition-band label has unsupported phase")
            label = int(row.get("label", -1))
            if label not in (0, 1):
                raise ValueError("transition-band label must be binary")
            previous = by_state.get(state_sha)
            if previous is not None:
                if int(previous["label"]) != label or previous["phase"] != row["phase"]:
                    raise ValueError("duplicate physical state has conflicting transition-band label")
                continue
            by_state[state_sha] = row
    return list(by_state.values())


def phase_transition_band_readiness(
    rows: Sequence[Mapping[str, Any]], criteria: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Evaluate predeclared phase-wise success/failure support minima."""
    result: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        phase_rows = [row for row in rows if row["phase"] == phase]
        positives = [row for row in phase_rows if int(row["label"]) == 1]
        negatives = [row for row in phase_rows if int(row["label"]) == 0]
        positive_groups = {str(row["parent_group_id"]) for row in positives}
        negative_groups = {str(row["parent_group_id"]) for row in negatives}
        ready = (
            len(positives) >= int(criteria["minimum_positive_candidates"])
            and len(negatives) >= int(criteria["minimum_negative_candidates"])
            and len(positive_groups)
            >= int(criteria["minimum_parent_groups_with_positive"])
            and len(negative_groups)
            >= int(criteria["minimum_parent_groups_with_negative"])
        )
        result[phase] = {
            "candidate_count": len(phase_rows),
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "positive_parent_group_count": len(positive_groups),
            "negative_parent_group_count": len(negative_groups),
            "ready": bool(ready),
        }
    return result


def _validate_inner_shell(
    config: Mapping[str, Any], policy_record: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inner = config["completed_inner_shell"]
    labels_path = Path(inner["labels_path"])
    summary_path = labels_path.parent / "summary.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    summary = _read_json(summary_path)
    if not isinstance(labels, list):
        raise ValueError("completed inner-shell labels must be a JSON array")
    if summary.get("status") != "completed":
        raise ValueError("completed inner-shell summary is not completed")
    if summary.get("protocol_sha256") != inner["expected_label_protocol_sha256"]:
        raise ValueError("inner-shell label protocol SHA-256 drift")
    if summary.get("candidate_catalog_protocol_sha256") != inner[
        "expected_acquisition_protocol_sha256"
    ]:
        raise ValueError("inner-shell acquisition protocol SHA-256 drift")
    for key, summary_key in (
        ("expected_candidate_count", "candidate_count"),
        ("expected_positive_count", "positive_count"),
        ("expected_negative_count", "negative_count"),
    ):
        if int(summary.get(summary_key, -1)) != int(inner[key]):
            raise ValueError(f"inner-shell {summary_key} drift")
    if len(labels) != int(inner["expected_candidate_count"]):
        raise ValueError("inner-shell label file count drift")
    if summary.get("policy_actor_sha256") != policy_record["actor_sha256"]:
        raise ValueError("inner-shell actor identity drift")
    if summary.get("policy_payload_sha256") != policy_record["payload_sha256"]:
        raise ValueError("inner-shell policy payload identity drift")
    if any(row.get("policy_actor_sha256") != policy_record["actor_sha256"] for row in labels):
        raise ValueError("inner-shell label actor identity drift")
    identity = {
        "labels_path": str(labels_path),
        "labels_file_sha256": file_sha256(labels_path),
        "summary_file_sha256": file_sha256(summary_path),
        "label_protocol_sha256": summary["protocol_sha256"],
        "acquisition_protocol_sha256": summary["candidate_catalog_protocol_sha256"],
        "candidate_count": len(labels),
        "positive_count": int(summary["positive_count"]),
        "negative_count": int(summary["negative_count"]),
    }
    return [dict(row) for row in labels], identity


def _load_completed_shell_labels(shell_root: Path) -> list[dict[str, Any]] | None:
    labels_path = shell_root / "labels" / "labels.json"
    summary_path = shell_root / "labels" / "summary.json"
    if not labels_path.exists() and not summary_path.exists():
        return None
    if not labels_path.exists() or not summary_path.exists():
        raise ValueError(f"cannot resume incomplete shell labeling: {shell_root}")
    summary = _read_json(summary_path)
    if summary.get("status") != "completed":
        raise ValueError(f"cannot resume non-completed shell labeling: {shell_root}")
    rows = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != int(summary.get("label_count", -1)):
        raise ValueError(f"resumed shell label count drift: {shell_root}")
    return [dict(row) for row in rows]


def _shell_phase_stats(
    labels: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    source = {str(row["candidate_id"]): row for row in catalog.get("entries", ())}
    result: dict[str, Any] = {}
    for phase in PHASES:
        phase_rows = [row for row in labels if row["phase"] == phase]
        positives = sum(int(row["label"]) for row in phase_rows)
        failures = Counter(str(row["outcome_class"]) for row in phase_rows if not row["label"])
        by_duration: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "total": 0})
        for row in phase_rows:
            candidate = source[str(row["candidate_id"])]
            duration = str(int(candidate["perturbation"]["duration"]))
            by_duration[duration]["total"] += 1
            by_duration[duration]["positive"] += int(row["label"])
        result[phase] = {
            "candidate_count": len(phase_rows),
            "positive_count": positives,
            "negative_count": len(phase_rows) - positives,
            "negative_outcomes": dict(sorted(failures.items())),
            "by_duration": dict(sorted(by_duration.items(), key=lambda item: int(item[0]))),
        }
    return result


def search_unified_transition_band(
    config_path: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the full predeclared TRAIN transition-band search without manual shell decisions."""
    config_path = Path(config_path)
    config = load_transition_band_search_config(config_path)
    output = Path(config["output_dir"])

    frozen_path = Path(config["frozen_policy"])
    frozen = load_frozen_unified_manifest(frozen_path)
    policy_record = frozen["policy"]
    if int(policy_record["iteration"]) != int(config["iteration"]):
        raise ValueError("transition-band config/frozen-policy iteration mismatch")

    formal = load_unified_formal_config(Path(policy_record["formal_config"]))
    if formal.config_sha256 != policy_record["formal_config_sha256"]:
        raise ValueError("transition-band frozen policy/formal config drift")
    artifact = load_soft_tube(Path(formal.soft_tube_path))
    if artifact.manifest["manifest_sha256"] != formal.soft_tube_manifest_sha256:
        raise ValueError("transition-band source Tube manifest drift")
    expected_tube = config.get("source_tube", {})
    if expected_tube.get("manifest_sha256") != artifact.manifest["manifest_sha256"]:
        raise ValueError("transition-band config/source Tube identity drift")

    inner_labels, inner_identity = _validate_inner_shell(config, policy_record)
    anchors, anchor_audit = select_tube_boundary_anchors(
        artifact,
        max_per_phase=int(config["fixed_acquisition"]["anchors_per_phase"]),
        frontier_score_ceiling=float(config["fixed_acquisition"]["frontier_score_ceiling"]),
    )
    if not anchors:
        raise ValueError("transition-band search selected no source Tube anchors")

    directions = len(config["fixed_acquisition"]["action_names"]) * len(
        config["fixed_acquisition"]["signs"]
    )
    strengths = len(config["fixed_acquisition"]["strengths"])
    max_acquisition = 0
    max_labeling = 0
    for shell in config["outer_shells"]:
        duration_sum = sum(int(x) for x in shell["durations"])
        max_acquisition += len(anchors) * directions * strengths * duration_sum
        max_variants = len(anchors) * directions * strengths * len(shell["durations"])
        max_labeling += max_variants * int(config["continuation_labeling"]["max_ticks"])

    protocol_base = {
        "schema": TRANSITION_BAND_SEARCH_PROTOCOL_SCHEMA,
        "status": "predeclared",
        "purpose": "automated_phasewise_duration_only_transition_band_localization",
        "iteration": int(config["iteration"]),
        "repository_head": _repository_head(),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "frozen_policy_path": str(frozen_path),
        "frozen_policy_file_sha256": file_sha256(frozen_path),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "policy_formal_config_sha256": policy_record["formal_config_sha256"],
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "inner_shell": inner_identity,
        "anchor_audit": anchor_audit,
        "exact_anchor_identities": [
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
        ],
        "fixed_acquisition": config["fixed_acquisition"],
        "continuation_labeling": config["continuation_labeling"],
        "readiness": config["readiness"],
        "outer_shells": config["outer_shells"],
        "maximum_outer_acquisition_environment_interactions": max_acquisition,
        "maximum_outer_labeling_environment_interactions": max_labeling,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": config["claim_boundary"],
    }
    protocol_sha = _canonical_sha256(protocol_base)
    protocol = {**protocol_base, "protocol_sha256": protocol_sha}

    if output.exists():
        if not resume:
            raise FileExistsError(f"transition-band search output already exists: {output}")
        existing = _read_json(output / "search_protocol.json")
        if existing != protocol:
            raise ValueError("cannot resume transition-band search under a different protocol")
        completed_summary = output / "summary.json"
        if completed_summary.exists():
            summary = _read_json(completed_summary)
            if summary.get("status") in ("transition_band_ready", "search_exhausted"):
                return summary
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "search_protocol.json", protocol)

    if jax.default_backend() != "gpu":
        raise RuntimeError("automated transition-band search requires the visible JAX GPU")
    runtime_config, runtime_artifact, env = build_unified_formal_environment(
        Path(policy_record["formal_config"])
    )
    if runtime_config.config_sha256 != formal.config_sha256:
        raise ValueError("transition-band runtime formal config drift")
    if runtime_artifact.manifest["manifest_sha256"] != artifact.manifest["manifest_sha256"]:
        raise ValueError("transition-band runtime source Tube drift")
    if env._bundle.xml_sha256 != policy_record["xml_sha256"]:
        raise ValueError("transition-band runtime XML drift")
    payload = load_checkpoint(
        Path(policy_record["checkpoint"]), expected=checkpoint_identity(runtime_config, env)
    )
    if int(payload.training_transitions) != int(policy_record["source_training_transitions"]):
        raise ValueError("transition-band checkpoint transition drift")
    if file_sha256(Path(policy_record["checkpoint"]) / "payload.pkl") != policy_record[
        "payload_sha256"
    ]:
        raise ValueError("transition-band checkpoint payload SHA-256 drift")
    policy: Callable[[Any, Any], Any] = jax.jit(
        make_checkpoint_policy(env, payload, deterministic=True)
    )

    label_sets: list[list[dict[str, Any]]] = [inner_labels]
    shell_reports: list[dict[str, Any]] = []
    total_acquisition_interactions = 0
    total_labeling_interactions = int(inner_identity.get("environment_interactions", 0))

    # Resume completed shells in order and reconstruct accumulated readiness.
    for shell_index, shell in enumerate(config["outer_shells"], start=1):
        shell_root = output / f"shell_{shell_index:02d}_{shell['shell_id']}"
        resumed = _load_completed_shell_labels(shell_root) if shell_root.exists() else None
        if resumed is None:
            break
        label_sets.append(resumed)
        acq_summary = _read_json(shell_root / "acquisition" / "summary.json")
        label_summary = _read_json(shell_root / "labels" / "summary.json")
        total_acquisition_interactions += int(acq_summary["environment_interactions"])
        total_labeling_interactions += int(label_summary["environment_interactions"])
        catalog = _read_json(shell_root / "acquisition" / "catalog.json")
        shell_reports.append(
            {
                "shell_id": shell["shell_id"],
                "durations": shell["durations"],
                "status": "completed",
                "resumed": True,
                "acquisition_environment_interactions": acq_summary[
                    "environment_interactions"
                ],
                "labeling_environment_interactions": label_summary[
                    "environment_interactions"
                ],
                "candidate_count": label_summary["candidate_count"],
                "phase_stats": _shell_phase_stats(resumed, catalog),
            }
        )

    accumulated = unique_label_rows(label_sets)
    readiness = phase_transition_band_readiness(accumulated, config["readiness"])
    completed_shell_count = len(shell_reports)

    for shell_index, shell in enumerate(
        config["outer_shells"][completed_shell_count:], start=completed_shell_count + 1
    ):
        unresolved = {phase for phase in PHASES if not readiness[phase]["ready"]}
        if not unresolved:
            break
        shell_root = output / f"shell_{shell_index:02d}_{shell['shell_id']}"
        if shell_root.exists():
            raise ValueError(f"cannot resume incomplete shell directory: {shell_root}")
        shell_root.mkdir(parents=True, exist_ok=False)
        active_anchors: tuple[TubeBoundaryAnchor, ...] = tuple(
            anchor for anchor in anchors if anchor.phase in unresolved
        )
        if not active_anchors:
            raise ValueError("unresolved transition-band phase has no source Tube anchor")

        acquisition_dir = shell_root / "acquisition"
        acquisition = collect_unified_boundary_candidates(
            active_anchors,
            acquisition_dir,
            env=env,
            policy=policy,
            policy_record=policy_record,
            frozen_manifest_sha256=file_sha256(frozen_path),
            protocol_seed=int(shell["acquisition_protocol_seed"]),
            frontier_score_ceiling=float(config["fixed_acquisition"]["frontier_score_ceiling"]),
            strengths=tuple(float(x) for x in config["fixed_acquisition"]["strengths"]),
            durations=tuple(int(x) for x in shell["durations"]),
            action_names=tuple(config["fixed_acquisition"]["action_names"]),
            signs=tuple(int(x) for x in config["fixed_acquisition"]["signs"]),
        )
        total_acquisition_interactions += int(acquisition["environment_interactions"])
        _write_json(
            acquisition_dir / "anchor_audit.json",
            {
                **anchor_audit,
                "active_phases": sorted(unresolved),
                "selected_anchor_count": len(active_anchors),
            },
        )

        candidate_count = int(acquisition["candidate_count"])
        if candidate_count == 0:
            shell_report = {
                "shell_id": shell["shell_id"],
                "durations": shell["durations"],
                "status": "no_candidates",
                "active_phases": sorted(unresolved),
                "acquisition_environment_interactions": acquisition[
                    "environment_interactions"
                ],
                "candidate_count": 0,
            }
            shell_reports.append(shell_report)
            _write_json(shell_root / "shell_summary.json", shell_report)
            break

        labels_dir = shell_root / "labels"
        label_report = label_unified_continuations(
            acquisition_dir / "catalog.json",
            labels_dir,
            env=env,
            policy=policy,
            policy_record=policy_record,
            frozen_manifest_sha256=file_sha256(frozen_path),
            max_ticks=int(config["continuation_labeling"]["max_ticks"]),
            protocol_seed=int(shell["label_protocol_seed"]),
        )
        total_labeling_interactions += int(label_report["environment_interactions"])
        shell_labels = json.loads((labels_dir / "labels.json").read_text(encoding="utf-8"))
        label_sets.append([dict(row) for row in shell_labels])
        accumulated = unique_label_rows(label_sets)
        readiness = phase_transition_band_readiness(accumulated, config["readiness"])
        shell_report = {
            "shell_id": shell["shell_id"],
            "durations": shell["durations"],
            "status": "completed",
            "active_phases": sorted(unresolved),
            "acquisition_environment_interactions": acquisition[
                "environment_interactions"
            ],
            "labeling_environment_interactions": label_report[
                "environment_interactions"
            ],
            "candidate_count": label_report["candidate_count"],
            "phase_stats": _shell_phase_stats(shell_labels, acquisition),
            "accumulated_readiness_after_shell": readiness,
        }
        shell_reports.append(shell_report)
        _write_json(shell_root / "shell_summary.json", shell_report)
        _write_json(
            output / "progress.json",
            {
                "schema": TRANSITION_BAND_SEARCH_SUMMARY_SCHEMA,
                "status": "searching",
                "protocol_sha256": protocol_sha,
                "completed_shell_count": len(shell_reports),
                "accumulated_unique_label_count": len(accumulated),
                "readiness": readiness,
                "shells": shell_reports,
                "training_transitions": 0,
            },
        )

    accumulated = unique_label_rows(label_sets)
    readiness = phase_transition_band_readiness(accumulated, config["readiness"])
    all_ready = all(readiness[phase]["ready"] for phase in PHASES)
    status = "transition_band_ready" if all_ready else "search_exhausted"
    _write_json(output / "accumulated_train_labels.json", {"entries": accumulated})
    summary = {
        "schema": TRANSITION_BAND_SEARCH_SUMMARY_SCHEMA,
        "status": status,
        "protocol_sha256": protocol_sha,
        "iteration": int(config["iteration"]),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "completed_shell_count": len(shell_reports),
        "shells": shell_reports,
        "accumulated_unique_label_count": len(accumulated),
        "readiness": readiness,
        "outer_acquisition_environment_interactions": total_acquisition_interactions,
        "outer_labeling_environment_interactions": total_labeling_interactions,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "next_scientific_gate": (
            "freeze accumulated expansion TRAIN labels and design group-disjoint expansion "
            "validation before fitting/calibrating C_up^0 and C_down^0"
            if all_ready
            else "stop automatic search and make a new explicit method decision; do not change "
            "strengths, axes, reward, physics, policy, or Tube_0 post hoc"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output / "summary.json", summary)
    return summary
