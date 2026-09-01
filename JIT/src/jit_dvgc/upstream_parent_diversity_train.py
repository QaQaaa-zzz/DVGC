"""Expand upstream TRAIN parent checkpoint diversity under frozen pi_0.

This stage is triggered only after the first C_up^0 field passed TRAIN seed-group
LOGO but failed the already-consumed independent validation.  It adds new TRAIN
parents from different pi_up checkpoint domains without reading consumed
validation rows or outcomes, generates bounded real-dynamics probes under the
frozen unified policy, and labels them with the existing unified continuation
labeler.

It does not reselect a continuation field, construct Tube_1, train PPO, inspect
TEST/final data, or restore authority to the consumed validation bank.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import jax
from jax import numpy as jp
import numpy as np

from .checkpoint import load_checkpoint
from .config import file_sha256
from .constants import ACTION_ORDER, END_REASONS
from .expansion_validation_runtime import restore_validation_anchor_as_unified
from .handoff_snapshot import HandoffSnapshot, load_snapshot
from .iteration_train_evidence import canonical_sha256, load_frozen_iteration_train_evidence
from .ppo import make_checkpoint_policy
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


CONFIG_SCHEMA = "jit_upstream_parent_diversity_train_config_v1"
PROTOCOL_SCHEMA = "jit_upstream_parent_diversity_train_protocol_v1"
ACQUISITION_PROTOCOL_SCHEMA = "jit_upstream_parent_diversity_acquisition_protocol_v1"
SUMMARY_SCHEMA = "jit_upstream_parent_diversity_train_summary_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(value: Any, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be SHA-256")
    return text


def _repository_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("upstream parent diversity runtime requires a Git checkout") from exc


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
    return all(np.isfinite(array).all() for array in arrays)


def load_upstream_parent_diversity_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported upstream parent diversity config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("upstream parent diversity protocol drift")
    if protocol.get("status") != "predeclared_after_train_logo_domain_shift":
        raise ValueError("upstream parent diversity protocol status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("upstream parent diversity policy identity drift")
    for field in (
        "frozen_policy_file_sha256",
        "policy_actor_sha256",
        "policy_payload_sha256",
        "xml_sha256",
        "frozen_train_manifest_sha256",
        "prior_train_logo_protocol_sha256",
        "prior_train_logo_summary_sha256",
        "source_catalog_file_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if protocol.get("source_role") != "ascending_entry":
        raise ValueError("upstream parent diversity source role drift")
    if protocol.get("source_checkpoint_transitions") != [7987200, 9977856]:
        raise ValueError("upstream parent diversity checkpoint-domain contract drift")
    if protocol.get("source_seeds") != [1000001, 1000002, 1000003, 1000004, 1000005]:
        raise ValueError("upstream parent diversity TRAIN seed contract drift")
    if int(protocol.get("forbidden_consumed_validation_seed", -1)) != 1000006:
        raise ValueError("upstream parent diversity consumed-validation seed guard drift")
    expected_panel = {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.025, 0.1],
        "durations": [4, 8, 16],
        "terminal_clipping": True,
        "max_label_ticks": 400,
    }
    if protocol.get("panel") != expected_panel:
        raise ValueError("upstream parent diversity fixed panel drift")
    if int(protocol.get("acquisition_seed", -1)) != 9520000:
        raise ValueError("upstream parent diversity acquisition seed drift")
    if int(protocol.get("label_seed", -1)) != 9521000:
        raise ValueError("upstream parent diversity label seed drift")
    expected_near = {
        "reject_frozen_train_exact_state": True,
        "reject_frozen_train_actor_observation_near_duplicate": True,
        "actor_observation_atol": 0.01,
        "reject_duplicate_new_train_state": True,
    }
    if protocol.get("near_duplicate_audit") != expected_near:
        raise ValueError("upstream parent diversity duplicate policy drift")
    expected_budget = {
        "parent_count": 10,
        "attempt_count": 480,
        "maximum_acquisition_environment_interactions": 4480,
        "maximum_successful_labeling_environment_interactions": 192000,
        "training_transitions": 0,
    }
    if protocol.get("interaction_budget") != expected_budget:
        raise ValueError("upstream parent diversity interaction budget drift")
    expected_data = {
        "train_only": True,
        "source_catalog_metadata_only_no_expert_outcomes": True,
        "consumed_validation_rows_read": False,
        "consumed_validation_predictions_read": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    if protocol.get("data_policy") != expected_data:
        raise ValueError("upstream parent diversity data policy drift")
    expected_claims = {
        "checkpoint_domain_train_expansion_only": True,
        "continuation_field_reselected": False,
        "fresh_validation_authorized": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }
    if protocol.get("claim_boundary") != expected_claims:
        raise ValueError("upstream parent diversity claim boundary drift")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("upstream parent diversity scientific protocol SHA drift")
    if not str(config.get("output_dir", "")):
        raise ValueError("upstream parent diversity output path missing")
    return config


def _select_source_anchors(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    catalog_path = Path(str(protocol["source_catalog"]))
    if file_sha256(catalog_path) != protocol["source_catalog_file_sha256"]:
        raise ValueError("upstream source catalog SHA drift")
    catalog = _read_object(catalog_path)
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise ValueError("upstream source catalog entries missing")
    selected: list[dict[str, Any]] = []
    for transition in protocol["source_checkpoint_transitions"]:
        for seed in protocol["source_seeds"]:
            parent = f"transition_{int(transition)}__{int(seed)}"
            matches = [
                dict(row)
                for row in entries
                if str(row.get("parent_group_id")) == parent
                and str(row.get("role")) == protocol["source_role"]
                and int(row.get("seed", -1)) == int(seed)
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one upstream source anchor for {parent}")
            row = matches[0]
            if int(seed) == int(protocol["forbidden_consumed_validation_seed"]):
                raise ValueError("consumed validation seed entered TRAIN source selection")
            if str(row.get("parent_trajectory")) != parent:
                raise ValueError("upstream source parent trajectory drift")
            _sha(row.get("state_sha256"), field="upstream source state_sha256")
            if not str(row.get("source_bank", "")) or not str(row.get("snapshot", "")):
                raise ValueError("upstream source snapshot identity missing")
            selected.append(
                {
                    **row,
                    "parent_domain_id": f"transition_{int(transition)}",
                    "source_checkpoint_transition": int(transition),
                }
            )
    if len(selected) != int(protocol["interaction_budget"]["parent_count"]):
        raise ValueError("upstream parent selection count drift")
    if len({str(row["parent_group_id"]) for row in selected}) != len(selected):
        raise ValueError("upstream parent selection repeats a group")
    return tuple(selected)


def enumerate_parent_diversity_attempts(
    protocol: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    panel = protocol["panel"]
    attempts: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(anchors):
        for duration in panel["durations"]:
            for strength in panel["strengths"]:
                for action_name in panel["action_names"]:
                    action_index = ACTION_ORDER.index(str(action_name))
                    for sign in panel["signs"]:
                        basis = [0.0] * len(ACTION_ORDER)
                        basis[action_index] = float(sign)
                        attempts.append(
                            {
                                "attempt_index": len(attempts),
                                "anchor_index": int(anchor_index),
                                "parent_group_id": str(anchor["parent_group_id"]),
                                "parent_domain_id": str(anchor["parent_domain_id"]),
                                "parent_state_sha256": str(anchor["state_sha256"]),
                                "source_bank": str(anchor["source_bank"]),
                                "source_snapshot": str(anchor["snapshot"]),
                                "source_checkpoint_transition": int(
                                    anchor["source_checkpoint_transition"]
                                ),
                                "seed": int(anchor["seed"]),
                                "role": str(anchor["role"]),
                                "tick": int(anchor["tick"]),
                                "action_name": str(action_name),
                                "action_index": int(action_index),
                                "sign": int(sign),
                                "basis_vector": basis,
                                "strength": float(strength),
                                "duration": int(duration),
                            }
                        )
    if len(attempts) != int(protocol["interaction_budget"]["attempt_count"]):
        raise ValueError("upstream parent diversity attempt count drift")
    return tuple(attempts)


def audit_upstream_parent_diversity(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_upstream_parent_diversity_config(config_path)
    protocol = config["protocol"]
    train_manifest, train_rows = load_frozen_iteration_train_evidence(
        Path(str(protocol["frozen_train_evidence"]))
    )
    if train_manifest["manifest_sha256"] != protocol["frozen_train_manifest_sha256"]:
        raise ValueError("upstream parent diversity frozen TRAIN manifest drift")
    if train_manifest["policy_actor_sha256"] != protocol["policy_actor_sha256"]:
        raise ValueError("upstream parent diversity TRAIN actor drift")
    if train_manifest["policy_payload_sha256"] != protocol["policy_payload_sha256"]:
        raise ValueError("upstream parent diversity TRAIN payload drift")
    prior = _read_object(Path(str(protocol["prior_train_logo_summary"])))
    if prior.get("summary_sha256") != protocol["prior_train_logo_summary_sha256"]:
        raise ValueError("upstream parent diversity prior LOGO summary drift")
    if prior.get("protocol_sha256") != protocol["prior_train_logo_protocol_sha256"]:
        raise ValueError("upstream parent diversity prior LOGO protocol drift")
    if prior.get("train_group_generalization_supported") is not True:
        raise ValueError("upstream parent diversity requires successful TRAIN LOGO")
    if prior.get("diagnosis") != (
        "current_linear_model_generalizes_across_existing_train_groups_but_failed_"
        "independent_validation_indicating_train_parent_domain_shift"
    ):
        raise ValueError("upstream parent diversity prior diagnosis drift")
    if prior.get("consumed_validation_rows_reused") is not False:
        raise ValueError("prior LOGO unexpectedly reused validation rows")
    if prior.get("consumed_validation_predictions_reused") is not False:
        raise ValueError("prior LOGO unexpectedly reused validation predictions")
    anchors = _select_source_anchors(protocol)
    existing_groups = {str(row["parent_group_id"]) for row in train_rows}
    if existing_groups.intersection(str(row["parent_group_id"]) for row in anchors):
        raise ValueError("new upstream parent group already exists in frozen TRAIN")
    attempts = enumerate_parent_diversity_attempts(protocol, anchors)
    return {
        "schema": "jit_upstream_parent_diversity_audit_v1",
        "status": "ready",
        "scientific_protocol_sha256": str(config["expected_protocol_sha256"]),
        "selected_parent_count": len(anchors),
        "selected_parent_group_ids": [str(row["parent_group_id"]) for row in anchors],
        "selected_checkpoint_domains": sorted(
            {str(row["parent_domain_id"]) for row in anchors}
        ),
        "attempt_count": len(attempts),
        "maximum_acquisition_environment_interactions": int(
            protocol["interaction_budget"]["maximum_acquisition_environment_interactions"]
        ),
        "maximum_successful_labeling_environment_interactions": int(
            protocol["interaction_budget"][
                "maximum_successful_labeling_environment_interactions"
            ]
        ),
        "consumed_validation_rows_read": False,
        "consumed_validation_predictions_read": False,
        "environment_interactions": 0,
        "training_transitions": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }


def _acquisition_protocol(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
) -> dict[str, Any]:
    scientific = config["protocol"]
    attempts = enumerate_parent_diversity_attempts(scientific, anchors)
    base = {
        "schema": ACQUISITION_PROTOCOL_SCHEMA,
        "status": "predeclared",
        "purpose": "upstream_checkpoint_domain_train_parent_expansion",
        "repository_head": _repository_head(),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "scientific_protocol_sha256": str(config["expected_protocol_sha256"]),
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "frozen_unified_manifest_sha256": str(frozen_manifest_sha256),
        "frozen_train_manifest_sha256": str(scientific["frozen_train_manifest_sha256"]),
        "source_catalog_file_sha256": str(scientific["source_catalog_file_sha256"]),
        "selected_parent_group_ids": [str(row["parent_group_id"]) for row in anchors],
        "selected_checkpoint_domains": sorted(
            {str(row["parent_domain_id"]) for row in anchors}
        ),
        "attempt_schedule_sha256": canonical_sha256({"attempts": list(attempts)}),
        "attempt_count": len(attempts),
        "acquisition_seed": int(scientific["acquisition_seed"]),
        "terminal_clipping": True,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    return {**base, "protocol_sha256": canonical_sha256(base)}


def _near_existing(observation: np.ndarray, train_observations: np.ndarray, atol: float) -> bool:
    return bool(
        np.any(np.all(np.abs(train_observations - observation) <= float(atol), axis=1))
    )


def _collect_candidates(
    *,
    protocol: Mapping[str, Any],
    acquisition_protocol: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
    train_rows: Sequence[Mapping[str, Any]],
    output: Path,
    step_fn: Callable[[Any, Any], Any],
) -> dict[str, Any]:
    attempts = enumerate_parent_diversity_attempts(protocol, anchors)
    catalog_path = Path(str(protocol["source_catalog"]))
    anchor_snapshots: dict[int, HandoffSnapshot] = {}
    for index, anchor in enumerate(anchors):
        snapshot = load_snapshot(
            catalog_path.parent / str(anchor["source_bank"]) / str(anchor["snapshot"])
        )
        if snapshot.xml_sha256 != protocol["xml_sha256"]:
            raise ValueError("upstream parent source XML drift")
        if snapshot.parent_trajectory != str(anchor["parent_group_id"]):
            raise ValueError("upstream parent source trajectory drift")
        anchor_snapshots[index] = snapshot

    train_states = {str(row["state_sha256"]) for row in train_rows}
    train_observations = np.asarray(
        [row["actor_observation"] for row in train_rows], dtype=np.float32
    )
    tolerance = float(protocol["near_duplicate_audit"]["actor_observation_atol"])
    bank = output / "candidate_bank"
    (bank / "snapshots").mkdir(parents=True, exist_ok=False)
    base_key = jax.random.PRNGKey(int(protocol["acquisition_seed"]))
    entries: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    exclusions: Counter[str] = Counter()
    terminal_outcomes: Counter[str] = Counter()
    interactions = 0

    for attempt in attempts:
        state = restore_validation_anchor_as_unified(
            anchor_snapshots[int(attempt["anchor_index"])],
            phase="upstream",
            env=env,
            parent_group_index=int(attempt["attempt_index"]),
        )
        candidate_state = None
        terminal_clipped = False
        terminal_meta: dict[str, Any] | None = None
        rejected: str | None = None
        executed = 0
        attempt_key = jax.random.fold_in(base_key, int(attempt["attempt_index"]))
        for perturb_step in range(int(attempt["duration"])):
            previous = state
            action_key = jax.random.fold_in(attempt_key, int(perturb_step))
            result = policy(state.obs, action_key)
            nominal_device = result[0] if isinstance(result, tuple) else result
            nominal = np.asarray(jax.device_get(nominal_device), dtype=np.float32).reshape(-1)
            if nominal.shape != (4,) or not np.isfinite(nominal).all():
                raise ValueError("frozen pi_0 returned invalid parent-diversity action")
            requested = nominal + np.asarray(attempt["basis_vector"], dtype=np.float32) * np.float32(
                attempt["strength"]
            )
            perturbed = np.clip(requested, -1.0, 1.0).astype(np.float32)
            state = step_fn(state, jp.asarray(perturbed))
            jax.block_until_ready(state)
            interactions += 1
            executed += 1
            if _truth(state.info["expert_switching_used"]):
                raise ValueError("parent-diversity acquisition used expert switching")
            if not _finite_state(state, perturbed):
                rejected = "nonfinite"
                break
            if _integer(state.info["active_phase"]) != 0:
                rejected = "phase_transition"
                break
            if _truth(state.done):
                if not _finite_state(previous):
                    rejected = "terminal_without_finite_predecessor"
                    break
                candidate_state = previous
                terminal_clipped = True
                end_code = _integer(state.info["end_code"])
                end_reason = END_REASONS.get(end_code, f"unknown_{end_code}")
                terminal_outcomes[end_reason] += 1
                terminal_meta = {
                    "end_code": end_code,
                    "end_reason": end_reason,
                    "success": _truth(state.info["success"]),
                    "physical_failure": _truth(state.info["physical_failure"]),
                    "timeout": _truth(state.info["timeout"]),
                    "terminal_interaction_index": executed,
                }
                break
        if rejected is not None:
            exclusions[rejected] += 1
            continue
        if candidate_state is None:
            candidate_state = state
        if _truth(candidate_state.done) or _integer(candidate_state.info["active_phase"]) != 0:
            exclusions["invalid_candidate_phase_or_terminal"] += 1
            continue
        unified_snapshot = capture_unified_envelope_snapshot(
            candidate_state,
            env=env,
            parent_trajectory=str(attempt["parent_group_id"]),
            parent_state_sha256=str(attempt["parent_state_sha256"]),
            config_sha256=str(policy_record["formal_config_sha256"]),
            policy_actor_sha256=str(policy_record["actor_sha256"]),
            policy_payload_sha256=str(policy_record["payload_sha256"]),
            policy_iteration=int(policy_record["iteration"]),
        )
        state_sha = physical_state_sha256(unified_snapshot)
        actor_observation = np.asarray(unified_snapshot.observation, dtype=np.float32)
        if state_sha in train_states:
            exclusions["frozen_train_exact_state"] += 1
            continue
        if _near_existing(actor_observation, train_observations, tolerance):
            exclusions["frozen_train_near_duplicate_observation"] += 1
            continue
        if state_sha in seen_states:
            exclusions["duplicate_new_train_state"] += 1
            continue
        seen_states.add(state_sha)
        relative = Path("snapshots") / f"candidate_{len(entries):06d}"
        save_unified_envelope_snapshot(bank / relative, unified_snapshot)
        entries.append(
            {
                "candidate_id": f"pi0_up_parentdiv_{len(entries):06d}",
                "candidate_kind": "reachable_unified_frontier_probe",
                "split": "train",
                "phase": "upstream",
                "phase_index": 0,
                "snapshot": str(relative),
                "source_bank": "candidate_bank",
                "state_sha256": state_sha,
                "parent_group_id": str(attempt["parent_group_id"]),
                "parent_domain_id": str(attempt["parent_domain_id"]),
                "parent_state_sha256": str(attempt["parent_state_sha256"]),
                "policy_iteration": int(policy_record["iteration"]),
                "policy_actor_sha256": str(policy_record["actor_sha256"]),
                "policy_payload_sha256": str(policy_record["payload_sha256"]),
                "protocol_sha256": str(acquisition_protocol["protocol_sha256"]),
                "source_anchor": {
                    "source_bank": str(attempt["source_bank"]),
                    "snapshot": str(attempt["source_snapshot"]),
                    "source_checkpoint_transition": int(
                        attempt["source_checkpoint_transition"]
                    ),
                    "seed": int(attempt["seed"]),
                    "role": str(attempt["role"]),
                    "tick": int(attempt["tick"]),
                },
                "actor_observation": actor_observation.tolist(),
                "perturbation": {
                    "action_name": str(attempt["action_name"]),
                    "action_index": int(attempt["action_index"]),
                    "sign": int(attempt["sign"]),
                    "strength": float(attempt["strength"]),
                    "duration": int(attempt["duration"]),
                    "executed_interactions": int(executed),
                    "terminal_clipped": bool(terminal_clipped),
                    "terminal_probe_outcome": terminal_meta,
                },
            }
        )

    maximum = int(protocol["interaction_budget"]["maximum_acquisition_environment_interactions"])
    if interactions > maximum:
        raise ValueError("upstream parent diversity acquisition exceeded interaction ceiling")
    catalog = {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "frozen_unified_manifest_sha256": str(frozen_manifest_sha256),
        "protocol_sha256": str(acquisition_protocol["protocol_sha256"]),
        "attempt_count": len(attempts),
        "candidate_count": len(entries),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "terminal_probe_outcomes": dict(sorted(terminal_outcomes.items())),
        "environment_interactions": int(interactions),
        "maximum_environment_interactions": maximum,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
        "entries": entries,
    }
    _write_json(output / "candidate_catalog.json", catalog)
    return catalog


def _domain_stats(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain in ("transition_7987200", "transition_9977856"):
        rows = [row for row in labels if str(row["parent_group_id"]).startswith(domain + "__")]
        positive = sum(int(row["label"]) for row in rows)
        positive_groups = {
            str(row["parent_group_id"]) for row in rows if int(row["label"]) == 1
        }
        negative_groups = {
            str(row["parent_group_id"]) for row in rows if int(row["label"]) == 0
        }
        result[domain] = {
            "candidate_count": len(rows),
            "positive_count": positive,
            "negative_count": len(rows) - positive,
            "positive_parent_group_count": len(positive_groups),
            "negative_parent_group_count": len(negative_groups),
        }
    return result


def execute_upstream_parent_diversity_train(
    config_path: Path, *, resume: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path)
    audit = audit_upstream_parent_diversity(config_path)
    config = load_upstream_parent_diversity_config(config_path)
    protocol = config["protocol"]
    output = Path(str(config["output_dir"]))
    frozen_path = Path(str(protocol["frozen_policy"]))
    frozen_manifest_sha256 = file_sha256(frozen_path)
    if frozen_manifest_sha256 != protocol["frozen_policy_file_sha256"]:
        raise ValueError("upstream parent diversity frozen policy file drift")
    frozen = load_frozen_unified_manifest(frozen_path)
    policy_record = frozen["policy"]
    if policy_record["actor_sha256"] != protocol["policy_actor_sha256"]:
        raise ValueError("upstream parent diversity actor drift")
    if policy_record["payload_sha256"] != protocol["policy_payload_sha256"]:
        raise ValueError("upstream parent diversity payload drift")
    train_manifest, train_rows = load_frozen_iteration_train_evidence(
        Path(str(protocol["frozen_train_evidence"]))
    )
    anchors = _select_source_anchors(protocol)
    acquisition_protocol = _acquisition_protocol(
        config_path=config_path,
        config=config,
        anchors=anchors,
        policy_record=policy_record,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )

    if output.exists():
        if not resume:
            raise FileExistsError(f"upstream parent diversity output already exists: {output}")
        if not (output / "protocol.json").is_file():
            raise ValueError("cannot resume incomplete parent-diversity acquisition")
        existing_protocol = _read_object(output / "protocol.json")
        if existing_protocol != acquisition_protocol:
            raise ValueError("cannot resume parent diversity under protocol drift")
        if (output / "summary.json").is_file():
            summary = _read_object(output / "summary.json")
            if summary.get("status") == "completed":
                return summary
        if not (output / "candidate_catalog.json").is_file():
            raise ValueError("parent-diversity acquisition was interrupted before catalog completion")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "protocol.json", acquisition_protocol)
        _write_json(output / "audit.json", audit)

    if jax.default_backend() != "gpu":
        raise RuntimeError("upstream parent diversity runtime requires the visible JAX GPU")
    formal = load_unified_formal_config(Path(policy_record["formal_config"]))
    runtime_config, runtime_artifact, env = build_unified_formal_environment(
        Path(policy_record["formal_config"])
    )
    if runtime_config.config_sha256 != formal.config_sha256:
        raise ValueError("upstream parent diversity formal config drift")
    if runtime_artifact.manifest["manifest_sha256"] != train_manifest["source_tube_manifest_sha256"]:
        raise ValueError("upstream parent diversity source Tube drift")
    if env._bundle.xml_sha256 != protocol["xml_sha256"]:
        raise ValueError("upstream parent diversity XML drift")
    payload = load_checkpoint(
        Path(policy_record["checkpoint"]), expected=checkpoint_identity(runtime_config, env)
    )
    if file_sha256(Path(policy_record["checkpoint"]) / "payload.pkl") != policy_record["payload_sha256"]:
        raise ValueError("upstream parent diversity checkpoint payload drift")
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))
    step_fn = jax.jit(env.step)

    if (output / "candidate_catalog.json").is_file():
        catalog = _read_object(output / "candidate_catalog.json")
        validate_unified_boundary_catalog(
            catalog,
            policy_record=policy_record,
            frozen_manifest_sha256=frozen_manifest_sha256,
        )
    else:
        catalog = _collect_candidates(
            protocol=protocol,
            acquisition_protocol=acquisition_protocol,
            anchors=anchors,
            env=env,
            policy=policy,
            policy_record=policy_record,
            frozen_manifest_sha256=frozen_manifest_sha256,
            train_rows=train_rows,
            output=output,
            step_fn=step_fn,
        )

    failed_label_interactions = 0
    label_dirs = sorted(output.glob("labels_*"))
    completed_dir: Path | None = None
    for directory in label_dirs:
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        record = _read_object(summary_path)
        if record.get("status") == "completed":
            completed_dir = directory
            break
        if record.get("status") == "engineering_error":
            failed_label_interactions += int(record.get("environment_interactions", 0))
    if completed_dir is None:
        attempt_index = len(label_dirs)
        if attempt_index > 0 and not resume:
            raise FileExistsError("failed parent-diversity label attempt exists; use --resume")
        label_dir = output / (
            "labels_attempt_00" if attempt_index == 0 else f"labels_retry_{attempt_index:02d}"
        )
        try:
            label_unified_continuations(
                output / "candidate_catalog.json",
                label_dir,
                env=env,
                policy=policy,
                policy_record=policy_record,
                frozen_manifest_sha256=frozen_manifest_sha256,
                max_ticks=int(protocol["panel"]["max_label_ticks"]),
                protocol_seed=int(protocol["label_seed"]),
                compiled_step_fn=step_fn,
            )
        except BaseException:
            raise
        completed_dir = label_dir
    label_summary = _read_object(completed_dir / "summary.json")
    labels_value = json.loads((completed_dir / "labels.json").read_text(encoding="utf-8"))
    if not isinstance(labels_value, list):
        raise ValueError("parent-diversity completed labels must be an array")
    labels = [dict(row) for row in labels_value]
    domain_stats = _domain_stats(labels)
    successful_label_interactions = int(label_summary["environment_interactions"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "upstream_checkpoint_domain_train_parent_expansion",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "scientific_protocol_sha256": str(config["expected_protocol_sha256"]),
        "acquisition_protocol_sha256": str(acquisition_protocol["protocol_sha256"]),
        "selected_parent_count": len(anchors),
        "selected_checkpoint_domains": audit["selected_checkpoint_domains"],
        "attempt_count": int(catalog["attempt_count"]),
        "candidate_count": int(catalog["candidate_count"]),
        "positive_count": int(label_summary["positive_count"]),
        "negative_count": int(label_summary["negative_count"]),
        "domain_stats": domain_stats,
        "acquisition_exclusion_counts": dict(catalog["exclusion_counts"]),
        "terminal_probe_outcomes": dict(catalog["terminal_probe_outcomes"]),
        "acquisition_environment_interactions": int(catalog["environment_interactions"]),
        "successful_labeling_environment_interactions": successful_label_interactions,
        "failed_labeling_environment_interactions": int(failed_label_interactions),
        "environment_interactions": int(
            catalog["environment_interactions"]
            + successful_label_interactions
            + failed_label_interactions
        ),
        "training_transitions": 0,
        "expert_switching_used": False,
        "consumed_validation_rows_read": False,
        "consumed_validation_predictions_read": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "labels_dir": str(completed_dir),
        "next_scientific_gate": (
            "combine these TRAIN-only labels with the frozen Iteration-0 TRAIN evidence, "
            "then evaluate upstream leave-one-checkpoint-domain-out generalization before "
            "any fresh validation bank is predeclared"
        ),
        "fresh_validation_authorized": False,
        "tube_1_authorized": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    _write_json(output / "summary.json", summary)
    return summary
