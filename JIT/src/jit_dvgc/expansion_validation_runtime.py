"""Execute the locked group-disjoint expansion validation under frozen pi_k.

The scientific protocol is defined by ``expansion_validation_protocol.py``.
This runtime consumes that immutable declaration, restores only its audited
held-out snapshots, generates the fixed perturbation panel through authoritative
unified dynamics, and labels every accepted validation candidate under the same
frozen deterministic unified policy.

Validation outcomes are never used to alter the panel while the run is active.
They may later calibrate C_up^0/C_down^0, but validation rows never become TRAIN
or Tube supervision.  This module performs no PPO updates and no expert
switching.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
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
from .expansion_validation_protocol import (
    audit_expansion_validation_protocol,
    canonical_sha256,
    load_expansion_validation_protocol_config,
)
from .handoff_snapshot import HandoffSnapshot, compatibility_identity, load_snapshot
from .iteration_train_evidence import load_frozen_iteration_train_evidence
from .ppo import make_checkpoint_policy
from .tube_rsi import PHASE_DOWNSTREAM, PHASE_UPSTREAM
from .unified_continuation_labels import (
    classify_unified_continuation_outcome,
    fresh_unified_continuation_start,
)
from .unified_envelope_snapshot import (
    UnifiedEnvelopeSnapshot,
    capture_unified_envelope_snapshot,
    load_unified_envelope_snapshot,
    physical_state_sha256,
    save_unified_envelope_snapshot,
)
from .unified_formal import build_unified_formal_environment, load_unified_formal_config
from .unified_policy_freeze import load_frozen_unified_manifest
from .unified_training import checkpoint_identity


RUNTIME_PROTOCOL_SCHEMA = "jit_expansion_validation_runtime_protocol_v1"
CANDIDATE_CATALOG_SCHEMA = "jit_expansion_validation_candidate_catalog_v1"
LABEL_SCHEMA = "jit_expansion_validation_labels_v1"
SUMMARY_SCHEMA = "jit_expansion_validation_runtime_summary_v1"
PHASE_INDEX = {"upstream": PHASE_UPSTREAM, "downstream": PHASE_DOWNSTREAM}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"JSON array required: {path}")
    return [dict(row) for row in value]


def _write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _repository_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("expansion validation requires a Git checkout") from exc


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


def _sha256_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [dict(row) for row in rows],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def enumerate_validation_attempts(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expand the immutable validation panel into its deterministic 160 attempts."""
    attempts: list[dict[str, Any]] = []
    for phase in ("upstream", "downstream"):
        source = protocol["sources"][phase]
        panel = protocol["panels"][phase]
        for anchor_index, anchor in enumerate(source["anchors"]):
            for duration in panel["durations"]:
                for strength in panel["strengths"]:
                    for action_name in panel["action_names"]:
                        if action_name not in ACTION_ORDER:
                            raise ValueError("validation panel contains an unknown action axis")
                        action_index = ACTION_ORDER.index(action_name)
                        for sign in panel["signs"]:
                            basis = [0.0] * len(ACTION_ORDER)
                            basis[action_index] = float(sign)
                            attempts.append(
                                {
                                    "attempt_index": len(attempts),
                                    "phase": phase,
                                    "phase_index": int(PHASE_INDEX[phase]),
                                    "anchor_index": int(anchor_index),
                                    "parent_group_id": str(anchor["parent_group_id"]),
                                    "parent_state_sha256": str(anchor["state_sha256"]),
                                    "source_bank": str(anchor["source_bank"]),
                                    "source_snapshot": str(anchor["snapshot"]),
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
    declared = int(protocol["interaction_budget"]["attempt_count"])
    if len(attempts) != declared:
        raise ValueError("validation expanded attempt count drift")
    return tuple(attempts)


def _anchor_snapshot_path(protocol: Mapping[str, Any], attempt: Mapping[str, Any]) -> Path:
    source = protocol["sources"][str(attempt["phase"])]
    catalog_path = Path(str(source["catalog_path"]))
    return catalog_path.parent / str(attempt["source_bank"]) / str(attempt["source_snapshot"])


def load_validation_anchor_snapshots(
    protocol: Mapping[str, Any],
) -> dict[tuple[str, int], HandoffSnapshot]:
    """Load the five protocol-bound legacy snapshots without inspecting outcomes."""
    result: dict[tuple[str, int], HandoffSnapshot] = {}
    for phase in ("upstream", "downstream"):
        for index, anchor in enumerate(protocol["sources"][phase]["anchors"]):
            attempt_stub = {
                "phase": phase,
                "source_bank": anchor["source_bank"],
                "source_snapshot": anchor["snapshot"],
            }
            snapshot = load_snapshot(_anchor_snapshot_path(protocol, attempt_stub))
            if snapshot.xml_sha256 != protocol["xml_sha256"]:
                raise ValueError("validation anchor XML identity drift")
            expected_parent = str(
                anchor.get("snapshot_parent_trajectory", anchor["parent_group_id"])
            )
            if snapshot.parent_trajectory != expected_parent:
                raise ValueError("validation anchor snapshot parent drift")
            result[(phase, index)] = snapshot
    return result


def restore_validation_anchor_as_unified(
    snapshot: HandoffSnapshot,
    *,
    phase: str,
    env: Any,
    parent_group_index: int,
) -> Any:
    """Restore one audited legacy anchor with the exact Tube-start semantics.

    The real qpos/qvel/control, FIFO, last action and upstream event context are
    preserved.  Administrative counters start fresh, and downstream events are
    initialized exactly as the existing unified Tube reset does.
    """
    if phase not in PHASE_INDEX:
        raise ValueError("unsupported validation phase")
    expected_compatibility = compatibility_identity(env)
    if snapshot.compatibility_identity is not None and snapshot.compatibility_identity != expected_compatibility:
        raise ValueError("validation anchor runtime compatibility drift")
    if snapshot.xml_sha256 != env._bundle.xml_sha256:
        raise ValueError("validation anchor runtime XML drift")
    if np.asarray(snapshot.observation).ndim != 1:
        raise ValueError("validation anchor stored actor observation is invalid")

    rng = jax.random.wrap_key_data(jp.asarray(snapshot.rng, dtype=jp.uint32))
    minus_one = jp.asarray(-1, jp.int32)
    sample = {
        "qpos": jp.asarray(snapshot.qpos),
        "qvel": jp.asarray(snapshot.qvel),
        "ctrl": jp.asarray(snapshot.ctrl),
        "observation_fifo": jp.asarray(snapshot.observation_fifo),
        "history_valid_count": jp.asarray(snapshot.history_valid_count, jp.int32),
        "events": {name: jp.asarray(value) for name, value in snapshot.events.items()},
        "tube_phase": jp.asarray(PHASE_INDEX[phase], jp.int32),
        "rng": rng,
        "last_action": jp.asarray(snapshot.last_action),
        "tick": jp.asarray(snapshot.tick, jp.int32),
        "parent_group_index": jp.asarray(parent_group_index, jp.int32),
        "tube_entry_index": minus_one,
        "tube_global_index": minus_one,
    }
    state = env._reset_from_tube_sample(sample)
    # This is a held-out validation reset, not a TRAIN Tube or natural reset.
    info = {**state.info, "reset_from_soft_tube": jp.asarray(False)}
    zero = jp.asarray(0.0, jp.float32)
    metrics = {
        **state.metrics,
        "reset/source_soft_tube": zero,
        "reset/source_natural": zero,
        "reset/tube_phase_upstream": zero,
        "reset/tube_phase_downstream": zero,
    }
    state = state.replace(info=info, metrics=metrics)
    restored = np.asarray(jax.device_get(state.obs["state"]), dtype=np.float32)
    stored = np.asarray(snapshot.observation, dtype=np.float32)
    if restored.shape != stored.shape or not np.allclose(restored, stored, rtol=0.0, atol=1.0e-5):
        raise ValueError("validation anchor actor observation reconstruction drift")
    if _integer(state.info["active_phase"]) != int(PHASE_INDEX[phase]):
        raise ValueError("validation anchor active phase drift")
    if _integer(state.info["episode_step"]) != 0 or _integer(state.info["phase_episode_step"]) != 0:
        raise ValueError("validation anchor administrative counters were not reset")
    if _truth(state.info["expert_switching_used"]):
        raise ValueError("validation anchor restore used expert switching")
    if not _finite_state(state):
        raise ValueError("validation anchor restore is nonfinite")
    return state


def _near_train_observation(
    observation: np.ndarray,
    train_observations: np.ndarray,
    *,
    atol: float,
) -> bool:
    if observation.shape != (train_observations.shape[1],):
        raise ValueError("validation candidate observation size drift")
    return bool(np.any(np.all(np.abs(train_observations - observation) <= float(atol), axis=1)))


def _runtime_protocol(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
    policy_record: Mapping[str, Any],
    train_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    scientific = config["protocol"]
    base = {
        "schema": RUNTIME_PROTOCOL_SCHEMA,
        "status": "predeclared",
        "purpose": "execute_locked_group_disjoint_expansion_validation",
        "repository_head": _repository_head(),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "scientific_protocol_sha256": str(audit["protocol_sha256"]),
        "iteration": int(scientific["iteration"]),
        "policy_name": str(scientific["policy_name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "frozen_policy_file_sha256": str(scientific["frozen_policy_file_sha256"]),
        "frozen_train_manifest_sha256": str(train_manifest["manifest_sha256"]),
        "source_tube_manifest_sha256": str(train_manifest["source_tube_manifest_sha256"]),
        "xml_sha256": str(policy_record["xml_sha256"]),
        "validation_seed": int(scientific["validation_seed"]),
        "attempt_schedule_sha256": _sha256_rows(enumerate_validation_attempts(scientific)),
        "attempt_count": int(scientific["interaction_budget"]["attempt_count"]),
        "maximum_acquisition_environment_interactions": int(
            scientific["interaction_budget"]["maximum_acquisition_environment_interactions"]
        ),
        "maximum_labeling_environment_interactions": int(
            scientific["interaction_budget"]["maximum_labeling_environment_interactions"]
        ),
        "policy_mode": "deterministic",
        "policy_key_rule": "fold_in(validation_seed, attempt_index) then fold_in(tick)",
        "candidate_start_semantics": (
            "restore audited held-out legacy snapshot with exact qpos/qvel/control/FIFO/last-action/"
            "upstream-event context; initialize unified downstream event state exactly as Tube reset; "
            "generate candidate only through authoritative env.step; terminal clipping saves the last "
            "finite nonterminal phase-local state; continuation labeling restarts a fresh 400-tick budget"
        ),
        "candidate_leakage_filter": {
            "reject_frozen_train_exact_physical_state": True,
            "reject_frozen_train_actor_observation_near_duplicate": True,
            "actor_observation_atol": float(
                scientific["near_duplicate_audit"]["actor_observation_atol"]
            ),
            "reject_duplicate_validation_physical_state": True,
            "no_replacement_after_exclusion": True,
        },
        "resume_semantics": (
            "reuse only a fully completed acquisition catalog bound to this runtime protocol; "
            "resume labeling from sequential completed labels; failed-label interactions remain "
            "accounted and the interrupted candidate may be deterministically replayed"
        ),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "data_policy": dict(scientific["data_policy"]),
        "claim_boundary": dict(scientific["claim_boundary"]),
    }
    return {**base, "protocol_sha256": canonical_sha256(base)}


def _collect_candidates(
    *,
    protocol: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    anchors: Mapping[tuple[str, int], HandoffSnapshot],
    train_rows: Sequence[Mapping[str, Any]],
    output: Path,
    step_fn: Callable[[Any, Any], Any],
) -> dict[str, Any]:
    attempts = enumerate_validation_attempts(protocol)
    train_states = {str(row["state_sha256"]) for row in train_rows}
    train_observations = np.asarray(
        [row["actor_observation"] for row in train_rows], dtype=np.float32
    )
    tolerance = float(protocol["near_duplicate_audit"]["actor_observation_atol"])
    bank = output / "candidate_bank"
    (bank / "snapshots").mkdir(parents=True, exist_ok=False)

    base_key = jax.random.PRNGKey(int(protocol["validation_seed"]))
    entries: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    exclusions: Counter[str] = Counter()
    terminal_outcomes: Counter[str] = Counter()
    interactions = 0

    for attempt in attempts:
        snapshot = anchors[(str(attempt["phase"]), int(attempt["anchor_index"]))]
        state = restore_validation_anchor_as_unified(
            snapshot,
            phase=str(attempt["phase"]),
            env=env,
            parent_group_index=int(attempt["attempt_index"]),
        )
        candidate_state = None
        terminal_clipped = False
        terminal_meta: dict[str, Any] | None = None
        nominal_actions: list[list[float]] = []
        perturbed_actions: list[list[float]] = []
        effective_deltas: list[list[float]] = []
        rejected: str | None = None
        executed = 0
        attempt_key = jax.random.fold_in(base_key, int(attempt["attempt_index"]))

        for perturb_step in range(int(attempt["duration"])):
            previous = state
            action_key = jax.random.fold_in(attempt_key, int(perturb_step))
            result = policy(state.obs, action_key)
            nominal_device = result[0] if isinstance(result, tuple) else result
            nominal = np.asarray(jax.device_get(nominal_device), dtype=np.float32).reshape(-1)
            if nominal.shape != (len(ACTION_ORDER),) or not np.isfinite(nominal).all():
                raise ValueError("frozen pi_0 returned invalid validation acquisition action")
            requested = nominal + np.asarray(attempt["basis_vector"], dtype=np.float32) * np.float32(
                attempt["strength"]
            )
            perturbed = np.clip(requested, -1.0, 1.0).astype(np.float32)
            state = step_fn(state, jp.asarray(perturbed))
            jax.block_until_ready(state)
            interactions += 1
            executed += 1
            nominal_actions.append(nominal.tolist())
            perturbed_actions.append(perturbed.tolist())
            effective_deltas.append((perturbed - nominal).tolist())
            if _truth(state.info["expert_switching_used"]):
                raise ValueError("validation acquisition used expert switching")
            if not _finite_state(state, perturbed):
                rejected = "nonfinite"
                break
            if _integer(state.info["active_phase"]) != int(attempt["phase_index"]):
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
                    "done": True,
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
        if _truth(candidate_state.done):
            raise ValueError("validation candidate unexpectedly terminal")
        if _integer(candidate_state.info["active_phase"]) != int(attempt["phase_index"]):
            exclusions["candidate_phase_transition"] += 1
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
            exclusions["train_exact_state"] += 1
            continue
        if _near_train_observation(actor_observation, train_observations, atol=tolerance):
            exclusions["train_near_duplicate_observation"] += 1
            continue
        if state_sha in seen_states:
            exclusions["duplicate_validation_state"] += 1
            continue
        seen_states.add(state_sha)

        relative = Path("snapshots") / f"candidate_{len(entries):06d}"
        save_unified_envelope_snapshot(bank / relative, unified_snapshot)
        entries.append(
            {
                "candidate_id": f"pi0_validation_{len(entries):06d}",
                "candidate_kind": "group_disjoint_validation_frontier_probe",
                "split": "validation",
                "phase": str(attempt["phase"]),
                "phase_index": int(attempt["phase_index"]),
                "snapshot": str(relative),
                "source_bank": "candidate_bank",
                "state_sha256": state_sha,
                "parent_group_id": str(attempt["parent_group_id"]),
                "parent_state_sha256": str(attempt["parent_state_sha256"]),
                "source_anchor": {
                    "source_bank": str(attempt["source_bank"]),
                    "snapshot": str(attempt["source_snapshot"]),
                    "role": str(attempt["role"]),
                    "tick": int(attempt["tick"]),
                    "anchor_index": int(attempt["anchor_index"]),
                },
                "attempt_index": int(attempt["attempt_index"]),
                "policy_iteration": int(policy_record["iteration"]),
                "policy_actor_sha256": str(policy_record["actor_sha256"]),
                "policy_payload_sha256": str(policy_record["payload_sha256"]),
                "scientific_protocol_sha256": str(runtime_protocol["scientific_protocol_sha256"]),
                "runtime_protocol_sha256": str(runtime_protocol["protocol_sha256"]),
                "actor_observation": actor_observation.tolist(),
                "perturbation": {
                    "action_name": str(attempt["action_name"]),
                    "action_index": int(attempt["action_index"]),
                    "sign": int(attempt["sign"]),
                    "basis_vector": list(attempt["basis_vector"]),
                    "strength": float(attempt["strength"]),
                    "duration": int(attempt["duration"]),
                    "executed_interactions": int(executed),
                    "terminal_clipped": bool(terminal_clipped),
                    "terminal_probe_outcome": terminal_meta,
                    "nominal_actions": nominal_actions,
                    "perturbed_actions": perturbed_actions,
                    "effective_deltas": effective_deltas,
                },
            }
        )

    maximum = int(protocol["interaction_budget"]["maximum_acquisition_environment_interactions"])
    if interactions > maximum:
        raise ValueError("validation acquisition exceeded the locked interaction ceiling")
    if len(attempts) != int(protocol["interaction_budget"]["attempt_count"]):
        raise ValueError("validation attempt accounting drift")
    report = {
        "schema": CANDIDATE_CATALOG_SCHEMA,
        "status": "completed",
        "artifact_role": "group_disjoint_expansion_validation_candidates",
        "split": "validation",
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "scientific_protocol_sha256": str(runtime_protocol["scientific_protocol_sha256"]),
        "runtime_protocol_sha256": str(runtime_protocol["protocol_sha256"]),
        "attempt_count": len(attempts),
        "candidate_count": len(entries),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "terminal_probe_outcomes": dict(sorted(terminal_outcomes.items())),
        "environment_interactions": interactions,
        "maximum_environment_interactions": maximum,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": True,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
        "entries": entries,
    }
    _write_json(output / "candidate_catalog.json", report)
    _write_json(
        output / "candidate_summary.json", {key: value for key, value in report.items() if key != "entries"}
    )
    return report


def _validate_completed_acquisition(
    *,
    output: Path,
    runtime_protocol: Mapping[str, Any],
    policy_record: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = _read_object(output / "candidate_catalog.json")
    if catalog.get("schema") != CANDIDATE_CATALOG_SCHEMA or catalog.get("status") != "completed":
        raise ValueError("cannot resume invalid validation acquisition catalog")
    if catalog.get("runtime_protocol_sha256") != runtime_protocol["protocol_sha256"]:
        raise ValueError("validation acquisition/runtime protocol drift")
    if catalog.get("scientific_protocol_sha256") != runtime_protocol["scientific_protocol_sha256"]:
        raise ValueError("validation acquisition/scientific protocol drift")
    if catalog.get("policy_actor_sha256") != policy_record["actor_sha256"]:
        raise ValueError("validation acquisition actor identity drift")
    if catalog.get("policy_payload_sha256") != policy_record["payload_sha256"]:
        raise ValueError("validation acquisition payload identity drift")
    entries = catalog.get("entries")
    if not isinstance(entries, list) or len(entries) != int(catalog.get("candidate_count", -1)):
        raise ValueError("validation acquisition candidate count drift")
    seen: set[str] = set()
    for row in entries:
        if row.get("split") != "validation":
            raise ValueError("validation acquisition contains non-validation candidate")
        state_sha = str(row.get("state_sha256", ""))
        if state_sha in seen:
            raise ValueError("validation acquisition contains duplicate physical state")
        seen.add(state_sha)
        snapshot = load_unified_envelope_snapshot(
            output / str(row["source_bank"]) / str(row["snapshot"])
        )
        if physical_state_sha256(snapshot) != state_sha:
            raise ValueError("validation acquisition snapshot physical-state drift")
        if snapshot.policy_actor_sha256 != policy_record["actor_sha256"]:
            raise ValueError("validation acquisition snapshot actor drift")
        if snapshot.policy_payload_sha256 != policy_record["payload_sha256"]:
            raise ValueError("validation acquisition snapshot payload drift")
    return catalog


def _validate_progress_rows(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    runtime_protocol: Mapping[str, Any],
) -> None:
    if len(rows) > len(candidates):
        raise ValueError("validation label progress exceeds candidate count")
    for index, row in enumerate(rows):
        candidate = candidates[index]
        if int(row.get("candidate_index", -1)) != index:
            raise ValueError("validation label progress order drift")
        if row.get("candidate_id") != candidate.get("candidate_id"):
            raise ValueError("validation label progress candidate identity drift")
        if row.get("state_sha256") != candidate.get("state_sha256"):
            raise ValueError("validation label progress state identity drift")
        if row.get("split") != "validation":
            raise ValueError("validation label progress split drift")
        if row.get("runtime_protocol_sha256") != runtime_protocol["protocol_sha256"]:
            raise ValueError("validation label progress runtime protocol drift")


def _label_candidates(
    *,
    protocol: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    catalog: Mapping[str, Any],
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    output: Path,
    step_fn: Callable[[Any, Any], Any],
    resume: bool,
) -> dict[str, Any]:
    candidates = [dict(row) for row in catalog["entries"]]
    progress_path = output / "label_progress.json"
    failure_path = output / "label_failure.json"
    completed: list[dict[str, Any]] = []
    prior_failed_interactions = 0
    if progress_path.exists():
        if not resume:
            raise FileExistsError("validation label progress already exists; use --resume")
        progress = _read_object(progress_path)
        completed = [dict(row) for row in progress.get("completed_labels", [])]
        _validate_progress_rows(completed, candidates, runtime_protocol=runtime_protocol)
        prior_failed_interactions = int(progress.get("failed_attempt_interactions", 0))
    elif resume and failure_path.exists():
        # A failure file without progress means zero labels completed; preserve its spent interactions.
        failure = _read_object(failure_path)
        prior_failed_interactions = int(failure.get("total_labeling_environment_interactions", 0))

    successful_interactions = sum(int(row["environment_interactions"]) for row in completed)
    labeling_interactions = successful_interactions + prior_failed_interactions
    maximum = int(protocol["interaction_budget"]["maximum_labeling_environment_interactions"])
    outcome_counts: Counter[str] = Counter(str(row["outcome_class"]) for row in completed)
    phase_counts: Counter[str] = Counter(str(row["phase"]) for row in completed)
    phase_positive: Counter[str] = Counter()
    parent_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in completed:
        phase_positive[str(row["phase"])] += int(row["label"])
        parent_stats[str(row["parent_group_id"])]["candidate_count"] += 1
        parent_stats[str(row["parent_group_id"])]["positive_count"] += int(row["label"])

    base_key = jax.random.PRNGKey(int(protocol["validation_seed"]) + 1)
    current_candidate_interactions = 0
    try:
        for candidate_index in range(len(completed), len(candidates)):
            candidate = candidates[candidate_index]
            snapshot = load_unified_envelope_snapshot(
                output / str(candidate["source_bank"]) / str(candidate["snapshot"])
            )
            if physical_state_sha256(snapshot) != candidate["state_sha256"]:
                raise ValueError("validation label snapshot state identity drift")
            state = fresh_unified_continuation_start(snapshot, env)
            if not _finite_state(state):
                raise ValueError("validation continuation start is nonfinite")
            start_phase = _integer(state.info["active_phase"])
            if start_phase != int(candidate["phase_index"]):
                raise ValueError("validation continuation start phase drift")
            if _truth(state.info["expert_switching_used"]):
                raise ValueError("validation continuation start used expert switching")

            apex_seen = _truth(state.info["up_events"].apex_seen)
            phase_transitioned = _truth(state.info["phase_transitioned"])
            recovery_success = _truth(state.info["down_events"].recovery_success)
            valid_contact_seen = _truth(state.info["down_events"].valid_contact_seen)
            current_candidate_interactions = 0
            candidate_key = jax.random.fold_in(base_key, int(candidate["attempt_index"]))
            max_ticks = int(protocol["panels"][str(candidate["phase"])]["max_label_ticks"])

            for tick in range(max_ticks):
                action_key = jax.random.fold_in(candidate_key, int(tick))
                result = policy(state.obs, action_key)
                action = result[0] if isinstance(result, tuple) else result
                action_array = np.asarray(jax.device_get(action), dtype=np.float32).reshape(-1)
                if action_array.shape != (4,) or not np.isfinite(action_array).all():
                    raise ValueError("frozen pi_0 returned invalid validation labeling action")
                state = step_fn(state, action)
                jax.block_until_ready(state)
                labeling_interactions += 1
                current_candidate_interactions += 1
                if not _finite_state(state, action):
                    raise ValueError("validation continuation rollout became nonfinite")
                if _truth(state.info["expert_switching_used"]):
                    raise ValueError("validation continuation rollout used expert switching")
                apex_seen |= _truth(state.info["up_events"].apex_seen)
                phase_transitioned |= _truth(state.info["phase_transitioned"])
                recovery_success |= _truth(state.info["down_events"].recovery_success)
                valid_contact_seen |= _truth(state.info["down_events"].valid_contact_seen)
                if _truth(state.done):
                    break

            done = _truth(state.done)
            terminal_success = _truth(state.info["success"])
            physical_failure = _truth(state.info["physical_failure"])
            timeout = _truth(state.info["timeout"])
            reached_horizon = current_candidate_interactions >= max_ticks and not done
            positive, outcome_class = classify_unified_continuation_outcome(
                start_phase=start_phase,
                terminal_success=terminal_success,
                physical_failure=physical_failure,
                timeout=timeout,
                done=done,
                apex_seen=apex_seen,
                phase_transitioned=phase_transitioned,
                recovery_success=recovery_success,
                reached_rollout_horizon=reached_horizon,
            )
            end_code = _integer(state.info["end_code"])
            end_reason = END_REASONS.get(end_code, f"unknown_{end_code}")
            row = {
                "candidate_index": int(candidate_index),
                "candidate_id": str(candidate["candidate_id"]),
                "split": "validation",
                "phase": str(candidate["phase"]),
                "phase_index": int(candidate["phase_index"]),
                "state_sha256": str(candidate["state_sha256"]),
                "parent_group_id": str(candidate["parent_group_id"]),
                "parent_state_sha256": str(candidate["parent_state_sha256"]),
                "attempt_index": int(candidate["attempt_index"]),
                "actor_observation": list(candidate["actor_observation"]),
                "perturbation": dict(candidate["perturbation"]),
                "label": int(positive),
                "continuation_success": bool(positive),
                "outcome_class": outcome_class,
                "environment_interactions": int(current_candidate_interactions),
                "terminal_done": bool(done),
                "terminal_success": bool(terminal_success),
                "physical_failure": bool(physical_failure),
                "timeout": bool(timeout),
                "end_code": int(end_code),
                "end_reason": str(end_reason),
                "apex_seen": bool(apex_seen),
                "phase_transitioned": bool(phase_transitioned),
                "valid_contact_seen": bool(valid_contact_seen),
                "recovery_success": bool(recovery_success),
                "final_active_phase": _integer(state.info["active_phase"]),
                "policy_iteration": int(policy_record["iteration"]),
                "policy_actor_sha256": str(policy_record["actor_sha256"]),
                "policy_payload_sha256": str(policy_record["payload_sha256"]),
                "scientific_protocol_sha256": str(runtime_protocol["scientific_protocol_sha256"]),
                "runtime_protocol_sha256": str(runtime_protocol["protocol_sha256"]),
            }
            completed.append(row)
            outcome_counts[outcome_class] += 1
            phase_counts[str(candidate["phase"])] += 1
            phase_positive[str(candidate["phase"])] += int(positive)
            parent_stats[str(candidate["parent_group_id"])]["candidate_count"] += 1
            parent_stats[str(candidate["parent_group_id"])]["positive_count"] += int(positive)
            current_candidate_interactions = 0
            _write_json(
                progress_path,
                {
                    "schema": LABEL_SCHEMA,
                    "status": "labeling",
                    "runtime_protocol_sha256": runtime_protocol["protocol_sha256"],
                    "completed_candidate_count": len(completed),
                    "candidate_count": len(candidates),
                    "successful_labeling_interactions": sum(
                        int(item["environment_interactions"]) for item in completed
                    ),
                    "failed_attempt_interactions": int(prior_failed_interactions),
                    "total_labeling_environment_interactions": int(labeling_interactions),
                    "completed_labels": completed,
                },
            )

        if labeling_interactions > maximum + prior_failed_interactions:
            raise ValueError("validation labeling exceeded locked successful interaction ceiling")
        report = {
            "schema": LABEL_SCHEMA,
            "status": "completed",
            "artifact_role": "pi_k_conditioned_expansion_validation_labels",
            "split": "validation",
            "iteration": int(policy_record["iteration"]),
            "policy_name": str(policy_record["name"]),
            "policy_actor_sha256": str(policy_record["actor_sha256"]),
            "policy_payload_sha256": str(policy_record["payload_sha256"]),
            "scientific_protocol_sha256": str(runtime_protocol["scientific_protocol_sha256"]),
            "runtime_protocol_sha256": str(runtime_protocol["protocol_sha256"]),
            "candidate_catalog_file_sha256": file_sha256(output / "candidate_catalog.json"),
            "candidate_count": len(candidates),
            "label_count": len(completed),
            "positive_count": sum(int(row["label"]) for row in completed),
            "negative_count": sum(1 - int(row["label"]) for row in completed),
            "phase_candidate_counts": dict(sorted(phase_counts.items())),
            "phase_positive_counts": dict(sorted(phase_positive.items())),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "parent_group_stats": {
                parent: {
                    "candidate_count": int(stats["candidate_count"]),
                    "positive_count": int(stats["positive_count"]),
                    "negative_count": int(stats["candidate_count"] - stats["positive_count"]),
                }
                for parent, stats in sorted(parent_stats.items())
            },
            "successful_labeling_environment_interactions": sum(
                int(row["environment_interactions"]) for row in completed
            ),
            "failed_attempt_environment_interactions": int(prior_failed_interactions),
            "environment_interactions": int(labeling_interactions),
            "maximum_successful_environment_interactions": maximum,
            "training_transitions": 0,
            "expert_switching_used": False,
            "validation_data_used": True,
            "test_data_used": False,
            "final_evaluation_data_used": False,
            "claim_boundary": dict(protocol["claim_boundary"]),
        }
        _write_json(output / "labels.json", completed)
        _write_json(output / "label_summary.json", report)
        progress_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        return report
    except BaseException as exc:
        # Interactions from an interrupted candidate are preserved for accounting.
        failed_total = int(prior_failed_interactions + current_candidate_interactions)
        successful = sum(int(row["environment_interactions"]) for row in completed)
        _write_json(
            progress_path,
            {
                "schema": LABEL_SCHEMA,
                "status": "engineering_error",
                "runtime_protocol_sha256": runtime_protocol["protocol_sha256"],
                "completed_candidate_count": len(completed),
                "candidate_count": len(candidates),
                "successful_labeling_interactions": successful,
                "failed_attempt_interactions": failed_total,
                "total_labeling_environment_interactions": successful + failed_total,
                "completed_labels": completed,
            },
        )
        _write_json(
            failure_path,
            {
                "schema": LABEL_SCHEMA,
                "status": "engineering_error",
                "runtime_protocol_sha256": runtime_protocol["protocol_sha256"],
                "completed_candidate_count": len(completed),
                "successful_labeling_interactions": successful,
                "failed_attempt_interactions": failed_total,
                "total_labeling_environment_interactions": successful + failed_total,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def execute_expansion_validation(
    config_path: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the locked Iteration-0 validation protocol exactly once."""
    config_path = Path(config_path)
    audit = audit_expansion_validation_protocol(config_path)
    config = load_expansion_validation_protocol_config(config_path)
    protocol = config["protocol"]
    output = Path(str(config["output_dir"]))

    frozen_path = Path(str(protocol["frozen_policy"]))
    frozen = load_frozen_unified_manifest(frozen_path)
    policy_record = frozen["policy"]
    formal = load_unified_formal_config(Path(policy_record["formal_config"]))
    if formal.config_sha256 != policy_record["formal_config_sha256"]:
        raise ValueError("validation frozen policy/formal config drift")
    train_manifest, train_rows = load_frozen_iteration_train_evidence(
        Path(str(protocol["frozen_train_evidence"]))
    )
    runtime_protocol = _runtime_protocol(
        config_path=config_path,
        config=config,
        audit=audit,
        policy_record=policy_record,
        train_manifest=train_manifest,
    )

    if output.exists():
        if not resume:
            raise FileExistsError(f"expansion validation output already exists: {output}")
        existing = _read_object(output / "runtime_protocol.json")
        if existing != runtime_protocol:
            raise ValueError("cannot resume expansion validation under a different runtime protocol")
        if (output / "summary.json").exists():
            summary = _read_object(output / "summary.json")
            if summary.get("status") == "completed":
                return summary
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "runtime_protocol.json", runtime_protocol)
        _write_json(output / "protocol_audit.json", audit)

    if jax.default_backend() != "gpu":
        raise RuntimeError("expansion validation runtime requires the visible JAX GPU")
    runtime_config, runtime_artifact, env = build_unified_formal_environment(
        Path(policy_record["formal_config"])
    )
    if runtime_config.config_sha256 != formal.config_sha256:
        raise ValueError("validation runtime formal config drift")
    if runtime_artifact.manifest["manifest_sha256"] != train_manifest["source_tube_manifest_sha256"]:
        raise ValueError("validation runtime source Tube drift")
    if env._bundle.xml_sha256 != policy_record["xml_sha256"]:
        raise ValueError("validation runtime XML drift")
    payload = load_checkpoint(
        Path(policy_record["checkpoint"]), expected=checkpoint_identity(runtime_config, env)
    )
    if int(payload.training_transitions) != int(policy_record["source_training_transitions"]):
        raise ValueError("validation checkpoint transition drift")
    if file_sha256(Path(policy_record["checkpoint"]) / "payload.pkl") != policy_record["payload_sha256"]:
        raise ValueError("validation checkpoint payload SHA-256 drift")
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))
    step_fn = jax.jit(env.step)

    anchors = load_validation_anchor_snapshots(protocol)
    # Zero-interaction restore preflight for all five held-out anchors.
    for phase in ("upstream", "downstream"):
        for index, _anchor in enumerate(protocol["sources"][phase]["anchors"]):
            state = restore_validation_anchor_as_unified(
                anchors[(phase, index)],
                phase=phase,
                env=env,
                parent_group_index=index,
            )
            jax.block_until_ready(state)

    if (output / "candidate_catalog.json").exists():
        if not resume:
            raise FileExistsError("validation acquisition already exists; use --resume")
        catalog = _validate_completed_acquisition(
            output=output,
            runtime_protocol=runtime_protocol,
            policy_record=policy_record,
        )
    else:
        catalog = _collect_candidates(
            protocol=protocol,
            runtime_protocol=runtime_protocol,
            env=env,
            policy=policy,
            policy_record=policy_record,
            anchors=anchors,
            train_rows=train_rows,
            output=output,
            step_fn=step_fn,
        )

    labels = _label_candidates(
        protocol=protocol,
        runtime_protocol=runtime_protocol,
        catalog=catalog,
        env=env,
        policy=policy,
        policy_record=policy_record,
        output=output,
        step_fn=step_fn,
        resume=resume,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "group_disjoint_expansion_validation_evidence",
        "split": "validation",
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "scientific_protocol_sha256": str(runtime_protocol["scientific_protocol_sha256"]),
        "runtime_protocol_sha256": str(runtime_protocol["protocol_sha256"]),
        "attempt_count": int(catalog["attempt_count"]),
        "candidate_count": int(catalog["candidate_count"]),
        "label_count": int(labels["label_count"]),
        "positive_count": int(labels["positive_count"]),
        "negative_count": int(labels["negative_count"]),
        "phase_candidate_counts": dict(labels["phase_candidate_counts"]),
        "phase_positive_counts": dict(labels["phase_positive_counts"]),
        "outcome_counts": dict(labels["outcome_counts"]),
        "parent_group_stats": dict(labels["parent_group_stats"]),
        "candidate_exclusion_counts": dict(catalog["exclusion_counts"]),
        "terminal_probe_outcomes": dict(catalog["terminal_probe_outcomes"]),
        "acquisition_environment_interactions": int(catalog["environment_interactions"]),
        "labeling_environment_interactions": int(labels["environment_interactions"]),
        "total_environment_interactions": int(
            catalog["environment_interactions"] + labels["environment_interactions"]
        ),
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": True,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "validation_rows_may_enter_train_or_tube": False,
        "continuation_field_trained": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
        "next_scientific_gate": (
            "freeze this validation artifact, inspect phase/group/panel label support, and predeclare "
            "a low-complexity C_up^0/C_down^0 fitting+calibration protocol; do not modify the validation "
            "panel after seeing these outcomes"
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary
