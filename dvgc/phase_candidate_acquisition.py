"""Evidence gates for real online Phase U candidate snapshot acquisition."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np


@dataclass(frozen=True)
class AcquisitionParentSummary:
    """Auditable identity and outcome for one stochastic online parent rollout."""

    seed: int
    trajectory_hash: str
    success: bool
    contract_valid: bool
    candidate_count: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("parent seed must be an integer")
        if (
            not isinstance(self.trajectory_hash, str)
            or len(self.trajectory_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.trajectory_hash)
        ):
            raise ValueError("trajectory_hash must be lowercase SHA-256")
        if not isinstance(self.success, bool) or not isinstance(self.contract_valid, bool):
            raise ValueError("parent outcome flags must be boolean")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 0
        ):
            raise ValueError("candidate_count must be a non-negative integer")


def evaluate_candidate_acquisition_gate(
    fixed_evaluation: Mapping[str, Any],
    parents: Sequence[AcquisitionParentSummary],
    *,
    minimum_independent_successful_parents: int = 8,
) -> dict[str, Any]:
    """Require held-out evidence plus distinct successful online parents."""
    if (
        isinstance(minimum_independent_successful_parents, bool)
        or not isinstance(minimum_independent_successful_parents, int)
        or minimum_independent_successful_parents <= 0
    ):
        raise ValueError("minimum independent parent count must be positive")
    if not all(isinstance(parent, AcquisitionParentSummary) for parent in parents):
        raise TypeError("parents must contain AcquisitionParentSummary values")
    physical = fixed_evaluation.get("physical_metrics")
    physical = physical if isinstance(physical, Mapping) else {}
    try:
        fixed_rate = float(physical.get("apex_band_success_rate", 0.0))
    except (TypeError, ValueError):
        fixed_rate = math.nan
    fixed_success = math.isfinite(fixed_rate) and fixed_rate > 0.0
    successful = [parent for parent in parents if parent.success]
    successful_with_candidates = [
        parent for parent in successful if parent.candidate_count > 0
    ]
    seeds = {parent.seed for parent in successful}
    trajectories = {parent.trajectory_hash for parent in successful}
    contracts_valid = all(parent.contract_valid for parent in parents)
    failed = []
    if not fixed_success:
        failed.append("fixed_apex_success")
    if len(successful) < minimum_independent_successful_parents:
        failed.append("minimum_successful_parents")
    if len(successful_with_candidates) < minimum_independent_successful_parents:
        failed.append("successful_parent_candidate_coverage")
    if len(seeds) < minimum_independent_successful_parents:
        failed.append("unique_successful_seeds")
    if len(trajectories) < minimum_independent_successful_parents:
        failed.append("unique_successful_trajectories")
    if not contracts_valid:
        failed.append("parent_contracts")
    return {
        "eligible": not failed,
        "fixed_apex_success": fixed_success,
        "successful_parent_count": len(successful),
        "successful_parent_candidate_count": len(successful_with_candidates),
        "unique_successful_seed_count": len(seeds),
        "unique_successful_trajectory_count": len(trajectories),
        "all_parent_contracts_valid": contracts_valid,
        "minimum_independent_successful_parents": minimum_independent_successful_parents,
        "failed": failed,
    }


def require_candidate_acquisition_integrity(gate: Mapping[str, Any]) -> None:
    """Escalate snapshot-contract corruption instead of treating it as low coverage."""
    if gate.get("all_parent_contracts_valid") is not True:
        raise RuntimeError("candidate snapshot timing or provenance contract violation")


def build_provisional_continuation_label(
    outcomes: Sequence[Mapping[str, str]],
    *,
    phase: str,
    source_policy_hash: str,
    protocol_hash: str,
) -> dict[str, Any]:
    """Build a closed checkpoint-policy-dependent continuation screen label."""
    categories = ("success", "physical_failure", "timeout", "other_failure")
    if phase not in {"propulsion_ascent", "descent_recovery"}:
        raise ValueError("continuation phase is invalid")
    for name, value in (
        ("source_policy_hash", source_policy_hash),
        ("protocol_hash", protocol_hash),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be lowercase SHA-256")
    if not outcomes:
        raise ValueError("continuation outcomes must be nonempty")
    counts = {name: 0 for name in categories}
    reasons: dict[str, int] = {}
    for row in outcomes:
        outcome = row.get("outcome")
        reason = row.get("termination_reason")
        if outcome not in counts:
            raise ValueError("continuation outcome category is invalid")
        if not isinstance(reason, str) or not reason:
            raise ValueError("continuation termination reason is required")
        counts[outcome] += 1
        reasons[reason] = reasons.get(reason, 0) + 1
    total = len(outcomes)
    successes = counts["success"]
    return {
        "contract_version": 1,
        "phase": phase,
        "num_rollouts": total,
        "num_successes": successes,
        "empirical_rate": successes / total,
        "physical_failure_rate": counts["physical_failure"] / total,
        "timeout_rate": counts["timeout"] / total,
        "outcome_counts": counts,
        "termination_reason_counts": dict(sorted(reasons.items())),
        "label_source_policy_hash": source_policy_hash,
        "label_protocol_hash": protocol_hash,
        "provisional": True,
        "formal_tube_authority": False,
    }


def build_continuation_branch_provenance(
    *,
    seed: int,
    seed_namespace: str,
    source_policy_hash: str,
    protocol_hash: str,
) -> dict[str, Any]:
    """Bind one stochastic continuation branch to reproducible authority."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("continuation branch seed must be an integer")
    if not isinstance(seed_namespace, str) or not seed_namespace:
        raise ValueError("continuation branch seed namespace is required")
    for name, value in (
        ("source_policy_hash", source_policy_hash),
        ("protocol_hash", protocol_hash),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be lowercase SHA-256")
    return {
        "policy_mode": "stochastic",
        "branch_seed_namespace": seed_namespace,
        "branch_seeds": [seed],
        "source_policy_hash": source_policy_hash,
        "label_protocol_hash": protocol_hash,
    }


@dataclass(frozen=True)
class PhaseUCandidateAcquisitionResult:
    gate: Mapping[str, Any]
    parents: tuple[AcquisitionParentSummary, ...]
    records: tuple[Mapping[str, Any], ...]
    environment_transitions: int


def pytree_sha256(value: Any) -> str:
    """Hash a JAX/NumPy pytree by stable leaf order, dtype, shape, and bytes."""
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(jax.device_get(value)):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def classify_phase_u_candidate_strata(
    *,
    events: Mapping[str, bool],
    signals: Any,
    thresholds: Any,
    window_active: bool,
) -> tuple[str, ...]:
    """Classify explicit task, outcome-boundary, and mistiming strata."""
    strata: list[str] = []
    window_entered = bool(events.get("jump_window_entered"))
    liftoff = bool(events.get("liftoff_seen"))
    stable_airborne = bool(events.get("stable_airborne"))
    ascending = bool(events.get("ascending"))
    apex = bool(events.get("apex_band_entered"))
    if window_entered and not liftoff:
        strata.append("propulsion")
    if (
        abs(float(signals.roll)) >= 0.8 * thresholds.max_abs_roll
        or abs(float(signals.pitch)) >= 0.8 * thresholds.max_abs_pitch
    ):
        strata.append("high_attitude")
    if window_entered and float(signals.clearance) < thresholds.min_clearance:
        strata.append("low_clearance")
    if (
        window_entered
        and float(signals.forward_velocity) < thresholds.min_forward_velocity
    ):
        strata.append("low_forward_speed")
    if not window_entered and float(signals.com_vz) > 0.0:
        strata.append("too_early_ascent")
    if window_entered and not window_active and not ascending:
        strata.append("too_late_ascent")
    if (
        stable_airborne
        and ascending
        and abs(float(signals.com_vz)) <= 1.5 * thresholds.max_abs_com_vz
        and not apex
    ):
        strata.append("apex_band_boundary")
    if (
        stable_airborne
        and ascending
        and float(signals.clearance) >= thresholds.min_clearance
        and float(signals.forward_velocity) >= thresholds.min_forward_velocity
        and not apex
    ):
        strata.append("near_success")
    return tuple(strata)


def _trajectory_hash(states: Sequence[tuple[np.ndarray, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for qpos, qvel in states:
        for array in (qpos, qvel):
            value = np.asarray(array, dtype=np.float32)
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def acquire_phase_u_candidate_parents(
    environment: Any,
    params: Any,
    *,
    fixed_evaluation: Mapping[str, Any],
    seeds: Sequence[int],
    horizon: int,
    provenance: Mapping[str, Any],
    minimum_independent_successful_parents: int = 8,
    transition_observer: Callable[[int], None] | None = None,
) -> PhaseUCandidateAcquisitionResult:
    """Run stochastic online parents and capture only real timing-explicit v4 states."""
    from .feasibility import validate_phase_snapshot
    from .runtime import build_inference

    required_provenance = {
        "xml_sha256",
        "config_sha256",
        "action_mapping_version",
        "policy_params_sha256",
        "policy_config_sha256",
        "policy_manifest_sha256",
        "normalizer_sha256",
        "source_fingerprint",
    }
    if set(provenance) != required_provenance:
        raise ValueError("candidate acquisition provenance is incomplete")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("candidate acquisition horizon must be positive")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("candidate acquisition seeds must be nonempty and unique")

    inference = build_inference(environment, params, deterministic=False)
    step = jax.jit(environment.step)
    parent_summaries: list[AcquisitionParentSummary] = []
    all_records: list[Mapping[str, Any]] = []
    transitions = 0

    for seed in seeds:
        key = jax.random.PRNGKey(int(seed))
        state = environment.reset(key)
        trajectory_states: list[tuple[np.ndarray, np.ndarray]] = []
        records: list[Mapping[str, Any]] = []
        pre_window_record: Mapping[str, Any] | None = None
        last_valid_record: Mapping[str, Any] | None = None
        previous_events = {
            "jump_window_entered": False,
            "liftoff_seen": False,
            "stable_airborne": False,
            "ascending": False,
            "apex_band_entered": False,
        }
        captured_strata: set[str] = set()
        ascent_age = 0
        success = False
        contract_valid = True
        post_success_ticks = 0
        parent_id = f"phase-u-parent-{int(seed)}"

        def capture(
            current_state: Any,
            policy_action: Any,
            *,
            tick: int,
            stratum: str,
            event_name: str,
            event_position: str = "event",
            terminated: bool = False,
            truncated: bool = False,
            termination_reason: str = "none",
        ) -> Mapping[str, Any] | None:
            nonlocal contract_valid
            if int(current_state.info["actor_packet_fifo_valid"]) != 3:
                return None
            try:
                record = environment._base_env.snapshot_record_v4(
                    current_state, "takeoff", policy_action, dict(provenance)
                )
                record["id"] = f"{parent_id}-{tick}-{stratum}"
                record["two_phase_context"] = {
                    "contract_version": 1,
                    "source_phase": "propulsion_ascent",
                    "parent_trajectory_id": parent_id,
                    "trajectory_id": parent_id,
                    "time_index": int(tick),
                    "event_names": [event_name],
                    "event_position": event_position,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "termination_reason": termination_reason,
                    "source_policy_hash": provenance["policy_params_sha256"],
                    "source_xml_hash": provenance["xml_sha256"],
                    "source_config_hash": provenance["config_sha256"],
                }
                record["candidate_acquisition"] = {
                    "status": "unlabeled_candidate",
                    "stratum": stratum,
                    "terminal_outcome": (
                        "physical_failure"
                        if terminated and str(termination_reason).startswith("end_code_")
                        and bool(current_state.info["phase_expert/physical_failure"])
                        else "other_failure"
                        if terminated
                        else "timeout"
                        if truncated
                        else None
                    ),
                    "policy_dependent_label": None,
                    "tube_membership": False,
                    "certified_safe": False,
                }
                event_state = {}
                from .two_phase_runtime import TwoPhaseEventState

                for name in TwoPhaseEventState._fields:
                    value = np.asarray(
                        jax.device_get(current_state.info[f"phase_expert/event/{name}"])
                    )
                    event_state[name] = value.item() if value.ndim == 0 else value.copy()
                record["two_phase_event_state"] = event_state
                validation = validate_phase_snapshot(record)
                if not validation["valid"]:
                    contract_valid = False
                    return None
                return record
            except (KeyError, TypeError, ValueError):
                contract_valid = False
                return None

        def append_stratum(
            source: Mapping[str, Any] | None,
            stratum: str,
            *,
            event_name: str | None = None,
            event_position: str | None = None,
        ) -> None:
            if source is None or stratum in captured_strata:
                return
            record = copy.deepcopy(source)
            record["id"] = (
                f"{parent_id}-{record['two_phase_context']['time_index']}-{stratum}"
            )
            record["candidate_acquisition"] = dict(record["candidate_acquisition"]) | {
                "stratum": stratum
            }
            if event_name is not None:
                record["two_phase_context"]["event_names"] = [event_name]
            if event_position is not None:
                record["two_phase_context"]["event_position"] = event_position
            records.append(record)
            captured_strata.add(stratum)

        key, action_key = jax.random.split(key)
        action, _ = inference(state.obs, action_key)
        for tick in range(1, horizon + 1):
            state = step(state, action)
            jax.block_until_ready(state)
            transitions += 1
            if transition_observer is not None:
                transition_observer(1)
            qpos = np.asarray(jax.device_get(state.data.qpos), np.float32)
            qvel = np.asarray(jax.device_get(state.data.qvel), np.float32)
            trajectory_states.append((qpos.copy(), qvel.copy()))
            key, next_action_key = jax.random.split(key)
            next_action, _ = inference(state.obs, next_action_key)
            events = {
                name: bool(state.info[f"phase_expert/event/{name}"])
                for name in previous_events
            }
            physical_failure = bool(state.info["phase_expert/physical_failure"])
            task_failure = bool(state.info["phase_expert/task_failure"])
            timeout = bool(state.info["phase_expert/timeout"])
            terminal_reason = (
                f"end_code_{int(state.info['end_code'])}"
                if physical_failure or task_failure
                else "continuation_horizon"
                if timeout
                else "none"
            )
            anchor_event = (
                "apex_band_entered"
                if events["apex_band_entered"]
                else "ascending"
                if events["ascending"]
                else "stable_airborne"
                if events["stable_airborne"]
                else "liftoff_seen"
                if events["liftoff_seen"]
                else "jump_window_entered"
            )
            anchor_position = (
                "nearest"
                if events["apex_band_entered"]
                else "event"
                if any(events.values())
                else "pre"
            )
            prior_valid_record = last_valid_record
            current_record = capture(
                state,
                next_action,
                tick=tick,
                stratum="online_tick",
                event_name=anchor_event,
                event_position=anchor_position,
                terminated=physical_failure or task_failure,
                truncated=timeout,
                termination_reason=terminal_reason,
            )
            if current_record is not None:
                last_valid_record = current_record
                if not events["jump_window_entered"]:
                    pre_window_record = current_record
            if events["jump_window_entered"] and not previous_events["jump_window_entered"]:
                if pre_window_record is not None:
                    append_stratum(
                        pre_window_record,
                        "pre_window_approach",
                        event_name="jump_window_entered",
                        event_position="pre",
                    )
                if current_record is not None:
                    append_stratum(
                        current_record,
                        "window_entry",
                        event_name="jump_window_entered",
                        event_position="event",
                    )
            if events["liftoff_seen"] and not previous_events["liftoff_seen"] and current_record is not None:
                append_stratum(
                    current_record,
                    "liftoff",
                    event_name="liftoff_seen",
                    event_position="event",
                )
            if events["ascending"]:
                ascent_age += 1
                stratum = {1: "early_ascent", 3: "middle_ascent", 5: "late_ascent"}.get(
                    ascent_age
                )
                if stratum is not None and current_record is not None:
                    append_stratum(
                        current_record,
                        stratum,
                        event_name="ascending",
                        event_position="event",
                    )
            event_now = environment._event_from_info(state.info) if hasattr(
                environment, "_event_from_info"
            ) else None
            apex_signals, _ = environment._extract_signals(
                state,
                environment._geometry,
                (
                    event_now.recovery_hold_count
                    if event_now is not None
                    else state.info["phase_expert/event/recovery_hold_count"]
                ),
            )
            apex_host = jax.device_get(apex_signals)
            apex_thresholds = environment._thresholds.apex
            for physical_stratum in classify_phase_u_candidate_strata(
                events=events,
                signals=apex_host,
                thresholds=apex_thresholds,
                window_active=bool(environment._window_active(state)),
            ):
                append_stratum(
                    current_record,
                    physical_stratum,
                    event_name=(
                        "apex_band_entered"
                        if physical_stratum == "apex_band_boundary"
                        else None
                    ),
                    event_position=(
                        "pre"
                        if physical_stratum == "apex_band_boundary"
                        else None
                    ),
                )
            if events["apex_band_entered"] and not previous_events["apex_band_entered"]:
                success = True
                append_stratum(
                    prior_valid_record,
                    "apex_pre",
                    event_name="apex_band_entered",
                    event_position="pre",
                )
                append_stratum(
                    last_valid_record,
                    "apex_nearest",
                    event_name="apex_band_entered",
                    event_position="nearest",
                )
                post_success_ticks = 1
            elif post_success_ticks > 0:
                append_stratum(
                    current_record,
                    "apex_post",
                    event_name="apex_band_entered",
                    event_position="post",
                )
                break
            if bool(state.done) and not success:
                append_stratum(current_record, "failure_terminal")
            previous_events = events
            action = next_action
            if bool(state.done) and not success:
                break

        trajectory_hash = _trajectory_hash(trajectory_states)
        parent_summaries.append(
            AcquisitionParentSummary(
                seed=int(seed),
                trajectory_hash=trajectory_hash,
                success=success,
                contract_valid=contract_valid,
                candidate_count=len(records),
            )
        )
        all_records.extend(records)

    parents = tuple(parent_summaries)
    gate = evaluate_candidate_acquisition_gate(
        fixed_evaluation,
        parents,
        minimum_independent_successful_parents=minimum_independent_successful_parents,
    )
    return PhaseUCandidateAcquisitionResult(
        gate=gate,
        parents=parents,
        records=tuple(all_records) if gate["eligible"] else (),
        environment_transitions=transitions,
    )


def probe_phase_u_continuations(
    environment: Any,
    params: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    horizon: int,
    source_policy_hash: str,
    protocol_hash: str,
    seed_namespace: str,
    transition_observer: Callable[[int], None] | None = None,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    """Run one bounded formal-restore continuation screen per candidate state."""
    from jax import numpy as jp

    from .phase_expert_training import _END_REASON, _event_info
    from .rollout import restore_snapshot_mode
    from .runtime import build_inference
    from .two_phase_runtime import TwoPhaseEventState

    if len(records) != len(seeds):
        raise ValueError("continuation records and seeds must have equal length")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("continuation horizon must be positive")
    inference = None
    step = None
    labeled: list[Mapping[str, Any]] = []
    transitions = 0
    for record_value, seed in zip(records, seeds, strict=True):
        record = copy.deepcopy(dict(record_value))
        acquisition = record.get("candidate_acquisition")
        terminal_outcome = (
            acquisition.get("terminal_outcome")
            if isinstance(acquisition, Mapping)
            else None
        )
        if terminal_outcome in {"physical_failure", "timeout", "other_failure"}:
            reason = str(
                record.get("two_phase_context", {}).get(
                    "termination_reason", "candidate_terminal_state"
                )
            )
            record["continuation_label"] = build_provisional_continuation_label(
                ({"outcome": terminal_outcome, "termination_reason": reason},),
                phase="propulsion_ascent",
                source_policy_hash=source_policy_hash,
                protocol_hash=protocol_hash,
            )
            record["continuation_provenance"] = build_continuation_branch_provenance(
                seed=int(seed),
                seed_namespace=seed_namespace,
                source_policy_hash=source_policy_hash,
                protocol_hash=protocol_hash,
            )
            labeled.append(record)
            continue
        if inference is None or step is None:
            inference = build_inference(environment, params, deterministic=False)
            step = jax.jit(environment.step)
        saved_event = record.get("two_phase_event_state")
        if not isinstance(saved_event, Mapping) or set(saved_event) != set(
            TwoPhaseEventState._fields
        ):
            raise ValueError("candidate snapshot lacks complete two-phase event state")
        event = TwoPhaseEventState(
            *(jp.asarray(saved_event[name]) for name in TwoPhaseEventState._fields)
        )
        key = jax.random.PRNGKey(int(seed))
        base_state = restore_snapshot_mode(
            environment._base_env,
            record,
            key,
            observation_mode="timing_explicit_independent_reconstruction",
        )
        info = base_state.info | _event_info(event) | {
            "phase_expert/source_phase_id": jp.asarray(0, jp.int32),
            "phase_expert/reset_valid": jp.asarray(True),
            "phase_expert/episode_step": jp.asarray(0, jp.int32),
            "phase_expert/success": jp.asarray(event.apex_band_entered),
            "phase_expert/physical_failure": jp.asarray(False),
            "phase_expert/task_failure": jp.asarray(False),
            "phase_expert/timeout": jp.asarray(False),
        }
        zero = jp.asarray(0.0, jp.float32)
        state = base_state.replace(
            reward=zero,
            done=jp.zeros_like(base_state.done),
            metrics=environment._metrics(
                reward=zero,
                success=event.apex_band_entered,
                physical=False,
                task=False,
                timeout=False,
            ),
            info=info,
        )
        outcome = "timeout"
        reason = "continuation_horizon"
        if bool(event.apex_band_entered):
            outcome = "success"
            reason = "apex_band_entered"
        else:
            for _ in range(horizon):
                key, action_key = jax.random.split(key)
                action, _ = inference(state.obs, action_key)
                state = step(state, action)
                jax.block_until_ready(state)
                transitions += 1
                if transition_observer is not None:
                    transition_observer(1)
                if bool(state.done):
                    info = jax.device_get(state.info)
                    if bool(info["phase_expert/success"]):
                        outcome, reason = "success", "apex_band_entered"
                    elif bool(info["phase_expert/physical_failure"]):
                        outcome = "physical_failure"
                        reason = _END_REASON.get(int(info["end_code"]), "unknown")
                    elif bool(info["phase_expert/timeout"]):
                        outcome, reason = "timeout", "continuation_horizon"
                    else:
                        outcome = "other_failure"
                        reason = _END_REASON.get(int(info["end_code"]), "unknown")
                    break
        record["continuation_label"] = build_provisional_continuation_label(
            ({"outcome": outcome, "termination_reason": reason},),
            phase="propulsion_ascent",
            source_policy_hash=source_policy_hash,
            protocol_hash=protocol_hash,
        )
        record["continuation_provenance"] = build_continuation_branch_provenance(
            seed=int(seed),
            seed_namespace=seed_namespace,
            source_policy_hash=source_policy_hash,
            protocol_hash=protocol_hash,
        )
        labeled.append(record)
    return tuple(labeled), transitions
