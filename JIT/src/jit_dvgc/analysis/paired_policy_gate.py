"""Paired non-final policy gate for iterative empirical envelope expansion.

This capability compares two exact frozen unified policies on one locked bank.
It is an iteration-selection diagnostic, not final JCE/JEL evidence.  The bank
is fixed before either policy is evaluated by this runner:

* core states are the complete declared source-Tube core;
* boundary states are baseline-negative TRAIN audit states locked before the
  candidate policy is trained/evaluated and absent from the target Tube.

Protocol v1 remains readable for historical gates that selected negatives from
frozen iteration TRAIN evidence.  Protocol v2 consumes an already-locked fresh
negative acceptance bank and resolves every snapshot from that bank's original
acquisition provenance.  Both forms are iteration-generic ``pi_k -> pi_(k+1)``.

The gate never trains, switches experts, consumes validation, or touches TEST.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from ..checkpoint import CheckpointIdentity, load_checkpoint
from ..config import file_sha256, load_config
from ..continuation import NEGATIVE_ACCEPTANCE_BANK_SCHEMA, validate_candidate_snapshot
from ..iteration_train_evidence import canonical_sha256, load_frozen_iteration_train_evidence
from ..ppo import make_checkpoint_policy
from ..soft_tube import load_soft_tube
from ..unified_continuation_labels import (
    classify_unified_continuation_outcome,
    fresh_unified_continuation_start,
)
from ..unified_envelope_snapshot import (
    load_unified_envelope_snapshot,
    physical_state_sha256 as unified_state_sha256,
)
from ..unified_env import UnifiedTubeRSIEnv
from ..unified_formal import load_unified_formal_config
from ..unified_policy_freeze import load_frozen_unified_manifest


CONFIG_SCHEMA = "jit_paired_policy_gate_config_v1"
PROTOCOL_SCHEMA = "jit_paired_policy_gate_protocol_v1"
CONFIG_SCHEMA_V2 = "jit_paired_policy_gate_config_v2"
PROTOCOL_SCHEMA_V2 = "jit_paired_policy_gate_protocol_v2"
REPORT_SCHEMA = "jit_paired_policy_gate_report_v1"

LEGACY_BOUNDARY_SELECTION = "baseline_train_continuation_negative_only"
LOCKED_BOUNDARY_SELECTION = "locked_baseline_continuation_negative_bank"
LOCKED_SNAPSHOT_PROVENANCE = "locked_bank_source_catalog_root"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_state(state: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(jax.device_get(state.data.qpos)).tobytes())
    digest.update(np.asarray(jax.device_get(state.data.qvel)).tobytes())
    return digest.hexdigest()


def _is_v2(config: Mapping[str, Any], protocol: Mapping[str, Any]) -> bool:
    pair = (config.get("schema"), protocol.get("schema"))
    if pair == (CONFIG_SCHEMA, PROTOCOL_SCHEMA):
        return False
    if pair == (CONFIG_SCHEMA_V2, PROTOCOL_SCHEMA_V2):
        return True
    raise ValueError("paired policy gate config/protocol schema drift")


def _validate_common_protocol(protocol: Mapping[str, Any]) -> tuple[int, int]:
    if protocol.get("status") != "predeclared_before_gate_execution":
        raise ValueError("paired policy gate protocol must be predeclared")
    source_iteration = int(protocol.get("source_iteration", -1))
    candidate_iteration = int(protocol.get("candidate_iteration", -1))
    if source_iteration < 0 or candidate_iteration != source_iteration + 1:
        raise ValueError("paired policy gate requires generic k -> k+1 iteration order")

    policies = protocol.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != {"baseline", "candidate"}:
        raise ValueError("paired policy gate requires baseline and candidate policies")
    for role, expected_iteration in (
        ("baseline", source_iteration),
        ("candidate", candidate_iteration),
    ):
        value = policies[role]
        if not isinstance(value, Mapping):
            raise ValueError(f"paired policy gate {role} policy declaration missing")
        if int(value.get("iteration", -1)) != expected_iteration:
            raise ValueError(f"paired policy gate {role} iteration drift")
        if value.get("name") != f"pi_{expected_iteration}":
            raise ValueError(f"paired policy gate {role} policy name drift")
        for field in ("actor_sha256", "payload_sha256"):
            text = str(value.get(field, ""))
            if len(text) != 64:
                raise ValueError(f"paired policy gate {role} {field} invalid")
        if not str(value.get("frozen_manifest", "")):
            raise ValueError(f"paired policy gate {role} frozen manifest missing")

    core = protocol.get("core")
    if not isinstance(core, Mapping):
        raise ValueError("paired policy gate core declaration missing")
    if not str(core.get("source_tube", "")):
        raise ValueError("paired policy gate source core Tube missing")
    if len(str(core.get("source_tube_manifest_sha256", ""))) != 64:
        raise ValueError("paired policy gate source core Tube manifest SHA-256 invalid")
    if core.get("selection") != "all_source_tube_entries":
        raise ValueError("paired policy gate core selection drift")
    if core.get("preservation_rule") != "zero_baseline_success_to_candidate_failure":
        raise ValueError("paired policy gate core preservation rule drift")
    if core.get("require_baseline_success_each_phase") is not True:
        raise ValueError("paired policy gate must reject vacuous core preservation")

    boundary = protocol.get("boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("paired policy gate boundary declaration missing")
    if not str(boundary.get("target_tube", "")):
        raise ValueError("paired policy gate target Tube missing")
    if len(str(boundary.get("target_tube_manifest_sha256", ""))) != 64:
        raise ValueError("paired policy gate target Tube manifest SHA-256 invalid")
    if boundary.get("require_baseline_negative_reproduction") is not True:
        raise ValueError("paired policy gate must reproduce baseline boundary failures")
    if int(boundary.get("minimum_candidate_success_parent_groups", 0)) <= 0:
        raise ValueError("paired policy gate boundary parent-group minimum invalid")

    data_policy = protocol.get("data_policy")
    if data_policy != {
        "split": "train_audit_only",
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "training_transitions": 0,
        "expert_switching_used": False,
    }:
        raise ValueError("paired policy gate data policy drift")
    claims = protocol.get("claim_boundary")
    if claims != {
        "iteration_selection_gate_only": True,
        "empirical_envelope_expansion_claim_requires_both_gates": True,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("paired policy gate claim boundary drift")
    return source_iteration, candidate_iteration


def _validate_paired_policy_gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(config)
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("paired policy gate protocol missing")
    is_v2 = _is_v2(config, protocol)
    _validate_common_protocol(protocol)

    boundary = protocol["boundary"]
    if is_v2:
        if boundary.get("selection") != LOCKED_BOUNDARY_SELECTION:
            raise ValueError("paired policy gate v2 must consume a locked negative bank")
        if not str(boundary.get("locked_bank", "")):
            raise ValueError("paired policy gate locked bank path missing")
        for field in (
            "locked_bank_sha256",
            "source_catalog_file_sha256",
            "source_catalog_protocol_sha256",
        ):
            if len(str(boundary.get(field, ""))) != 64:
                raise ValueError(f"paired policy gate boundary {field} invalid")
        if not str(boundary.get("source_catalog_root", "")):
            raise ValueError("paired policy gate source catalog root missing")
        if boundary.get("snapshot_provenance") != LOCKED_SNAPSHOT_PROVENANCE:
            raise ValueError("paired policy gate locked-bank snapshot provenance drift")

        runtime = protocol.get("runtime")
        if not isinstance(runtime, Mapping) or set(runtime) != {
            "policy_mode",
            "max_ticks",
            "protocol_seed",
        }:
            raise ValueError("paired policy gate runtime contract drift")
        if runtime.get("policy_mode") != "deterministic":
            raise ValueError("paired policy gate must use deterministic policies")
        if int(runtime.get("max_ticks", 0)) <= 0:
            raise ValueError("paired policy gate runtime horizon invalid")
        if int(runtime.get("protocol_seed", -1)) < 0:
            raise ValueError("paired policy gate protocol seed invalid")
    else:
        if boundary.get("selection") != LEGACY_BOUNDARY_SELECTION:
            raise ValueError("paired policy gate v1 boundary selection drift")
        if not str(boundary.get("frozen_train_evidence", "")):
            raise ValueError("paired policy gate TRAIN evidence path missing")
        if len(str(boundary.get("frozen_train_manifest_sha256", ""))) != 64:
            raise ValueError("paired policy gate TRAIN evidence manifest SHA-256 invalid")
        roots = boundary.get("snapshot_search_roots")
        if not isinstance(roots, list) or not roots:
            raise ValueError("paired policy gate snapshot roots missing")
        runtime = protocol.get("runtime")
        if runtime != {
            "policy_mode": "deterministic",
            "max_ticks": 400,
            "protocol_seed": int(runtime.get("protocol_seed", -1))
            if isinstance(runtime, Mapping)
            else -1,
        }:
            raise ValueError("paired policy gate runtime contract drift")
        if int(runtime["protocol_seed"]) < 0:
            raise ValueError("paired policy gate protocol seed invalid")

    if not str(config.get("output_dir", "")):
        raise ValueError("paired policy gate output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("paired policy gate protocol SHA-256 drift")
    return config


def load_paired_policy_gate_config(path: Path) -> dict[str, Any]:
    return _validate_paired_policy_gate_config(_read_object(Path(path)))


def summarize_paired_gate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_candidate_success_parent_groups: int,
    require_baseline_success_each_phase: bool = True,
) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    core = [row for row in rows if row.get("bank_role") == "core"]
    boundary = [row for row in rows if row.get("bank_role") == "boundary"]
    if not core or not boundary:
        raise ValueError("paired policy gate requires non-empty core and boundary banks")
    phases = ("upstream", "downstream")
    core_phase_baseline_success = {
        phase: sum(bool(row["baseline_success"]) for row in core if row["phase"] == phase)
        for phase in phases
    }
    core_regressions = [
        row for row in core if bool(row["baseline_success"]) and not bool(row["candidate_success"])
    ]
    core_improvements = [
        row for row in core if not bool(row["baseline_success"]) and bool(row["candidate_success"])
    ]
    core_nonvacuous = all(core_phase_baseline_success[phase] > 0 for phase in phases)
    core_pass = len(core_regressions) == 0 and (
        core_nonvacuous if require_baseline_success_each_phase else True
    )

    boundary_reproduction_failures = [row for row in boundary if bool(row["baseline_success"])]
    candidate_boundary_successes = [row for row in boundary if bool(row["candidate_success"])]
    successful_parent_groups = {
        str(row["parent_group_id"]) for row in candidate_boundary_successes
    }
    boundary_pass = (
        not boundary_reproduction_failures
        and len(candidate_boundary_successes) > 0
        and len(successful_parent_groups) >= int(minimum_candidate_success_parent_groups)
    )

    def _phase_counts(source: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for phase in phases:
            phase_rows = [row for row in source if row["phase"] == phase]
            result[phase] = {
                "state_count": len(phase_rows),
                "baseline_success_count": sum(bool(row["baseline_success"]) for row in phase_rows),
                "candidate_success_count": sum(bool(row["candidate_success"]) for row in phase_rows),
                "improvement_count": sum(
                    (not bool(row["baseline_success"])) and bool(row["candidate_success"])
                    for row in phase_rows
                ),
                "regression_count": sum(
                    bool(row["baseline_success"]) and (not bool(row["candidate_success"]))
                    for row in phase_rows
                ),
            }
        return result

    return {
        "core": {
            "state_count": len(core),
            "baseline_success_count": sum(bool(row["baseline_success"]) for row in core),
            "candidate_success_count": sum(bool(row["candidate_success"]) for row in core),
            "regression_count": len(core_regressions),
            "improvement_count": len(core_improvements),
            "baseline_success_each_phase": core_phase_baseline_success,
            "nonvacuous": core_nonvacuous,
            "passed": core_pass,
            "phase_counts": _phase_counts(core),
        },
        "boundary": {
            "state_count": len(boundary),
            "baseline_reproduction_failure_count": len(boundary_reproduction_failures),
            "candidate_success_count": len(candidate_boundary_successes),
            "candidate_success_parent_group_count": len(successful_parent_groups),
            "minimum_candidate_success_parent_groups": int(
                minimum_candidate_success_parent_groups
            ),
            "passed": boundary_pass,
            "phase_counts": _phase_counts(boundary),
        },
        "accepted": bool(core_pass and boundary_pass),
    }


def _load_policy(
    protocol: Mapping[str, Any], role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration = protocol["policies"][role]
    manifest_path = Path(str(declaration["frozen_manifest"]))
    manifest = load_frozen_unified_manifest(manifest_path)
    record = dict(manifest["policy"])
    for field in ("iteration", "name", "actor_sha256", "payload_sha256"):
        if record.get(field) != declaration[field]:
            raise ValueError(f"paired policy gate {role} frozen {field} drift")
    return manifest, record


def _build_runtime(
    protocol: Mapping[str, Any],
    baseline_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    baseline_config = load_unified_formal_config(
        Path(str(baseline_record["formal_config"]))
    )
    candidate_config = load_unified_formal_config(
        Path(str(candidate_record["formal_config"]))
    )
    for field in ("up_config_sha256", "down_config_sha256", "runtime_naccdmax"):
        if getattr(baseline_config, field) != getattr(candidate_config, field):
            raise ValueError(f"paired policy gate policy runtime {field} mismatch")
    if baseline_config.ppo.episode_horizon != candidate_config.ppo.episode_horizon:
        raise ValueError("paired policy gate policy horizon mismatch")
    if int(protocol["runtime"]["max_ticks"]) != baseline_config.ppo.episode_horizon:
        raise ValueError("paired policy gate runtime horizon drift")
    up_config = load_config(Path(baseline_config.up_config_path))
    down_config = load_config(Path(baseline_config.down_config_path))
    core_tube = load_soft_tube(Path(str(protocol["core"]["source_tube"])))
    if (
        core_tube.manifest.get("manifest_sha256")
        != protocol["core"]["source_tube_manifest_sha256"]
    ):
        raise ValueError("paired policy gate source core Tube identity drift")
    target_tube = load_soft_tube(Path(str(protocol["boundary"]["target_tube"])))
    if (
        target_tube.manifest.get("manifest_sha256")
        != protocol["boundary"]["target_tube_manifest_sha256"]
    ):
        raise ValueError("paired policy gate target Tube identity drift")
    env = UnifiedTubeRSIEnv(
        up_config,
        down_config,
        core_tube,
        runtime_naccdmax=baseline_config.runtime_naccdmax,
        natural_reset_probability=0.0,
    )
    if (
        env._bundle.xml_sha256 != baseline_record["xml_sha256"]
        or env._bundle.xml_sha256 != candidate_record["xml_sha256"]
    ):
        raise ValueError("paired policy gate XML identity mismatch")
    return env, core_tube, target_tube


def _checkpoint_policy(env: Any, record: Mapping[str, Any]):
    identity = CheckpointIdentity(
        config_sha256=str(record["formal_config_sha256"]),
        xml_sha256=str(record["xml_sha256"]),
        actor_frame_fields=tuple(record["actor_frame_fields"]),
        actor_task_fields=tuple(record["actor_task_fields"]),
        action_order=tuple(record["action_order"]),
    )
    payload = load_checkpoint(Path(str(record["checkpoint"])), expected=identity)
    if int(payload.training_transitions) != int(record["source_training_transitions"]):
        raise ValueError("paired policy gate checkpoint transition drift")
    return jax.jit(make_checkpoint_policy(env, payload, deterministic=True))


def _resolve_unified_snapshots(
    search_roots: Sequence[Path], target_states: set[str]
) -> dict[str, Path]:
    matches: dict[str, list[Path]] = {state: [] for state in target_states}
    for root in search_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"paired policy gate snapshot root missing: {root}")
        for identity_path in root.rglob("identity.json"):
            try:
                identity = _read_object(identity_path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if identity.get("schema") != "jit_unified_envelope_snapshot_v1":
                continue
            state_sha = str(identity.get("physical_state_sha256", ""))
            if state_sha in matches:
                matches[state_sha].append(identity_path.parent.resolve())
    unresolved = sorted(state for state, paths in matches.items() if not paths)
    if unresolved:
        raise FileNotFoundError(
            f"paired policy gate cannot resolve {len(unresolved)} boundary snapshots; "
            f"first={unresolved[0]}"
        )
    return {state: sorted(paths)[0] for state, paths in matches.items()}


def _core_bank_rows(core_tube: Any) -> list[dict[str, Any]]:
    core_rows: list[dict[str, Any]] = []
    phase_local = Counter()
    for global_index, row in enumerate(core_tube.entries):
        phase = str(row["phase"])
        if phase not in ("upstream", "downstream"):
            raise ValueError("paired policy gate core phase drift")
        core_rows.append(
            {
                "bank_role": "core",
                "phase": phase,
                "phase_index": 0 if phase == "upstream" else 1,
                "entry_index": int(phase_local[phase]),
                "global_index": global_index,
                "state_sha256": str(row["state_sha256"]),
                "parent_group_id": str(
                    row.get("parent_group_id", f"core:{phase}:{global_index}")
                ),
            }
        )
        phase_local[phase] += 1
    if len(core_rows) != len(core_tube.entries):
        raise ValueError("paired policy gate core bank count drift")
    return core_rows


def _legacy_boundary_rows(
    protocol: Mapping[str, Any],
    target_tube: Any,
    baseline_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    boundary = protocol["boundary"]
    evidence_root = Path(str(boundary["frozen_train_evidence"]))
    manifest, rows = load_frozen_iteration_train_evidence(evidence_root)
    if manifest.get("manifest_sha256") != boundary["frozen_train_manifest_sha256"]:
        raise ValueError("paired policy gate TRAIN evidence manifest drift")
    if int(manifest.get("iteration", -1)) != int(baseline_record["iteration"]):
        raise ValueError("paired policy gate TRAIN evidence iteration drift")
    if manifest.get("policy_actor_sha256") != baseline_record["actor_sha256"]:
        raise ValueError("paired policy gate TRAIN evidence actor drift")
    if manifest.get("policy_payload_sha256") != baseline_record["payload_sha256"]:
        raise ValueError("paired policy gate TRAIN evidence payload drift")
    selected = [dict(row) for row in rows if int(row.get("label", -1)) == 0]
    if not selected:
        raise ValueError("paired policy gate has no baseline-negative boundary states")
    selected_states = {str(row["state_sha256"]) for row in selected}
    target_states = {str(row["state_sha256"]) for row in target_tube.entries}
    overlap = sorted(selected_states.intersection(target_states))
    if overlap:
        raise ValueError(
            "paired policy gate boundary bank contains a state already admitted "
            "to target Tube"
        )
    resolved = _resolve_unified_snapshots(
        [Path(str(root)) for root in boundary["snapshot_search_roots"]],
        selected_states,
    )
    boundary_rows: list[dict[str, Any]] = []
    for row in sorted(
        selected,
        key=lambda value: (
            str(value["phase"]),
            str(value["parent_group_id"]),
            str(value["state_sha256"]),
        ),
    ):
        state_sha = str(row["state_sha256"])
        snapshot_path = resolved[state_sha]
        snapshot = load_unified_envelope_snapshot(snapshot_path)
        if unified_state_sha256(snapshot) != state_sha:
            raise ValueError(
                "paired policy gate boundary snapshot physical-state drift"
            )
        if snapshot.policy_iteration != int(baseline_record["iteration"]):
            raise ValueError("paired policy gate boundary snapshot iteration drift")
        if snapshot.policy_actor_sha256 != baseline_record["actor_sha256"]:
            raise ValueError("paired policy gate boundary snapshot actor drift")
        if snapshot.policy_payload_sha256 != baseline_record["payload_sha256"]:
            raise ValueError("paired policy gate boundary snapshot payload drift")
        boundary_rows.append(
            {
                "bank_role": "boundary",
                "phase": str(row["phase"]),
                "phase_index": int(row["phase_index"]),
                "state_sha256": state_sha,
                "parent_group_id": str(row["parent_group_id"]),
                "source_label": 0,
                "snapshot": str(snapshot_path),
            }
        )
    return boundary_rows


def _acceptance_bank_path(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path / "acceptance_bank.json"
    return path


def _locked_snapshot_path(
    bank: Mapping[str, Any], row: Mapping[str, Any]
) -> Path:
    return (
        Path(str(bank["source_catalog_root"]))
        / str(row["source_bank"])
        / str(row["snapshot"])
    )


def _load_locked_negative_bank(
    boundary: Mapping[str, Any],
    *,
    baseline_record: Mapping[str, Any],
    target_tube: Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    bank_path = _acceptance_bank_path(Path(str(boundary["locked_bank"])))
    bank = _read_object(bank_path)
    if bank.get("schema") != NEGATIVE_ACCEPTANCE_BANK_SCHEMA:
        raise ValueError("paired policy gate locked negative bank schema drift")
    if bank.get("status") not in {
        "locked_before_repair_training",
        "locked_before_candidate_training",
    }:
        raise ValueError("paired policy gate negative bank was not locked pre-training")
    declared_bank_sha = str(bank.get("bank_sha256", ""))
    if len(declared_bank_sha) != 64:
        raise ValueError("paired policy gate locked bank self-hash missing")
    if (
        canonical_sha256(
            {key: value for key, value in bank.items() if key != "bank_sha256"}
        )
        != declared_bank_sha
    ):
        raise ValueError("paired policy gate locked bank self-hash drift")
    if declared_bank_sha != str(boundary["locked_bank_sha256"]):
        raise ValueError("paired policy gate locked bank identity drift")

    if bank.get("artifact_role") != (
        "fresh_nonfinal_baseline_negative_iteration_acceptance_bank"
    ):
        raise ValueError("paired policy gate locked bank artifact role drift")
    if bank.get("split") != "train_audit_only":
        raise ValueError("paired policy gate locked bank split drift")
    if bank.get("selection") != "all_baseline_continuation_negative_candidates":
        raise ValueError("paired policy gate locked bank selection drift")
    if bank.get("baseline_policy_actor_sha256") != baseline_record["actor_sha256"]:
        raise ValueError("paired policy gate locked bank baseline actor drift")
    if (
        bank.get("baseline_policy_payload_sha256")
        != baseline_record["payload_sha256"]
    ):
        raise ValueError("paired policy gate locked bank baseline payload drift")
    if str(bank.get("target_tube", "")) != str(boundary["target_tube"]):
        raise ValueError("paired policy gate locked bank target Tube path drift")
    if (
        bank.get("target_tube_manifest_sha256")
        != boundary["target_tube_manifest_sha256"]
    ):
        raise ValueError("paired policy gate locked bank target Tube identity drift")
    if (
        target_tube.manifest.get("manifest_sha256")
        != bank["target_tube_manifest_sha256"]
    ):
        raise ValueError("paired policy gate loaded target Tube/bank identity drift")

    for key in (
        "training_transitions",
        "environment_interactions",
    ):
        if int(bank.get(key, -1)) != 0:
            raise ValueError(f"paired policy gate locked bank {key} drift")
    for key in (
        "expert_switching_used",
        "validation_data_used",
        "test_data_used",
        "final_evaluation_data_used",
    ):
        if bank.get(key) is not False:
            raise ValueError(f"paired policy gate locked bank {key} drift")
    claims = bank.get("claim_boundary")
    if not isinstance(claims, Mapping):
        raise ValueError("paired policy gate locked bank claim boundary missing")
    if claims.get("candidate_policy_outcomes_inspected") is not False:
        raise ValueError(
            "paired policy gate locked bank inspected candidate outcomes pre-lock"
        )
    if claims.get("nonfinal_iteration_acceptance_audit_only") is not True:
        raise ValueError("paired policy gate locked bank role claim drift")
    if claims.get("jce_jel_claim") is not False:
        raise ValueError("paired policy gate locked bank used final JCE/JEL")

    source_catalog_raw = str(bank.get("source_catalog", ""))
    source_root_raw = str(bank.get("source_catalog_root", ""))
    if not source_catalog_raw or not source_root_raw:
        raise ValueError("paired policy gate locked bank source catalog provenance missing")
    source_catalog = Path(source_catalog_raw)
    source_root = Path(source_root_raw)
    if source_catalog.parent.resolve() != source_root.resolve():
        raise ValueError("paired policy gate source catalog root/path drift")
    if str(source_root) != str(boundary["source_catalog_root"]):
        raise ValueError("paired policy gate configured source catalog root drift")
    if (
        str(bank.get("source_catalog_file_sha256", ""))
        != str(boundary["source_catalog_file_sha256"])
    ):
        raise ValueError("paired policy gate source catalog file identity drift")
    if file_sha256(source_catalog) != str(bank["source_catalog_file_sha256"]):
        raise ValueError("paired policy gate source catalog file SHA-256 drift")
    if (
        str(bank.get("source_catalog_protocol_sha256", ""))
        != str(boundary["source_catalog_protocol_sha256"])
    ):
        raise ValueError("paired policy gate source catalog protocol drift")
    source_catalog_payload = _read_object(source_catalog)
    if source_catalog_payload.get("status") != "completed":
        raise ValueError("paired policy gate source catalog is not completed")
    if (
        source_catalog_payload.get("protocol_sha256")
        != bank["source_catalog_protocol_sha256"]
    ):
        raise ValueError("paired policy gate source catalog semantic protocol drift")
    catalog_rows = source_catalog_payload.get("entries")
    if not isinstance(catalog_rows, list) or not catalog_rows:
        raise ValueError("paired policy gate source catalog has no entries")
    catalog_by_state: dict[str, Mapping[str, Any]] = {}
    for catalog_row in catalog_rows:
        if not isinstance(catalog_row, Mapping):
            raise ValueError("paired policy gate source catalog entry drift")
        catalog_state = str(catalog_row.get("state_sha256", ""))
        if len(catalog_state) != 64 or catalog_state in catalog_by_state:
            raise ValueError("paired policy gate source catalog state identity drift")
        catalog_by_state[catalog_state] = catalog_row

    rows_raw = bank.get("entries")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("paired policy gate locked bank has no entries")
    if int(bank.get("entry_count", -1)) != len(rows_raw):
        raise ValueError("paired policy gate locked bank entry count drift")
    selection_audit = bank.get("selection_audit")
    if not isinstance(selection_audit, Mapping):
        raise ValueError("paired policy gate locked bank selection audit missing")
    if int(selection_audit.get("locked_negative_count", -1)) != len(rows_raw):
        raise ValueError("paired policy gate locked bank selection count drift")

    target_states = {str(row["state_sha256"]) for row in target_tube.entries}
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    phases: set[str] = set()
    for source in rows_raw:
        row = dict(source)
        if row.get("split") != "train" or int(row.get("label", -1)) != 0:
            raise ValueError(
                "paired policy gate locked bank contains a nonnegative/non-TRAIN row"
            )
        phase = str(row.get("phase", ""))
        phase_index = int(row.get("phase_index", -1))
        if (phase, phase_index) not in (("upstream", 0), ("downstream", 1)):
            raise ValueError("paired policy gate locked bank phase identity drift")
        state_sha = str(row.get("state_sha256", ""))
        if len(state_sha) != 64 or state_sha in seen:
            raise ValueError("paired policy gate locked bank state identity drift")
        if state_sha in target_states:
            raise ValueError(
                "paired policy gate locked boundary state is already in target Tube"
            )
        if int(row.get("policy_iteration", -1)) != int(
            baseline_record["iteration"]
        ):
            raise ValueError("paired policy gate locked bank policy iteration drift")
        if row.get("policy_actor_sha256") != baseline_record["actor_sha256"]:
            raise ValueError("paired policy gate locked bank row actor drift")
        if row.get("policy_payload_sha256") != baseline_record["payload_sha256"]:
            raise ValueError("paired policy gate locked bank row payload drift")
        if (
            row.get("acquisition_protocol_sha256")
            != bank["acquisition_protocol_sha256"]
        ):
            raise ValueError("paired policy gate locked bank acquisition protocol drift")
        if row.get("label_protocol_sha256") != bank["label_protocol_sha256"]:
            raise ValueError("paired policy gate locked bank label protocol drift")
        if not str(row.get("source_bank", "")) or not str(row.get("snapshot", "")):
            raise ValueError("paired policy gate locked bank snapshot provenance missing")
        if not str(row.get("parent_group_id", "")):
            raise ValueError("paired policy gate locked bank parent group missing")
        catalog_row = catalog_by_state.get(state_sha)
        if catalog_row is None:
            raise ValueError("paired policy gate locked state is absent from source catalog")
        for field in (
            "candidate_id",
            "phase",
            "phase_index",
            "source_bank",
            "snapshot",
            "state_sha256",
            "parent_group_id",
            "parent_state_sha256",
        ):
            if catalog_row.get(field) != row.get(field):
                raise ValueError(
                    f"paired policy gate locked row/source catalog {field} drift"
                )
        seen.add(state_sha)
        phases.add(phase)
        rows.append(row)
    if phases != {"upstream", "downstream"}:
        raise ValueError("paired policy gate locked bank must cover both phases")
    return bank, tuple(rows)


def _locked_boundary_rows(
    protocol: Mapping[str, Any],
    target_tube: Any,
    baseline_record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    boundary = protocol["boundary"]
    bank, selected = _load_locked_negative_bank(
        boundary,
        baseline_record=baseline_record,
        target_tube=target_tube,
    )
    boundary_rows: list[dict[str, Any]] = []
    for row in sorted(
        selected,
        key=lambda value: (
            str(value["phase"]),
            str(value["parent_group_id"]),
            str(value["state_sha256"]),
        ),
    ):
        snapshot_path = _locked_snapshot_path(bank, row)
        if not snapshot_path.is_dir():
            raise FileNotFoundError(
                f"paired policy gate locked snapshot missing: {snapshot_path}"
            )
        snapshot = load_unified_envelope_snapshot(snapshot_path)
        validate_candidate_snapshot(
            snapshot,
            row,
            policy_record=baseline_record,
        )
        if unified_state_sha256(snapshot) != row["state_sha256"]:
            raise ValueError(
                "paired policy gate locked snapshot physical-state identity drift"
            )
        boundary_rows.append(
            {
                "bank_role": "boundary",
                "phase": str(row["phase"]),
                "phase_index": int(row["phase_index"]),
                "candidate_id": str(row.get("candidate_id", "")),
                "state_sha256": str(row["state_sha256"]),
                "parent_group_id": str(row["parent_group_id"]),
                "source_label": 0,
                "source_acceptance_bank_sha256": str(bank["bank_sha256"]),
                "source_catalog_protocol_sha256": str(
                    bank["source_catalog_protocol_sha256"]
                ),
                "snapshot": str(snapshot_path),
            }
        )
    return boundary_rows, bank


def _lock_bank(
    protocol: Mapping[str, Any],
    core_tube: Any,
    target_tube: Any,
    baseline_record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    core_rows = _core_bank_rows(core_tube)
    selection = str(protocol["boundary"]["selection"])
    if selection == LEGACY_BOUNDARY_SELECTION:
        boundary_rows = _legacy_boundary_rows(
            protocol, target_tube, baseline_record
        )
        source = {
            "selection": LEGACY_BOUNDARY_SELECTION,
            "frozen_train_evidence": str(
                protocol["boundary"]["frozen_train_evidence"]
            ),
            "frozen_train_manifest_sha256": str(
                protocol["boundary"]["frozen_train_manifest_sha256"]
            ),
        }
    elif selection == LOCKED_BOUNDARY_SELECTION:
        boundary_rows, bank = _locked_boundary_rows(
            protocol, target_tube, baseline_record
        )
        source = {
            "selection": LOCKED_BOUNDARY_SELECTION,
            "locked_bank": str(protocol["boundary"]["locked_bank"]),
            "locked_bank_sha256": str(bank["bank_sha256"]),
            "source_catalog_root": str(bank["source_catalog_root"]),
            "source_catalog_file_sha256": str(bank["source_catalog_file_sha256"]),
            "source_catalog_protocol_sha256": str(
                bank["source_catalog_protocol_sha256"]
            ),
        }
    else:
        raise ValueError("paired policy gate boundary selection unsupported")
    if {row["phase"] for row in boundary_rows} != {"upstream", "downstream"}:
        raise ValueError("paired policy gate boundary bank must cover both phases")
    return core_rows, boundary_rows, source


def prepare_locked_bank_gate_config(
    *,
    baseline_frozen_manifest: Path,
    candidate_frozen_manifest: Path,
    core_tube_path: Path,
    locked_bank_path: Path,
    output_dir: Path,
    minimum_candidate_success_parent_groups: int,
    protocol_seed: int,
) -> dict[str, Any]:
    """Build a v2 immutable gate config from already-frozen k/k+1 artifacts.

    This only binds identities and criteria; it performs no environment rollout
    and inspects no candidate-policy outcome.
    """
    minimum_groups = int(minimum_candidate_success_parent_groups)
    if minimum_groups <= 0:
        raise ValueError("minimum candidate success parent groups must be positive")
    protocol_seed = int(protocol_seed)
    if protocol_seed < 0:
        raise ValueError("paired policy gate protocol seed must be nonnegative")

    baseline_manifest = load_frozen_unified_manifest(
        Path(baseline_frozen_manifest)
    )
    candidate_manifest = load_frozen_unified_manifest(
        Path(candidate_frozen_manifest)
    )
    baseline_record = dict(baseline_manifest["policy"])
    candidate_record = dict(candidate_manifest["policy"])
    source_iteration = int(baseline_record["iteration"])
    candidate_iteration = int(candidate_record["iteration"])
    if source_iteration < 0 or candidate_iteration != source_iteration + 1:
        raise ValueError("paired policy gate preparation requires generic k -> k+1")
    if baseline_record["name"] != f"pi_{source_iteration}":
        raise ValueError("paired policy gate baseline name/iteration drift")
    if candidate_record["name"] != f"pi_{candidate_iteration}":
        raise ValueError("paired policy gate candidate name/iteration drift")
    if baseline_record["xml_sha256"] != candidate_record["xml_sha256"]:
        raise ValueError("paired policy gate policy XML mismatch")
    for role, record in (("baseline", baseline_record), ("candidate", candidate_record)):
        if record.get("policy_role") != "envelope_expansion_authority":
            raise ValueError(
                f"paired policy gate {role} frozen policy is not an expansion authority"
            )

    baseline_config = load_unified_formal_config(
        Path(str(baseline_record["formal_config"]))
    )
    candidate_config = load_unified_formal_config(
        Path(str(candidate_record["formal_config"]))
    )
    if baseline_config.ppo.episode_horizon != candidate_config.ppo.episode_horizon:
        raise ValueError("paired policy gate policy horizon mismatch")

    core_tube = load_soft_tube(Path(core_tube_path))
    bank_file = _acceptance_bank_path(Path(locked_bank_path))
    raw_bank = _read_object(bank_file)
    if raw_bank.get("schema") != NEGATIVE_ACCEPTANCE_BANK_SCHEMA:
        raise ValueError("paired policy gate preparation locked bank schema drift")
    bank_sha = str(raw_bank.get("bank_sha256", ""))
    if (
        len(bank_sha) != 64
        or canonical_sha256(
            {
                key: value
                for key, value in raw_bank.items()
                if key != "bank_sha256"
            }
        )
        != bank_sha
    ):
        raise ValueError("paired policy gate preparation locked bank self-hash drift")
    target_tube_raw = str(raw_bank.get("target_tube", ""))
    if not target_tube_raw:
        raise ValueError("paired policy gate preparation target Tube missing")
    target_tube_path = Path(target_tube_raw)
    target_tube = load_soft_tube(target_tube_path)
    source_catalog_root = str(raw_bank.get("source_catalog_root", ""))
    source_catalog_file_sha256 = str(raw_bank.get("source_catalog_file_sha256", ""))
    source_catalog_protocol_sha256 = str(
        raw_bank.get("source_catalog_protocol_sha256", "")
    )
    if not source_catalog_root:
        raise ValueError("paired policy gate preparation source catalog root missing")
    if len(source_catalog_file_sha256) != 64:
        raise ValueError("paired policy gate preparation source catalog file SHA invalid")
    if len(source_catalog_protocol_sha256) != 64:
        raise ValueError("paired policy gate preparation source catalog protocol SHA invalid")

    boundary = {
        "selection": LOCKED_BOUNDARY_SELECTION,
        "locked_bank": str(bank_file),
        "locked_bank_sha256": bank_sha,
        "target_tube": str(target_tube_path),
        "target_tube_manifest_sha256": str(
            raw_bank["target_tube_manifest_sha256"]
        ),
        "source_catalog_root": source_catalog_root,
        "source_catalog_file_sha256": source_catalog_file_sha256,
        "source_catalog_protocol_sha256": source_catalog_protocol_sha256,
        "snapshot_provenance": LOCKED_SNAPSHOT_PROVENANCE,
        "require_baseline_negative_reproduction": True,
        "minimum_candidate_success_parent_groups": minimum_groups,
    }
    _load_locked_negative_bank(
        boundary,
        baseline_record=baseline_record,
        target_tube=target_tube,
    )

    protocol = {
        "schema": PROTOCOL_SCHEMA_V2,
        "status": "predeclared_before_gate_execution",
        "source_iteration": source_iteration,
        "candidate_iteration": candidate_iteration,
        "policies": {
            "baseline": {
                "iteration": source_iteration,
                "name": str(baseline_record["name"]),
                "frozen_manifest": str(baseline_frozen_manifest),
                "actor_sha256": str(baseline_record["actor_sha256"]),
                "payload_sha256": str(baseline_record["payload_sha256"]),
            },
            "candidate": {
                "iteration": candidate_iteration,
                "name": str(candidate_record["name"]),
                "frozen_manifest": str(candidate_frozen_manifest),
                "actor_sha256": str(candidate_record["actor_sha256"]),
                "payload_sha256": str(candidate_record["payload_sha256"]),
            },
        },
        "core": {
            "source_tube": str(core_tube_path),
            "source_tube_manifest_sha256": str(
                core_tube.manifest["manifest_sha256"]
            ),
            "selection": "all_source_tube_entries",
            "preservation_rule": "zero_baseline_success_to_candidate_failure",
            "require_baseline_success_each_phase": True,
        },
        "boundary": boundary,
        "runtime": {
            "policy_mode": "deterministic",
            "max_ticks": int(baseline_config.ppo.episode_horizon),
            "protocol_seed": protocol_seed,
        },
        "data_policy": {
            "split": "train_audit_only",
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
            "training_transitions": 0,
            "expert_switching_used": False,
        },
        "claim_boundary": {
            "iteration_selection_gate_only": True,
            "empirical_envelope_expansion_claim_requires_both_gates": True,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    config = {
        "schema": CONFIG_SCHEMA_V2,
        "output_dir": str(output_dir),
        "expected_protocol_sha256": canonical_sha256(protocol),
        "protocol": protocol,
    }
    return _validate_paired_policy_gate_config(config)


def _rollout(
    env: Any,
    policy: Any,
    state: Any,
    *,
    step_fn: Any,
    start_phase: int,
    max_ticks: int,
    seed: int,
) -> dict[str, Any]:
    if bool(np.asarray(jax.device_get(state.done))):
        raise ValueError("paired policy gate start state is terminal")
    if bool(np.asarray(jax.device_get(state.info["expert_switching_used"]))):
        raise ValueError("paired policy gate start used expert switching")
    apex_seen = bool(np.asarray(jax.device_get(state.info["up_events"].apex_seen)))
    phase_transitioned = bool(
        np.asarray(jax.device_get(state.info["phase_transitioned"]))
    )
    recovery_success = bool(
        np.asarray(jax.device_get(state.info["down_events"].recovery_success))
    )
    interactions = 0
    for tick in range(int(max_ticks)):
        key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(tick))
        result = policy(state.obs, key)
        action = result[0] if isinstance(result, tuple) else result
        action_array = np.asarray(jax.device_get(action), dtype=np.float32).reshape(-1)
        if action_array.shape != (4,) or not np.isfinite(action_array).all():
            raise ValueError("paired policy gate policy returned invalid action")
        state = step_fn(state, action)
        jax.block_until_ready(state)
        interactions += 1
        if bool(np.asarray(jax.device_get(state.info["expert_switching_used"]))):
            raise ValueError("paired policy gate rollout used expert switching")
        apex_seen |= bool(
            np.asarray(jax.device_get(state.info["up_events"].apex_seen))
        )
        phase_transitioned |= bool(
            np.asarray(jax.device_get(state.info["phase_transitioned"]))
        )
        recovery_success |= bool(
            np.asarray(jax.device_get(state.info["down_events"].recovery_success))
        )
        if bool(np.asarray(jax.device_get(state.done))):
            break
    done = bool(np.asarray(jax.device_get(state.done)))
    terminal_success = bool(np.asarray(jax.device_get(state.info["success"])))
    physical_failure = bool(
        np.asarray(jax.device_get(state.info["physical_failure"]))
    )
    timeout = bool(np.asarray(jax.device_get(state.info["timeout"])))
    positive, outcome = classify_unified_continuation_outcome(
        start_phase=int(start_phase),
        terminal_success=terminal_success,
        physical_failure=physical_failure,
        timeout=timeout,
        done=done,
        apex_seen=apex_seen,
        phase_transitioned=phase_transitioned,
        recovery_success=recovery_success,
        reached_rollout_horizon=(interactions >= int(max_ticks) and not done),
    )
    return {
        "success": bool(positive),
        "outcome_class": str(outcome),
        "environment_interactions": interactions,
        "terminal_success": terminal_success,
        "physical_failure": physical_failure,
        "timeout": timeout,
        "apex_seen": apex_seen,
        "phase_transitioned": phase_transitioned,
        "recovery_success": recovery_success,
    }


def run_paired_policy_gate(config_path: Path) -> dict[str, Any]:
    """Execute a predeclared paired core-preservation/boundary-gain audit."""
    config_path = Path(config_path)
    config = load_paired_policy_gate_config(config_path)
    protocol = dict(config["protocol"])
    is_v2 = _is_v2(config, protocol)
    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "protocol.json",
        {**protocol, "protocol_sha256": canonical_sha256(protocol)},
    )
    interactions = 0
    try:
        if jax.default_backend() != "gpu":
            raise RuntimeError(
                "paired policy gate requires the visible JAX GPU backend"
            )
        _, baseline_record = _load_policy(protocol, "baseline")
        _, candidate_record = _load_policy(protocol, "candidate")
        if baseline_record["xml_sha256"] != candidate_record["xml_sha256"]:
            raise ValueError("paired policy gate frozen policy XML mismatch")
        env, core_tube, target_tube = _build_runtime(
            protocol, baseline_record, candidate_record
        )
        core_bank, boundary_bank, boundary_source = _lock_bank(
            protocol, core_tube, target_tube, baseline_record
        )
        bank = {
            "schema": (
                "jit_paired_policy_gate_bank_v2"
                if is_v2
                else "jit_paired_policy_gate_bank_v1"
            ),
            "status": "locked_before_policy_rollout",
            "source_iteration": int(protocol["source_iteration"]),
            "candidate_iteration": int(protocol["candidate_iteration"]),
            "core_count": len(core_bank),
            "boundary_count": len(boundary_bank),
            "boundary_source": boundary_source,
            "core": core_bank,
            "boundary": boundary_bank,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        }
        bank["bank_sha256"] = canonical_sha256(bank)
        _write_json(output / "bank.json", bank)

        baseline_policy = _checkpoint_policy(env, baseline_record)
        candidate_policy = _checkpoint_policy(env, candidate_record)
        max_ticks = int(protocol["runtime"]["max_ticks"])
        base_seed = int(protocol["runtime"]["protocol_seed"])
        reset_tube = jax.jit(env.reset_tube_index)
        step_fn = jax.jit(env.step)
        records: list[dict[str, Any]] = []

        for index, row in enumerate([*core_bank, *boundary_bank]):
            if row["bank_role"] == "core":
                baseline_state = reset_tube(
                    np.int32(row["phase_index"]), np.int32(row["entry_index"])
                )
                candidate_state = reset_tube(
                    np.int32(row["phase_index"]), np.int32(row["entry_index"])
                )
                if (
                    _sha256_state(baseline_state) != row["state_sha256"]
                    or _sha256_state(candidate_state) != row["state_sha256"]
                ):
                    raise ValueError(
                        "paired policy gate core reset physical-state drift"
                    )
            else:
                snapshot = load_unified_envelope_snapshot(Path(str(row["snapshot"])))
                baseline_state = fresh_unified_continuation_start(snapshot, env)
                candidate_state = fresh_unified_continuation_start(snapshot, env)
            state_seed = base_seed + index * 10_000
            baseline = _rollout(
                env,
                baseline_policy,
                baseline_state,
                step_fn=step_fn,
                start_phase=int(row["phase_index"]),
                max_ticks=max_ticks,
                seed=state_seed,
            )
            interactions += int(baseline["environment_interactions"])
            candidate = _rollout(
                env,
                candidate_policy,
                candidate_state,
                step_fn=step_fn,
                start_phase=int(row["phase_index"]),
                max_ticks=max_ticks,
                seed=state_seed,
            )
            interactions += int(candidate["environment_interactions"])
            records.append(
                {
                    **row,
                    "baseline_success": bool(baseline["success"]),
                    "candidate_success": bool(candidate["success"]),
                    "baseline_outcome_class": baseline["outcome_class"],
                    "candidate_outcome_class": candidate["outcome_class"],
                    "baseline_environment_interactions": baseline[
                        "environment_interactions"
                    ],
                    "candidate_environment_interactions": candidate[
                        "environment_interactions"
                    ],
                }
            )

        gates = summarize_paired_gate_records(
            records,
            minimum_candidate_success_parent_groups=int(
                protocol["boundary"]["minimum_candidate_success_parent_groups"]
            ),
            require_baseline_success_each_phase=bool(
                protocol["core"]["require_baseline_success_each_phase"]
            ),
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "completed",
            "source_iteration": int(protocol["source_iteration"]),
            "candidate_iteration": int(protocol["candidate_iteration"]),
            "baseline_policy_name": str(baseline_record["name"]),
            "baseline_actor_sha256": str(baseline_record["actor_sha256"]),
            "baseline_payload_sha256": str(baseline_record["payload_sha256"]),
            "candidate_policy_name": str(candidate_record["name"]),
            "candidate_actor_sha256": str(candidate_record["actor_sha256"]),
            "candidate_payload_sha256": str(candidate_record["payload_sha256"]),
            "protocol_sha256": canonical_sha256(protocol),
            "bank_sha256": bank["bank_sha256"],
            "boundary_source": boundary_source,
            "core_gate": gates["core"],
            "boundary_gate": gates["boundary"],
            "iteration_accepted": bool(gates["accepted"]),
            "empirical_envelope_expansion_accepted": bool(gates["accepted"]),
            "environment_interactions": interactions,
            "training_transitions": 0,
            "expert_switching_used": False,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        }
        _write_json(output / "records.json", {"records": records})
        _write_json(output / "summary.json", report)
        return report
    except BaseException as exc:
        failure = {
            "schema": REPORT_SCHEMA,
            "status": "engineering_error",
            "source_iteration": int(protocol["source_iteration"]),
            "candidate_iteration": int(protocol["candidate_iteration"]),
            "protocol_sha256": canonical_sha256(protocol),
            "environment_interactions": interactions,
            "training_transitions": 0,
            "expert_switching_used": False,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(output / "summary.json", failure)
        raise
