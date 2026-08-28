"""Validation-only application of a locked TRAIN V_up boundary protocol."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .constants import ACTION_ORDER
from .handoff_snapshot import HandoffSnapshot, load_snapshot, save_snapshot
from .upstream_boundary import (
    BOUNDARY_CATALOG_SCHEMA,
    AuditedAnchor,
    action_basis_directions,
    canonical_sha256,
    file_sha256,
    label_reasons,
    physical_state_sha256,
    row_identity,
    snapshot_path,
    validate_durations,
    validate_strengths,
)
from .upstream_boundary_lock import load_boundary_lock

VALIDATION_PROTOCOL_SCHEMA = "jit_upstream_boundary_validation_protocol_v1"


def _truth(value: Any) -> bool:
    return bool(np.asarray(value))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _label_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("entries", payload.get("labels", []))
    if not isinstance(payload, list):
        raise ValueError("labels must be a list")
    return [dict(row) for row in payload]


def select_locked_validation_anchors(
    nominal_catalog: Path,
    nominal_labels: Path,
    lock_path: Path,
) -> tuple[list[AuditedAnchor], dict[str, Any]]:
    """Select exactly the held-out validation failure anchor(s) named by the lock."""
    lock = load_boundary_lock(lock_path)
    if file_sha256(nominal_catalog) != str(lock["nominal_catalog_sha256"]):
        raise ValueError("nominal catalog differs from the locked TRAIN identity")
    if file_sha256(nominal_labels) != str(lock["nominal_labels_sha256"]):
        raise ValueError("nominal labels differ from the locked TRAIN identity")

    validation_seeds = tuple(int(seed) for seed in lock.get("validation_seeds", ()))
    test_seeds = {int(seed) for seed in lock.get("test_seeds", ())}
    if not validation_seeds or set(validation_seeds).intersection(test_seeds):
        raise ValueError("invalid locked validation/test seed partition")

    labels_by_id: dict[tuple[Any, ...], dict[str, Any]] = {}
    for label in _label_rows(nominal_labels):
        identity = row_identity(label)
        if identity in labels_by_id:
            raise ValueError(f"duplicate nominal label identity: {identity}")
        labels_by_id[identity] = label

    entries = list(_load_json(nominal_catalog).get("entries", []))
    selected: list[AuditedAnchor] = []
    selected_rows = []
    for seed in validation_seeds:
        matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for raw in entries:
            row = dict(raw)
            if int(row.get("seed", -1)) != seed:
                continue
            if int(row.get("source_training_transitions", -1)) != int(lock["source_training_transitions"]):
                continue
            if str(row.get("role", "")) != str(lock["negative_role"]):
                continue
            label = labels_by_id.get(row_identity(row))
            if label is None or str(label.get("split", "")) != "validation":
                continue
            if int(label.get("success_count", 0)) != 0:
                continue
            if str(lock["failure_reason"]) not in label_reasons(label):
                continue
            matches.append((row, label))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one locked validation failure anchor for seed {seed}, found {len(matches)}")
        row, label = matches[0]
        anchor = AuditedAnchor(
            row=row,
            label=label,
            snapshot=load_snapshot(snapshot_path(nominal_catalog, row)),
            anchor_kind="failure_anchor",
        )
        selected.append(anchor)
        selected_rows.append(
            {
                "seed": seed,
                "parent_group_id": row["parent_group_id"],
                "role": row["role"],
                "tick": int(row["tick"]),
                "state_sha256": row.get("state_sha256"),
                "reasons": list(label_reasons(label)),
            }
        )

    audit = {
        "schema": "jit_upstream_boundary_validation_anchor_audit_v1",
        "target": "V_up",
        "split": "validation",
        "lock_sha256": lock["lock_sha256"],
        "validation_seeds": list(validation_seeds),
        "selected_anchor_count": len(selected),
        "selected_failure_anchors": selected_rows,
        "test_seed_interaction_count": 0,
        "training_transitions": 0,
    }
    return selected, audit


def collect_locked_validation_candidates(
    anchors: Sequence[AuditedAnchor],
    output_dir: Path,
    *,
    lock_path: Path,
    restore: Callable[[HandoffSnapshot], Any],
    policy_action: Callable[[Any, int, int], Any],
    step: Callable[[Any, Any], Any],
    capture: Callable[[Any, AuditedAnchor], HandoffSnapshot],
    protocol_seed: int = 820404,
) -> dict[str, Any]:
    """Apply the locked recipe once to validation anchors without retuning."""
    lock = load_boundary_lock(lock_path)
    if not anchors:
        raise ValueError("validation boundary collection requires anchors")
    if any(str(anchor.label.get("split", "")) != "validation" for anchor in anchors):
        raise ValueError("validation collection accepts validation anchors only")

    strengths = validate_strengths(lock["strengths"])
    durations = validate_durations(lock["durations"])
    directions = action_basis_directions(
        action_names=tuple(lock["selected_action_names"]),
        signs=tuple(int(x) for x in lock["selected_signs"]),
    )
    protocol = {
        "schema": VALIDATION_PROTOCOL_SCHEMA,
        "target": "V_up",
        "split": "validation",
        "lock_sha256": lock["lock_sha256"],
        "train_protocol_sha256": lock["train_protocol_sha256"],
        "frozen_pi_up_actor_sha256": lock["frozen_pi_up_actor_sha256"],
        "frozen_pi_up_payload_sha256": lock["frozen_pi_up_payload_sha256"],
        "frozen_pi_up_config_sha256": lock["frozen_pi_up_config_sha256"],
        "xml_sha256": lock["xml_sha256"],
        "source_training_transitions": lock["source_training_transitions"],
        "negative_role": lock["negative_role"],
        "failure_reason": lock["failure_reason"],
        "selected_action_names": list(lock["selected_action_names"]),
        "selected_signs": list(lock["selected_signs"]),
        "strengths": list(strengths),
        "durations": list(durations),
        "protocol_seed": int(protocol_seed),
        "state_generation": lock["state_generation"],
        "retuning_allowed": False,
        "test_interaction_allowed": False,
        "training_transitions": 0,
    }
    protocol_hash = canonical_sha256(protocol)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    bank_name = "boundary_bank"
    bank_dir = output_dir / bank_name
    (bank_dir / "snapshots").mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(
        json.dumps({**protocol, "protocol_sha256": protocol_hash}, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    exclusions = {"terminal": 0, "apex": 0, "duplicate": 0}
    attempted = transitions = variant_index = 0
    for anchor in sorted(anchors, key=lambda item: item.sort_key):
        for duration in durations:
            for strength in strengths:
                for direction in directions:
                    attempted += 1
                    state = restore(anchor.snapshot)
                    nominal_actions, perturbed_actions, effective_deltas = [], [], []
                    rejected: str | None = None
                    for perturb_step in range(duration):
                        nominal = np.asarray(policy_action(state, variant_index, perturb_step), dtype=np.float32).reshape(-1)
                        if nominal.shape != (len(ACTION_ORDER),):
                            raise ValueError(f"policy action shape must be {(len(ACTION_ORDER),)}, got {nominal.shape}")
                        requested = nominal + np.asarray(direction["basis_vector"], dtype=np.float32) * np.float32(strength)
                        perturbed = np.clip(requested, -1.0, 1.0).astype(np.float32)
                        state = step(state, perturbed)
                        transitions += 1
                        nominal_actions.append(nominal.tolist())
                        perturbed_actions.append(perturbed.tolist())
                        effective_deltas.append((perturbed - nominal).tolist())
                        if _truth(getattr(state.info["events"], "apex_seen")):
                            rejected = "apex"
                            break
                        if _truth(state.info.get("terminated", False)) or _truth(state.info.get("truncated", False)) or _truth(state.info.get("timeout", False)):
                            rejected = "terminal"
                            break
                    current_variant, variant_index = variant_index, variant_index + 1
                    if rejected is not None:
                        exclusions[rejected] += 1
                        continue

                    snapshot = capture(state, anchor)
                    state_hash = physical_state_sha256(snapshot)
                    if state_hash in seen:
                        exclusions["duplicate"] += 1
                        continue
                    seen.add(state_hash)
                    relative = Path("snapshots") / f"candidate_{len(entries):06d}"
                    save_snapshot(bank_dir / relative, snapshot)
                    row = anchor.row
                    qpos, qvel = np.asarray(snapshot.qpos), np.asarray(snapshot.qvel)
                    entries.append(
                        {
                            "snapshot": str(relative),
                            "source_bank": bank_name,
                            "candidate_kind": "reachable_boundary_validation",
                            "parent_group_id": str(row["parent_group_id"]),
                            "parent_trajectory": str(row["parent_group_id"]),
                            "seed": int(row["seed"]),
                            "role": str(row["role"]),
                            "tick": int(snapshot.tick),
                            "state_sha256": state_hash,
                            "result_state_sha256": state_hash,
                            "anchor_kind": "failure_anchor",
                            "anchor_source_bank": str(row["source_bank"]),
                            "anchor_snapshot": str(row["snapshot"]),
                            "anchor_parent_group_id": str(row["parent_group_id"]),
                            "anchor_seed": int(row["seed"]),
                            "anchor_role": str(row["role"]),
                            "anchor_tick": int(row["tick"]),
                            "anchor_state_sha256": str(row.get("state_sha256", "")),
                            "anchor_source_checkpoint": str(row.get("source_checkpoint", "")),
                            "anchor_source_training_transitions": int(row.get("source_training_transitions", -1)),
                            "anchor_source_actor_sha256": str(row.get("source_actor_sha256", "")),
                            "anchor_split": "validation",
                            "perturbation": {
                                **direction,
                                "strength": float(strength),
                                "duration": int(duration),
                                "variant_index": current_variant,
                                "nominal_actions": nominal_actions,
                                "perturbed_actions": perturbed_actions,
                                "effective_deltas": effective_deltas,
                            },
                            "protocol_seed": int(protocol_seed),
                            "protocol_sha256": protocol_hash,
                            "lock_sha256": lock["lock_sha256"],
                            "z": float(qpos[2]),
                            "vx": float(qvel[0]),
                            "vz": float(qvel[2]),
                            "pitch": float(qpos[4]) if qpos.size > 4 else 0.0,
                            "pitch_rate": float(qvel[4]) if qvel.size > 4 else 0.0,
                        }
                    )

    report = {
        "schema": BOUNDARY_CATALOG_SCHEMA,
        "status": "completed",
        "target": "V_up",
        "selection": "validation_locked_reachable_boundary_action_basis_subset",
        "split": "validation",
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": protocol_hash,
        "training_transitions": 0,
        "test_interaction_count": 0,
        "anchor_count": len(anchors),
        "attempted_candidate_count": attempted,
        "candidate_count": len(entries),
        "collection_transitions": transitions,
        "exclusion_counts": exclusions,
        "source_bank": bank_name,
        "entries": entries,
    }
    (output_dir / "catalog.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({key: value for key, value in report.items() if key != "entries"}, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
