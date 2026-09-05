"""Train-only reachable boundary expansion for V_up continuation feasibility.

Boundary candidates are generated only by restoring real nominal snapshots and
stepping the real environment under bounded action-basis perturbations.  The
module never mutates qpos/qvel to synthesize states.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .constants import ACTION_ORDER
from .handoff_snapshot import HandoffSnapshot, load_snapshot, save_snapshot

BOUNDARY_CATALOG_SCHEMA = "jit_upstream_boundary_catalog_v1"
DEFAULT_BOUNDARY_STRENGTHS = (0.025, 0.05, 0.10)
DEFAULT_BOUNDARY_DURATIONS = (1, 2)
DEFAULT_NEAR_ATOL = 1.0e-5


def _truth(value: Any) -> bool:
    return bool(np.asarray(value))


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_state_sha256(snapshot: HandoffSnapshot) -> str:
    """Use the same qpos+qvel byte contract as the nominal BankCollector."""
    digest = hashlib.sha256()
    digest.update(np.asarray(snapshot.qpos).tobytes())
    digest.update(np.asarray(snapshot.qvel).tobytes())
    return digest.hexdigest()


def validate_strengths(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(x) for x in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("boundary strengths must be non-empty and unique")
    if any(not 0.0 < x <= 1.0 for x in result):
        raise ValueError("boundary strengths must lie in (0, 1]")
    return result


def validate_durations(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(x) for x in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("boundary durations must be non-empty and unique")
    if any(x <= 0 for x in result):
        raise ValueError("boundary durations must be positive")
    return result


def action_basis_directions(
    action_order: Sequence[str] = ACTION_ORDER,
    *,
    action_names: Sequence[str] | None = None,
    signs: Sequence[int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return deterministic one-hot perturbation directions, optionally filtered.

    Filtering is deliberately explicit so a TRAIN-only refinement can focus on a
    crossing direction discovered by a broader pilot without changing the
    default all-axis protocol.
    """
    order = tuple(str(name) for name in action_order)
    names = order if action_names is None else tuple(str(name) for name in action_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("action_names must be non-empty and unique")
    unknown = sorted(set(names).difference(order))
    if unknown:
        raise ValueError(f"unknown action_names: {unknown}")

    selected_signs = (-1, 1) if signs is None else tuple(int(sign) for sign in signs)
    if not selected_signs or len(set(selected_signs)) != len(selected_signs):
        raise ValueError("signs must be non-empty and unique")
    if any(sign not in (-1, 1) for sign in selected_signs):
        raise ValueError("signs must contain only -1 or +1")

    name_set = set(names)
    sign_set = set(selected_signs)
    result = []
    for dimension, name in enumerate(order):
        if name not in name_set:
            continue
        for sign in (-1, 1):
            if sign not in sign_set:
                continue
            vector = [0.0] * len(order)
            vector[dimension] = float(sign)
            result.append(
                {
                    "action_dimension": dimension,
                    "action_name": str(name),
                    "sign": sign,
                    "basis_vector": vector,
                }
            )
    return tuple(result)


def _catalog_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ValueError("catalog must contain an entries list")
    return payload


def _label_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("entries", payload.get("labels", []))
    if not isinstance(payload, list):
        raise ValueError("labels must be a list")
    return [dict(row) for row in payload]


def row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("source_bank", "")),
        str(row.get("parent_group_id", "")),
        int(row.get("seed", -1)),
        str(row.get("role", "")),
        int(row.get("tick", -1)),
        str(row.get("state_sha256", "")),
    )


def snapshot_path(catalog: Path, row: Mapping[str, Any]) -> Path:
    return Path(catalog).parent / str(row["source_bank"]) / str(row["snapshot"])


def label_reasons(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted({str(branch.get("reason", "")) for branch in row.get("branches", [])})
    )


def snapshot_distance(a: HandoffSnapshot, b: HandoffSnapshot) -> dict[str, float | bool]:
    def linf(left: Any, right: Any) -> float:
        x, y = np.asarray(left, np.float64), np.asarray(right, np.float64)
        if x.shape != y.shape:
            return float("inf")
        return float(np.max(np.abs(x - y))) if x.size else 0.0

    return {
        "qpos_linf": linf(a.qpos, b.qpos),
        "qvel_linf": linf(a.qvel, b.qvel),
        "observation_linf": linf(a.observation, b.observation),
        "exact": bool(
            np.array_equal(a.qpos, b.qpos)
            and np.array_equal(a.qvel, b.qvel)
            and np.array_equal(a.observation, b.observation)
        ),
    }


def are_near_duplicates(
    a: HandoffSnapshot,
    b: HandoffSnapshot,
    *,
    qpos_atol: float = DEFAULT_NEAR_ATOL,
    qvel_atol: float = DEFAULT_NEAR_ATOL,
    observation_atol: float = DEFAULT_NEAR_ATOL,
) -> bool:
    d = snapshot_distance(a, b)
    return bool(
        d["qpos_linf"] <= qpos_atol
        and d["qvel_linf"] <= qvel_atol
        and d["observation_linf"] <= observation_atol
    )


@dataclass(frozen=True)
class AuditedAnchor:
    row: dict[str, Any]
    label: dict[str, Any]
    snapshot: HandoffSnapshot
    anchor_kind: str

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (
            int(self.row["seed"]),
            str(self.row["parent_group_id"]),
            int(self.row["tick"]),
            str(self.row.get("state_sha256", "")),
        )


def _cluster_near_duplicates(
    anchors: Sequence[AuditedAnchor],
    *,
    qpos_atol: float,
    qvel_atol: float,
    observation_atol: float,
) -> list[list[AuditedAnchor]]:
    clusters: list[list[AuditedAnchor]] = []
    for anchor in sorted(anchors, key=lambda x: x.sort_key):
        for cluster in clusters:
            if are_near_duplicates(
                anchor.snapshot,
                cluster[0].snapshot,
                qpos_atol=qpos_atol,
                qvel_atol=qvel_atol,
                observation_atol=observation_atol,
            ):
                cluster.append(anchor)
                break
        else:
            clusters.append([anchor])
    return clusters


def audit_and_select_train_anchors(
    catalog: Path,
    labels: Path,
    *,
    train_seeds: Sequence[int],
    source_training_transitions: int,
    negative_role: str = "ascending_entry",
    failure_reason: str = "pitch_limit",
    guard_roles: Sequence[str] = ("jump_zone_entry", "height_entry"),
    max_negative_anchors: int = 1,
    qpos_atol: float = DEFAULT_NEAR_ATOL,
    qvel_atol: float = DEFAULT_NEAR_ATOL,
    observation_atol: float = DEFAULT_NEAR_ATOL,
) -> tuple[list[AuditedAnchor], dict[str, Any]]:
    """Audit only TRAIN rows, cluster pseudo-diverse failures, choose representatives."""
    if max_negative_anchors <= 0:
        raise ValueError("max_negative_anchors must be positive")
    if min(qpos_atol, qvel_atol, observation_atol) < 0.0:
        raise ValueError("near-duplicate tolerances must be nonnegative")
    train = {int(seed) for seed in train_seeds}
    if not train:
        raise ValueError("train_seeds must be non-empty")

    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for label in _label_rows(labels):
        identity = row_identity(label)
        if identity in index:
            raise ValueError(f"duplicate continuation label identity: {identity}")
        index[identity] = label

    joined: list[AuditedAnchor] = []
    for raw in _catalog_payload(catalog)["entries"]:
        row = dict(raw)
        if int(row.get("source_training_transitions", -1)) != int(source_training_transitions):
            continue
        label = index.get(row_identity(row))
        if label is None:
            continue
        seed = int(row["seed"])
        if str(label.get("split")) == "train" and seed not in train:
            raise ValueError("train-labeled row uses a seed outside train_seeds")
        if seed not in train or str(label.get("split")) != "train":
            continue
        joined.append(
            AuditedAnchor(
                row,
                label,
                load_snapshot(snapshot_path(catalog, row)),
                "nominal",
            )
        )

    negatives = [
        x
        for x in joined
        if str(x.row.get("role")) == negative_role
        and int(x.label.get("success_count", 0)) == 0
        and failure_reason in label_reasons(x.label)
    ]
    if not negatives:
        raise ValueError("no matching TRAIN failure anchors found")
    clusters = _cluster_near_duplicates(
        negatives,
        qpos_atol=qpos_atol,
        qvel_atol=qvel_atol,
        observation_atol=observation_atol,
    )
    representatives = [cluster[0] for cluster in clusters][:max_negative_anchors]
    selected = [
        AuditedAnchor(x.row, x.label, x.snapshot, "failure_anchor")
        for x in representatives
    ]

    neighbor_outcomes = []
    for failure_anchor in tuple(selected):
        for role in guard_roles:
            matches = sorted(
                (
                    x
                    for x in joined
                    if x.row["parent_group_id"] == failure_anchor.row["parent_group_id"]
                    and x.row["role"] == role
                ),
                key=lambda x: x.sort_key,
            )
            if not matches:
                neighbor_outcomes.append(
                    {
                        "parent_group_id": failure_anchor.row["parent_group_id"],
                        "role": str(role),
                        "present": False,
                    }
                )
                continue
            item = matches[0]
            positive = int(item.label.get("success_count", 0)) == int(
                item.label.get("branch_count", 1)
            )
            neighbor_outcomes.append(
                {
                    "parent_group_id": item.row["parent_group_id"],
                    "seed": int(item.row["seed"]),
                    "role": item.row["role"],
                    "present": True,
                    "positive": positive,
                    "reasons": list(label_reasons(item.label)),
                    "state_sha256": item.row.get("state_sha256"),
                }
            )
            if positive:
                selected.append(
                    AuditedAnchor(item.row, item.label, item.snapshot, "positive_guard")
                )

    ordered = sorted(negatives, key=lambda x: x.sort_key)
    pairwise = []
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            pairwise.append(
                {
                    "left_seed": int(left.row["seed"]),
                    "right_seed": int(right.row["seed"]),
                    "left_state_sha256": left.row.get("state_sha256"),
                    "right_state_sha256": right.row.get("state_sha256"),
                    **snapshot_distance(left.snapshot, right.snapshot),
                    "near_duplicate": are_near_duplicates(
                        left.snapshot,
                        right.snapshot,
                        qpos_atol=qpos_atol,
                        qvel_atol=qvel_atol,
                        observation_atol=observation_atol,
                    ),
                }
            )

    audit = {
        "schema": "jit_upstream_boundary_audit_v1",
        "split": "train",
        "source_training_transitions": int(source_training_transitions),
        "negative_role": negative_role,
        "failure_reason": failure_reason,
        "train_seeds": sorted(train),
        "matching_negative_count": len(negatives),
        "near_duplicate_tolerances": {
            "qpos_atol": qpos_atol,
            "qvel_atol": qvel_atol,
            "observation_atol": observation_atol,
        },
        "near_duplicate_cluster_count": len(clusters),
        "near_duplicate_clusters": [
            {
                "representative_seed": int(c[0].row["seed"]),
                "representative_state_sha256": c[0].row.get("state_sha256"),
                "member_seeds": [int(x.row["seed"]) for x in c],
                "member_state_sha256": [x.row.get("state_sha256") for x in c],
            }
            for c in clusters
        ],
        "pairwise_distances": pairwise,
        "selected_failure_anchors": [
            {
                "seed": int(x.row["seed"]),
                "parent_group_id": x.row["parent_group_id"],
                "role": x.row["role"],
                "tick": int(x.row["tick"]),
                "state_sha256": x.row.get("state_sha256"),
                "reasons": list(label_reasons(x.label)),
            }
            for x in selected
            if x.anchor_kind == "failure_anchor"
        ],
        "neighbor_outcomes": neighbor_outcomes,
        "selected_anchor_count": len(selected),
        "selected_anchor_kinds": [x.anchor_kind for x in selected],
    }
    return selected, audit


def collect_reachable_boundary_candidates(
    anchors: Sequence[AuditedAnchor],
    output_dir: Path,
    *,
    restore: Callable[[HandoffSnapshot], Any],
    policy_action: Callable[[Any, int, int], Any],
    step: Callable[[Any, Any], Any],
    capture: Callable[[Any, AuditedAnchor], HandoffSnapshot],
    protocol: Mapping[str, Any],
    strengths: Sequence[float] = DEFAULT_BOUNDARY_STRENGTHS,
    durations: Sequence[int] = DEFAULT_BOUNDARY_DURATIONS,
    action_names: Sequence[str] | None = None,
    signs: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Generate dynamically reachable candidates; terminal/Apex states are rejected."""
    strengths = validate_strengths(strengths)
    durations = validate_durations(durations)
    directions = action_basis_directions(action_names=action_names, signs=signs)
    if not anchors:
        raise ValueError("boundary candidate collection requires anchors")
    if any(str(x.label.get("split")) != "train" for x in anchors):
        raise ValueError("boundary pilot accepts TRAIN anchors only")

    selected_action_names = list(dict.fromkeys(row["action_name"] for row in directions))
    selected_signs = sorted({int(row["sign"]) for row in directions})
    protocol_payload = {
        **dict(protocol),
        "schema": "jit_upstream_boundary_protocol_v1",
        "split": "train",
        "action_order": list(ACTION_ORDER),
        "direction_family": "action_basis" if len(directions) == 2 * len(ACTION_ORDER) else "action_basis_subset",
        "selected_action_names": selected_action_names,
        "selected_signs": selected_signs,
        "strengths": list(strengths),
        "durations": list(durations),
        "training_transitions": 0,
    }
    protocol_hash = canonical_sha256(protocol_payload)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    bank_name = "boundary_bank"
    bank_dir = output_dir / bank_name
    (bank_dir / "snapshots").mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {**protocol_payload, "protocol_sha256": protocol_hash},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    entries, seen = [], set()
    exclusions = {"terminal": 0, "apex": 0, "duplicate": 0}
    attempted = transitions = variant_index = 0
    for anchor in sorted(anchors, key=lambda x: (x.anchor_kind, x.sort_key)):
        for duration in durations:
            for strength in strengths:
                for direction in directions:
                    attempted += 1
                    state = restore(anchor.snapshot)
                    nominal_actions, perturbed_actions, effective_deltas = [], [], []
                    rejected = None
                    for perturb_step in range(duration):
                        nominal = np.asarray(
                            policy_action(state, variant_index, perturb_step), np.float32
                        ).reshape(-1)
                        if nominal.shape != (len(ACTION_ORDER),):
                            raise ValueError(
                                f"policy action shape must be {(len(ACTION_ORDER),)}, got {nominal.shape}"
                            )
                        requested = nominal + np.asarray(
                            direction["basis_vector"], np.float32
                        ) * np.float32(strength)
                        perturbed = np.clip(requested, -1.0, 1.0).astype(np.float32)
                        state = step(state, perturbed)
                        transitions += 1
                        nominal_actions.append(nominal.tolist())
                        perturbed_actions.append(perturbed.tolist())
                        effective_deltas.append((perturbed - nominal).tolist())
                        if _truth(getattr(state.info["events"], "apex_seen")):
                            rejected = "apex"
                            break
                        if (
                            _truth(state.info.get("terminated", False))
                            or _truth(state.info.get("truncated", False))
                            or _truth(state.info.get("timeout", False))
                        ):
                            rejected = "terminal"
                            break
                    current_variant, variant_index = variant_index, variant_index + 1
                    if rejected:
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
                            "candidate_kind": "reachable_boundary",
                            "parent_group_id": str(row["parent_group_id"]),
                            "parent_trajectory": str(row["parent_group_id"]),
                            "seed": int(row["seed"]),
                            "role": str(row["role"]),
                            "tick": int(snapshot.tick),
                            "state_sha256": state_hash,
                            "result_state_sha256": state_hash,
                            "anchor_kind": anchor.anchor_kind,
                            "anchor_source_bank": str(row["source_bank"]),
                            "anchor_snapshot": str(row["snapshot"]),
                            "anchor_parent_group_id": str(row["parent_group_id"]),
                            "anchor_seed": int(row["seed"]),
                            "anchor_role": str(row["role"]),
                            "anchor_tick": int(row["tick"]),
                            "anchor_state_sha256": str(row.get("state_sha256", "")),
                            "anchor_source_checkpoint": str(row.get("source_checkpoint", "")),
                            "anchor_source_training_transitions": int(
                                row.get("source_training_transitions", -1)
                            ),
                            "anchor_source_actor_sha256": str(
                                row.get("source_actor_sha256", "")
                            ),
                            "anchor_split": "train",
                            "perturbation": {
                                **direction,
                                "strength": float(strength),
                                "duration": int(duration),
                                "variant_index": current_variant,
                                "nominal_actions": nominal_actions,
                                "perturbed_actions": perturbed_actions,
                                "effective_deltas": effective_deltas,
                            },
                            "protocol_seed": int(
                                protocol_payload.get("protocol_seed", 0)
                            ),
                            "protocol_sha256": protocol_hash,
                            "z": float(qpos[2]),
                            "vx": float(qvel[0]),
                            "vz": float(qvel[2]),
                            "pitch": float(qpos[4]) if qpos.size > 4 else 0.0,
                            "pitch_rate": float(qvel[4]) if qvel.size > 4 else 0.0,
                        }
                    )

    selection = (
        "train_unique_reachable_boundary_action_basis"
        if len(directions) == 2 * len(ACTION_ORDER)
        else "train_unique_reachable_boundary_action_basis_subset"
    )
    report = {
        "schema": BOUNDARY_CATALOG_SCHEMA,
        "status": "completed",
        "target": "V_up",
        "selection": selection,
        "split": "train",
        "protocol_sha256": protocol_hash,
        "training_transitions": 0,
        "anchor_count": len(anchors),
        "attempted_candidate_count": attempted,
        "candidate_count": len(entries),
        "collection_transitions": transitions,
        "exclusion_counts": exclusions,
        "source_bank": bank_name,
        "entries": entries,
    }
    (output_dir / "catalog.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "entries"}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
