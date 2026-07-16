"""Bounded, parent-aware local support expansion for pre-Tube descent."""
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from .bank import SnapshotBank
from .config import file_sha256
from .flight_augmentation import tangent_difference


BOOTSTRAP_GROUPS=("provisional_safe","boundary","successful_anchor")


def robust_scale(values, floor=1e-6):
    x=np.asarray(values,np.float64); center=np.median(x,axis=0); mad=1.4826*np.median(np.abs(x-center),axis=0); std=x.std(axis=0)
    return center,np.maximum(np.maximum(mad,.25*std),float(floor))


def tangent_factor(rows: Sequence[Mapping], allowed: np.ndarray) -> np.ndarray:
    by_parent={}
    for row in rows: by_parent.setdefault(str(row.get("entry_source_id",row["id"])),[]).append(row)
    differences=[]
    for group in by_parent.values():
        ordered=sorted(group,key=lambda row:(int(row.get("proposal_step",0)),str(row["id"])))
        differences.extend(tangent_difference(a,b) for a,b in zip(ordered[:-1],ordered[1:]))
    if len(differences)<2:
        raise ValueError("Local descent covariance needs at least two temporal differences")
    x=np.asarray(differences,np.float64); x[:,~np.asarray(allowed,bool)]=0.0; covariance=np.cov(x,rowvar=False)
    values,vectors=np.linalg.eigh(covariance); keep=values>max(float(values.max())*1e-10,1e-14)
    if not np.any(keep): raise ValueError("Local descent covariance is degenerate")
    return vectors[:,keep]*np.sqrt(values[keep])[None,:]


def difficulty_layers(rows: Sequence[Mapping], distances: Mapping[str,float]) -> dict[str,str]:
    features=np.asarray([row["physical_feature"] for row in rows],np.float64); center,scale=robust_scale(features,1e-4)
    scored=[]; max_step=max(int(row.get("proposal_step",0)) for row in rows)
    for row in rows:
        f=np.asarray(row["physical_feature"],np.float64); stability=np.mean(np.abs((f[[3,4,6,8,9,10,13,14]]-center[[3,4,6,8,9,10,13,14]])/scale[[3,4,6,8,9,10,13,14]]))
        score=float(distances[row["id"]])+0.25*stability+0.20*(max_step-int(row.get("proposal_step",0)))
        scored.append((score,row["id"]))
    scored.sort(); n=len(scored); result={}
    for rank,(_,identifier) in enumerate(scored): result[identifier]="late" if rank<n/3 else "middle" if rank<2*n/3 else "early"
    return result


def balanced_parent(rows, children: Counter, rng, cap):
    eligible=[row for row in rows if children[row["id"]]<int(cap)]
    if not eligible: return None
    source_min=min(sum(children[row["id"]] for row in eligible if row.get("entry_source_id",row["id"])==source) for source in {row.get("entry_source_id",row["id"]) for row in eligible})
    sources=sorted({row.get("entry_source_id",row["id"]) for row in eligible if sum(children[x["id"]] for x in eligible if x.get("entry_source_id",x["id"])==row.get("entry_source_id",row["id"]))==source_min})
    source=sources[int(rng.integers(len(sources)))]; subset=[row for row in eligible if row.get("entry_source_id",row["id"])==source]; minimum=min(children[row["id"]] for row in subset); subset=[row for row in subset if children[row["id"]]==minimum]
    return subset[int(rng.integers(len(subset)))]


def build_candidate_bootstrap_bank(
    bank: SnapshotBank, bank_path: str, cfg: Mapping
) -> tuple[SnapshotBank, dict]:
    """Build the auditable pre-Tube reset distribution.

    Mass is assigned to the three declared evidence groups, then equally to
    distinct parent snapshots, then equally to records belonging to a parent.
    Thus local children add coverage but cannot multiply a parent's reset mass.
    """
    rows = [
        row for row in bank.records_for_phase("flight", include_training_only=False)
        if row.get("local_bootstrap_eligible")
    ]
    masses = {
        "provisional_safe": float(cfg.descent_local_reset_safe_mass),
        "boundary": float(cfg.descent_local_reset_boundary_mass),
        "successful_anchor": float(cfg.descent_local_reset_anchor_mass),
    }
    if set(row.get("bootstrap_group") for row in rows) - set(masses):
        raise ValueError("Eligible descent records contain an undeclared bootstrap group")
    if any(value <= 0 for value in masses.values()) or not np.isclose(sum(masses.values()), 1.0):
        raise ValueError("Descent bootstrap group masses must be positive and sum to one")
    source_hash = file_sha256(bank_path)
    output, parent_weights = [], {}
    for group, group_mass in masses.items():
        group_rows = [row for row in rows if row["bootstrap_group"] == group]
        by_parent = {}
        for row in group_rows:
            parent = str(row.get("parent_candidate_id", row["id"]))
            by_parent.setdefault(parent, []).append(row)
        if not by_parent:
            raise ValueError(f"Descent bootstrap group {group} is empty")
        parent_mass = group_mass / len(by_parent)
        for parent, parent_rows in sorted(by_parent.items()):
            parent_weights[f"{group}:{parent}"] = parent_mass
            for row in parent_rows:
                item = copy.deepcopy(row)
                item.setdefault("origin_phase", item.get("source_phase", "flight"))
                item.update({
                    "reset_source": "flight_curriculum",
                    "reset_weight": parent_mass / len(parent_rows),
                    "reset_parent_id": parent,
                    "original_bank_path": str(bank_path),
                    "original_bank_sha256": source_hash,
                })
                output.append(item)
    metadata = copy.deepcopy(bank.metadata)
    metadata["reset_source_protocol"] = {
        "version": 1,
        "mode": "candidate_guided_local_bootstrap",
        "source_bank_sha256": source_hash,
        "group_masses": masses,
        "parent_balanced": True,
    }
    actual_group = {
        group: float(sum(row["reset_weight"] for row in output if row["bootstrap_group"] == group))
        for group in masses
    }
    actual_layer = {
        layer: float(sum(row["reset_weight"] for row in output if row["descent_layer"] == layer))
        for layer in ("late", "middle", "early")
    }
    report = {
        "records": len(output),
        "group_counts": dict(Counter(row["bootstrap_group"] for row in output)),
        "layer_counts": dict(Counter(row["descent_layer"] for row in output)),
        "expected_group_reset_ratio": actual_group,
        "expected_layer_reset_ratio": actual_layer,
        "parent_count": len(parent_weights),
        "parent_reset_weights": parent_weights,
        "maximum_parent_reset_weight": max(parent_weights.values()),
        "source_bank_sha256": source_hash,
    }
    return SnapshotBank(output, metadata), report
