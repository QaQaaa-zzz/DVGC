"""Bounded, parent-aware local support expansion for pre-Tube descent."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

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
