"""Constrained, label-free local diversification of valid Flight anchors."""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import numpy as np


REGIONS = ("ascent", "apex", "descent")


def proportional_quotas(counts: Mapping[str, int], target: int) -> dict[str, int]:
    """Largest-remainder allocation with deterministic region ordering."""
    total = sum(int(counts[name]) for name in REGIONS)
    if total <= 0 or target < len(REGIONS):
        raise ValueError("Flight quota inputs are invalid")
    raw = {name: target * int(counts[name]) / total for name in REGIONS}
    quota = {name: int(np.floor(raw[name])) for name in REGIONS}
    for name in sorted(REGIONS, key=lambda key: (-(raw[key] - quota[key]), REGIONS.index(key))):
        if sum(quota.values()) >= target:
            break
        quota[name] += 1
    return quota


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("Invalid quaternion")
    return q / norm


def quat_multiply(a, b) -> np.ndarray:
    aw, ax, ay, az = quat_normalize(a); bw, bx, by, bz = quat_normalize(b)
    return quat_normalize(np.array([
        aw*bw-ax*bx-ay*by-az*bz,
        aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,
        aw*bz+ax*by-ay*bx+az*bw,
    ]))


def quat_slerp(a, b, fraction: float) -> np.ndarray:
    a = quat_normalize(a); b = quat_normalize(b); dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1.0, 1.0)); t = float(fraction)
    if dot > 0.9995:
        return quat_normalize((1.0-t)*a+t*b)
    angle = np.arccos(dot); sine = np.sin(angle)
    return quat_normalize(np.sin((1.0-t)*angle)/sine*a + np.sin(t*angle)/sine*b)


def quat_log(q) -> np.ndarray:
    q = quat_normalize(q)
    if q[0] < 0: q = -q
    length = float(np.linalg.norm(q[1:]))
    if length < 1e-12: return np.zeros(3)
    return q[1:] / length * (2.0*np.arctan2(length, float(q[0])))


def quat_exp(vector) -> np.ndarray:
    vector = np.asarray(vector, np.float64); angle = float(np.linalg.norm(vector))
    if angle < 1e-12: return np.array([1.0, 0.0, 0.0, 0.0])
    axis = vector/angle
    return np.r_[np.cos(angle/2.0), axis*np.sin(angle/2.0)]


def interpolate_state(parent: Mapping, neighbor: Mapping, fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q0=np.asarray(parent["qpos"],np.float64); q1=np.asarray(neighbor["qpos"],np.float64)
    v0=np.asarray(parent["qvel"],np.float64); v1=np.asarray(neighbor["qvel"],np.float64)
    c0=np.asarray(parent["ctrl"],np.float64); c1=np.asarray(neighbor["ctrl"],np.float64)
    t=float(fraction); q=(1-t)*q0+t*q1
    q[3:7]=quat_slerp(q0[3:7],q1[3:7],t)
    return q,(1-t)*v0+t*v1,(1-t)*c0+t*c1


def tangent_difference(a: Mapping, b: Mapping) -> np.ndarray:
    qa=np.asarray(a["qpos"],np.float64); qb=np.asarray(b["qpos"],np.float64)
    relative=quat_multiply(qb[3:7],np.r_[qa[3],-qa[4:7]])
    return np.r_[qb[:3]-qa[:3],quat_log(relative),qb[7:]-qa[7:],np.asarray(b["qvel"],np.float64)-np.asarray(a["qvel"],np.float64)]


def apply_tangent(qpos, qvel, delta) -> tuple[np.ndarray, np.ndarray]:
    qpos=np.asarray(qpos,np.float64).copy(); qvel=np.asarray(qvel,np.float64).copy(); delta=np.asarray(delta,np.float64)
    scalar_count=len(qpos)-7; expected=3+3+scalar_count+len(qvel)
    if len(delta)!=expected: raise ValueError(f"Expected tangent size {expected}, got {len(delta)}")
    qpos[:3]+=delta[:3]
    qpos[3:7]=quat_multiply(quat_exp(delta[3:6]),qpos[3:7])
    qpos[7:]+=delta[6:6+scalar_count]
    qvel+=delta[6+scalar_count:]
    return qpos,qvel


def covariance_factor(rows: Sequence[Mapping]) -> np.ndarray:
    ordered=sorted(rows,key=lambda row:int(row["source_index"]))
    differences=np.asarray([tangent_difference(a,b) for a,b in zip(ordered[:-1],ordered[1:])],np.float64)
    if len(differences)<2: raise ValueError("At least three anchors are required for local covariance")
    covariance=np.cov(differences,rowvar=False)
    values,vectors=np.linalg.eigh(covariance)
    keep=values>max(float(values.max())*1e-10,1e-14)
    return vectors[:,keep]*np.sqrt(values[keep])[None,:]


def normalized_distance(feature, existing, scale) -> float:
    feature=np.asarray(feature,np.float64); existing=np.asarray(existing,np.float64); scale=np.asarray(scale,np.float64)
    if not len(existing): return float("inf")
    return float(np.min(np.linalg.norm((existing-feature)/scale,axis=1)))


def feature_scale(rows: Sequence[Mapping]) -> np.ndarray:
    features=np.asarray([row["physical_feature"] for row in rows],np.float64)
    std=np.std(features,axis=0)
    ordered=sorted(rows,key=lambda row:(str(row.get("flight_subinterval")),int(row["source_index"])))
    local=np.asarray([np.abs(np.asarray(b["physical_feature"])-np.asarray(a["physical_feature"])) for a,b in zip(ordered[:-1],ordered[1:]) if a.get("flight_subinterval")==b.get("flight_subinterval")])
    local_scale=np.median(local,axis=0) if len(local) else np.zeros_like(std)
    positive=np.r_[std[std>1e-8],local_scale[local_scale>1e-8]]
    floor=float(np.median(positive))*1e-3 if len(positive) else 1e-6
    return np.maximum(np.maximum(std,local_scale),floor)


def clone_as_anchors(records: Sequence[Mapping]) -> list[dict]:
    result=copy.deepcopy(list(records))
    for row in result:
        row["candidate_kind"]="reference_anchor"
        row["source_index"]=int(row.get("source_index",row["reference_index"]))
    return result
