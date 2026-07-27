"""Schema, deduplication and stratified sampling for provisional Descent RSI.

These records are training proposals.  They are deliberately independent of
the legacy Descent matcher and never carry certified Tube/JEL semantics.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "descent_provisional_candidate_v1"
LABELS = ("provisional_core", "provisional_frontier", "rejected_physical")
LAYERS = ("early", "middle", "late")
SOURCES = ("natural_continuous", "local_rsi_perturbation")
FEATURE_NAMES = (
    "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz",
    "wx", "wy", "wz", "steering", "hip", "knee", "wheel_speed",
)
REQUIRED_POLICY_STATE = (
    "last_action", "obs_history", "actor_observation", "filter_phase",
    "phase_probs", "delay_buffer", "prev_acc_z", "prev_vz",
)


def state_bytes(record: Mapping[str, Any]) -> bytes:
    """Canonical complete reset-state bytes, excluding labels/provenance."""
    pieces = []
    for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
        value = np.ascontiguousarray(np.asarray(record[name], np.float32))
        pieces.extend((name.encode(), repr(value.shape).encode(), value.tobytes()))
    policy = record["policy_state"]
    for name in REQUIRED_POLICY_STATE:
        value = policy[name]
        if np.isscalar(value):
            value = np.asarray([value], np.float32)
        else:
            value = np.ascontiguousarray(np.asarray(value, np.float32))
        pieces.extend((name.encode(), repr(value.shape).encode(), value.tobytes()))
    return b"".join(pieces)


def state_identity(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(state_bytes(record)).hexdigest()


def validate_candidate(record: Mapping[str, Any]) -> None:
    if record.get("candidate_schema") != SCHEMA_VERSION:
        raise ValueError("invalid provisional Descent candidate schema")
    if record.get("provisional_label") not in LABELS:
        raise ValueError("invalid provisional label")
    if record.get("descent_layer") not in LAYERS:
        raise ValueError("invalid Descent layer")
    if record.get("candidate_source") not in SOURCES:
        raise ValueError("invalid candidate source")
    if record.get("artifact_role") != "proposal_support_bank":
        raise ValueError("provisional candidate must use proposal_support_bank role")
    for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "physical_feature"):
        if name not in record or not np.isfinite(np.asarray(record[name])).all():
            raise ValueError(f"missing or nonfinite {name}")
    policy = record.get("policy_state", {})
    missing = [name for name in REQUIRED_POLICY_STATE if name not in policy]
    if missing:
        raise ValueError(f"incomplete PolicyState: {missing}")
    if record.get("formal_tube_member") or record.get("formal_jel_member"):
        raise ValueError("provisional candidate cannot be a formal member")


def tolerance_unique(
    records: Sequence[Mapping[str, Any]], scale: Sequence[float], distance: float,
) -> tuple[list[dict[str, Any]], int]:
    """Greedy deterministic normalized physical-feature deduplication."""
    kept: list[dict[str, Any]] = []
    rejected = 0
    scale_array = np.maximum(np.asarray(scale, np.float64), 1e-8)
    for source in records:
        row = copy.deepcopy(dict(source))
        feature = np.asarray(row["physical_feature"], np.float64)
        if any(np.linalg.norm((feature - np.asarray(old["physical_feature"]))
                              / scale_array) < float(distance) for old in kept):
            rejected += 1
            continue
        kept.append(row)
    return kept, rejected


def greedy_clusters(
    records: Sequence[Mapping[str, Any]], scale: Sequence[float], radius: float,
) -> list[int]:
    """Stable online state-space clustering for compact coverage reporting."""
    centers: list[np.ndarray] = []
    assignments: list[int] = []
    scale_array = np.maximum(np.asarray(scale, np.float64), 1e-8)
    for row in records:
        value = np.asarray(row["physical_feature"], np.float64) / scale_array
        distances = [float(np.linalg.norm(value - center)) for center in centers]
        if not distances or min(distances) > float(radius):
            centers.append(value.copy())
            assignments.append(len(centers) - 1)
        else:
            assignments.append(int(np.argmin(distances)))
    return assignments


class StratifiedRSISampler:
    """Resumable host-side sampler balanced by label and Descent time layer."""

    def __init__(self, records: Sequence[Mapping[str, Any]], seed: int = 0):
        self.records = list(records)
        self.rng = np.random.default_rng(int(seed))
        self.draws = 0
        buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(self.records):
            if row.get("provisional_label") == "rejected_physical":
                continue
            buckets[(str(row["provisional_label"]), str(row["descent_layer"]))].append(index)
        self.buckets = {key: value for key, value in sorted(buckets.items()) if value}
        if not self.buckets:
            raise ValueError("RSI sampler has no eligible provisional candidates")

    def sample_indices(self, count: int) -> list[int]:
        keys = list(self.buckets)
        result = []
        for offset in range(int(count)):
            key = keys[(self.draws + offset) % len(keys)]
            result.append(int(self.rng.choice(self.buckets[key])))
        self.draws += int(count)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "descent_provisional_rsi_sampler_v1",
            "draws": self.draws,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "descent_provisional_rsi_sampler_v1":
            raise ValueError("invalid RSI sampler state")
        self.draws = int(state["draws"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    def state_sha256(self) -> str:
        payload = json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
