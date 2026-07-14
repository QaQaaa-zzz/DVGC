"""Versioned Physical-Belief snapshot bank and dual Beta certification."""
from __future__ import annotations

import copy
import math
import os
import pickle
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_dist

from .config import PHASES, SNAPSHOT_SCHEMA

BANK_VERSION = 5
LABELS = ("safe", "boundary", "dead", "unknown")


def beta_posterior(
    successes: int,
    failures: int,
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict[str, float]:
    a = float(alpha0 + int(successes))
    b = float(beta0 + int(failures))
    lo = float(beta_dist.ppf(q_low, a, b))
    hi = float(beta_dist.ppf(q_high, a, b))
    return {
        "alpha": a,
        "beta": b,
        "mean": a / (a + b),
        "lower": lo,
        "upper": hi,
        "width": hi - lo,
        "q_low": float(q_low),
        "q_high": float(q_high),
    }


def posterior_label(
    posterior: Mapping[str, float],
    branches: int,
    *,
    min_branches: int,
    safe_threshold: float,
    dead_threshold: float,
    boundary_max_width: float,
) -> str:
    if int(branches) < int(min_branches):
        return "unknown"
    mean = float(posterior["mean"])
    if float(posterior["lower"]) >= float(safe_threshold):
        return "safe"
    if float(posterior["upper"]) < float(dead_threshold):
        return "dead"
    if (
        float(dead_threshold) <= mean <= float(safe_threshold)
        and float(posterior["width"]) <= float(boundary_max_width)
    ):
        return "boundary"
    return "unknown"


def empty_outcome() -> dict[str, Any]:
    p = beta_posterior(0, 0)
    return {"successes": 0, "failures": 0, "branches": 0, "posterior": p, "label": "unknown"}


def _as_float32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


class SnapshotBank:
    """Atomic, policy-versioned bank.

    Every state stores both the physical simulator state and PolicyState.  Chain
    and Final Recovery Bernoulli statistics are independent and are never
    merged across policy/estimator versions.
    """

    def __init__(self, records: Sequence[dict[str, Any]] | None = None, metadata: Mapping[str, Any] | None = None):
        self.records = [self._normalize_record(r) for r in (records or [])]
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("snapshot_schema", SNAPSHOT_SCHEMA)
        self.metadata.setdefault("created_at", time.time())

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotBank":
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or int(payload.get("version", -1)) != BANK_VERSION:
            raise ValueError(f"Incompatible clean DVGC bank: {path}")
        return cls(payload.get("records", []), payload.get("metadata", {}))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(
                    {"version": BANK_VERSION, "metadata": self.metadata, "records": self.records},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
        r = copy.deepcopy(dict(record))
        r.setdefault("id", uuid.uuid4().hex)
        phase = str(r.get("source_phase", r.get("source_stage", "landing"))).lower()
        if phase not in PHASES:
            raise ValueError(f"Invalid source phase: {phase}")
        r["source_phase"] = phase
        for key in ("qpos", "qvel", "ctrl", "physical_feature"):
            if key not in r and key == "physical_feature" and "feature" in r:
                r[key] = r["feature"]
            if key not in r:
                raise KeyError(f"Snapshot missing {key}")
            r[key] = _as_float32(r[key])
        r.setdefault("policy_state", {})
        ps = r["policy_state"]
        for key in ("last_action", "obs_history", "actor_observation", "phase_probs", "contact_probs", "estimator_hidden", "delay_buffer"):
            if key in ps:
                ps[key] = _as_float32(ps[key])
        r.setdefault("training_only", False)
        r.setdefault("bootstrap_eligible", True)
        r.setdefault("candidate_kind", "physical")
        r.setdefault("chain", empty_outcome())
        r.setdefault("final", empty_outcome())
        r.setdefault("policy_version", None)
        r.setdefault("estimator_version", None)
        r.setdefault("tube_version", None)
        r.setdefault("certification_branches", [])
        r.setdefault("label_timestamp", None)
        r.setdefault("label_age", 0)
        r.setdefault("reset_use_count", 0)
        r.setdefault("connection_flag", False)
        r.setdefault("support_score", None)
        return r

    def add(self, record: Mapping[str, Any], *, deduplicate: bool = True, distance: float = 0.12) -> bool:
        row = self._normalize_record(record)
        if deduplicate:
            for old in self.records_for_phase(row["source_phase"]):
                if np.linalg.norm(old["physical_feature"] - row["physical_feature"]) < float(distance):
                    return False
        self.records.append(row)
        return True

    def get(self, record_id: str) -> dict[str, Any]:
        for r in self.records:
            if r["id"] == record_id:
                return r
        raise KeyError(record_id)

    def records_for_phase(
        self,
        phase: str,
        *,
        final_labels: Sequence[str] | None = None,
        chain_labels: Sequence[str] | None = None,
        include_training_only: bool = True,
    ) -> list[dict[str, Any]]:
        rows = [r for r in self.records if r["source_phase"] == str(phase).lower()]
        if final_labels is not None:
            allowed = set(final_labels)
            rows = [r for r in rows if r["final"]["label"] in allowed]
        if chain_labels is not None:
            allowed = set(chain_labels)
            rows = [r for r in rows if r["chain"]["label"] in allowed]
        if not include_training_only:
            rows = [r for r in rows if not bool(r.get("training_only", False))]
        return rows

    def update_certification(
        self,
        record_id: str,
        *,
        chain_successes: int,
        chain_failures: int,
        final_successes: int,
        final_failures: int,
        policy_version: str,
        estimator_version: str,
        tube_version: str,
        protocol: Mapping[str, Any],
        seed_namespace: str,
        branch_evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        r = self.get(record_id)
        evidence = copy.deepcopy([dict(row) for row in branch_evidence])
        expected_branches = int(chain_successes) + int(chain_failures)
        if expected_branches != int(final_successes) + int(final_failures):
            raise ValueError("Chain and Final certification must use the same branches")
        if len(evidence) != expected_branches:
            raise ValueError(
                f"Expected {expected_branches} branch evidence rows, got {len(evidence)}"
            )
        evidence_chain = sum(bool(row.get("chain_success")) for row in evidence)
        evidence_final = sum(bool(row.get("final_recovery")) for row in evidence)
        if evidence_chain != int(chain_successes) or evidence_final != int(final_successes):
            raise ValueError("Branch evidence does not match Chain/Final success counts")
        if any(str(row.get("seed_namespace")) != str(seed_namespace) for row in evidence):
            raise ValueError("Branch evidence seed namespace does not match certification")
        old_version = (r.get("policy_version"), r.get("estimator_version"))
        new_version = (str(policy_version), str(estimator_version))
        if old_version != (None, None) and old_version != new_version:
            r.setdefault("certification_history", []).append(
                {
                    "policy_version": old_version[0],
                    "estimator_version": old_version[1],
                    "chain": copy.deepcopy(r["chain"]),
                    "final": copy.deepcopy(r["final"]),
                    "certification_branches": copy.deepcopy(r["certification_branches"]),
                    "label_timestamp": r.get("label_timestamp"),
                }
            )
        def outcome(s: int, f: int) -> dict[str, Any]:
            p = beta_posterior(
                s,
                f,
                alpha0=float(protocol["alpha0"]),
                beta0=float(protocol["beta0"]),
                q_low=float(protocol["q_low"]),
                q_high=float(protocol["q_high"]),
            )
            n = int(s) + int(f)
            return {
                "successes": int(s),
                "failures": int(f),
                "branches": n,
                "posterior": p,
                "label": posterior_label(
                    p,
                    n,
                    min_branches=int(protocol["min_branches"]),
                    safe_threshold=float(protocol["safe_threshold"]),
                    dead_threshold=float(protocol["dead_threshold"]),
                    boundary_max_width=float(protocol["boundary_max_width"]),
                ),
            }
        r["chain"] = outcome(chain_successes, chain_failures)
        r["final"] = outcome(final_successes, final_failures)
        r["policy_version"] = str(policy_version)
        r["estimator_version"] = str(estimator_version)
        r["tube_version"] = str(tube_version)
        r["certification_branches"] = evidence
        r["label_timestamp"] = time.time()
        r["label_age"] = 0
        r["seed_namespace"] = str(seed_namespace)
        r["connection_flag"] = r["chain"]["label"] == "safe"
        return r

    def invalidate_phase(self, phase: str, reason: str) -> int:
        count = 0
        for r in self.records_for_phase(phase):
            if r["chain"]["branches"] or r["final"]["branches"]:
                r.setdefault("certification_history", []).append(
                    {
                        "policy_version": r.get("policy_version"),
                        "estimator_version": r.get("estimator_version"),
                        "chain": copy.deepcopy(r["chain"]),
                        "final": copy.deepcopy(r["final"]),
                        "certification_branches": copy.deepcopy(r["certification_branches"]),
                        "reason": reason,
                    }
                )
            r["chain"] = empty_outcome()
            r["final"] = empty_outcome()
            r["policy_version"] = None
            r["estimator_version"] = None
            r["tube_version"] = None
            r["certification_branches"] = []
            r["label_age"] = 0
            count += 1
        return count

    def age_labels(self) -> None:
        for r in self.records:
            if r["final"]["branches"] > 0:
                r["label_age"] = int(r.get("label_age", 0)) + 1

    def validate_certification_provenance(
        self,
        phase: str,
        *,
        policy_version: str,
        estimator_version: str,
        require_evidence: bool = True,
    ) -> str | None:
        """Reject policy, estimator, or Tube-version mixtures in one phase."""
        rows = [
            r
            for r in self.records_for_phase(phase, include_training_only=False)
            if r["final"]["branches"] > 0
        ]
        if require_evidence and not rows:
            raise ValueError(f"Bank has no certified {phase} candidates")
        mismatched_policy = [
            r["id"] for r in rows if r.get("policy_version") != str(policy_version)
        ]
        if mismatched_policy:
            raise ValueError(
                f"Bank contains {phase} labels from another policy: {mismatched_policy[:3]}"
            )
        mismatched_estimator = [
            r["id"] for r in rows if r.get("estimator_version") != str(estimator_version)
        ]
        if mismatched_estimator:
            raise ValueError(
                f"Bank contains {phase} labels from another estimator: {mismatched_estimator[:3]}"
            )
        tube_versions = {str(r.get("tube_version")) for r in rows}
        if len(tube_versions) > 1:
            raise ValueError(f"Bank mixes {phase} Tube versions: {sorted(tube_versions)}")
        tube_version = next(iter(tube_versions), None)
        metadata_policy = self.metadata.get("last_policy_version")
        metadata_tube = self.metadata.get("last_tube_version")
        if rows and metadata_policy != str(policy_version):
            raise ValueError("Bank policy metadata does not match its certified records")
        if rows and metadata_tube != tube_version:
            raise ValueError("Bank Tube metadata does not match its certified records")
        return tube_version

    def reset_distribution(
        self,
        phase: str,
        *,
        safe_mass: float,
        boundary_mass: float,
        aux_mass: float,
        rehearsal_mass: float,
        tube_activation_min_safe: int,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        rows = self.records_for_phase(phase)
        if not rows:
            return [], np.empty((0,), np.float32)
        safe = [r for r in rows if r["final"]["label"] == "safe" and not r["training_only"]]
        boundary = [r for r in rows if r["final"]["label"] == "boundary" and not r["training_only"]]
        rehearsal = [
            r
            for r in rows
            if r["training_only"] and r.get("candidate_kind") == "downstream_rehearsal"
        ]
        aux = [
            r
            for r in rows
            if r["training_only"] and r.get("candidate_kind") != "downstream_rehearsal"
        ]
        if len(safe) < int(tube_activation_min_safe):
            base = [r for r in rows if bool(r.get("bootstrap_eligible", True)) and not r["training_only"]]
            base_mass = max(0.0, 1.0 - float(aux_mass) - float(rehearsal_mass))
            groups = [
                (base, base_mass),
                (aux, float(aux_mass)),
                (rehearsal, float(rehearsal_mass)),
            ]
        else:
            groups = [
                (safe, safe_mass),
                (boundary, boundary_mass),
                (aux, aux_mass),
                (rehearsal, rehearsal_mass),
            ]
        selected: list[dict[str, Any]] = []
        weights: list[float] = []
        for group, mass in groups:
            if not group or mass <= 0:
                continue
            raw = np.asarray(
                [max(0.05, 1.0 + 0.25 * math.log1p(int(r.get("reset_use_count", 0)))) for r in group],
                np.float64,
            )
            raw /= raw.sum()
            selected.extend(group)
            weights.extend((float(mass) * raw).tolist())
        if not selected:
            return [], np.empty((0,), np.float32)
        w = np.asarray(weights, np.float32)
        w /= w.sum()
        return selected, w

    def prioritized_for_relabel(self, phase: str, budget: int) -> list[dict[str, Any]]:
        rows = self.records_for_phase(phase, include_training_only=False)
        def score(r: Mapping[str, Any]) -> float:
            boundary = 2.0 if r["final"]["label"] in ("boundary", "unknown") else 0.0
            width = float(r["final"]["posterior"].get("width", 1.0))
            use = math.log1p(int(r.get("reset_use_count", 0)))
            age = float(r.get("label_age", 0))
            connection = 1.0 if r.get("connection_flag", False) else 0.0
            model_unc = float(r.get("model_uncertainty", 0.0) or 0.0)
            return boundary + width + 0.35 * use + 0.25 * age + 0.5 * connection + model_unc
        return sorted(rows, key=score, reverse=True)[: int(budget)]

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for phase in PHASES:
            rows = self.records_for_phase(phase)
            out[phase] = {
                "total": len(rows),
                "training_only": sum(bool(r["training_only"]) for r in rows),
                "chain": {label: sum(r["chain"]["label"] == label for r in rows) for label in LABELS},
                "final": {label: sum(r["final"]["label"] == label for r in rows) for label in LABELS},
            }
        return out


class TubeMatcher:
    """Phase-specific robust standardized nearest-neighbour matcher."""

    def __init__(self, records: Iterable[Mapping[str, Any]], *, eps: float = 1e-4):
        rows = list(records)
        if not rows:
            self.features = np.zeros((1, 1), np.float32)
            self.center = np.zeros((1,), np.float32)
            self.scale = np.ones((1,), np.float32)
            self.ids: list[str] = []
            return
        x = np.asarray([r["physical_feature"] for r in rows], np.float32)
        center = np.median(x, axis=0)
        mad = np.median(np.abs(x - center[None, :]), axis=0) * 1.4826
        std = x.std(axis=0)
        scale = np.maximum(np.maximum(mad, std * 0.25), eps)
        self.features = ((x - center) / scale).astype(np.float32)
        self.center = center.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.ids = [str(r["id"]) for r in rows]

    def transform(self, feature: np.ndarray) -> np.ndarray:
        return (np.asarray(feature, np.float32) - self.center) / self.scale

    def nearest(self, feature: np.ndarray) -> tuple[float, str | None]:
        if not self.ids:
            return float("inf"), None
        z = self.transform(feature)
        d = np.linalg.norm(self.features - z[None, :], axis=1)
        i = int(np.argmin(d))
        return float(d[i]), self.ids[i]
