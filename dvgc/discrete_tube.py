"""Exact, policy-conditioned empirical Tube membership.

This module intentionally provides no distance-based generalization.  A state
is a member only when both its immutable record id and complete snapshot hash
appear in the frozen manifest for the same policy and certification artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np


def snapshot_identity(record: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
        value = np.ascontiguousarray(record[key]); digest.update(key.encode()); digest.update(value.dtype.str.encode()); digest.update(repr(value.shape).encode()); digest.update(value.tobytes())
    for key in sorted(record.get("policy_state", {})):
        value = np.ascontiguousarray(record["policy_state"][key]); digest.update(key.encode()); digest.update(value.dtype.str.encode()); digest.update(repr(value.shape).encode()); digest.update(value.tobytes())
    for key in ("oracle_phase", "had_airborne", "had_valid_landing", "contact_age", "airborne_count", "recovery_count"):
        digest.update(f"{key}:{record.get(key)}".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExactTubeMembership:
    policy_hash: str
    certification_hash: str
    members: Mapping[str, str]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "ExactTubeMembership":
        if manifest.get("membership_type") != "exact_snapshot_identity":
            raise ValueError("Manifest is not an exact empirical Tube")
        members = {str(row["id"]): str(row["snapshot_sha256"]) for row in manifest["members"]}
        if len(members) != len(manifest["members"]):
            raise ValueError("Exact Tube manifest contains duplicate state ids")
        return cls(str(manifest["policy_hash"]), str(manifest["certification_hash"]), members)

    def contains(self, record: Mapping[str, Any], *, policy_hash: str, certification_hash: str) -> bool:
        if str(policy_hash) != self.policy_hash or str(certification_hash) != self.certification_hash:
            return False
        expected = self.members.get(str(record.get("id")))
        return bool(expected is not None and expected == snapshot_identity(record))


def prediction_cannot_promote(record: Mapping[str, Any]) -> bool:
    """Network scores are acquisition metadata, never empirical membership."""
    return str(record.get("empirical_label", record.get("final", {}).get("label", "unknown"))) == "safe"
