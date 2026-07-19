"""Immutable artifact roles and handoff-decomposition semantics."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


CERTIFIED_TUBE = "certified_tube"
PROPOSAL_SUPPORT_BANK = "proposal_support_bank"
EXPERT_CONDITIONED_PROVISIONAL_ENVELOPE = "expert_conditioned_provisional_envelope"
FINAL_SHARED_POLICY_JEL = "final_shared_policy_jel"


FORMAL_SAFE_ROLES = frozenset({CERTIFIED_TUBE, FINAL_SHARED_POLICY_JEL})


def permits_formal_jel_claim(metadata: Mapping) -> bool:
    """Only a frozen shared-policy recertification may define final JEL."""
    role = metadata.get("artifact_role") or metadata.get("entry_bank_role")
    return bool(
        role == FINAL_SHARED_POLICY_JEL
        and metadata.get("controller_mode") == "single_shared_actor"
        and metadata.get("independent_recertification") is True
    )


def validate_artifact_role(metadata: Mapping, *, expected: str) -> None:
    role = metadata.get("artifact_role") or metadata.get("entry_bank_role")
    if role != expected:
        raise ValueError(f"Expected artifact role {expected!r}, got {role!r}")


def proposal_training_weight(label: str, lower: float, safe_threshold: float = 0.7) -> float:
    """Return an auditable RSI weight without promoting proposals to safe."""
    if label == "safe":
        return 1.0
    if label == "boundary":
        return 0.55 if float(lower) >= float(safe_threshold) - 0.1 else 0.25
    if label == "unknown":
        return 0.05
    if label == "dead":
        return 0.0
    raise ValueError(f"Unknown empirical label: {label}")


def classify_handoff(*, has_contact: bool, event_order_valid: bool,
                     matched_c_l: bool, landing_label: str) -> str:
    """Classify a handoff branch without changing Chain or Final semantics."""
    if not has_contact or not event_order_valid:
        return "H4"
    recoverable = landing_label == "safe"
    if recoverable and not matched_c_l:
        return "H1"
    if matched_c_l and not recoverable:
        return "H2"
    if not matched_c_l and not recoverable:
        return "H3"
    return "matched_recoverable"


def summarize_handoff(rows: Iterable[Mapping]) -> dict:
    rows = list(rows)
    branch = Counter(str(row["handoff_class"]) for row in rows)
    by_class = {}
    for name in sorted(branch):
        subset = [row for row in rows if row["handoff_class"] == name]
        states = {row["candidate_id"] for row in subset}
        parents = {row.get("parent_id") or row["candidate_id"] for row in subset}
        layers = Counter(str(row.get("layer") or "unknown") for row in subset)
        by_class[name] = {
            "branches": len(subset),
            "branch_fraction": len(subset) / len(rows) if rows else 0.0,
            "states": len(states),
            "parents": len(parents),
            "layers": dict(layers),
        }
    return {"branches": len(rows), "classes": by_class}
