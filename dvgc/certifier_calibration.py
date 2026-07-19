"""Predeclared Final-Recovery admission rules for held-out calibration."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Mapping

from .bank import beta_posterior


PROTOCOL_VERSION = "final-admission-calibration-v1"


def _successes(branches: Iterable[Mapping]) -> tuple[int, int]:
    values = [bool(row["final_recovery"]) for row in branches]
    return sum(values), len(values)


def _beta_lower(branches: Iterable[Mapping]) -> float:
    s, n = _successes(branches)
    if not n:
        return 0.0
    return float(beta_posterior(s, n - s, alpha0=1.0, beta0=1.0,
                                q_low=.05, q_high=.95)["lower"])


def wilson_lower(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    den = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (center - radius) / den


def current_conjunction(branches: list[Mapping], safe_threshold: float) -> bool:
    """Two independent ordered halves must each meet the target LCB."""
    if len(branches) < 16:
        return False
    split = len(branches) // 2
    return (_beta_lower(branches[:split]) >= safe_threshold and
            _beta_lower(branches[split:]) >= safe_threshold)


def batch_lcb(branches: list[Mapping], safe_threshold: float) -> bool:
    return len(branches) >= 8 and _beta_lower(branches) >= safe_threshold


def pooled_heterogeneity(branches: list[Mapping], safe_threshold: float,
                         max_rate_gap: float = .25) -> bool:
    if len(branches) < 16 or _beta_lower(branches) < safe_threshold:
        return False
    split = len(branches) // 2
    a, na = _successes(branches[:split]); b, nb = _successes(branches[split:])
    return abs(a / na - b / nb) <= max_rate_gap


def sequential_confidence(branches: list[Mapping], safe_threshold: float) -> bool:
    """Bonferroni-adjusted one-sided Wilson stopping over fixed checkpoints."""
    checkpoints = [n for n in (8, 16, 24, 32, 48, 64) if n <= len(branches)]
    for n in checkpoints:
        s, _ = _successes(branches[:n])
        if wilson_lower(s, n, z=2.3939797998185104) >= safe_threshold:
            return True
    return False


RULES = {
    "stage_conjunction": current_conjunction,
    "pooled_lcb_heterogeneity": pooled_heterogeneity,
    "sequential_adaptive": sequential_confidence,
}


def protocol_hash(selected_rule: str, safe_threshold: float) -> str:
    payload = {"version": PROTOCOL_VERSION, "rule": selected_rule,
               "safe_threshold": float(safe_threshold),
               "pooled_max_rate_gap": .25,
               "sequential_checkpoints": [8, 16, 24, 32, 48, 64],
               "sequential_z": 2.3939797998185104}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
