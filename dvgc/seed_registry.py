"""Exact seed-set registry and allocator for independent experiment tasks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from dvgc.certification import branch_seed
from dvgc.runtime import save_json


def normalized_seeds(values: Iterable[int]) -> list[int]:
    seeds = sorted({int(value) for value in values})
    if any(value < 0 for value in seeds):
        raise ValueError("Registered seeds must be non-negative")
    return seeds


def seed_set_sha256(values: Iterable[int]) -> str:
    payload = json.dumps(normalized_seeds(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def branch_seed_grid(base_seed: int, state_count: int, branches: int) -> list[int]:
    return normalized_seeds(
        branch_seed(base_seed, state_index, branch_index)
        for state_index in range(int(state_count))
        for branch_index in range(int(branches))
    )


def report_branch_seeds(payload: Mapping) -> list[int]:
    seeds = []
    for row in payload.get("rows", []):
        evidence = row.get("branch_evidence", row.get("certification_branches", []))
        seeds.extend(int(branch["branch_seed"]) for branch in evidence)
    return normalized_seeds(seeds)


def make_claim(name: str, category: str, seeds: Iterable[int], **metadata) -> dict:
    values = normalized_seeds(seeds)
    return {
        "name": str(name),
        "category": str(category),
        "seed_count": len(values),
        "seed_set_sha256": seed_set_sha256(values),
        "seeds": values,
        **metadata,
    }


def exact_intersection_proof(candidate: Mapping, existing: Iterable[Mapping]) -> dict:
    candidate_seeds = set(normalized_seeds(candidate.get("seeds", [])))
    comparisons = []
    union = set()
    for claim in existing:
        other = set(normalized_seeds(claim.get("seeds", [])))
        overlap = sorted(candidate_seeds & other)
        union.update(other)
        comparisons.append({
            "claim": claim["name"],
            "category": claim["category"],
            "registered_seed_count": len(other),
            "intersection_count": len(overlap),
            "intersection_preview": overlap[:10],
        })
    overlap = sorted(candidate_seeds & union)
    return {
        "status": "PASS" if not overlap else "CONFLICT",
        "candidate_claim": candidate["name"],
        "candidate_seed_count": len(candidate_seeds),
        "candidate_seed_set_sha256": seed_set_sha256(candidate_seeds),
        "registered_union_seed_count": len(union),
        "intersection_count": len(overlap),
        "intersection_preview": overlap[:10],
        "comparisons": comparisons,
    }


def allocate_disjoint_grid(
    initial_base_seed: int,
    state_count: int,
    branches: int,
    existing: Iterable[Mapping],
    *,
    reallocation_stride: int = 100_000_000,
    max_reallocations: int = 3,
) -> tuple[int, dict, list[dict]]:
    """Allocate an exact disjoint grid, recording every rejected proposal."""
    attempts = []
    for offset in range(int(max_reallocations) + 1):
        base_seed = int(initial_base_seed) + offset * int(reallocation_stride)
        claim = make_claim("allocation_candidate", "branch_randomness", branch_seed_grid(base_seed, state_count, branches))
        proof = exact_intersection_proof(claim, existing)
        attempts.append({"base_seed": base_seed, "proof": proof})
        if proof["status"] == "PASS":
            return base_seed, proof, attempts
    raise RuntimeError("Seed allocator exhausted three automatic reallocations")


def save_registry(path: Path, claims: Iterable[Mapping], **metadata) -> dict:
    rows = list(claims)
    intersections = []
    for index, claim in enumerate(rows):
        proof = exact_intersection_proof(claim, rows[:index])
        if proof["intersection_count"]:
            intersections.append(proof)
    payload = {
        "schema_version": 1,
        "claims": rows,
        "historical_intersections": intersections,
        **metadata,
    }
    save_json(path, payload)
    return payload
