import pytest

from dvgc.seed_registry import (
    allocate_disjoint_grid,
    branch_seed_grid,
    exact_intersection_proof,
    make_claim,
    report_branch_seeds,
)


def test_exact_intersection_checks_full_global_index_grid():
    construction = make_claim("construction", "certification", branch_seed_grid(9_630_000, 139, 32))
    collided = make_claim("old_audit", "audit", branch_seed_grid(9_330_000, 98, 32))
    replacement = make_claim("new_audit", "audit", branch_seed_grid(200_000_000, 98, 32))
    assert exact_intersection_proof(collided, [construction])["intersection_count"] > 0
    assert exact_intersection_proof(replacement, [construction])["intersection_count"] == 0


def test_allocator_rejects_collision_then_uses_next_exact_namespace():
    occupied = [make_claim("used", "audit", branch_seed_grid(10_000, 2, 2))]
    seed, proof, attempts = allocate_disjoint_grid(
        10_000, 2, 2, occupied, reallocation_stride=1_000_000, max_reallocations=3
    )
    assert seed == 1_010_000
    assert proof["status"] == "PASS"
    assert [row["proof"]["status"] for row in attempts] == ["CONFLICT", "PASS"]


def test_report_seed_extraction_is_unique_and_supports_both_schemas():
    payload = {"rows": [
        {"branch_evidence": [{"branch_seed": 3}, {"branch_seed": 2}]},
        {"certification_branches": [{"branch_seed": 3}, {"branch_seed": 4}]},
    ]}
    assert report_branch_seeds(payload) == [2, 3, 4]


def test_allocator_fuses_after_three_reallocations():
    seeds = []
    for base in (10, 110, 210, 310):
        seeds.extend(branch_seed_grid(base, 1, 1))
    occupied = [make_claim("used", "audit", seeds)]
    with pytest.raises(RuntimeError, match="three automatic reallocations"):
        allocate_disjoint_grid(10, 1, 1, occupied, reallocation_stride=100, max_reallocations=3)
