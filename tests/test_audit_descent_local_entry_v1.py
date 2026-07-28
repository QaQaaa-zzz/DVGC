import numpy as np

from cli.audit_descent_local_entry_v1 import audit_offsets


def test_audit_offsets_are_fixed_inside_and_outside_shells():
    rows = audit_offsets(np.ones(16), .1)
    assert [row[2] for row in rows].count("inside") == 8
    assert [row[2] for row in rows].count("outside") == 4
    radii = [np.linalg.norm(row[:2]) for row in rows]
    assert max(radii[:8]) < .1
    assert min(radii[8:]) > .1
