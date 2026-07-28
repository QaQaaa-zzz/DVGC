from cli.pilot_descent_local_entry_v1 import select_pilot_anchors
from dvgc.trajectory_mining import canonical_state_byte_hash


def test_pilot_anchor_selection_covers_regions_and_extension():
    rows = [
        {"id": "e", "descent_region": "early", "final": {"successes": 4, "branches": 4}},
        {"id": "m0", "descent_region": "middle", "final": {"successes": 3, "branches": 4}},
        {"id": "m1", "descent_region": "middle", "final": {"successes": 4, "branches": 4}},
        {"id": "l", "descent_region": "late", "final": {"successes": 4, "branches": 4}},
        {"id": "x", "descent_region": None, "target_distance": 1., "final": {"successes": 4, "branches": 4}},
    ]
    assert [row["id"] for row in select_pilot_anchors(rows)] == ["e", "m1", "l", "x"]


def test_generated_snapshot_hash_does_not_require_declared_sidecar():
    row = {"qpos": [1., 2.], "qvel": [3.], "ctrl": [0.], "qacc_warmstart": [0.]}
    assert canonical_state_byte_hash(row) == canonical_state_byte_hash(dict(row))


def test_local_entry_pilot_supports_explicit_tube_anchors():
    from pathlib import Path
    text=Path('cli/pilot_descent_local_entry_v1.py').read_text()
    assert '--anchor-id' in text and 'declared local-entry anchor missing' in text
