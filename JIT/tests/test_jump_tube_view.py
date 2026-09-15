from __future__ import annotations

from jit_dvgc.analysis.jump_tube_view import _eligible


def _row(phase: str, x: float, vz: float) -> dict:
    return {
        "phase": phase,
        "coordinates": {
            "root_x_m": x,
            "root_vz_mps": vz,
        },
    }


def test_jump_tube_semantic_filter_rejects_late_and_non_descending_downstream() -> None:
    assert _eligible(_row("upstream", 2.8, +0.4), x_min=2.5, x_max=4.1)
    assert _eligible(_row("downstream", 3.7, -0.2), x_min=2.5, x_max=4.1)
    assert not _eligible(_row("downstream", 3.7, +0.2), x_min=2.5, x_max=4.1)
    assert not _eligible(_row("downstream", 4.5, -0.2), x_min=2.5, x_max=4.1)
    assert not _eligible(_row("upstream", 2.4, +0.2), x_min=2.5, x_max=4.1)
