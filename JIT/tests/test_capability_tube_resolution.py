from __future__ import annotations

from jit_dvgc.analysis.capability_tube import (
    ROOT_GEOMETRY_FIELDS,
    quantize_coordinates,
    resolution_contract,
)


def _base_coordinates() -> dict[str, float]:
    return {
        "root_x_m": 1.00,
        "root_y_m": 0.00,
        "root_z_m": 0.80,
        "root_vx_mps": 2.00,
        "root_vy_mps": 0.00,
        "root_vz_mps": 0.40,
        "roll_deg": 1.00,
        "pitch_deg": 3.00,
        "yaw_deg": 0.00,
        "root_wx_degps": 4.00,
        "root_wy_degps": 8.00,
        "root_wz_degps": 0.00,
    }


def test_resolution_contract_uses_two_degree_per_second_angular_bins() -> None:
    contract = resolution_contract()
    resolutions = contract["resolutions"]
    assert resolutions["root_wx_degps"]["resolution"] == 2.0
    assert resolutions["root_wy_degps"]["resolution"] == 2.0
    assert resolutions["root_wz_degps"]["resolution"] == 2.0
    assert resolutions["pitch_deg"]["resolution"] == 0.5
    assert resolutions["root_x_m"]["resolution"] == 0.1
    assert resolutions["root_vx_mps"]["resolution"] == 0.1


def test_subresolution_neighbour_states_share_root_geometry_cell() -> None:
    first = _base_coordinates()
    second = dict(first)
    second.update(
        {
            "root_x_m": 1.03,
            "root_z_m": 0.82,
            "root_vx_mps": 2.03,
            "pitch_deg": 3.10,
            "root_wy_degps": 8.60,
        }
    )
    assert quantize_coordinates(first, ROOT_GEOMETRY_FIELDS) == quantize_coordinates(
        second, ROOT_GEOMETRY_FIELDS
    )


def test_physically_resolved_change_moves_to_another_cell() -> None:
    first = _base_coordinates()
    second = dict(first)
    second["root_x_m"] = 1.11
    assert quantize_coordinates(first, ROOT_GEOMETRY_FIELDS) != quantize_coordinates(
        second, ROOT_GEOMETRY_FIELDS
    )


def test_two_degree_per_second_rate_change_is_resolved() -> None:
    first = _base_coordinates()
    second = dict(first)
    second["root_wy_degps"] = 10.1
    first_bins = quantize_coordinates(first, ROOT_GEOMETRY_FIELDS)
    second_bins = quantize_coordinates(second, ROOT_GEOMETRY_FIELDS)
    assert first_bins["root_wy_degps"] != second_bins["root_wy_degps"]
