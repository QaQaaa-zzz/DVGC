from __future__ import annotations


def test_c1_engineering_selection_is_locked_to_exact_64x64_artifact() -> None:
    from jit_dvgc.c1_engineering_selection import (
        SELECTED_PROFILE,
        SELECTED_UPSTREAM_AUC,
        SELECTED_UPSTREAM_CALIBRATION_SHA256,
        SELECTED_UPSTREAM_FIELD_SHA256,
        SELECTED_UPSTREAM_MANIFEST_SHA256,
    )
    from jit_dvgc.iterative_continuation_fields import MODEL_PROFILES

    profile = MODEL_PROFILES[SELECTED_PROFILE]
    assert SELECTED_PROFILE == "standard_mlp_64x64_tanh"
    assert profile["hidden_sizes"] == [64, 64]
    assert profile["parameter_count"] == 9153
    assert profile["architecture"] == "76->64_tanh->64_tanh->1"
    assert SELECTED_UPSTREAM_AUC == 0.6903137789904502
    assert SELECTED_UPSTREAM_FIELD_SHA256 == "94528aed8bfb4e6db5c01a2bd4231297a0cd3252f198a889c929bca4ee8aac07"
    assert SELECTED_UPSTREAM_MANIFEST_SHA256 == "f010f1cafd17dc7e981f1c0c0d62f55dbeda75ea8d65185568041ac8f199955d"
    assert SELECTED_UPSTREAM_CALIBRATION_SHA256 == "3b1de2557eba250fdaa1df6e4b6f05082e2f808e9979a3c10ce843d53469cdf3"


def test_engineering_override_does_not_change_formal_auc_gate() -> None:
    from jit_dvgc.c1_engineering_selection import SELECTED_UPSTREAM_AUC
    from jit_dvgc.iterative_continuation_fields import CALIBRATION_CONTRACT

    assert CALIBRATION_CONTRACT["minimum_roc_auc"] == 0.70
    assert SELECTED_UPSTREAM_AUC < CALIBRATION_CONTRACT["minimum_roc_auc"]
