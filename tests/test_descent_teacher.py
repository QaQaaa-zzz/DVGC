import numpy as np

from dvgc.descent_teacher import (
    fixed_candidate_folds, nearest_neighbor_audit, relabel_support_gate,
    trajectory_support_radius,
)


def test_candidate_folds_are_deterministic_and_never_split_candidates():
    rows = [
        {"id": f"c{i}", "provisional_label": "core" if i % 2 else "frontier",
         "descent_layer": ("early", "middle", "late")[i % 3]}
        for i in range(9)
    ]
    first = fixed_candidate_folds(rows)
    second = fixed_candidate_folds(list(reversed(rows)))
    assert first == second
    assert sorted(sum(first, [])) == [f"c{i}" for i in range(9)]
    assert not (set(first[0]) & set(first[1]) or set(first[1]) & set(first[2]))


def test_relabel_requires_support_discrete_state_and_counterfactual_gain():
    accepted, reasons = relabel_support_gate(
        normalized_distance=.05, support_p95=.10, phase_equal=True,
        contact_mode_equal=True, delay_buffer_equal=True, precursor_equal=True,
        counterfactual_survival_gain=1, counterfactual_margin_gain=0.0,
        excluded_or_heldout=False,
    )
    assert accepted and not reasons
    rejected, reasons = relabel_support_gate(
        normalized_distance=.11, support_p95=.10, phase_equal=True,
        contact_mode_equal=True, delay_buffer_equal=True, precursor_equal=True,
        counterfactual_survival_gain=0, counterfactual_margin_gain=0.0,
        excluded_or_heldout=False,
    )
    assert not rejected
    assert set(reasons) == {"outside_teacher_support_p95", "counterfactual_not_strictly_better"}


def test_relabel_never_accepts_excluded_or_heldout_state():
    accepted, reasons = relabel_support_gate(
        normalized_distance=0, support_p95=1, phase_equal=True,
        contact_mode_equal=True, delay_buffer_equal=True, precursor_equal=True,
        counterfactual_survival_gain=5, counterfactual_margin_gain=1,
        excluded_or_heldout=True,
    )
    assert not accepted and reasons == ["excluded_or_heldout"]


def test_nearest_neighbor_audit_and_support_radius_are_finite():
    obs = np.arange(32, dtype=np.float32).reshape(8, 4) / 10
    actions = np.ones((8, 4), np.float32) * .1
    report = nearest_neighbor_audit(obs, actions, ["a"] * 4 + ["b"] * 4)
    assert report["representable"]
    assert 0 < trajectory_support_radius(obs) < 2
