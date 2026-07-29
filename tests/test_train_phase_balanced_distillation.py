import numpy as np
import pytest

from cli.train_phase_balanced_distillation import (
    downstream_fidelity_pass,
    summarize_error,
    summarize_fidelity,
    validate_dataset,
)


def _payload():
    examples = []
    for phase in ("takeoff", "ascent", "apex", "descent", "landing"):
        examples.append({
            "phase": phase, "observation": np.zeros(140, np.float32),
            "action": np.zeros(4, np.float32), "training_weight": .2,
        })
    return {"schema": "dvgc_phase_balanced_distillation_teacher_v1",
            "artifact_role": "phase_balanced_distillation_teacher_dataset",
            "formal_tube_or_jel": False,
            "expert_controller_identities": {"descent": "adapter:identity"},
            "examples": examples}


def test_validate_dataset_requires_equal_five_phase_mass():
    obs, actions, weights, phases = validate_dataset(_payload())
    assert obs.shape == (5, 140); assert actions.shape == (5, 4)
    assert weights.sum() == pytest.approx(1.0)
    assert len(set(phases)) == 5
    bad = _payload(); bad["examples"][0]["training_weight"] = .1
    with pytest.raises(ValueError, match="not phase balanced"):
        validate_dataset(bad)


def test_error_summary_is_phase_conditioned():
    target = np.zeros((2, 4)); prediction = np.asarray([[1, 0, 0, 0], [0, 2, 0, 0]])
    result = summarize_error(prediction, target, ["takeoff", "landing"])
    assert result["takeoff"]["mean_action_l2"] == pytest.approx(1.0)
    assert result["landing"]["mean_action_l2"] == pytest.approx(2.0)
    assert result["all"]["max_action_l2"] == pytest.approx(2.0)


def test_downstream_teacher_fidelity_is_a_hard_gate():
    phases = [stage for stage in
              ("takeoff", "ascent", "apex", "descent", "landing") for _ in range(2)]
    target = np.zeros((10, 4), np.float32)
    prediction = target.copy()
    fidelity = summarize_fidelity(prediction, target, phases)
    assert downstream_fidelity_pass(fidelity)
    prediction[6, 0] = .051
    fidelity = summarize_fidelity(prediction, target, phases)
    assert fidelity["descent"]["max"] > .05
    assert not downstream_fidelity_pass(fidelity)
