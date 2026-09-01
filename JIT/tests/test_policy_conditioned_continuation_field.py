from __future__ import annotations

import numpy as np


def _row(index: int, *, phase: str, label: int, group: str, split: str, x0: float):
    obs = np.zeros(76, dtype=np.float32)
    obs[0] = x0
    obs[1] = float(index) * 0.01
    return {
        "candidate_id": f"c{index}",
        "state_sha256": f"{index + 1:064x}",
        "phase": phase,
        "phase_index": 0 if phase == "upstream" else 1,
        "label": label,
        "parent_group_id": group,
        "split": split,
        "actor_observation": obs.tolist(),
        "policy_actor_sha256": "a" * 64,
        "policy_payload_sha256": "b" * 64,
    }


def test_cell_balanced_weights_equalize_parent_label_cells():
    from jit_dvgc.policy_conditioned_continuation_field import _cell_balanced_weights

    groups = ("g1", "g1", "g1", "g2", "g2", "g2", "g2", "g2")
    y = np.asarray([0, 1, 1, 0, 0, 1, 1, 1], dtype=np.float32)
    w = _cell_balanced_weights(groups, y)
    masses = {}
    for group in ("g1", "g2"):
        for label in (0, 1):
            mask = np.asarray(
                [g == group and int(v) == label for g, v in zip(groups, y)]
            )
            masses[(group, label)] = float(np.sum(w[mask]))
    assert max(masses.values()) - min(masses.values()) < 1e-6
    assert np.isclose(np.mean(w), 1.0)


def test_real_c0_config_is_linear_fixed_and_validation_single_use(jit_root):
    from jit_dvgc.policy_conditioned_continuation_field import (
        load_continuation_field_config,
    )

    config = load_continuation_field_config(
        jit_root / "configs/envelope_iter0_continuation_field.json"
    )
    p = config["protocol"]
    assert p["model"]["family"] == "linear_logistic"
    assert p["model"]["observation_size"] == 76
    assert p["model"]["sample_weighting"] == "equal_parent_label_cell_mass"
    assert p["calibration"]["validation_hyperparameter_search"] is False
    assert p["calibration"]["threshold_is_safety_certificate"] is False
    assert p["expected_counts"]["train"]["downstream"]["negative_count"] == 30
    assert p["expected_counts"]["validation"]["downstream"] == {
        "candidate_count": 16,
        "positive_count": 4,
        "negative_count": 12,
    }


def test_synthetic_linear_field_conservative_calibration(tmp_path):
    from jit_dvgc.policy_conditioned_continuation_field import _fit_phase

    train = []
    validation = []
    index = 0
    for group, offset in (("g1", 0.0), ("g2", 0.05)):
        for _ in range(6):
            train.append(
                _row(
                    index,
                    phase="upstream",
                    label=0,
                    group=group,
                    split="train",
                    x0=-1.0 + offset,
                )
            )
            index += 1
            train.append(
                _row(
                    index,
                    phase="upstream",
                    label=1,
                    group=group,
                    split="train",
                    x0=1.0 + offset,
                )
            )
            index += 1
        validation.append(
            _row(
                index,
                phase="upstream",
                label=0,
                group=group,
                split="validation",
                x0=-0.9 + offset,
            )
        )
        index += 1
        validation.append(
            _row(
                index,
                phase="upstream",
                label=1,
                group=group,
                split="validation",
                x0=0.9 + offset,
            )
        )
        index += 1

    protocol = {
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": "a" * 64,
        "policy_payload_sha256": "b" * 64,
        "model": {
            "l2_weight": 0.01,
            "optimizer": "adam_full_batch_fixed_schedule",
            "steps": 300,
            "learning_rate": 0.03,
            "seeds": {"upstream": 1, "downstream": 2},
        },
        "calibration": {
            "decision_rule": "accept_if_score_strictly_greater_than_max_validation_negative_score",
            "minimum_validation_roc_auc": 0.70,
            "minimum_validation_positive_recall": 0.20,
            "require_accepted_positive_in_every_validation_parent": True,
        },
        "claim_boundary": {},
    }
    manifest = _fit_phase(
        phase="upstream",
        train_rows=train,
        validation_rows=validation,
        protocol=protocol,
        output=tmp_path,
    )
    assert manifest["parameter_count"] == 77
    assert manifest["calibration_passed"] is True
    assert manifest["validation_metrics"]["roc_auc"] == 1.0
    assert manifest["calibration_gate"]["accepted_validation_negative_count_zero"] is True
    assert set(manifest["accepted_validation_positive_groups"]) == {"g1", "g2"}
