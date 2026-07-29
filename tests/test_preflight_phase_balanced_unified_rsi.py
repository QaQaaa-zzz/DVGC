import numpy as np

from cli.preflight_phase_balanced_unified_rsi import STAGES, validate_static_contract
from dvgc.bank import SnapshotBank


def _bank():
    records = []
    for stage in STAGES:
        records.append({
            "id": stage, "source_phase": "takeoff" if stage == "takeoff" else
            ("landing" if stage == "landing" else "flight"),
            "qpos": np.zeros(13), "qvel": np.zeros(12), "ctrl": np.zeros(4),
            "physical_feature": np.zeros(16), "phase_rsi_stage": stage,
            "reset_weight": .2, "training_only": True,
            "artifact_role": "proposal_support_bank",
        })
    return SnapshotBank(records, {
        "artifact_role": "phase_balanced_tube_rsi_reset_bank", "formal_tube_or_jel": False,
    })


def test_static_contract_requires_balanced_training_only_proposals():
    bank = _bank(); report = {"stage_weight_mass": {stage: .2 for stage in STAGES}}
    checks = validate_static_contract(bank, report)
    assert all(checks.values())
    bank.records[0]["certified_safe"] = True
    assert not validate_static_contract(bank, report)["no_embedded_safe_claim"]


def test_static_contract_rejects_unequal_stage_mass():
    bank = _bank(); masses = {stage: .2 for stage in STAGES}; masses["apex"] = .1
    assert not validate_static_contract(bank, {"stage_weight_mass": masses})["equal_stage_mass"]
