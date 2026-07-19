import json
import sys

import numpy as np

from cli.prepare_decoupled_bootstrap import CANONICAL_C_L_ROLES, main
from dvgc.bank import SnapshotBank, empty_outcome
from dvgc.config import file_sha256, load_config
from dvgc.policy import save_bundle


def _policy(path, stage):
    cfg = load_config("configs/default.json", {"training_stage": stage})
    return save_bundle(
        path,
        params={"stage": stage},
        config=cfg,
        xml_path=cfg.xml_path,
        candidate_bank=None,
        downstream_bank=None,
        policy_version=f"pi-{stage}",
        extra={"stage": stage},
    )


def test_freeze_contract_keeps_cl_and_policy_ownership(monkeypatch, tmp_path):
    landing = _policy(tmp_path / "landing", "landing")
    flight = _policy(tmp_path / "flight", "flight")
    final = empty_outcome()
    final.update({"label": "safe", "branches": 32, "successes": 32, "failures": 0})
    entry = tmp_path / "c_l.pkl"
    SnapshotBank(
        [{
            "id": "entry", "source_phase": "landing", "qpos": np.zeros(1),
            "qvel": np.zeros(1), "ctrl": np.zeros(1),
            "physical_feature": np.zeros(16), "entry_feature": np.zeros(20),
            "final": final,
        }],
        {
            "entry_bank_role": "canonical_certified_landing_entry_set",
            "last_policy_version": "pi-landing", "last_tube_version": "c-l-v1",
            "entry_matcher": {"version": "matcher", "radius": 1.0, "entry_window_steps": 3},
        },
    ).save(entry)
    bank = tmp_path / "flight.pkl"
    SnapshotBank().save(bank)
    cfg = load_config("configs/default.json")
    gate = tmp_path / "runtime.json"
    gate.write_text(json.dumps({
        "status": "PASS", "source_fingerprint": "runtime",
        "xml_sha256": file_sha256(cfg.xml_path),
    }))
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "prepare", "--landing-policy", str(landing), "--flight-policy", str(flight),
        "--landing-entry-set", str(entry), "--flight-bank", str(bank),
        "--runtime-gate", str(gate), "--output-root", str(out),
    ])
    main()
    contract = json.loads((out / "frozen_contract.json").read_text())
    assert contract["canonical_c_l"]["sha256"] == file_sha256(entry)
    assert contract["canonical_c_l"]["final_safe_count"] == 1
    assert contract["flight_initial"]["landing_retention_required"] is False
    assert contract["expert_certification_role"] == "expert_conditioned_provisional_envelope"
    assert contract["formal_jel_role"] == "final_shared_policy_jel"


def test_current_extended_canonical_role_is_explicitly_supported():
    assert CANONICAL_C_L_ROLES == {
        "canonical_certified_landing_entry_set",
        "canonical_certified_landing_entry_set_extended",
    }
