from types import SimpleNamespace

import numpy as np
import pytest

from cli.fit_viability import parent_split
from cli.build_descent_entries import snapshot_identity as legacy_snapshot_identity
from dvgc.discrete_tube import ExactTubeMembership, prediction_cannot_promote, snapshot_identity
from dvgc.descent_membership import FrozenViabilityThreshold, LocalTrustRegions


def _row(identifier="a", parent="p", label="safe"):
    return {"id":identifier,"entry_source_id":parent,"source_phase":"flight",
            "qpos":np.zeros(2,np.float32),"qvel":np.zeros(2,np.float32),"ctrl":np.zeros(1,np.float32),
            "qacc_warmstart":np.zeros(2,np.float32),"physical_feature":np.zeros(16,np.float32),
            "oracle_phase":2,"had_airborne":1,"had_valid_landing":0,"contact_age":0,
            "airborne_count":1,"recovery_count":0,"policy_state":{"last_action":np.zeros(1,np.float32)},
            "final":{"label":label,"branches":8,"posterior":{"mean":.8}}}


def test_exact_tube_requires_id_snapshot_policy_and_certification_hash():
    row=_row();manifest={"membership_type":"exact_snapshot_identity","policy_hash":"policy","certification_hash":"cert",
                         "members":[{"id":"a","snapshot_sha256":snapshot_identity(row)}]}
    assert snapshot_identity(row)==legacy_snapshot_identity(row)
    tube=ExactTubeMembership.from_manifest(manifest)
    assert tube.contains(row,policy_hash="policy",certification_hash="cert")
    changed={**row,"qvel":np.ones(2,np.float32)}
    assert not tube.contains(changed,policy_hash="policy",certification_hash="cert")
    assert not tube.contains(row,policy_hash="other",certification_hash="cert")


def test_prediction_score_cannot_upgrade_unknown_to_empirical_safe():
    row=_row(label="unknown");row["viability_probability"]=.999
    assert not prediction_cannot_promote(row)
    row["empirical_label"]="safe";assert prediction_cannot_promote(row)


def test_parent_group_split_has_no_leakage_and_uses_snapshots():
    rows=[_row(str(i),f"p{i//2}",("safe","boundary","dead")[i%3]) for i in range(12)]
    split,parents=parent_split(rows,7)
    assert sum(map(len,split.values()))==12
    assert not set(parents["train"])&set(parents["validation"])
    assert not set(parents["train"])&set(parents["acquisition"])
    assert not set(parents["validation"])&set(parents["acquisition"])


def test_local_regions_keep_isolated_safe_radius_zero():
    model=LocalTrustRegions(np.asarray([[0.,0.],[2.,2.]]),np.ones((2,2)),np.asarray([0.,.5]))
    assert model.contains([0.,0.]) and not model.contains([.01,0.])
    assert model.contains([2.2,2.2])


def test_viability_threshold_is_frozen_and_rejects_uncertainty():
    model=FrozenViabilityThreshold(.9,.1,"calibration")
    assert model.contains(.95,.05,calibration_hash="calibration")
    assert not model.contains(.95,.2,calibration_hash="calibration")
    with pytest.raises(ValueError):model.contains(.95,.05,calibration_hash="audit-tuned")
