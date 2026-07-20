import math

import numpy as np

from dvgc.config import default_config
from dvgc.stage_reachability import CANONICAL_PHASE, evaluate_entry, next_branch_budget, protocol_payload, reachability_label


def sample(**kw):
    f=np.zeros(16,np.float32);f[0]=4.;f[2]=.55;f[6]=3.;f[8]=0.
    row={"physical_feature":f,"canonical_phase":"flight","dual_wheel_airborne":True}
    row.update(kw);return row


def support(row):
    f=np.asarray(row["physical_feature"],np.float32)
    return {"stage_entry_matcher":{"center":np.zeros(16).tolist(),"scale":np.ones(16).tolist(),"radius":.5,
            "reference_envelope":{"x":{"min":3.5,"max":5.0},"z":{"min":.35,"max":.65}},
            "descent_vz_min":-2.5,"descent_vz_max":-.05,"max_abs_roll_rate":2.,"max_abs_pitch_rate":2.,
            "envelope_tolerance_x":0.,"envelope_tolerance_z":0.},"support_features":[f.tolist()]}


def test_flight_substages_map_to_existing_canonical_phase():
    assert CANONICAL_PHASE["ascent"]==CANONICAL_PHASE["apex"]==CANONICAL_PHASE["descent"]=="flight"


def test_takeoff_ascent_requires_real_airborne_and_upward_motion():
    cfg=default_config();row=sample(canonical_phase="takeoff",dual_wheel_airborne=True);row["physical_feature"][8]=.3
    assert evaluate_entry("takeoff",row,cfg)["valid"]
    row["dual_wheel_airborne"]=False
    assert not evaluate_entry("takeoff",row,cfg)["valid"]


def test_apex_entry_and_descent_crossing_are_event_aligned():
    cfg=default_config();assert evaluate_entry("ascent",sample(),cfg)["valid"]
    row=sample(previous_vz=.1,apex_seen=True);row["physical_feature"][8]=-.1;meta=support(row)
    assert evaluate_entry("apex",row,cfg,meta)["valid"]
    row["apex_seen"]=False
    assert not evaluate_entry("apex",row,cfg,meta)["valid"]


def test_apex_rejects_falling_contact_and_missing_support():
    cfg=default_config();row=sample(apex_seen=True);row["physical_feature"][8]=-.2;meta=support(row)
    assert evaluate_entry("apex",row,cfg,meta)["valid"]
    row["body_terrain_contact"]=True
    assert not evaluate_entry("apex",row,cfg,meta)["valid"]
    row.pop("body_terrain_contact")
    assert not evaluate_entry("apex",row,cfg)["valid"]


def test_descent_entry_rejects_body_contact_and_bad_support():
    cfg=default_config();row=sample(canonical_phase="landing",first_valid_landing=True,support=True)
    row["physical_feature"][2]=.42
    assert evaluate_entry("descent",row,cfg)["valid"]
    row["body_terrain_contact"]=True
    assert not evaluate_entry("descent",row,cfg)["valid"]


def test_landing_stable_uses_existing_hold_gate():
    cfg=default_config();row=sample(canonical_phase="landing",support=True,recovery_count=cfg.recovery_hold_steps)
    assert evaluate_entry("landing",row,cfg)["valid"]


def test_soft_labels_do_not_call_controller_failure_physically_unreachable():
    x=reachability_label(stage="descent",successes=0,branches=8,branch_records=[],controller_bank_exhausted=True)
    assert x["label"]=="negative_under_current_controller_bank"
    assert next_branch_budget(reachability_label(stage="apex",successes=2,branches=4,branch_records=[]))==8
    assert len(protocol_payload(default_config())["protocol_sha256"])==64

def test_four_zero_success_branches_remain_unknown_for_adaptive_funnel():
    x=reachability_label(stage="apex",successes=0,branches=4,branch_records=[],controller_bank_exhausted=True)
    assert x["label"]=="unknown" and next_branch_budget(x)==4
