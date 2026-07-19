import numpy as np

from cli.stage_reachability_inventory import infer_stage, state_hash


def row(value=0):
    return {k:np.full(2,value,np.float32) for k in ("qpos","qvel","ctrl","qacc_warmstart")}


def test_state_hash_is_full_state_and_deterministic():
    assert state_hash(row())==state_hash(row())
    assert state_hash(row())!=state_hash(row(1))


def test_infer_stage_keeps_flight_substages_separate():
    assert infer_stage({"source_phase":"flight","flight_subinterval":"apex"},"flight_candidates")=="apex"
    assert infer_stage({"source_phase":"flight","descent_layer":"late"},"pool")=="descent"
    assert infer_stage({"source_phase":"landing"},"landing")=="landing"
