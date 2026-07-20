import json
from pathlib import Path
import pytest

from dvgc.config import load_config
from dvgc.experts import StageExpertRegistry
from dvgc.policy import copy_bundle,save_bundle


def make_policy(path,stage,responsibility=None):
    cfg=load_config("configs/default.json",{"training_stage":stage})
    return save_bundle(path,params={"stage":stage},config=cfg,xml_path=cfg.xml_path,candidate_bank=None,downstream_bank=None,policy_version=f"pi-{stage}-{responsibility or stage}",extra={"stage":stage,"expert_responsibility":responsibility or stage})


def test_registry_owns_distinct_immutable_policy_files(tmp_path):
    landing=make_policy(tmp_path/"landing","landing"); flight=make_policy(tmp_path/"flight","flight"); entry=tmp_path/"entry.pkl"; entry.write_bytes(b"fixed-entry")
    registry=StageExpertRegistry.build({"landing":landing,"flight":flight},{"flight":entry},runtime_source_fingerprint="runtime")
    assert registry.specs["landing"].policy_hash!=registry.specs["flight"].policy_hash
    assert registry.specs["flight"].downstream_entry_set_sha256
    assert registry.specs["flight"].downstream_controller_stack_hash==registry.specs["landing"].controller_stack_hash
    registry.save(tmp_path/"registry.json"); StageExpertRegistry.load(tmp_path/"registry.json")
    (landing/"params.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError,match="policy changed"): registry.validate_files()


def test_policy_clone_refuses_to_overwrite(tmp_path):
    source=make_policy(tmp_path/"source","flight"); copy_bundle(source,tmp_path/"owned")
    with pytest.raises(FileExistsError): copy_bundle(source,tmp_path/"owned")


def test_registry_rejects_stage_or_schema_mismatch(tmp_path):
    landing=make_policy(tmp_path/"landing","landing"); wrong=make_policy(tmp_path/"wrong","takeoff"); entry=tmp_path/"entry"; entry.write_bytes(b"x")
    with pytest.raises(ValueError,match="declares stage"): StageExpertRegistry.build({"landing":landing,"flight":wrong},{"flight":entry},runtime_source_fingerprint="runtime")

def test_registry_supports_distinct_flight_substage_responsibilities(tmp_path):
    landing=make_policy(tmp_path/"landing","landing")
    ascent=make_policy(tmp_path/"ascent","flight","ascent_to_apex")
    apex=make_policy(tmp_path/"apex","flight","apex_to_descent")
    e1=tmp_path/"apex_support";e2=tmp_path/"descent_support";e1.write_bytes(b"a");e2.write_bytes(b"d")
    registry=StageExpertRegistry.build({"landing":landing,"apex_to_descent":apex,"ascent_to_apex":ascent},{"apex_to_descent":e2,"ascent_to_apex":e1},runtime_source_fingerprint="runtime")
    assert registry.specs["apex_to_descent"].stage=="flight"
    assert registry.specs["ascent_to_apex"].responsibility=="ascent_to_apex"
