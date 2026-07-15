import json
from pathlib import Path
import pytest

from dvgc.config import load_config
from dvgc.experts import StageExpertRegistry
from dvgc.policy import copy_bundle,save_bundle


def make_policy(path,stage):
    cfg=load_config("configs/default.json",{"training_stage":stage})
    return save_bundle(path,params={"stage":stage},config=cfg,xml_path=cfg.xml_path,candidate_bank=None,downstream_bank=None,policy_version=f"pi-{stage}",extra={"stage":stage})


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
