import pytest

from dvgc.config import load_config
from dvgc.policy import load_bundle, save_bundle, verify_manifest_artifact

def test_policy_bundle_provenance(tmp_path):
    cfg=load_config("configs/default.json")
    root=save_bundle(tmp_path/"policy",params={"x":1},config=cfg,xml_path="assets/orange_bike_4kg_horizontal.xml",candidate_bank=None,downstream_bank=None,policy_version="test")
    params,stored,manifest=load_bundle(root,verify_files=True)
    assert params["x"]==1
    assert stored["action_mapping_version"]==manifest["action_mapping_version"]


def test_manifest_artifact_verification_rejects_a_different_bank(tmp_path):
    declared=tmp_path/"declared.pkl"; declared.write_bytes(b"declared")
    other=tmp_path/"other.pkl"; other.write_bytes(b"other")
    cfg=load_config()
    root=save_bundle(tmp_path/"policy",params={"x":1},config=cfg,xml_path="assets/orange_bike_4kg_horizontal.xml",candidate_bank=declared,downstream_bank=None,policy_version="test")
    _,_,manifest=load_bundle(root)
    verify_manifest_artifact(manifest,"candidate_bank",declared)
    verify_manifest_artifact(manifest,"downstream_bank",None,required=False)
    with pytest.raises(ValueError,match="candidate_bank mismatch"):
        verify_manifest_artifact(manifest,"candidate_bank",other)
