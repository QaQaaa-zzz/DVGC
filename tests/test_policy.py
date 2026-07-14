from dvgc.config import load_config
from dvgc.policy import load_bundle, save_bundle

def test_policy_bundle_provenance(tmp_path):
    cfg=load_config("configs/default.json")
    root=save_bundle(tmp_path/"policy",params={"x":1},config=cfg,xml_path="assets/orange_bike_4kg_horizontal.xml",candidate_bank=None,downstream_bank=None,policy_version="test")
    params,stored,manifest=load_bundle(root,verify_files=True)
    assert params["x"]==1
    assert stored["action_mapping_version"]==manifest["action_mapping_version"]
