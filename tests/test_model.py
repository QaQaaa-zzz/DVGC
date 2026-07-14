from dvgc.model import inspect_model


def test_original_model_contract():
    model = inspect_model("assets/orange_bike_4kg_horizontal.xml")
    assert model["xml_path"].endswith("orange_bike_4kg_horizontal.xml")
    assert abs(model["step"]["front_x"] - 3.6) < 1e-9
    assert abs(model["step"]["top_z"] - 0.16) < 1e-9
    assert model["named_masses_kg"]["load"] == 4.0
    assert all(asset["exists_in_current_copy"] for asset in model["mesh_assets"])
    names = [a["name"] for a in model["actuators"]]
    assert names == ["cmd_steering_v", "cmd_rearwheel_f", "cmd_hip_f", "cmd_knee_f"]
    knee = next(j for j in model["joints"] if j["name"] == "knee_joint")
    assert knee["range"] == "-1.5 2.5"
    force_ranges = {a["name"]: a["forcerange"] for a in model["actuators"]}
    assert force_ranges["cmd_hip_f"] == "-50 50"
    assert force_ranges["cmd_knee_f"] == "-50 50"
