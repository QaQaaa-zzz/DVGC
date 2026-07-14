from dvgc.model import inspect_model


def test_original_model_contract():
    model = inspect_model("assets/orange_bike_2kg_horizontal.xml")
    assert model["xml_path"].endswith("orange_bike_2kg_horizontal.xml")
    assert abs(model["step"]["front_x"] - 3.6) < 1e-9
    assert abs(model["step"]["top_z"] - 0.16) < 1e-9
    names = [a["name"] for a in model["actuators"]]
    assert names == ["cmd_steering_v", "cmd_rearwheel_f", "cmd_hip_f", "cmd_knee_f"]
    knee = next(j for j in model["joints"] if j["name"] == "knee_joint")
    assert knee["range"] == "-1.5 2.5"
