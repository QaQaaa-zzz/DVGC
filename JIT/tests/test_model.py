from __future__ import annotations

import numpy as np
import pytest

from jit_dvgc.config import load_config
from jit_dvgc.constants import EXPECTED_XML_SHA256
from jit_dvgc.model import load_host_model


def test_authoritative_model_contract_is_audited_before_conversion(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_smoke.json")
    bundle = load_host_model(config)

    assert bundle.xml_sha256 == EXPECTED_XML_SHA256
    assert bundle.mj_model.opt.timestep == pytest.approx(0.005)
    assert bundle.actuator_names == (
        "cmd_steering_v",
        "cmd_rearwheel_f",
        "cmd_hip_f",
        "cmd_knee_f",
    )
    np.testing.assert_array_equal(
        bundle.mj_model.actuator_forcerange[2:],
        np.array([[-50.0, 50.0], [-50.0, 50.0]]),
    )
    assert bundle.payload_mass == pytest.approx(2.0)
    assert bundle.model_index.keyframe_id == 0
    assert bundle.model_index.knee_qpos_address == 11


def test_config_loader_rejects_a_wrong_declared_model_identity(jit_root, tmp_path):
    source = jit_root / "configs" / "phase_u_smoke.json"
    payload = source.read_text().replace(EXPECTED_XML_SHA256, "0" * 64)
    config_path = tmp_path / "wrong_identity.json"
    config_path.write_text(payload)
    with pytest.raises(ValueError, match="approved v2 model"):
        load_config(config_path)
