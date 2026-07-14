from pathlib import Path

def test_key_fixes_present():
    s=Path("dvgc/env.py").read_text()
    assert 'data, action, phase_probs1, had_landing' in s
    assert 'chain_ever = (state.info["chain_ever"] > 0) | chain' in s
    assert 'feature_z = (feature - self._safe_center) / self._safe_scale' in s
    assert 'truncated = timeout & (~terminated)' in s
    assert 'takeoff_total = jp.where(takeoff_local' in s
    assert 'self._bank_phase_probs[idx]' in s


def test_original_xml_is_the_only_model_path():
    env = Path("dvgc/env.py").read_text()
    cfg = Path("dvgc/config.py").read_text()
    assert "MjModel.from_xml_path(self._xml_path)" in env
    assert "orange_bike_4kg_horizontal.xml" in cfg
    assert "orange_bike_runtime.xml" not in env + cfg
    assert "prepare_runtime_xml" not in Path("dvgc/model.py").read_text()


def test_incremental_knee_mapping_contract():
    source = Path("dvgc/env.py").read_text()
    assert "q_target = clip(q_current - action_3 * delta_q" in source
    assert "knee_action_target_delta" in source


def test_candidate_rng_and_incremental_training_log_contracts():
    candidates = Path("cli/build_candidates.py").read_text()
    training = Path("cli/train.py").read_text()
    assert "rng.uniform(0.35,0.85)" in candidates
    assert "np.random.uniform" not in candidates
    assert 'metric_log["status"]="running"' in training
    assert '"status":"failed"' in training
