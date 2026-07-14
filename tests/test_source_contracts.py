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


def test_formal_pipeline_contains_tube_rsi_refinement_and_recertification():
    script = Path("scripts/run_backward_bootstrap.sh").read_text()
    assert "_bootstrap_tube.pkl" in script
    assert "--require-final-safe-rsi" in script
    assert script.count("-m cli.certify") == 2
    assert "600000 400000" in script
    assert "cli.runtime_gate" in script and "--check-only" in script


def test_metric_contract_uses_synchronized_jit_warp_path():
    runtime = Path("dvgc/runtime.py").read_text()
    assert "next_state = jax.jit(env.step)" in runtime
    assert "jax.block_until_ready(next_state)" in runtime


def test_runtime_gate_has_bounded_warp_replay_tolerances_and_exact_semantics():
    gate = Path("cli/runtime_gate.py").read_text()
    assert "GATE_VERSION = 3" in gate
    assert '"qacc_warmstart": 1e-5' in gate
    assert '"qvel": 2e-3' in gate
    assert '"reward": 1e-5' in gate
    assert '"actor_obs": 2e-2' in gate
    assert "SNAPSHOT_DISCRETE_FIELDS" in gate
    assert "np.array_equal" in gate


def test_remaining_pipeline_is_resumable_and_stage_complete():
    controller = Path("scripts/run_remaining_pipeline.sh").read_text()
    assert "cli.pipeline_marker check" in controller
    assert "certify_chunked" in controller and "audit_chunked" in controller
    assert controller.index("run_stage flight") < controller.index("run_stage takeoff") < controller.index("run_stage approach")
    assert "natural_start_seed0" in controller
    env = Path("dvgc/env.py").read_text()
    analysis = Path("cli/analyze_training.py").read_text()
    for phase in ("approach", "takeoff", "flight", "landing"):
        assert f'"phase/{phase}"' in env
    assert 'eval/episode_reward/phase/{phase}' in analysis
