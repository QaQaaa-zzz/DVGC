import importlib.util
from pathlib import Path

import pytest


MODEL_ASSETS = [
    Path("assets/meshes/base_link.STL"),
    Path("assets/meshes/frontwheel.STL"),
    Path("assets/meshes/rearwheel.STL"),
    Path("assets/meshes/steer.STL"),
    Path("assets/meshes/downarm.STL"),
    Path("assets/meshes/uparm.STL"),
]
RUNTIME_READY = importlib.util.find_spec("mujoco_playground") is not None and all(p.is_file() for p in MODEL_ASSETS)


@pytest.mark.skipif(
    not RUNTIME_READY,
    reason="MuJoCo Playground and the user's original STL mesh directory are required for the dynamic smoke test.",
)
def test_environment_smoke():
    import jax

    from dvgc.bank import SnapshotBank
    from dvgc.config import load_config
    from dvgc.env import OrangeBikeDVGC

    cfg = load_config(
        "configs/default.json",
        {
            "training_stage": "landing",
            "use_bank_resets": False,
            "obs_noise_enable": False,
        },
    )
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs["state"].shape[-1] > 0


@pytest.mark.skipif(
    not RUNTIME_READY,
    reason="MuJoCo Playground and the user's original STL mesh directory are required for the dynamic smoke test.",
)
def test_final_safe_tube_reset_smoke():
    import jax
    import numpy as np

    from dvgc.bank import SnapshotBank
    from dvgc.config import STAGE_ID, load_config
    from dvgc.env import OrangeBikeDVGC

    base_cfg = load_config(
        "configs/default.json",
        {"training_stage": "landing", "use_bank_resets": False, "obs_noise_enable": False},
    )
    base_env = OrangeBikeDVGC(base_cfg, snapshot_bank=SnapshotBank())
    record = base_env.snapshot_record(base_env.reset(jax.random.PRNGKey(0)), "landing")
    record["oracle_phase"] = STAGE_ID["landing"]
    record["policy_state"]["filter_phase"] = STAGE_ID["landing"]
    record["policy_state"]["phase_probs"] = np.eye(4, dtype=np.float32)[STAGE_ID["landing"]]
    bank = SnapshotBank([record])
    bank.records[0]["final"]["label"] = "safe"
    rsi_cfg = load_config(
        "configs/default.json",
        {
            "training_stage": "landing",
            "use_bank_resets": True,
            "obs_noise_enable": False,
            "tube_activation_min_safe": 1,
        },
    )
    env = OrangeBikeDVGC(rsi_cfg, snapshot_bank=bank)
    state = env.reset(jax.random.PRNGKey(1))
    assert int(state.info["phase"]) == STAGE_ID["landing"]


@pytest.mark.skipif(
    not RUNTIME_READY,
    reason="The configured Brax runtime is required for the policy-network test.",
)
def test_policy_network_starts_neutral_and_low_variance():
    import jax
    import jax.numpy as jp
    import numpy as np

    from dvgc.runtime import POLICY_INITIAL_ACTION_STD, make_dvgc_ppo_networks

    networks=make_dvgc_ppo_networks(
        observation_size={"state":(140,),"privileged_state":(29,)},
        action_size=4,
        preprocess_observations_fn=lambda obs,_params:obs,
    )
    params=networks.policy_network.init(jax.random.PRNGKey(0))
    logits=networks.policy_network.apply(
        None,params,{"state":jp.ones((3,140),jp.float32)}
    )
    dist=networks.parametric_action_distribution.create_dist(logits)
    np.testing.assert_allclose(np.asarray(dist.loc),0.0,atol=0.0)
    np.testing.assert_allclose(
        np.asarray(dist.scale),POLICY_INITIAL_ACTION_STD,rtol=1e-6,atol=1e-7
    )
