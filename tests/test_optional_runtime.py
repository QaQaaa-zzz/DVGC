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


@pytest.mark.skipif(not RUNTIME_READY,reason="MuJoCo runtime required")
def test_multisource_sampler_preserves_origin_phase_and_declared_mass():
    import copy
    import jax
    import numpy as np
    from dvgc.bank import SnapshotBank
    from dvgc.config import STAGE_ID, load_config
    from dvgc.env import OrangeBikeDVGC, RESET_SOURCE

    cfg=load_config("configs/default.json",{"training_stage":"flight","obs_noise_enable":False})
    base=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank())
    record=base.snapshot_record(base.reset(jax.random.PRNGKey(0)),"flight")
    rows=[]
    for source,phase,weight in (
        ("flight_curriculum",STAGE_ID["flight"],.6),
        ("canonical_entry_rehearsal",STAGE_ID["landing"],.1),
        ("landing_tube_rehearsal",STAGE_ID["landing"],.3),
    ):
        row=copy.deepcopy(record); row["id"]=source; row["source_phase"]="landing" if phase==STAGE_ID["landing"] else "flight"
        row["origin_phase"]=row["source_phase"]; row["oracle_phase"]=phase
        row["reset_source"]=source; row["reset_weight"]=weight; rows.append(row)
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(rows,{"reset_source_protocol":{"version":1}}))
    np.testing.assert_allclose(np.exp(np.asarray(env._bank_logw)),[.6,.1,.3],rtol=1e-6)
    assert list(map(int,np.asarray(env._bank_phase)))==[STAGE_ID["flight"],STAGE_ID["landing"],STAGE_ID["landing"]]
    assert list(map(int,np.asarray(env._bank_reset_source)))==[
        RESET_SOURCE[x] for x in ("flight_curriculum","canonical_entry_rehearsal","landing_tube_rehearsal")
    ]


@pytest.mark.skipif(not RUNTIME_READY,reason="MuJoCo runtime required")
def test_landing_snapshot_reward_and_gates_match_in_flight_rehearsal():
    import jax
    import jax.numpy as jp
    import numpy as np
    from dvgc.bank import SnapshotBank
    from dvgc.config import STAGE_ID, load_config
    from dvgc.env import OrangeBikeDVGC

    def make(stage):
        cfg=load_config("configs/default.json",{"training_stage":stage,"use_bank_resets":False,"obs_noise_enable":False,"domain_randomization":False})
        return OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank())
    landing,flight=make("landing"),make("flight")
    source=landing.reset(jax.random.PRNGKey(3)); phase=jp.asarray(STAGE_ID["landing"],jp.int32)
    def restore(env):
        return env.reset_from_snapshot(source.data.qpos,source.data.qvel,source.data.ctrl,jax.random.PRNGKey(4),phase,jp.ones((),jp.int32),jp.ones((),jp.int32),jp.ones((),jp.int32),qacc_warmstart=source.data.qacc_warmstart)
    a=jp.zeros(landing.action_size,jp.float32); ls=landing.step(restore(landing),a); fs=flight.step(restore(flight),a)
    for key in ("reward/total","reward/recovery_shaping","reward/recovery_streak","reward/failure_penalty","reward/instability_penalty"):
        np.testing.assert_allclose(np.asarray(ls.metrics[key]),np.asarray(fs.metrics[key]),rtol=1e-6,atol=1e-6)
    assert int(ls.info["phase"])==int(fs.info["phase"])==STAGE_ID["landing"]
    assert int(ls.info["end_code"])==int(fs.info["end_code"])
    assert int(ls.info["recovery_success"])==int(fs.info["recovery_success"])
