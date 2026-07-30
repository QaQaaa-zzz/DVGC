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


@pytest.mark.skipif(not RUNTIME_READY,reason="MuJoCo Playground runtime required")
@pytest.mark.parametrize("stage,objective",[("takeoff","takeoff_to_ascent"),("flight","ascent_to_apex"),("flight","apex_to_descent")])
def test_stage_reachability_reward_step_and_snapshot_latches(stage,objective):
    import jax
    import jax.numpy as jp
    import numpy as np
    from dvgc.bank import SnapshotBank
    from dvgc.config import load_config
    from dvgc.env import OrangeBikeDVGC
    from dvgc.rollout import restore_snapshot
    cfg=load_config("configs/default.json",{"training_stage":stage,"stage_reachability_objective":objective,"use_bank_resets":False,"obs_noise_enable":False,"domain_randomization":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank());state=env.reset(jax.random.PRNGKey(710))
    next_state=env.step(state,jp.zeros(env.action_size,jp.float32))
    for key in ("reward/stage_entry_total","reward/stage_entry_shaping","reward/stage_entry_event","reward/stage_entry_failure_penalty"):
        assert np.isfinite(np.asarray(next_state.metrics[key])).all()
    record=env.snapshot_record(next_state,stage);restored=restore_snapshot(env,record,jax.random.PRNGKey(711))
    assert int(restored.info["stage_entry_ever"])==int(next_state.info["stage_entry_ever"])
    assert bool(restored.info["jump_signal_latched"])==bool(next_state.info["jump_signal_latched"])
    assert np.isclose(float(restored.info["jump_window_end_x"]),float(next_state.info["jump_window_end_x"]))


@pytest.mark.skipif(not RUNTIME_READY, reason="MuJoCo runtime required")
def test_phase_balanced_rsi_selects_per_reset_objective_without_actor_leakage():
    import copy
    import jax
    import jax.numpy as jp
    import numpy as np

    from dvgc.bank import SnapshotBank
    from dvgc.config import load_config
    from dvgc.env import OrangeBikeDVGC, PHASE_RSI_OBJECTIVE

    base_cfg = load_config("configs/default.json", {
        "training_stage": "flight", "use_bank_resets": False,
        "obs_noise_enable": False, "domain_randomization": False,
    })
    base = OrangeBikeDVGC(base_cfg, snapshot_bank=SnapshotBank())
    record = base.snapshot_record(base.reset(jax.random.PRNGKey(812)), "flight")

    rows = []
    for stage in ("takeoff", "ascent", "apex", "descent", "landing"):
        row = copy.deepcopy(record)
        row["id"] = f"phase-{stage}"
        row["phase_rsi_stage"] = stage
        row["reset_source"] = "flight_curriculum"
        row["reset_weight"] = .2
        rows.append(row)
    bank = SnapshotBank(rows, {"reset_source_protocol": {"version": 1}})
    cfg = load_config("configs/default.json", {
        "training_stage": "flight", "stage_reachability_objective": "phase_balanced_rsi",
        "use_bank_resets": True, "natural_prob_flight": 0.0,
        "obs_noise_enable": False, "domain_randomization": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=bank)
    assert list(map(int, np.asarray(env._bank_reachability_objective))) == [
        PHASE_RSI_OBJECTIVE[stage]
        for stage in ("takeoff", "ascent", "apex", "descent", "landing")
    ]

    def one(stage):
        row = copy.deepcopy(record)
        row["phase_rsi_stage"] = stage
        row["reset_source"] = "flight_curriculum"
        row["reset_weight"] = 1.0
        single = SnapshotBank([row], {"reset_source_protocol": {"version": 1}})
        return OrangeBikeDVGC(cfg, snapshot_bank=single)

    takeoff_env, landing_env = one("takeoff"), one("landing")
    key = jax.random.PRNGKey(813)
    takeoff_state, landing_state = takeoff_env.reset(key), landing_env.reset(key)
    assert int(takeoff_state.info["reachability_objective_id"]) == PHASE_RSI_OBJECTIVE["takeoff"]
    assert int(landing_state.info["reachability_objective_id"]) == PHASE_RSI_OBJECTIVE["landing"]
    np.testing.assert_array_equal(takeoff_state.obs["state"], landing_state.obs["state"])

    next_state = takeoff_env.step(
        takeoff_state, jp.zeros(takeoff_env.action_size, jp.float32)
    )
    stage_total = float(np.asarray(next_state.metrics["reward/stage_entry_total"]))
    assert np.isfinite(stage_total)
    assert stage_total != 0.0
    assert float(np.asarray(next_state.metrics[
        "reset/transition/objective/takeoff_to_ascent"
    ])) == 1.0


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


@pytest.mark.skipif(not RUNTIME_READY, reason="The configured Brax runtime is required.")
def test_frozen_normalizer_training_state_has_zero_mean_std_drift():
    import jax.numpy as jp
    import numpy as np
    from brax.training.acme import running_statistics

    from dvgc.runtime import (
        frozen_normalizer_training_params,
        normalizer_max_abs_difference,
    )

    state = running_statistics.init_state(jp.zeros((3,), jp.float32), std_eps=1e-6)
    state = running_statistics.update(
        state,
        jp.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]], jp.float32),
    )
    actor, critic = {"actor": jp.ones((1,))}, {"critic": jp.ones((1,))}
    frozen, returned_actor, returned_critic = frozen_normalizer_training_params(
        (state, actor, critic)
    )
    updated = running_statistics.update(
        frozen,
        jp.asarray([[100.0, -200.0, 300.0]], jp.float32),
        until_count=0,
    )
    drift = normalizer_max_abs_difference(state, updated)
    assert drift["mean"] == 0.0
    assert drift["std"] <= 1e-6
    assert returned_actor is actor and returned_critic is critic
    np.testing.assert_allclose(np.asarray(updated.mean), np.asarray(state.mean), atol=0.0)


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


@pytest.mark.skipif(not RUNTIME_READY,reason="MuJoCo runtime required")
def test_full_reset_resamples_bank_without_losing_terminal_source_metrics():
    import copy
    import jax
    import jax.numpy as jp
    import numpy as np
    from dvgc.bank import SnapshotBank
    from dvgc.config import STAGE_ID,load_config
    from dvgc.env import OrangeBikeDVGC,RESET_SOURCE
    from dvgc.wrappers import wrap_for_training

    cfg=load_config("configs/default.json",{"training_stage":"flight","obs_noise_enable":False})
    base=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank())
    record=base.snapshot_record(base.reset(jax.random.PRNGKey(0)),"flight"); rows=[]
    for source,phase,weight in (("flight_curriculum",STAGE_ID["flight"],.6),("canonical_entry_rehearsal",STAGE_ID["landing"],.1),("landing_tube_rehearsal",STAGE_ID["landing"],.3)):
        row=copy.deepcopy(record); row["id"]="full-reset-"+source; row["reset_source"]=source; row["reset_weight"]=weight; row["oracle_phase"]=phase; rows.append(row)
    raw=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(rows,{"reset_source_protocol":{"version":1}}))
    env=wrap_for_training(raw,episode_length=1); keys=jax.random.split(jax.random.PRNGKey(91),64)
    state=env.reset(keys); original=np.asarray(state.info["reset_source"]); state=env.step(state,jp.zeros((64,raw.action_size),jp.float32))
    assert np.all(np.asarray(state.info["episode_done"])==1)
    episode_total=sum(float(np.asarray(state.info["episode_metrics"][f"reset/episode/{name}"]).sum()) for name in RESET_SOURCE)
    transition_total=sum(float(np.asarray(state.info["episode_metrics"][f"reset/transition/{name}"]).sum()) for name in RESET_SOURCE)
    assert episode_total==64.0 and transition_total==64.0
    assert np.any(np.asarray(state.info["reset_source"])!=original)


@pytest.mark.skipif(not RUNTIME_READY,reason="MuJoCo runtime required")
def test_descent_local_reset_labels_are_auditable_but_not_actor_inputs():
    import jax
    import jax.numpy as jp
    import numpy as np
    from dvgc.bank import SnapshotBank
    from dvgc.config import STAGE_ID,load_config
    from dvgc.env import OrangeBikeDVGC

    cfg=load_config("configs/default.json",{"training_stage":"flight","use_bank_resets":False,"obs_noise_enable":False,"domain_randomization":False,"descent_local_reward_enable":True})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load("artifacts/landing_entry_tube_v2.pkl"))
    natural=env.reset(jax.random.PRNGKey(41)); phase=jp.asarray(STAGE_ID["flight"],jp.int32); common=dict(qacc_warmstart=natural.data.qacc_warmstart,reset_source=jp.asarray(0,jp.int32),descent_layer=jp.asarray(0,jp.int32),reset_parent=jp.asarray(0,jp.int32))
    safe=env.reset_from_snapshot(natural.data.qpos,natural.data.qvel,natural.data.ctrl,jax.random.PRNGKey(42),phase,jp.ones((),jp.int32),jp.zeros((),jp.int32),jp.zeros((),jp.int32),bootstrap_group=jp.asarray(0,jp.int32),**common)
    boundary=env.reset_from_snapshot(natural.data.qpos,natural.data.qvel,natural.data.ctrl,jax.random.PRNGKey(42),phase,jp.ones((),jp.int32),jp.zeros((),jp.int32),jp.zeros((),jp.int32),bootstrap_group=jp.asarray(1,jp.int32),**common)
    np.testing.assert_array_equal(np.asarray(safe.obs["state"]),np.asarray(boundary.obs["state"]))
    action=jp.zeros(env.action_size,jp.float32); stepped=env.step(safe,action)
    assert float(stepped.metrics["reset/transition/group/provisional_safe"])==1.0
    assert float(stepped.metrics["reset/transition/layer/late"])==1.0
    assert np.isfinite(float(stepped.metrics["reward/descent_local_shaping"]))


@pytest.mark.skipif(not RUNTIME_READY,reason="MuJoCo runtime required")
def test_nonfinite_action_is_explicit_finite_terminal_transition():
    import jax
    import jax.numpy as jp
    import numpy as np
    from dvgc.bank import SnapshotBank
    from dvgc.config import load_config
    from dvgc.env import END_NONFINITE,OrangeBikeDVGC

    cfg=load_config("configs/default.json",{"training_stage":"flight","use_bank_resets":False,"obs_noise_enable":False,"domain_randomization":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank()); state=env.reset(jax.random.PRNGKey(55))
    terminal=env.step(state,jp.full((env.action_size,),jp.nan))
    assert int(terminal.info["end_code"])==END_NONFINITE and int(terminal.info["terminated"])==1
    assert float(terminal.metrics["diag/nonfinite_transition"])==1.0
    assert np.isfinite(np.asarray(terminal.data.qpos)).all() and np.isfinite(np.asarray(terminal.data.qvel)).all()
    assert np.isfinite(np.asarray(terminal.obs["state"])).all() and np.isfinite(float(terminal.reward))


@pytest.mark.skipif(not RUNTIME_READY,reason="MuJoCo runtime required")
def test_composite_handoff_preserves_physics_and_policy_state():
    import jax
    import jax.numpy as jp
    import numpy as np
    from dvgc.bank import SnapshotBank
    from dvgc.composite import CompositeSession
    from dvgc.config import load_config
    from dvgc.env import OrangeBikeDVGC

    class Always:
        def match(self,state): return True,0.0
    def zero(obs,key): return jp.zeros(4,jp.float32),{}
    cfg=load_config("configs/default.json",{"training_stage":"flight","domain_randomization":False,"obs_noise_enable":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank()); key=jax.random.PRNGKey(123); initial=env.reset(key); step=jax.jit(env.step); action=jp.zeros(4,jp.float32)
    direct=step(initial,action); jax.block_until_ready(direct)
    session=CompositeSession(env,("flight","landing"),{"flight":zero,"landing":zero},{"flight":Always()},initial,key)
    handed=session.step(step_fn=step); jax.block_until_ready(handed)
    np.testing.assert_allclose(np.asarray(handed.data.qpos),np.asarray(direct.data.qpos),atol=5e-5)
    np.testing.assert_allclose(np.asarray(handed.data.qvel),np.asarray(direct.data.qvel),atol=3e-3)
    np.testing.assert_allclose(np.asarray(handed.info["obs_history"]),np.asarray(direct.info["obs_history"]),atol=1e-4)
    np.testing.assert_allclose(np.asarray(handed.info["last_action"]),np.asarray(direct.info["last_action"]),atol=1e-6)
    assert session.active_stage=="landing" and len(session.handoffs)==1


@pytest.mark.skipif(not RUNTIME_READY,reason="MuJoCo runtime required")
def test_expert_chain_entry_is_successful_nonphysical_termination(tmp_path):
    import copy
    import jax
    import jax.numpy as jp
    import numpy as np
    from dvgc.bank import SnapshotBank
    from dvgc.config import load_config
    from dvgc.env import END_CHAIN_ENTRY,OrangeBikeDVGC
    from dvgc.rollout import restore_snapshot

    source=SnapshotBank.load("artifacts/landing_entry_tube_v2.pkl"); bank=copy.deepcopy(source); bank.metadata["entry_matcher"]["radius"]=1e6; path=tmp_path/"entry.pkl"; bank.save(path)
    cfg=load_config("configs/default.json",{"training_stage":"flight","expert_chain_termination":True,"domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(path)); row=next(r for r in bank.records if r["final"]["label"]=="safe"); state=env.step(restore_snapshot(env,row,jax.random.PRNGKey(8)),jp.zeros(4,jp.float32))
    assert int(np.asarray(state.info["chain_success"]))==1
    assert int(np.asarray(state.done))==1 and int(np.asarray(state.info["terminated"]))==1 and int(np.asarray(state.info["truncated"]))==0
    assert int(np.asarray(state.info["end_code"]))==END_CHAIN_ENTRY and int(np.asarray(state.info["recovery_success"]))==0
