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
