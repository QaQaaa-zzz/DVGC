"""Regression coverage for Warp's shared scratch at curriculum autoreset."""
import jax
import jax.numpy as jp
import numpy as np
from flax import struct
from JIT.tests.test_skill_transfer_curriculum import _mock_reset


def test_warp_shared_scratch_survives_vmap_autoreset_scan():
    from mujoco.mjx.warp.types import DataWarp, DATA_NON_VMAP
    from jit_dvgc.recovery_curriculum import wrap_recovery_for_training
    @struct.dataclass
    class Data:
        qpos: object
        _impl: object
    reset = _mock_reset()
    values = {f.name: (4096 if f.name in ('naconmax','naccdmax','njmax') else 2 if f.name == 'nworld'
                      else jp.zeros(4096, jp.int32) if f.name in DATA_NON_VMAP
                      else jp.zeros((2,1), jp.float32)) for f in DataWarp.fields()}
    reset.pool = reset.pool.replace(data=Data(jp.array([[7.],[9.]]), DataWarp(**values)))
    def physical_step(state, action):
        return state.replace(data=state.data.replace(qpos=state.data.qpos+1),
            reward=jp.array(1.), done=jp.array(0.),
            info={**state.info, 'episode_step':state.info['episode_step']+1})
    reset.original_step = physical_step
    env = wrap_recovery_for_training(reset.env)
    with jax.checking_leaks():
        state = jax.jit(env.reset)(jax.random.split(jax.random.key(0), 2))
        def run(state):
            def tick(state, _):
                state = env.step(state, jp.zeros((2,1)))
                return state, state.done
            return jax.lax.scan(tick, state, None, length=3)
        result, done = jax.jit(run)(state)
    np.testing.assert_array_equal(done, [[0,0],[1,1],[0,0]])
    np.testing.assert_array_equal(result.data.qpos, [[8.],[8.]])
    assert result.data._impl.contact__dim.shape == (4096,)
