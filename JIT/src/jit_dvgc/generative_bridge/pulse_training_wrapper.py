"""Opt-in pulse RNG ownership at actual Brax full-reset boundaries.

Logical episode ID = (reset_count << 32) | declared_slot_id. The runner's
single-device/no-host-reset contract keeps this mapping stable. Inference-only
checkpoints do not contain this state and are not exact-training resumes.
"""
import jax
import jax.numpy as jp
from mujoco_playground._src import wrapper
from .pulse_protocol import _uint, _namespace, NAMESPACES


def validate_training_episode_protocol(pulse):
    protocol = pulse.get('logical_episode_rng')
    if protocol is None:
        return None
    if pulse.get('pulse_scope', pulse.get('scope')) != 'full_task_start_only':
        raise ValueError('logical episode RNG requires full_task_start_only')
    if not isinstance(protocol, dict):
        raise ValueError('logical_episode_rng must be an object')
    _uint(protocol['master_seed'], 32, 'master_seed')
    _uint(protocol['round'], 32, 'round')
    if protocol['role'] != 'student_ppo':
        raise ValueError('PPO logical episode role must be student_ppo')
    slots = protocol.get('slot_ids')
    if not isinstance(slots, list) or not slots:
        raise ValueError('explicit PPO slot_ids are required')
    for slot in slots:
        _uint(slot, 32, 'slot_id')
    if len(set(slots)) != len(slots):
        raise ValueError('duplicate PPO slot IDs')
    return protocol


class LogicalEpisodePulseAutoResetWrapper(wrapper.BraxAutoResetWrapper):
    def __init__(self, env, pulse):
        super().__init__(env, full_reset=True)
        self._pulse = pulse
        self._protocol = validate_training_episode_protocol(pulse)
        if self._protocol is None:
            raise ValueError('explicit logical_episode_rng required')
        if jax.process_count() != 1:
            raise ValueError('logical PPO slot mapping currently requires one process')

    def _initialize_pulses(self, state, counters):
        from ..iterative_probe_training import initialize_training_action_pulse
        protocol = self._protocol
        slots = jp.asarray(protocol['slot_ids'], jp.uint32)
        if state.done.shape != slots.shape:
            raise ValueError('declared PPO slot_ids must match all single-device environment lanes')
        def key_for(slot, counter):
            key = jax.random.PRNGKey(protocol['master_seed'])
            for component in (_namespace(protocol['role']), protocol['round'],
                              slot, counter, NAMESPACES['pulse']):
                key = jax.random.fold_in(key, component)
            return key
        keys = jax.vmap(key_for)(slots, counters)
        def initialize(full_start, key):
            return initialize_training_action_pulse({}, key, self._pulse,
                full_task_start=full_start, logical_key=key)
        # The auto-reset info also contains cached MJX/Warp data, whose shared
        # capacity arrays must never be vmapped as if they were environment lanes.
        pulse_info = jax.vmap(initialize)(state.info['reset_from_jump_start'], keys)
        pulse_info.update(training_episode_id_low=slots, training_episode_id_high=counters,
                          training_pulse_key=keys)
        return state.replace(info={**state.info, **pulse_info})

    def reset(self, rng):
        state = super().reset(rng)
        return self._initialize_pulses(state, jp.zeros(state.done.shape, jp.uint32))

    def step(self, state, action):
        prior_counters = state.info['training_episode_id_high']
        result = super().step(state, action)
        counters = prior_counters + result.done.astype(jp.uint32)
        initialized = self._initialize_pulses(result, counters)
        def choose(new, old):
            done = result.done.astype(bool).reshape(result.done.shape + (1,) * (new.ndim-result.done.ndim))
            return jp.where(done, new, old)
        fields = {name: choose(value, result.info[name])
                  for name, value in initialized.info.items()
                  if name.startswith(('training_pulse_', 'training_episode_id_'))}
        return result.replace(info={**result.info, **fields})
