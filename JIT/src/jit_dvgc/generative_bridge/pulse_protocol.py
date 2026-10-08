"""Opt-in logical-episode pulse requests; no simulator or training side effects."""
import hashlib
import numpy as np

# Fixed namespaces are part of the protocol, never Python's randomized hash().
NAMESPACES = {'train': 0x54524149, 'student_ppo': 0x5350504F, 'student_eval': 0x53504556, 'student_dev': 0x53544456,
              'generator_dev': 0x47454456, 'solver_dev': 0x534F4456,
              'test': 0x54455354, 'pulse': 0x50554C53, 'policy': 0x504F4C49,
              'generator': 0x47454E45, 'minibatch': 0x4D494E49,
              'split': 0x53504C49, 'condition': 0x434F4E44, 'reset': 0x52534554}


def _uint(value, bits, name):
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ValueError(f'{name} must be an unsigned {bits}-bit integer')
    return value


def _namespace(value):
    if isinstance(value, str):
        if value not in NAMESPACES:
            raise ValueError('unknown RNG namespace: ' + value)
        return NAMESPACES[value]
    return _uint(value, 32, 'namespace')


def logical_episode_key(master_seed, role, round_index, episode_id, purpose):
    """Installed JAX RNG; fold master/role/round/low32/high32/purpose in order."""
    import jax
    key = jax.random.PRNGKey(_uint(master_seed, 32, 'master_seed'))
    episode_id = _uint(episode_id, 64, 'episode_id')
    for component in (_namespace(role), _uint(round_index, 32, 'round'),
                      episode_id & 0xffffffff, episode_id >> 32, _namespace(purpose)):
        key = jax.random.fold_in(key, component)
    return key


def requested_draws(key, amplitude=1.):
    import jax
    return jax.random.uniform(key, (3, 4), minval=-1., maxval=1.) * amplitude


class EpisodeKeyRegistry:
    """Host-side reservation guard; share one registry across collection shards."""
    def __init__(self):
        self._seen = set()

    def claim(self, master_seed, role, round_index, episode_id, purpose):
        key = logical_episode_key(master_seed, role, round_index, episode_id, purpose)
        identity = np.asarray(key).tobytes()
        if identity in self._seen:
            raise ValueError('duplicate logical episode key')
        self._seen.add(identity)
        return key


def validate_ancestor_splits(splits):
    owners = {}
    for role, ancestors in splits.items():
        for ancestor in ancestors:
            if not isinstance(ancestor, str) or not ancestor:
                raise ValueError('nonempty ancestor identity required')
            if ancestor in owners and owners[ancestor] != role:
                raise ValueError('ancestor crosses data splits: ' + ancestor)
            owners[ancestor] = role


def collection_draws(spec, *, registry=None):
    """Validate before simulation; returns lane x 3 x 4 normalized requests + receipt."""
    import jax
    protocol = spec.get('pulse_protocol_v1_2')
    if protocol is None:
        return None, None
    if (spec.get('controller_mode') != 'fixed_random' or spec.get('pulse_steps') != 3
            or spec.get('pulse_event_schedule') or spec.get('initial_velocity_randomization')):
        raise ValueError('logical pulse protocol requires fixed_random three-step action-only pulses')
    ids = protocol['episode_ids']
    if len(ids) != spec['num_envs'] or not ids:
        raise ValueError('logical episode IDs must match physical lanes')
    if protocol['role'] == 'test':
        raise ValueError('final TEST keys remain unopened in this stage')
    registry = registry if registry is not None else EpisodeKeyRegistry()
    draws, records = [], []
    for episode_id in ids:
        key = registry.claim(protocol['master_seed'], protocol['role'], protocol['round'], episode_id, 'pulse')
        draw = np.asarray(requested_draws(key), dtype=np.float32)
        draws.append(draw)
        records.append(dict(episode_id=episode_id, key=np.asarray(key).tolist(),
                            key_sha256=hashlib.sha256(np.asarray(key).tobytes()).hexdigest(),
                            draw_sha256=hashlib.sha256(draw.tobytes()).hexdigest()))
    return np.stack(draws), dict(schema='jit_logical_episode_pulse_v1_2',
        master_seed=protocol['master_seed'], role=protocol['role'], round=protocol['round'],
        purpose='pulse', rng_impl=str(jax.config.jax_default_prng_impl), draw_shape=[3, 4],
        requested_pairing_only=True, episodes=records)


def pulse_delta_for_step(draws, applied_steps, active):
    """Select each lane's fixed request, suppressing inactive/finished episodes."""
    import jax.numpy as jp
    selected = draws[jp.arange(draws.shape[0]), jp.clip(applied_steps, 0, 2)]
    return jp.where((active & (applied_steps < 3))[:, None], selected, 0.)
