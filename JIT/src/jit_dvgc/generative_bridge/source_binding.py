"""Audited Phase checkpoint binding through the existing zero-training freezer."""
from copy import deepcopy
import json
from pathlib import Path

from .contracts import file_sha
from .protocol import atomic_json


def validate_runtime_contract(source, runtime, downstream):
    from ..config import canonical_sha256
    if runtime['inputs']['up_config_sha256'] != canonical_sha256(source):
        raise ValueError('source upstream configuration differs')
    if runtime.get('reward_mode') != 'original_all_phases':
        raise ValueError('original_all_phases reward required')
    if runtime.get('success_criterion') != 'stable_forward_recovery':
        raise ValueError('stable_forward_recovery endpoint required')
    for field in ('action',):
        if source[field] != downstream[field]:
            raise ValueError(f'up/down {field} differs')
    if source['model']['xml_sha256'] != downstream['model']['xml_sha256']:
        raise ValueError('up/down XML differs')
    recovery = downstream['descent']
    if not recovery['continuous_stability'] or recovery['recovery_ticks'] != 25:
        raise ValueError('25-tick continuous recovery required')


def bind_phase_source(*, checkpoint, source_config, runtime_template, output,
                      expected_source_transitions, name='bridge_baseline'):
    """Freeze unchanged parameters, verify action parity, and record physical identity.

    Runtime template support is a schema dependency only: its labels are not
    imported as v1.2 capabilities. This entry executes no simulation or training.
    """
    import jax
    import jax.numpy as jp
    import numpy as np
    import mujoco
    from types import SimpleNamespace
    from ..constants import SIM_DT, CTRL_DT, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, ACTION_ORDER
    from ..iterative_probe_training import load_phase_initializer
    from ..unified_policy_freeze import freeze_initialization_policy, _checkpoint_identity, _load_policy_formal_config
    from ..checkpoint import load_checkpoint
    from ..handoff_bank import pytree_sha256
    from ..ppo import make_checkpoint_policy
    read = lambda p: json.loads(Path(p).read_text())
    checkpoint, source_config, runtime_template, output = map(lambda p: Path(p).resolve(),
        (checkpoint, source_config, runtime_template, output))
    source, raw = read(source_config), deepcopy(read(runtime_template))
    downstream = read(raw['inputs']['down_config_path'])
    validate_runtime_contract(source, raw, downstream)
    descriptor = dict(schema='jit_phase_checkpoint_initialization_source_v1',
        source_checkpoint=str(checkpoint), source_phase_config=str(source_config),
        input_files={str(p):file_sha(p) for p in (source_config, checkpoint/'identity.json', checkpoint/'payload.pkl')})
    payload = load_phase_initializer(descriptor)
    if payload.training_transitions != expected_source_transitions:
        raise ValueError('source serialized training transitions differ')
    # Resolve the actual immutable model asset used by the historical config.
    xml = Path(source['model']['xml_path'])
    if not xml.is_absolute():
        xml = Path(__file__).resolve().parents[4]/xml
    if file_sha(xml) != payload.identity.xml_sha256:
        raise ValueError('physical XML identity differs')
    from ..model import load_host_model
    from ..config import ActionConfig
    bundle = load_host_model(SimpleNamespace(model=source['model'], action=ActionConfig(**source['action'])))
    model = bundle.mj_model
    if model.opt.timestep != SIM_DT or CTRL_DT != .020 or SIM_DT != .005:
        raise ValueError('physical/control timing differs')
    output.mkdir(parents=True, exist_ok=False)
    descriptor_path = output/'phase_source.json'
    atomic_json(descriptor_path, descriptor)
    raw['initialization'] = dict(actor='warm_start_phase_checkpoint', critic='fresh', optimizer='fresh',
        source_checkpoint=str(checkpoint), source_phase_config=str(source_config), source_frozen_policy=str(descriptor_path))
    raw['input_files'].update(descriptor['input_files'])
    raw['input_files'][str(descriptor_path)] = file_sha(descriptor_path)
    raw['run_declaration']['run_id'] = name
    raw['claim_boundary']['iteration'] = 0
    config_path = output/'runtime_config.json'
    atomic_json(config_path, raw)
    frozen = freeze_initialization_policy(output/'binding', config_path=config_path,
        source_frozen_policy=descriptor_path, name=name)
    rebound = load_checkpoint(Path(frozen['policy']['checkpoint']),
        expected=_checkpoint_identity(_load_policy_formal_config(config_path)))
    env = SimpleNamespace(actor_observation_size=76, privileged_observation_size=106, action_size=4)
    rng = np.random.default_rng(92812)
    observations = {'state':jp.asarray(rng.normal(size=(64,76)), dtype=jp.float32),
                    'privileged_state':jp.asarray(rng.normal(size=(64,106)), dtype=jp.float32)}
    a = np.asarray(make_checkpoint_policy(env,payload)(observations,jax.random.PRNGKey(0))[0])
    b = np.asarray(make_checkpoint_policy(env,rebound)(observations,jax.random.PRNGKey(0))[0])
    if not np.isfinite(a).all() or not np.array_equal(a,b):
        raise ValueError('zero-training action parity failed')
    shapes = lambda value: [dict(shape=list(np.shape(x)),dtype=str(np.asarray(x).dtype)) for x in jax.tree.leaves(value)]
    audit = dict(schema='jit_bridge_phase_source_audit_v1_2', source_checkpoint=str(checkpoint),
        source_config=str(source_config), source_training_transitions=payload.training_transitions,
        new_training_transitions=0, environment_interactions=0, expert_switching_used=False,
        frozen_policy=str(output/'binding/frozen_unified_policy.json'), runtime_config=str(config_path),
        runtime_template=str(runtime_template), runtime_template_sha256=file_sha(runtime_template),
        inherited_support_labels_used=False, baseline_success='not_evaluated',
        parameter_hashes={k:pytree_sha256(v) for k,v in [('actor',payload.actor_params),('critic',payload.critic_params),('normalizer',payload.observation_normalizer)]},
        parameter_shapes={k:shapes(v) for k,v in [('actor',payload.actor_params),('critic',payload.critic_params),('normalizer',payload.observation_normalizer)]},
        observation=dict(frame_fields=list(ACTOR_FRAME_FIELDS),task_fields=list(ACTOR_TASK_FIELDS),actor_dimension=76,privileged_dimension=106),
        action=dict(order=list(ACTION_ORDER),mapping=source['action']),
        physical=dict(xml=str(xml),xml_sha256=file_sha(xml),sim_dt=SIM_DT,ctrl_dt=CTRL_DT,
            body_mass=np.asarray(model.body_mass).tolist(),body_inertia=np.asarray(model.body_inertia).tolist()),
        reward_mode=raw['reward_mode'],success_criterion=raw['success_criterion'],recovery=downstream['descent'],
        phase_physical_limits=dict(upstream=source['physical_limits'],downstream=downstream['physical_limits'],
            explanation='Phase-specific termination configuration retained from the existing complete-task runtime; not a changed XML or reward'),
        action_parity=dict(observations=64,dimension=76,seed=92812,exact=True,max_absolute_difference=float(np.max(np.abs(a-b)))),
        limitations=['CPU inference parity is not full-task success', 'runtime template support labels are not new source evidence'])
    atomic_json(output/'source_audit.json', audit)
    return audit
