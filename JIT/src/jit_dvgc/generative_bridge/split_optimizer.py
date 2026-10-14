"""Opt-in, source-locked recovery PPO with independent atomic Adam branches."""
import ast
import hashlib
import inspect
from pathlib import Path
import pickle
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
import optax
from optax._src.combine import chain as unguarded_chain
from ..ppo_numerics import _finite, finite_optimizer_update

PINNED_BRAX_TRAIN_SHA256 = '32ed2754f3f9b545c7f381aba9953d1160ca82cc0d01deb59d1f5847872eea32'


class SplitOptimizerState(NamedTuple):
    actor: object
    critic: object


def split_optimizer(*, actor_learning_rate=3e-4, critic_learning_rate=3e-4,
                    actor_clip=1., critic_clip=1., run_dir=None):
    """Clip each branch, then accept the whole proposal or retain both states.

    The captured native chain deliberately avoids guard_ppo_updates' patched
    optax.chain: independently guarded branches could produce half an update.
    Its loss guard remains in place; the outer finite guard covers both branches.
    """
    if any(not np.isfinite(x) or x <= 0 for x in
           (actor_learning_rate, critic_learning_rate, actor_clip, critic_clip)):
        raise ValueError('finite positive split optimizer settings required')
    actor = unguarded_chain(optax.clip_by_global_norm(actor_clip), optax.adam(actor_learning_rate))
    critic = unguarded_chain(optax.clip_by_global_norm(critic_clip), optax.adam(critic_learning_rate))

    def init(params):
        return SplitOptimizerState(actor.init(params.policy), critic.init(params.value))

    def proposal(grads, state, params):
        a, sa = actor.update(grads.policy, state.actor, params.policy)
        c, sc = critic.update(grads.value, state.critic, params.value)
        return grads.replace(policy=a, value=c), SplitOptimizerState(sa, sc)

    raw = optax.GradientTransformation(init, proposal)

    def update(grads, state, params=None, *, batch=None):
        if params is None:
            raise ValueError('atomic split optimizer requires current PPO parameters')
        updates, next_state, accepted = finite_optimizer_update(raw, grads, state, params)
        if run_dir is not None:
            def capture(values):
                root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
                path = root/'nonfinite_split_optimizer.pkl'
                try:
                    with path.open('xb') as stream:
                        pickle.dump(jax.device_get(values), stream)
                except FileExistsError:
                    pass
                raise FloatingPointError('nonfinite split PPO proposal rejected atomically')
            def fail(_):
                jax.debug.callback(capture, (grads, state, params, batch))
                return jp.asarray(0)
            # Nonfinite gradients already trigger the original full batch guard.
            jax.lax.cond(_finite(grads) & ~accepted, fail, lambda _: jp.asarray(0), None)
        return updates, next_state
    return optax.GradientTransformation(init, update)


def instrument_split_optimizer(trainer, *, actor_learning_rate=3e-4,
                               critic_learning_rate=3e-4, actor_clip=1.,
                               critic_clip=1., run_dir=None,
                               expected_sha256=PINNED_BRAX_TRAIN_SHA256):
    """Replace the unique optimizer block and pass current params to its update."""
    from brax.training.agents.ppo import train as installed
    actual = hashlib.sha256(Path(inspect.getsourcefile(installed)).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f'unsupported installed Brax train source SHA: {actual}; expected {expected_sha256}')
    source = getattr(trainer, '_jit_source_text', None) or inspect.getsource(trainer)
    tree = ast.parse(source); counts = [0, 0]
    class Hooks(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            if node.name != trainer.__name__:
                return node
            starts = [i for i, x in enumerate(node.body) if isinstance(x, ast.Assign)
                      and len(x.targets) == 1 and ast.unparse(x.targets[0]) == 'base_optimizer'
                      and ast.unparse(x.value) == 'optax.adam(learning_rate=learning_rate)']
            ends = [i for i, x in enumerate(node.body) if isinstance(x, ast.If)
                    and ast.unparse(x.test) == 'max_grad_norm is not None'
                    and x.orelse and ast.unparse(x.orelse[0]) == 'optimizer = base_optimizer']
            if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
                raise ValueError('unsupported upstream PPO optimizer construction')
            replacement = ast.parse('lr_is_adaptive_kl = False\noptimizer = _recovery_split_optimizer()').body
            node.body[starts[0]:ends[0]+1] = replacement
            counts[0] += 1
            return node
        def visit_Call(self, node):
            if ast.unparse(node) == 'optimizer.update(grads, optimizer_state)':
                counts[1] += 1
                node.args.append(ast.Name(id='params', ctx=ast.Load()))
                node.keywords.append(ast.keyword(arg='batch', value=ast.Name(id='data', ctx=ast.Load())))
            return self.generic_visit(node)
    tree = Hooks().visit(tree)
    if counts != [1, 1]:
        raise ValueError('unsupported upstream PPO split optimizer layout')
    factory = lambda: split_optimizer(actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate, actor_clip=actor_clip,
        critic_clip=critic_clip, run_dir=run_dir)
    namespace = {**trainer.__globals__, '_recovery_split_optimizer': factory}
    exec(compile(ast.fix_missing_locations(tree), '<source locked split PPO optimizer>', 'exec'), namespace)
    adapted = namespace[trainer.__name__]
    adapted._jit_source_text = ast.unparse(tree)
    def wrapped(**kwargs):
        schedule = kwargs.get('learning_rate_schedule')
        if schedule not in (None, 'none'):
            raise ValueError('split PPO only supports the locked constant learning rate schedule')
        return adapted(**kwargs)
    wrapped._jit_source_text = adapted._jit_source_text
    wrapped._split_adapted_trainer = adapted
    return wrapped


def make_recovery_trainer(trainer, hooks, demo_manifest, contract, probe, sink, *,
                          transitions, run_dir=None, demo_coefficient_start=.2,
                          demo_coefficient_end=.05, demo_batch_size=256,
                          actor_learning_rate=3e-4, critic_learning_rate=3e-4,
                          actor_clip=1., critic_clip=1., usage_sink=None):
    """Compose full learner checkpoints, split optimizer, telemetry and demo loss.

    Call inside guard_ppo_updates. `contract` is the immutable trainer file SHA;
    fresh initialization verifies both zero Adam clocks and an actual critic
    different from the source. Continuation restores the complete learner and RNG
    and starts fresh physical episodes, as declared by LearnerHooks.
    """
    from .learner_continuation import instrument_trainer
    from .actual_update import instrument_updates, adam_update_counts
    from .student import make_joint_student_trainer
    from ..handoff_bank import pytree_sha256
    source_critic = [None]
    def initialize(state, key):
        if hooks.parent is None:
            counts = adam_update_counts(state.optimizer_state)
            if any(int(v) != 0 for v in counts.values()):
                raise ValueError('fresh R Adam counts must be zero')
            if source_critic[0] is None or pytree_sha256(state.params.value) == source_critic[0]:
                raise ValueError('R initial critic must differ from restored source critic')
        state, key = hooks.initialize(state, key)
        # Evidence is taken from the actual post-restore step-zero learner.
        if usage_sink is not None:
            usage_sink['initial_actor_sha256'] = pytree_sha256(state.params.policy)
            usage_sink['initial_critic_sha256'] = pytree_sha256(state.params.value)
            usage_sink['initial_adam_counts'] = {k: int(v) for k, v in adam_update_counts(state.optimizer_state).items()}
        return state, key
    wrapped = instrument_trainer(trainer, initialize, hooks.save)
    split = instrument_split_optimizer(wrapped, expected_sha256=contract, run_dir=run_dir,
        actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
        actor_clip=actor_clip, critic_clip=critic_clip)
    wrapped = instrument_updates(split._split_adapted_trainer, sink,
        max_grad_norm=actor_clip, probe=probe, record_optimizer_count=True,
        split_clip_norms=(actor_clip, critic_clip))
    wrapped = make_joint_student_trainer(wrapped, demo_manifest, retention=None,
        transitions=transitions, demo_coefficient_start=demo_coefficient_start,
        demo_coefficient_end=demo_coefficient_end, keep_coefficient=0,
        demo_batch_size=demo_batch_size, usage_sink=usage_sink,
        freeze_actor_normalizer=True, demo_clock='completed_transitions',
        actual_gradient_audit=True, record_update_clock=True)
    def train(**kwargs):
        if kwargs.get('learning_rate_schedule') not in (None, 'none'):
            raise ValueError('split PPO requires constant learning rate')
        restore = kwargs.get('restore_params')
        if restore is None or len(restore) != 3:
            raise ValueError('source Actor, normalizer and critic identity required')
        source_critic[0] = pytree_sha256(restore[2])
        if hooks.parent is None:
            if kwargs.get('restore_value_fn', False) is not False:
                raise ValueError('fresh R requires restore_value_fn=False')
            kwargs['restore_value_fn'] = False
        elif kwargs.get('restore_value_fn', True) is not True:
            raise ValueError('full learner continuation must seed the matching critic identity')
        return wrapped(**kwargs)
    return train
