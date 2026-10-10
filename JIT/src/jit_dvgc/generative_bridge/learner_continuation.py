"""Persist complete Brax PPO learners across changing TRAIN-support rounds.

Only host initialization/checkpoint hooks are inserted into the pinned upstream
trainer. PPO rollout, losses and optimizer math are unchanged. Simulator state
is intentionally reset on each new round's declared support.
"""
import ast
import hashlib
import inspect
import pickle
from pathlib import Path
import jax
import numpy as np
from .contracts import file_sha,digest
from .protocol import atomic_json
from .production import read
from ..handoff_bank import pytree_sha256


def trainer_sha(trainer):
    return hashlib.sha256(inspect.getsource(trainer).encode()).hexdigest()


def learner_contract(raw):
    return digest(dict(ppo={k:v for k,v in raw['ppo'].items() if k not in ('seed','requested_transitions')},
        inputs=raw['inputs'],reward_mode=raw['reward_mode'],reward_contract=raw['reward_contract'],
        snapshot_reset_contract=raw['snapshot_reset_contract']))


class LearnerHooks:
    def __init__(self,root,*,contract,trainer_sha,parent):
        self.root=Path(root);self.contract=contract;self.trainer_sha=trainer_sha
        self.parent=parent;self.offset=0;self.latest=None

    def initialize(self,state,key):
        if self.parent is not None:
            p=self.parent
            if file_sha(p['path'])!=p['sha256']:raise ValueError('learner manifest changed')
            m=read(p['path'])
            if m['contract_sha256']!=self.contract or m['trainer_sha256']!=self.trainer_sha:
                raise ValueError('learner contract or trainer changed')
            if file_sha(m['state_path'])!=m['state_sha256']:raise ValueError('learner state changed')
            with Path(m['state_path']).open('rb') as f:saved=pickle.load(f)
            restored=saved['training_state'];key=saved['rng']
            if jax.tree.structure(restored)!=jax.tree.structure(state):raise ValueError('learner structure changed')
            if any(np.shape(a)!=np.shape(b) for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(state))):
                raise ValueError('learner shapes changed')
            if pytree_sha256((restored.params,restored.normalizer_params))!=pytree_sha256((state.params,state.normalizer_params)):
                raise ValueError('learner does not match preceding Actor/critic/normalizer')
            self.offset=m['lifetime_transitions']
            state=restored.replace(env_steps=state.env_steps)
        if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((state,key))):
            raise ValueError('nonfinite learner state')
        atomic_json(self.root/'initialization.json',dict(
            mode='full_learner_continuation' if self.parent else 'one_time_legacy_optimizer_bootstrap',
            parent=self.parent,lifetime_transition_offset=self.offset,
            actor_sha256=pytree_sha256(state.params.policy),critic_sha256=pytree_sha256(state.params.value),
            normalizer_sha256=pytree_sha256(state.normalizer_params),optimizer_sha256=pytree_sha256(state.optimizer_state),
            rng_sha256=pytree_sha256(key),simulator_state_restored=False,
            simulator_reset_reason='new round declared TRAIN support; fresh physical episodes'))
        return state,key

    def save(self,step,state,key):
        if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((state,key))):
            raise ValueError('cannot save nonfinite learner')
        directory=self.root/f'transition_{step:06d}';directory.mkdir(parents=True,exist_ok=False)
        target=directory/'state.pkl';tmp=directory/'state.tmp'
        with tmp.open('wb') as f:pickle.dump(jax.device_get(dict(training_state=state,rng=key)),f)
        tmp.replace(target)
        manifest=directory/'manifest.json'
        atomic_json(manifest,dict(schema='jit_bridge_full_ppo_learner_v1',state_path=str(target.resolve()),
            state_sha256=file_sha(target),contract_sha256=self.contract,trainer_sha256=self.trainer_sha,
            local_transitions=int(step),lifetime_transitions=self.offset+int(step),
            actor_sha256=pytree_sha256(state.params.policy),critic_sha256=pytree_sha256(state.params.value),
            normalizer_sha256=pytree_sha256(state.normalizer_params),optimizer_sha256=pytree_sha256(state.optimizer_state),
            rng_sha256=pytree_sha256(key)))
        self.latest=dict(path=str(manifest.resolve()),sha256=file_sha(manifest),lifetime_transitions=self.offset+int(step))
        atomic_json(self.root/'latest.json',self.latest)

    def latest_receipt(self):return self.latest


def instrument_trainer(trainer,initialize,save):
    """Fail closed unless the installed trainer has exactly the expected hooks."""
    tree=ast.parse(inspect.getsource(trainer));counts=[0,0]
    class Hooks(ast.NodeTransformer):
        def visit_Assign(self,node):
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='training_state'
                and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='jax'
                and node.value.func.attr=='device_put_replicated'):
                counts[0]+=1
                return [*ast.parse('training_state, local_key = _learner_initialize(training_state, local_key)').body,node]
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='policy_params_fn':
                counts[1]+=1
                return [*ast.parse('_learner_save(current_step, _unpmap(training_state), local_key)').body,node]
            return self.generic_visit(node)
    tree=Hooks().visit(tree)
    if counts!=[1,2]:raise ValueError('unsupported upstream PPO hook layout')
    namespace={**trainer.__globals__,'_learner_initialize':initialize,'_learner_save':save}
    exec(compile(ast.fix_missing_locations(tree),'<pinned PPO learner continuation>','exec'),namespace)
    wrapped=namespace[trainer.__name__]
    wrapped._jit_source_text=ast.unparse(ast.fix_missing_locations(tree))
    return wrapped


def continuing_trainer(trainer,raw,run_dir):
    contract=raw['continuous_learner']
    if trainer_sha(trainer)!=contract['trainer_sha256']:raise ValueError('upstream PPO trainer identity changed')
    hooks=LearnerHooks(Path(run_dir)/'learner',contract=learner_contract(raw),
        trainer_sha=contract['trainer_sha256'],parent=contract['parent'])
    return instrument_trainer(trainer,hooks.initialize,hooks.save)


def validate_declaration(raw):
    c=raw['continuous_learner'];i=raw['initialization'];parent=c['parent']
    if (c.get('schema')!='jit_bridge_full_ppo_learner_v1' or i['actor']!='warm_start_frozen_unified'
            or i['critic']!='warm_start_frozen_unified'
            or i['optimizer']!=('resume_learner' if parent else 'fresh_once')
            or c.get('legacy_optimizer_bootstrap') is not (parent is None)
            or not isinstance(c.get('trainer_sha256'),str) or len(c['trainer_sha256'])!=64):
        raise ValueError('invalid continuous learner initialization')
    if parent:
        if raw['input_files'].get(parent['path'])!=parent['sha256'] or file_sha(parent['path'])!=parent['sha256']:
            raise ValueError('missing or changed learner input lock')


def validate_saved_learner(receipt,policy,*,expected_local_transitions):
    if file_sha(receipt['path'])!=receipt['sha256']:raise ValueError('learner manifest changed')
    m=read(receipt['path'])
    if file_sha(m['state_path'])!=m['state_sha256']:raise ValueError('learner payload changed')
    if m['local_transitions']!=expected_local_transitions:raise ValueError('incomplete learner checkpoint')
    for key in ('actor_sha256','critic_sha256','normalizer_sha256'):
        if m[key]!=policy[key]:raise ValueError('learner/inference identity mismatch: '+key)
    return m
