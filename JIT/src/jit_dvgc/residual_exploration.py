"""Bounded frozen-bank residual warm starts, without simulation or RL discovery.

The dataset is an explicit export of witnessed TRAIN action examples. Historical
candidate snapshots alone are insufficient: each action must have its original
observation/history/base action/goal, complete prefix and same-context suffix
identity. This module checks the declared export and its content hashes; it does
not replay physics or independently certify a producer's witness assertions.

Residual slew is measured per control tick. It constrains the internal requested
residual; action clipping can change the effective residual when the base moves.
Neither bound is a stability guarantee. History and previous residual are explicit
closed-loop state and must be saved/restored alongside the simulator snapshot.
"""
from __future__ import annotations

from collections import Counter
import base64
import hashlib
import json
from pathlib import Path
import time

from flax import linen as nn, serialization, struct
from flax.core import freeze
import jax
import jax.numpy as jnp
import numpy as np
import optax


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid SHA-256 identity')


def _verify_files(records):
    if not records:
        raise ValueError('explicit frozen source files required')
    for record in records:
        _sha(record['sha256'])
        path = Path(record['path'])
        if not path.is_absolute():
            raise ValueError('source paths must be absolute')
        if file_sha(path) != record['sha256']:
            raise ValueError(f'frozen source changed: {path}')


def validate_contract(contract):
    n = len(contract['observation_names'])
    g = len(contract['goal_names'])
    h = contract['history_steps']
    a = len(contract['action_names'])
    if n < 1 or g < 1 or a != 4 or isinstance(h, bool) or not isinstance(h, int) or h < 0:
        raise ValueError('explicit observation, four action, goal and history contract required')
    for key in ('observation_names', 'goal_names', 'action_names'):
        if len(set(contract[key])) != len(contract[key]) or not all(isinstance(x,str) and x for x in contract[key]):
            raise ValueError('feature names must be unique nonempty strings')
    for key in ('model_sha256', 'observation_contract_sha256', 'endpoint_protocol_sha256', 'physical_cell_schema_sha256'):
        _sha(contract[key])
    width = n * (h+1) + a + g
    for key, size in [('feature_mean',width), ('feature_scale',width),
                      ('delta_limit',a), ('slew_limit',a), ('action_low',a), ('action_high',a)]:
        value = np.asarray(contract[key], dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f'invalid {key} shape or nonfinite value')
    if np.any(np.asarray(contract['feature_scale']) <= 0):
        raise ValueError('feature scales must be positive')
    if any(np.any(np.asarray(contract[k]) < 0) for k in ('delta_limit','slew_limit')):
        raise ValueError('residual limits must be nonnegative')
    if np.any(np.asarray(contract['action_low']) >= np.asarray(contract['action_high'])):
        raise ValueError('action interval is empty')
    if not np.isfinite(contract['control_dt']) or contract['control_dt'] <= 0:
        raise ValueError('control_dt must be positive')
    return width


class ConditionalResidual(nn.Module):
    """Small feedback network; base action conditions behavior, never numeric IDs."""
    @nn.compact
    def __call__(self, features):
        x = nn.leaky_relu(nn.Dense(64)(features), negative_slope=.01)
        x = nn.leaky_relu(nn.Dense(32)(x), negative_slope=.01)
        # A new explorer is exactly the frozen base before fitting.
        return jnp.tanh(nn.Dense(4, kernel_init=nn.initializers.zeros,
                                bias_init=nn.initializers.zeros)(x))


@struct.dataclass
class ExplorerState:
    history: jax.Array
    previous_delta: jax.Array


def reset_state(contract):
    validate_contract(contract)
    return ExplorerState(jnp.zeros((contract['history_steps'],len(contract['observation_names']))),
                         jnp.zeros(4))


def initialize(contract, *, seed=0):
    width = validate_contract(contract)
    return freeze(ConditionalResidual().init(jax.random.PRNGKey(seed), jnp.zeros(width)))


def _features(observation, history, base_action, goal, contract):
    observation = jnp.asarray(observation)
    history = jnp.asarray(history)
    base_action = jnp.asarray(base_action)
    goal = jnp.asarray(goal)
    expected = (len(contract['observation_names']),)
    if observation.shape != expected or history.shape != (contract['history_steps'],expected[0]):
        raise ValueError('observation/history contract mismatch')
    if base_action.shape != (4,) or goal.shape != (len(contract['goal_names']),):
        raise ValueError('base action/goal contract mismatch')
    raw = jnp.concatenate([observation, history.reshape(-1), base_action, goal])
    return (raw-jnp.asarray(contract['feature_mean'])) / jnp.asarray(contract['feature_scale'])


def bound_delta(requested, base_action, previous_delta, contract):
    requested = jnp.asarray(requested)
    base_action = jnp.asarray(base_action)
    previous_delta = jnp.asarray(previous_delta)
    if requested.shape != (4,) or base_action.shape != (4,) or previous_delta.shape != (4,):
        raise ValueError('requested/base action/previous residual must each have four channels')
    limit = jnp.asarray(contract['delta_limit'])
    slew = jnp.asarray(contract['slew_limit'])
    delta = jnp.clip(requested, -limit, limit)
    delta = jnp.clip(delta, previous_delta-slew, previous_delta+slew)
    delta = jnp.clip(delta, -limit, limit)
    action = jnp.clip(jnp.asarray(base_action)+delta, jnp.asarray(contract['action_low']),
                      jnp.asarray(contract['action_high']))
    return action, delta


def apply_residual(variables, contract, observation, base_action, goal, state):
    features = _features(observation,state.history,base_action,goal,contract)
    requested = ConditionalResidual().apply(variables,features) * jnp.asarray(contract['delta_limit'])
    action,delta = bound_delta(requested,base_action,state.previous_delta,contract)
    history = state.history
    if contract['history_steps']:
        history = jnp.concatenate([history[1:],jnp.asarray(observation)[None,:]],axis=0)
    return action, ExplorerState(history,delta)


def load_training_data(dataset):
    """Validate all exports; select only cumulative novel witnessed action rows.

    Each new root cell has total weight one, even if its trajectory has many
    examples. No credit is assigned to an unfamiliar state without a suffix.
    The caller must freeze the named cumulative baseline for the fitting block.
    """
    if dataset.get('schema') != 'residual_warmstart_dataset_v1' or dataset.get('role') != 'TRAIN':
        raise ValueError('explicit TRAIN warm-start dataset required')
    contract = dataset['contract']
    validate_contract(contract)
    _verify_files(dataset['frozen_base_bank'])
    _verify_files(dataset['sources'])
    bank = {x['sha256'] for x in dataset['frozen_base_bank']}
    baseline = dataset['cumulative_baseline']
    if not isinstance(baseline.get('name'),str) or not baseline['name']:
        raise ValueError('named cumulative witnessed baseline required')
    if not isinstance(baseline.get('root_cells'),list) or not all(isinstance(x,str) and x for x in baseline['root_cells']):
        raise ValueError('explicit cumulative root cell list required')
    seen = set(baseline['root_cells'])
    rows=[]
    for source in dataset['sources']:
        doc = json.loads(Path(source['path']).read_text())
        if doc.get('schema') != 'residual_action_witnesses_v1' or doc.get('role') != 'TRAIN':
            raise ValueError('source must be an explicit TRAIN action witness export')
        if doc.get('contract_sha256') != digest(contract):
            raise ValueError('source observation/action contract mismatch')
        for row in doc['records']:
            if row.get('role') != 'TRAIN' or row.get('base_sha256') not in bank:
                raise ValueError('row role or frozen base identity mismatch')
            for key in ('state_sha256','context_sha256','acquisition_protocol_sha256'):
                _sha(row[key])
            if (row.get('forward_prefix_complete') is not True or
                    row.get('continuation_status') != 'success' or
                    row.get('endpoint') != 'first_valid_landing' or
                    row.get('continuation_endpoint_protocol_sha256') != contract['endpoint_protocol_sha256'] or
                    row.get('physical_failure') is not False or
                    row.get('continuation_state_sha256') != row['state_sha256'] or
                    row.get('continuation_context_sha256') != row['context_sha256']):
                raise ValueError('complete unconflicted same-context prefix/suffix witness required')
            if not isinstance(row.get('root_cell'),str) or not row['root_cell']:
                raise ValueError('root cell required')
            features = np.asarray(_features(row['observation'],row['history'],row['base_action'],row['goal'],contract))
            target = np.asarray(row['target_delta'],dtype=np.float32)
            previous = np.asarray(row['previous_delta'],dtype=np.float32)
            base = np.asarray(row['base_action'],dtype=np.float32)
            if target.shape != (4,) or previous.shape != (4,) or not all(np.isfinite(x).all() for x in (features,target,previous,base)):
                raise ValueError('invalid or nonfinite action example')
            if (np.any(abs(target)>np.asarray(contract['delta_limit'])+1e-7) or
                    np.any(abs(previous)>np.asarray(contract['delta_limit'])+1e-7) or
                    np.any(abs(target-previous)>np.asarray(contract['slew_limit'])+1e-7) or
                    np.any(base < np.asarray(contract['action_low'])) or
                    np.any(base > np.asarray(contract['action_high']))):
                raise ValueError('training action example exceeds bounds')
            if row['root_cell'] not in seen:
                rows.append((row,features,target,previous))
    if not rows:
        raise ValueError('zero positive cumulative witnessed novel-support examples; refuse fitting')
    counts = Counter(row['root_cell'] for row,_,_,_ in rows)
    return dict(features=np.stack([f for _,f,_,_ in rows]),
                targets=np.stack([t for _,_,t,_ in rows]),
                previous=np.stack([p for _,_,_,p in rows]),
                weights=np.asarray([1/counts[r['root_cell']] for r,_,_,_ in rows],dtype=np.float32),
                novel_cells=len(counts))


def fit(dataset, *, steps, max_steps, seed=0, learning_rate=.001):
    """Bounded full-batch supervised fit, starting from zero residual each block.

    This imitates supplied successful perturbations; it is not an RL optimizer,
    does not produce new witnesses, and does not establish exploration gains.
    """
    if isinstance(steps,bool) or not isinstance(steps,int) or not isinstance(max_steps,int) or not 0 < steps <= max_steps:
        raise ValueError('positive explicit optimizer-step budget required')
    if not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('learning rate must be positive')
    started=time.monotonic()
    batch=load_training_data(dataset)
    contract=dataset['contract']
    variables=initialize(contract,seed=seed)
    features=jnp.asarray(batch['features']); targets=jnp.asarray(batch['targets'])
    weights=jnp.asarray(batch['weights'])
    amplitude=jnp.asarray(contract['delta_limit'])
    def loss(params):
        delta=ConditionalResidual().apply(params,features)*amplitude
        # Regress the declared residual target before the runtime slew projection.
        # Clipping around historical previous_delta here would give zero gradients
        # whenever the fresh zero-output model starts outside that slew interval.
        return jnp.sum(weights*jnp.mean((delta-targets)**2,axis=-1))/jnp.sum(weights)
    optimizer=optax.adam(learning_rate)
    optimizer_state=optimizer.init(variables)
    @jax.jit
    def update(params,state):
        value,grad=jax.value_and_grad(loss)(params)
        updates,state=optimizer.update(grad,state,params)
        return optax.apply_updates(params,updates),state,value
    initial=float(loss(variables))
    for _ in range(steps):
        variables,optimizer_state,_=update(variables,optimizer_state)
    final=float(loss(variables))
    if not np.isfinite(final):
        raise ValueError('nonfinite fitting result')
    report=dict(training_kind='supervised_warm_start',loss_contract='requested_residual_regression_v1',
                optimizer_steps=steps,max_optimizer_steps=max_steps,
                seed=seed,learning_rate=learning_rate,initial_loss=initial,final_loss=final,
                examples=len(features),novel_cells=batch['novel_cells'],environment_interactions=0,
                ppo_transitions=0,wall_seconds=time.monotonic()-started,
                independent_discovery_evidence=False)
    return variables,report


def save_artifact(path, variables, dataset, report):
    load_training_data(dataset)  # Refuse publication after a source changed during fitting.
    payload=dict(schema='frozen_residual_explorer_v1', architecture='conditional_mlp_64_32_v1',
                 dataset=dataset, dataset_sha256=digest(dataset), fit_report=report,
                 parameters=base64.b64encode(serialization.to_bytes(variables)).decode('ascii'))
    identity=digest(payload)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(dict(payload=payload,explorer_sha256=identity),handle,indent=2,allow_nan=False)
        handle.write('\n')
    return identity


def load_artifact(path, *, expected_fingerprint=None):
    doc=json.loads(Path(path).read_text()); payload=doc['payload']
    identity=digest(payload)
    if identity!=doc['explorer_sha256'] or (expected_fingerprint is not None and identity!=expected_fingerprint):
        raise ValueError('explorer fingerprint mismatch')
    if payload.get('schema')!='frozen_residual_explorer_v1' or payload.get('architecture')!='conditional_mlp_64_32_v1':
        raise ValueError('unsupported explorer artifact')
    dataset=payload['dataset']
    if digest(dataset)!=payload['dataset_sha256']:
        raise ValueError('dataset fingerprint mismatch')
    validate_contract(dataset['contract'])
    if dataset.get('role')!='TRAIN':
        raise ValueError('artifact role mismatch')
    _verify_files(dataset['frozen_base_bank'])
    variables=serialization.from_bytes(initialize(dataset['contract']),base64.b64decode(payload['parameters']))
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(variables)):
        raise ValueError('nonfinite explorer parameters')
    return dict(variables=freeze(variables),contract=dataset['contract'],
                frozen_base_bank=dataset['frozen_base_bank'],explorer_sha256=identity)


class FrozenResidualPolicy:
    """Explicit-state adapter for existing Brax policy(obs, key) callables.

    The caller supplies a policy reconstructed from the verified base artifact;
    this adapter cannot introspect an arbitrary Python callable's provenance.
    No parameters from that callable are passed to the residual fit optimizer.
    """
    def __init__(self, artifact, base_policy, *, base_sha256):
        if base_sha256 not in {x['sha256'] for x in artifact['frozen_base_bank']}:
            raise ValueError('unknown frozen base policy')
        _verify_files(artifact['frozen_base_bank'])
        self.artifact=artifact
        self.base_policy=base_policy

    def reset(self):
        return reset_state(self.artifact['contract'])

    def step(self, observations, key, goal, state):
        result=self.base_policy(observations,key)
        base=result[0] if isinstance(result,tuple) else result
        obs=observations['state'] if isinstance(observations,dict) else observations
        return apply_residual(self.artifact['variables'],self.artifact['contract'],obs,base,goal,state)
