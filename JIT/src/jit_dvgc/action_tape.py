"""Complete causal action records. Hash validation does not certify simulator replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'jit_causal_action_tape_v1'
GOAL_NAMES = ['anchor_x_m', 'window_start_x_m', 'direction_steer',
              'direction_rear_wheel_drive', 'direction_hip', 'direction_knee', 'strength']


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid action tape SHA256')


def _array(value, shape=None):
    array = np.asarray(value, dtype=float)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError('invalid action tape array')
    return array


def validate_action_tape(doc):
    if doc.get('schema') != SCHEMA or not doc.get('trajectory_id'):
        raise ValueError('invalid action tape schema or trajectory')
    if doc.get('tape_sha256') != _digest({k:v for k,v in doc.items() if k != 'tape_sha256'}):
        raise ValueError('action tape self hash mismatch')
    for key in ('protocol_sha256', 'policy_actor_sha256', 'policy_payload_sha256',
                'initial_state_sha256', 'initial_context_sha256', 'final_state_sha256', 'final_context_sha256'):
        _sha(doc[key])
    goal = _array(doc['goal'], (7,))
    if goal[1] >= goal[0] or goal[-1] <= 0 or np.count_nonzero(goal[2:6]) != 1 or np.max(abs(goal[2:6])) != 1:
        raise ValueError('invalid exogenous action tape goal')
    records = doc['records']
    if not records or len(records) != doc['record_count'] or len(records) != doc['environment_interactions']:
        raise ValueError('action tape must cover all interactions')
    previous = np.zeros(4)
    physical, context = doc['initial_state_sha256'], doc['initial_context_sha256']
    width = len(records[0]['observation'])
    previous_explorer = None
    for tick, row in enumerate(records):
        if not isinstance(row['tick'], int) or isinstance(row['tick'], bool) or row['tick'] != tick:
            raise ValueError('action tape ticks must start at zero and be contiguous')
        for key in ('state_sha256', 'context_sha256', 'next_state_sha256', 'next_context_sha256'):
            _sha(row[key])
        if row['state_sha256'] != physical or row['context_sha256'] != context:
            raise ValueError('action tape physical/context chain mismatch')
        observation = _array(row['observation'], (width,))
        history = _array(row['controller_history'])
        if history.size and (history.ndim != 2 or history.shape[1] != width):
            raise ValueError('action tape history width mismatch')
        base, applied, requested, effective, prior = [_array(row[k], (4,)) for k in
            ('base_action', 'applied_action', 'requested_delta', 'effective_delta', 'previous_delta')]
        if any(np.any(abs(a) > 1.0000001) for a in (base, applied)):
            raise ValueError('action tape actions exceed bounds')
        if not np.allclose(applied-base, effective, atol=2e-7, rtol=0) or not np.allclose(np.clip(base+requested,-1,1),applied,atol=2e-7,rtol=0):
            raise ValueError('action tape effective/requested arithmetic mismatch')
        if not np.allclose(previous, prior, atol=1e-7, rtol=0):
            raise ValueError('action tape controller residual continuity mismatch')
        if 'explorer_sha256' in doc:
            _sha(doc['explorer_sha256'])
            before, after = row['explorer_state_before'], row['explorer_state_after']
            if previous_explorer is not None and before != previous_explorer:
                raise ValueError('action tape explorer state continuity mismatch')
            if not np.array_equal(_array(before['previous_delta'], (4,)), prior) or not np.array_equal(_array(after['previous_delta'],(4,)), requested):
                raise ValueError('action tape explorer residual mismatch')
            h = _array(before['history'])
            if not np.array_equal(h,history):
                raise ValueError('action tape explorer history mismatch')
            expected_history = np.concatenate([h[1:], observation[None,:]]) if h.size else h
            if not np.array_equal(_array(after['history']), expected_history):
                raise ValueError('action tape explorer history update mismatch')
            if tick == 0 and (np.any(h) or np.any(prior)):
                raise ValueError('action tape explorer reset missing')
            limit = _array(doc['delta_limit'],(4,)); slew = _array(doc['slew_limit'],(4,))
            if np.any(limit<0) or np.any(slew<2*limit) or np.any(abs(requested)>limit+1e-7) or np.any(abs(requested-prior)>slew+1e-7):
                raise ValueError('action tape explorer residual bounds mismatch')
            previous_explorer = after
        elif history.size or not (np.allclose(requested, 0, atol=1e-7, rtol=0) or
                                  np.allclose(requested, goal[2:6]*goal[-1], atol=1e-7, rtol=0)):
            raise ValueError('fixed perturbation controller mismatch')
        previous = requested
        physical, context = row['next_state_sha256'], row['next_context_sha256']
    if (physical,context) != (doc['final_state_sha256'],doc['final_context_sha256']):
        raise ValueError('action tape final identity mismatch')
    return doc


def load_action_tape(path, *, expected_sha256=None):
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('action tape file hash mismatch')
    try:
        return validate_action_tape(json.loads(raw))
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('malformed action tape') from exc


def save_action_tape(path, payload):
    doc = dict(payload, tape_sha256=_digest(payload))
    validate_action_tape(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(doc,handle,sort_keys=True,indent=2,allow_nan=False)
        handle.write('\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_action_tape_schedule(protocol):
    """Reconstruct exact trajectory goals from the locked anchors and variant order."""
    from .acquisition.causal_jump import _variant_specs, VARIANT_PARTITION_MODULUS
    from .unified_boundary import action_sparse_directions
    variants = _variant_specs(lookbacks_m=protocol['causal_lookbacks_m'], strengths=protocol['strengths'],
        directions=action_sparse_directions(action_names=tuple(protocol['selected_action_names']),
            signs=tuple(protocol['selected_signs']), active_action_dimensions=1))
    result = {}
    for anchor in protocol['action_tape_anchors']:
        family = anchor['proposal_family_index']
        if not isinstance(family, int) or isinstance(family, bool) or not 0 <= family < VARIANT_PARTITION_MODULUS:
            raise ValueError('invalid action tape schedule family')
        for variant in variants:
            if variant['ordinal'] % VARIANT_PARTITION_MODULUS != family:
                continue
            key = f"{anchor['parent_group_id']}/variant_{variant['ordinal']}"
            if key in result:
                raise ValueError('duplicate action tape schedule trajectory')
            x = float(anchor['x_target_m'])
            result[key] = [x, x-float(variant['lookback_m']), *variant['direction']['basis_vector'], float(variant['strength'])]
    if not result:
        raise ValueError('empty action tape schedule')
    return result


def validate_tape_candidate_observation(tape, row, snapshot):
    """A saved next-step-ready candidate must supply the next recorded Actor input."""
    tick = row['trajectory_step']
    if tick < len(tape['records']):
        observation = np.asarray(snapshot.observation)
        recorded = np.asarray(tape['records'][tick]['observation'], dtype=observation.dtype)
        if not np.array_equal(observation, recorded):
            raise ValueError('candidate action tape observation mismatch')


def validate_catalog_action_tapes(directory, catalog, protocol):
    """Bind every recorded interaction and saved candidate to its actual tape."""
    if not protocol.get('record_action_tape'):
        if catalog.get('record_action_tape') or catalog.get('residual_explorer'):
            raise ValueError('catalog action tape option absent from protocol')
        return
    if catalog.get('record_action_tape') is not True or catalog.get('controller_mode') != protocol.get('controller_mode'):
        raise ValueError('catalog action tape/controller protocol mismatch')
    if catalog.get('residual_explorer') != protocol.get('residual_explorer'):
        raise ValueError('catalog residual identity mismatch')
    expected_schedule = expected_action_tape_schedule(protocol)
    explorer_ref = protocol.get('residual_explorer')
    explorer_artifact = None
    if explorer_ref:
        from .residual_exploration import load_artifact, file_sha
        from .constants import ACTION_ORDER, CTRL_DT
        if file_sha(explorer_ref['path']) != explorer_ref['sha256']:
            raise ValueError('action tape residual artifact file hash mismatch')
        explorer_artifact = load_artifact(explorer_ref['path'], expected_fingerprint=explorer_ref['explorer_sha256'])
        contract = explorer_artifact['contract']
        if (contract['action_names'] != list(ACTION_ORDER) or contract['goal_names'] != GOAL_NAMES or
            contract['action_low'] != [-1.]*4 or contract['action_high'] != [1.]*4 or contract['control_dt'] != CTRL_DT or
            contract.get('goal_contract') != 'exogenous fixed perturbation schedule; no future outcome features' or
            protocol.get('controller_mode') != 'frozen_residual_spatial_window_v1' or
            protocol.get('residual_gate') != 'shared fixed spatial window; history advances each tick; zero delta outside; slew >= twice amplitude'):
            raise ValueError('action tape residual action/goal/window contract mismatch')
        if protocol['frozen_unified_manifest_sha256'] not in {r['sha256'] for r in explorer_artifact['frozen_base_bank']}:
            raise ValueError('action tape residual base contract mismatch')
    elif protocol.get('controller_mode') != 'fixed_perturbation_v1':
        raise ValueError('action tape fixed controller mode mismatch')
    tapes = {}
    directory = Path(directory).resolve()
    interactions = 0
    for receipt in catalog['trajectory_receipts']:
        identity = receipt['trajectory_id']
        if identity in tapes:
            raise ValueError('duplicate action tape trajectory')
        reference = receipt['action_tape']
        path = (directory / reference['path']).resolve()
        if Path(reference['path']).is_absolute() or not path.is_relative_to(directory):
            raise ValueError('action tape path escapes acquisition')
        tape = load_action_tape(path, expected_sha256=reference['sha256'])
        if tape['trajectory_id'] != identity or tape['environment_interactions'] != receipt['environment_interactions']:
            raise ValueError('action tape trajectory/interaction mismatch')
        for key in ('protocol_sha256', 'policy_actor_sha256', 'policy_payload_sha256'):
            if tape[key] != catalog[key]:
                raise ValueError('action tape acquisition/base identity mismatch')
        if tape['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('action tape protocol mismatch')
        explorer = protocol.get('residual_explorer')
        if tape.get('explorer_sha256') != (explorer['explorer_sha256'] if explorer else None):
            raise ValueError('action tape explorer identity mismatch')
        goal = np.asarray(tape['goal'])
        receipt_goal = [receipt['anchor_x_m'], receipt['anchor_x_m']-receipt['lookback_m'],
                        *receipt['direction']['basis_vector'], receipt['strength']]
        if (identity not in expected_schedule or not np.array_equal(goal, np.asarray(expected_schedule[identity])) or
                not np.array_equal(goal, np.asarray(receipt_goal))):
            raise ValueError('action tape goal differs from declared schedule/receipt')
        if explorer_artifact:
            contract = explorer_artifact['contract']
            if any(tape[key] != contract[key] for key in ('delta_limit','slew_limit')):
                raise ValueError('action tape residual limit contract mismatch')
            width, history_steps = len(contract['observation_names']), contract['history_steps']
            for record in tape['records']:
                if len(record['observation']) != width or len(record['controller_history']) != history_steps:
                    raise ValueError('action tape observation/history contract mismatch')
        interactions += tape['environment_interactions']
        tapes[identity] = tape
    if set(tapes) != set(expected_schedule):
        raise ValueError('action tape trajectory set differs from declared schedule')
    if interactions != catalog['environment_interactions']:
        raise ValueError('action tape acquisition interaction ledger mismatch')
    for row in catalog['entries']:
        tape = tapes[row['trajectory_id']]
        tick = row['trajectory_step']
        if not isinstance(tick,int) or isinstance(tick,bool) or not 0 < tick <= len(tape['records']) or row['episode_step'] != tick:
            raise ValueError('candidate action tape tick mismatch')
        for key in ('protocol_sha256', 'policy_actor_sha256', 'policy_payload_sha256'):
            if row[key] != tape[key]:
                raise ValueError('candidate action tape identity mismatch')
        previous = tape['records'][tick-1]
        if row['state_sha256'] != previous['next_state_sha256'] or row['snapshot_context_sha256'] != previous['next_context_sha256']:
            raise ValueError('candidate action tape state/context mismatch')
        for prefix_key, record_key in [('nominal_actions','base_action'),('perturbed_actions','applied_action'),('effective_deltas','effective_delta')]:
            prefix = _array(row['perturbation'][prefix_key],(tick,4))
            if not np.array_equal(prefix,np.asarray([r[record_key] for r in tape['records'][:tick]])):
                raise ValueError('candidate action tape prefix mismatch')
    return tapes
