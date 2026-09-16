"""Clean current-policy lineage: nominal support and explicit source adoption."""
from pathlib import Path
import numpy as np
from .jump_evidence_validation import read, write, file_sha
from .evidence_integrity import canonical_sha256


def validate_initial_bank(bank, source):
    if [m['name'] for m in bank['members']] != [source]:
        raise ValueError('current-policy experiment requires a singleton initial source bank')


def verify_stage_reuse(previous_spec, current_spec):
    """Match science inputs, resolving relocated candidate/bank manifests."""
    from .probe_bank import load_probe_bank
    def normalize(spec):
        result={k:v for k,v in spec.items() if k not in ('input_files','source_locks','resume_stage_root','resume_boundary','repo','code_commit')}
        if 'bank' in result:
            bank=load_probe_bank(Path(result['bank']))
            result['bank']={'task':bank['task'],'members':{m['name']:m['policy'] for m in bank['members']}}
        if 'neighborhood_map' in result:
            import hashlib
            digest=hashlib.sha256(Path(result['neighborhood_map']).read_bytes()).hexdigest()
            if digest!=result.get('neighborhood_map_sha256'):raise ValueError('neighborhood map drift')
            result['neighborhood_map']=digest
        if 'candidates' in result:
            result['candidates']=read(result['candidates'])
        return result
    if normalize(previous_spec)!=normalize(current_spec):
        raise ValueError('completed stage scientific contract differs; cannot reuse')


def successor_name(lineage_name, index, bank):
    """Allocate a deterministic free display name without changing old identities."""
    stem=f'{lineage_name}_repair_{index:04d}'
    occupied={member['name'] for member in bank['members']}
    name=stem
    ordinal=0
    while name in occupied:
        ordinal+=1
        name=f'{stem}_new_{ordinal:04d}'
    return name


def promotion_decision(rows, old, new, gains, minimum_retention):
    if not 0 <= minimum_retention <= 1 or not rows:
        raise ValueError('nonempty start/panel and valid retention threshold required')
    def label(row, name):
        return next((a['label'] for a in row['attempts'] if a['policy'] == name), None)
    prior = [r for r in rows[1:] if label(r, old) == 1]
    retained = sum(label(r, new) == 1 for r in prior)
    unknown = sum(label(r, new) is None for r in prior)
    fraction = retained / len(prior) if prior else 0.
    initial_success = label(rows[0], new) == 1
    return dict(promote=bool(gains > 0 and initial_success and prior and not unknown and
                            fraction >= minimum_retention),
                new_pending_successes=gains, old_positive=len(prior), retained=retained,
                lost=sum(label(r, new) == 0 for r in prior), unknown=unknown,
                retention_fraction=fraction, minimum_retention=minimum_retention,
                fixed_start_success=initial_success, whole_tube_guarantee=False)


def retention_candidates(support, initial, samples_per_phase):
    from .tube_rsi_smoke import fixed_indices
    result = [{**initial, 'index': 0, 'evaluation_group': 'fixed_start'}]
    for phase in ('upstream', 'downstream'):
        entries = sorted([r for r in support['entries'] if r['phase'] == phase], key=lambda r:r['key'])
        if not entries:
            raise ValueError('retention panel requires both witnessed phases')
        for i in fixed_indices(len(entries), min(samples_per_phase, len(entries))):
            result.append({**entries[i], 'index': len(result), 'prefix_terminal': False,
                           'evaluation_group': 'old_support'})
    return result


def seed_support(spec, output):
    """New real zero-residual prefix and same-source continuation labels only."""
    import jax
    from .pulse_exploration_runtime import collect, evaluate, networks
    from .exploration_continuation import snapshot_from_arrays
    from .unified_envelope_snapshot import save_unified_envelope_snapshot, snapshot_context_sha256, physical_state_sha256
    from .analysis.capability_tube import physical_coordinates_from_arrays, quantize_coordinates, ROOT_GEOMETRY_FIELDS, _cell_id
    from .pulse_exploration import support_row
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    nominal = {**spec, 'nominal_source_rollout': True, 'num_envs': 1, 'pulse_steps': spec['horizon'],
               'pulse_start_schedule': [0], 'round_index': 0,
               'delta_limit': [0., 0., 0., 0.], 'explorer_checkpoint': None}
    collect(nominal, output / 'nominal')
    env, _, member, _, _, _ = networks(spec)
    trace = output / 'nominal/prefixes.npz'
    tape = dict(np.load(trace))
    active = np.flatnonzero(tape['prefix_mask'][:, 0])
    end = int(active[-1])
    success_key = 'snap/down/recovery_success' if spec.get('success_criterion') == 'stable_forward_recovery' else 'snap/down/valid_contact_seen'
    if not (bool(tape[success_key][end, 0]) and
            not bool(tape['physical_failure'][end, 0])):
        if spec.get('allow_nominal_failure'):
            write(output/'status.json',dict(phase='completed',support_ready=False,reason='nominal_recovery_failed',charged_interactions=read(output/'nominal/status.json')['charged_interactions']))
            return
        raise ValueError('current source did not complete a conflict-free nominal jump')
    rows = []
    for t in active[::spec.get('seed_stride', 2)]:
        if tape['terminal'][t, 0]:
            continue
        arrays = {k[5:]:v[t, 0] for k,v in tape.items() if k.startswith('snap/')}
        snapshot = snapshot_from_arrays(arrays, env=env, record=member['policy'],
            parent_trajectory=str(trace) + '::0', parent_state_sha256=file_sha(trace))
        path = output / 'snapshots' / f'{int(t):05d}'
        save_unified_envelope_snapshot(path, snapshot)
        phase = 'upstream' if int(arrays['info/active_phase']) == 0 else 'downstream'
        coords = physical_coordinates_from_arrays(tape['qpos'][t,0], tape['qvel'][t,0], bundle=env._bundle)
        rows.append(dict(index=len(rows), snapshot=str(path), phase=phase,
            snapshot_context_sha256=snapshot_context_sha256(snapshot), state_sha256=physical_state_sha256(snapshot),
            cell=_cell_id(phase,'root_geometry_v1',quantize_coordinates(coords,ROOT_GEOMETRY_FIELDS)),
            coordinates=coords, prefix_file=str(trace), prefix_terminal=False))
    if not rows:
        raise ValueError('nominal trajectory produced no restart candidates')
    write(output / 'candidates.json', rows)
    del env
    jax.clear_caches()
    evaluate({**spec, 'candidates':str(output/'candidates.json'), 'order':[spec['proposer']],
              'budget':len(rows)*spec['horizon']}, output/'evaluation')
    results = read(output/'evaluation/results.json')
    entries = []
    inputs = {str(p):file_sha(p) for p in [trace, output/'evaluation/results.json']}
    for row in results:
        if row['label'] != 1:
            continue
        entry = support_row(row, True)
        entry.update(labels={spec['proposer']:1}, coordinates=row['coordinates'], trajectory_id=str(trace)+'::0')
        entries.append(entry)
        for filename in ('identity.json','snapshot.pkl'):
            p=Path(row['snapshot'])/filename;inputs[str(p)]=file_sha(p)
    if {r['phase'] for r in entries} != {'upstream','downstream'}:
        raise ValueError('source-only nominal support requires both witnessed phases')
    support=dict(schema='jit_iterative_witnessed_support_v1', role='train', final_test_used=False,
        status='completed', entries=entries, inputs=inputs,
        selection='new current-source zero-residual trajectory; source-only continuation positives',
        training_guidance_only=True, unwitnessed_resets_used=False)
    support['support_sha256']=canonical_sha256(support)
    write(output/'support.json',support)
    cost=read(output/'nominal/status.json')['charged_interactions']+read(output/'evaluation/status.json')['charged_interactions']
    write(output/'status.json',dict(phase='completed',charged_interactions=cost,
        source_policy=spec['proposer'],witnessed_rows=len(entries),historical_support_imported=False))
