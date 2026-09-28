"""Fresh-source collection declarations and provenance-preserving TRAIN support."""
from copy import deepcopy
import json
from pathlib import Path

from .contracts import file_sha
from .protocol import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def prepare_collection_plan(template, output, *, master_seed, round_index=0):
    """Prepare bounded children only; execute neither collection nor suffixes."""
    root = Path(output).resolve()
    if template.get('success_criterion') != 'stable_forward_recovery' or template.get('reward_mode') != 'original_all_phases':
        raise ValueError('full-task recovery and original reward required')
    root.mkdir(parents=True, exist_ok=False)
    groups = [('train', 'train', 1024, [0], False, 0),
              ('generator_dev', 'generator_dev', 64, [0], False, 0),
              ('student_dev_id', 'student_dev', 512, [0], True, 0),
              ('student_dev_temporal', 'student_dev', 384, [5,10,15], True, 512)]
    from .pulse_protocol import EpisodeKeyRegistry
    registry = EpisodeKeyRegistry()
    batches = []
    for panel, role, count, onsets, full, first_id in groups:
        per_onset = count//len(onsets)
        for onset_i,onset in enumerate(onsets):
            amplitudes = (.25,) if panel == 'student_dev_temporal' else (.1,.25)
            per_amplitude = per_onset//len(amplitudes)
            for amplitude_i,amplitude in enumerate(amplitudes):
                for local in range(per_amplitude//32):
                    start = first_id + onset_i*per_onset + amplitude_i*per_amplitude + local*32
                    batch = root/panel/f'onset_{onset:02d}_amplitude_{amplitude:g}_batch_{local:03d}'
                    batch.mkdir(parents=True)
                    spec = deepcopy(template)
                    for field in ('reuse_results','reuse_collection','reuse_prefix_collection','resume_evaluation_root',
                                  'phase_policy','bridge_action_plan','pulse_event_schedule','initial_velocity_randomization',
                                  'neighborhood','neighborhood_map_sha256','evaluation_batch_size'):
                        spec.pop(field,None)
                    spec.update(controller_mode='fixed_random',num_envs=32,horizon=400,pulse_steps=3,
                        pulse_start_schedule=[onset],pulse_batch_mode='single',round_index=round_index,
                        delta_limit=[amplitude]*4,explorer_checkpoint=None,full_episode_rollout=full,
                        nominal_source_rollout=False,record_actor_preobservations=True,
                        order=[template['proposer']],full_matrix=True,
                        pulse_protocol_v1_2=dict(master_seed=master_seed,role=role,round=round_index,
                                                episode_ids=list(range(start,start+32))))
                    for episode_id in range(start,start+32):
                        registry.claim(master_seed,role,round_index,episode_id,'pulse')
                    collection = batch/'collection'
                    atomic_json(batch/'collection_spec.json',spec)
                    item = dict(panel=panel,role=role,count=32,onset=onset,amplitude=amplitude,
                        episode_ids=list(range(start,start+32)),collection_spec=str(batch/'collection_spec.json'),
                        collection_output=str(collection),full_episode=full,
                        maximum_interactions=32*(400 if full else 403))
                    if not full:
                        suffix = {**spec,'candidates':str(collection/'candidates.json'),'horizon':400,'budget':32*400}
                        suffix.pop('pulse_protocol_v1_2',None)
                        atomic_json(batch/'suffix_spec.json',suffix)
                        item.update(suffix_spec=str(batch/'suffix_spec.json'),suffix_output=str(batch/'suffix'))
                    batches.append(item)
    manifest = dict(schema='jit_bridge_fresh_source_collections_v1_2',batches=batches,
        master_seed=master_seed,round=round_index,source_policy=template['proposer'],bank=template['bank'],
        role_counts={p:n for p,_,n,_,_,_ in groups},batch_size=32,
        maximum_interactions=sum(b['maximum_interactions'] for b in batches),
        suffix_horizon=400,prefix_horizon=3,extra_suffix_allowance_per_root=3,
        prior_labels_imported=False,final_test_used=False)
    atomic_json(root/'plan.json',manifest)
    return manifest


def aggregate_collection(plan_path, output, *, source_actor_sha256):
    """Keep the full declared denominator, including terminal prefixes and unknowns."""
    plan = read(plan_path)
    rows,inputs,logical_ids = [],{str(Path(plan_path).resolve()):file_sha(plan_path)},set()
    charged = 0
    for batch in plan['batches']:
        collection = Path(batch['collection_output'])
        paths = [collection/'candidates.json']
        if not batch['full_episode']:
            paths.append(Path(batch['suffix_output'])/'results.json')
        candidates = read(paths[0]); results = read(paths[-1])
        if len(candidates) != batch['count'] or len(results) != batch['count']:
            raise ValueError('declared denominator differs')
        for path in paths:
            inputs[str(path)] = file_sha(path)
        for folder in [collection]+([] if batch['full_episode'] else [Path(batch['suffix_output'])]):
            status = read(folder/'status.json')
            if status['phase'] != 'completed':
                raise ValueError('collection child incomplete')
            charged += status['charged_interactions']
        for original,result,episode_id in zip(candidates,results,batch['episode_ids']):
            if original['snapshot_context_sha256'] != result['snapshot_context_sha256']:
                raise ValueError('suffix root identity differs')
            row = deepcopy(result)
            logical = row.get('logical_episode',{})
            if (logical.get('episode_id') != episode_id or logical.get('role') != batch['role']
                    or logical.get('master_seed') != plan['master_seed'] or logical.get('round') != plan['round']):
                raise ValueError('logical episode receipt differs')
            logical_key = (batch['role'],episode_id)
            if logical_key in logical_ids:
                raise ValueError('duplicate logical episode')
            logical_ids.add(logical_key)
            if row.get('prefix_terminal'):
                row['label'] = row.get('prefix_label')
                ancestor = row['prefix_file']+'::'+str(row['index'])
            else:
                identity = Path(row['snapshot'])/'identity.json'
                ancestor = read(identity)['parent_trajectory']
                if ancestor != row['prefix_file']+'::'+str(row['index']):
                    raise ValueError('snapshot ancestor differs from real prefix')
                attempts = row.get('attempts',[])
                if (len(attempts)!=1 or attempts[0]['actor_sha256']!=source_actor_sha256
                        or attempts[0]['policy']!=plan['source_policy'] or attempts[0]['label']!=row['label']):
                    raise ValueError('fresh source-only continuation required')
                for name in ('identity.json','snapshot.pkl'):
                    path = Path(row['snapshot'])/name; inputs[str(path)] = file_sha(path)
            row.update(root_id=row['snapshot_context_sha256'],root_episode_id=ancestor,
                data_role=batch['role'],panel=batch['panel'],requested_amplitude=batch['amplitude'],
                requested_onset=batch['onset'],source_actor_sha256=source_actor_sha256)
            rows.append(row)
    if charged > plan['maximum_interactions']:
        raise ValueError('fresh source collection budget exceeded')
    artifact = dict(schema='jit_bridge_fresh_source_rows_v1_2',rows=rows,inputs=inputs,
        source_policy=plan['source_policy'],source_actor_sha256=source_actor_sha256,
        denominator=len(rows),charged_interactions=charged,prior_labels_imported=False,final_test_used=False)
    atomic_json(Path(output),artifact)
    return artifact


def build_training_support(seed_support_path, aggregated_path, output, *, source_policy):
    """Combine only freshly witnessed source states and every valid TRAIN negative."""
    from ..iterative_probe_training import candidate_support_view
    from ..pulse_exploration import support_row
    from ..evidence_integrity import canonical_sha256
    seed_support_path = Path(seed_support_path)
    seed_status = read(seed_support_path.parent/'status.json')
    if (seed_status.get('source_policy') != source_policy or seed_status.get('historical_support_imported') is not False
            or seed_status.get('phase') != 'completed'):
        raise ValueError('fresh same-source seed_support provenance required')
    witnessed = deepcopy(read(seed_support_path)); aggregated = read(aggregated_path)
    if aggregated['source_policy'] != source_policy or aggregated['prior_labels_imported'] is not False:
        raise ValueError('same-source fresh collection required')
    if {r['phase'] for r in witnessed['entries']} != {'upstream','downstream'}:
        raise ValueError('both fresh witnessed phases required')
    if any(r.get('labels') != {source_policy:1} for r in witnessed['entries']):
        raise ValueError('seed support contains other source labels')
    inputs = {**aggregated['inputs'],str(Path(aggregated_path).resolve()):file_sha(aggregated_path),
              str(seed_support_path.resolve()):file_sha(seed_support_path)}
    pending,seen = [],{r['key'] for r in witnessed['entries']}
    for row in aggregated['rows']:
        if row['data_role']!='train' or row.get('prefix_terminal') or row['label'] not in (0,1):
            continue
        if row['root_id'] in seen:
            raise ValueError('duplicate fresh support context')
        seen.add(row['root_id'])
        entry = support_row(row,row['label']==1)
        entry.update(trajectory_id=row['root_episode_id'],labels={source_policy:row['label']},coordinates=row['coordinates'])
        (witnessed['entries'] if row['label']==1 else pending).append(entry)
    witnessed['inputs'].update(inputs)
    witnessed.pop('support_sha256',None)
    witnessed['support_sha256'] = canonical_sha256(witnessed)
    if pending:
        support = candidate_support_view(witnessed,pending,inputs,pending_fraction=.5,
                                         max_pending_per_phase=len(pending))
        if sum(r['evidence_status']=='pending' for r in support['entries']) != len(pending):
            raise ValueError('pending support was downselected')
    else:
        support = witnessed
    atomic_json(Path(output),support)
    return support
