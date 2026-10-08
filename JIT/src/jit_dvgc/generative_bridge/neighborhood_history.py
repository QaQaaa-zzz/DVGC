"""TRAIN-only historical arrivals with policy-qualified evidence, frozen per round."""
from copy import deepcopy
from .contracts import digest
from .production import read
from .protocol import atomic_json


def neighborhood_config():
    from ..neighborhood import DEFAULT_CONFIG
    return {**DEFAULT_CONFIG,'medium_halfwidths':list(DEFAULT_CONFIG['medium_halfwidths']),
            'evidence_scope':'train_history_v1'}


def actor_evidence(rows,train_ancestors,provenance):
    records=[]
    for row in rows:
        if row.get('prefix_terminal'):continue
        ancestor=row['root_episode_id']
        if ancestor not in train_ancestors:raise ValueError('non-TRAIN neighborhood evidence')
        attempts=row.get('attempts',[])
        if len(attempts)!=1 or attempts[0]['label']!=row['label']:
            raise ValueError('single actual evaluated Actor identity required')
        records.append(dict(coordinates=deepcopy(row['coordinates']),phase=row['phase'],
            snapshot_context_sha256=row['snapshot_context_sha256'],root_episode_id=ancestor,
            evaluated_actor_sha256=attempts[0]['actor_sha256'],label=row['label'],data_role='train',
            evidence_kind='actor',provenance=provenance,
            arrival_source_actor_sha256=row.get('source_actor_sha256')))
    return records


def teacher_evidence(teachers,roots,provenance):
    byroot={r['root_id']:r for r in roots};records=[]
    for rid,teacher in teachers.items():
        if teacher['teacher_status']!='verified_solution':continue
        row=byroot[rid]
        if row['data_role']!='train':raise ValueError('non-TRAIN teacher map evidence')
        records.append(dict(coordinates=deepcopy(row['coordinates']),phase=row['phase'],
            snapshot_context_sha256=row['snapshot_context_sha256'],root_episode_id=row['root_episode_id'],
            evaluated_actor_sha256='teacher:'+digest(teacher['demo']),label=1,data_role='train',
            evidence_kind='verified_teacher',teacher_status='verified_solution',provenance=provenance,
            arrival_source_actor_sha256=teacher['source_actor_sha256']))
    return records


def source_recheck_evidence(teachers,roots,provenance):
    """Preserve known source repeats; 0/1 conflicts quarantine that policy."""
    byroot={r['root_id']:r for r in roots};records=[]
    for rid,teacher in teachers.items():
        row=byroot[rid]
        if row['data_role']!='train':raise ValueError('non-TRAIN source recheck')
        labels=list(teacher.get('source_control_labels',[]))+[teacher.get('source_recheck_label')]
        for label in labels:
            if label not in (0,1):continue
            records.append(dict(coordinates=deepcopy(row['coordinates']),phase=row['phase'],
                snapshot_context_sha256=row['snapshot_context_sha256'],root_episode_id=row['root_episode_id'],
                evaluated_actor_sha256=teacher['source_actor_sha256'],label=label,data_role='train',
                evidence_kind='actor',provenance=provenance,teacher_status=teacher['teacher_status'],
                source_recheck=True))
    return records


def seed_history(campaign):
    from pathlib import Path
    campaign=Path(campaign)
    phase=read(campaign/'stages/fresh_source_phase.json')['result']
    all_rows=read(phase['aggregate_path'])['rows'];rows=[r for r in all_rows if r['data_role']=='train']
    ancestors={r['root_episode_id'] for r in rows}
    records=actor_evidence(rows,ancestors,phase['aggregate_path'])
    reports=read(campaign/'acceptance_report.json')
    records+=actor_evidence(reports['C']['root_results'],ancestors,str(campaign/'acceptance_report.json'))
    records+=teacher_evidence(phase['teachers'],rows,str(campaign/'stages/fresh_source_phase.json'))
    records+=source_recheck_evidence(phase['teachers'],rows,str(campaign/'stages/fresh_source_phase.json'))
    return records


def freeze_history(path,records,actor,config,round_index):
    from ..neighborhood import FrozenNeighborhood
    index=FrozenNeighborhood(records,actor,config)
    atomic_json(path,dict(schema='jit_frozen_neighborhood_map_v2',source_actor_sha256=actor,
        config=config,rows=records,frozen_before_collection=True,round=round_index,
        source_scope='historical TRAIN arrivals; current Actor labels only when evaluated identity matches',
        flags=['current_actor_success','current_actor_failure','historical_actor_success','verified_teacher_success'],
        record_count=index.record_count,final_test_used=False))
