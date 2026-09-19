"""Build auditable witnessed reset-support views from completed TRAIN refinement.

This deliberately is not a PPO config: the training reset adapter and recipe
must consume and validate these exact snapshots before training is launched.
"""
from collections import Counter, defaultdict
from pathlib import Path
import traceback
from .jump_evidence_validation import read, write, file_sha, verify_hash
from .evidence_integrity import canonical_sha256
from .result_bundle import bundle

DEFAULT_SOURCE='JIT/runs/discovery/knee_boundary_v1_budgetfix'
DEFAULT_OUTPUT='JIT/runs/training_support/complementary_boundary_v1'
NAMES=['pi_0','pi_1','pi_2','pi_3']


def partition(entries, labels):
    from .analysis.frontier_evidence import classify_labels
    if set(labels)!=set(NAMES) or any(len(labels[n])!=len(entries) for n in NAMES):
        raise ValueError('four complete evaluator panels required')
    rows=[]
    for i,entry in enumerate(entries):
        values=[]
        for name in NAMES:
            row=labels[name][i]
            if any(row[k]!=entry[k] for k in ('candidate_id','state_sha256','phase')):
                raise ValueError('candidate identity/order drift')
            if row.get('success_criterion')!='first_valid_landing':
                raise ValueError('mixed endpoint')
            values.append(row['label'])
        status=classify_labels(values)
        rows.append(dict(candidate_id=entry['candidate_id'],state_sha256=entry['state_sha256'],
            snapshot_context_sha256=entry['snapshot_context_sha256'],
            trajectory_id=entry['trajectory_id'],phase=entry['phase'],status=status,
            witnessed=any(values),labels=dict(zip(NAMES,values))))
    scores={name:sum(row['labels'][name] for row in rows) for name in NAMES}
    initializer=min(NAMES,key=lambda name:(-scores[name],name))
    # Equal trajectory/phase groups within each witnessed stratum; long trajectories
    # must not dominate solely because more frames were retained.
    groups=defaultdict(Counter)
    for row in rows:
        if row['witnessed']:
            groups[row['status']][(row['trajectory_id'],row['phase'])]+=1
    for row in rows:
        group=groups[row['status']]
        row['within_stratum_probability']=(1/len(group)/group[(row['trajectory_id'],row['phase'])]
                                            if row['witnessed'] else 0.)
    return rows,scores,initializer


def build(source, output):
    from .policy_comparison_runtime import complete_output
    from .probe_bank import validate_probe_arrivals
    from .unified_continuation_labels import validate_unified_boundary_catalog
    from .unified_policy_freeze import load_frozen_unified_manifest
    source,output=Path(source).resolve(),Path(output).resolve()
    report=read(source/'summary.json');verify_hash(report,'report_sha256')
    if (report.get('status')!='completed' or report.get('scope')!='TRAIN local boundary refinement; no fair policy ranking'
        or report.get('final_test_used') is not False):
        raise ValueError('completed TRAIN boundary refinement required')
    child=source/'pi_1';plan=read(child/'plan.json');verify_hash(plan,'plan_sha256')
    if plan.get('role')!='train' or plan.get('proposer')!='pi_1':raise ValueError('source role/proposer drift')
    # Historical source hashes describe the run; current code need not equal them.
    inputs={}
    def lock(path):
        path=Path(path).resolve();inputs[str(path)]=file_sha(path)
    for path in (source/'summary.json',child/'plan.json',child/'analysis_inputs.json'):
        lock(path)
    for path,sha in plan['input_files'].items():
        if file_sha(path)!=sha:raise ValueError('locked historical input changed')
        lock(path)
    members=plan['members']
    if [m['policy']['name'] for m in members]!=NAMES:raise ValueError('source bank drift')
    for member in members:
        if file_sha(member['path'])!=member['file_sha256'] or load_frozen_unified_manifest(Path(member['path']))['policy']!=member['policy']:
            raise ValueError('frozen initializer/evaluator changed')
        lock(member['path'])
    manifest=read(child/'analysis_inputs.json');catalog_path=Path(manifest['catalog'])
    catalog=read(catalog_path);protocol=read(catalog_path.parent/'protocol.json')
    verify_hash(protocol,'protocol_sha256')
    if protocol.get('logical_role')!='train' or protocol.get('probe_bank_sha256')!=plan['bank_sha256'] or catalog.get('protocol_sha256')!=protocol['protocol_sha256']:
        raise ValueError('arrival protocol/bank/role drift')
    proposer=next(m for m in members if m['policy']['name']=='pi_1')
    entries=validate_unified_boundary_catalog(catalog,policy_record=proposer['policy'],
        frozen_manifest_sha256=proposer['file_sha256'],allow_empty=False)
    validate_probe_arrivals(catalog_path,entries,proposer['policy'])
    lock(catalog_path);lock(catalog_path.parent/'protocol.json')
    labels={}
    for member in members:
        name=member['policy']['name'];directory=Path(manifest['merged'][name])
        result=complete_output(directory,catalog_path,proposer,member,plan['horizon'],plan['label_seed'])
        if result is None:raise ValueError('incomplete source evaluator')
        labels[name]=result[1]
        for filename in ('summary.json','protocol.json','labels.json'):lock(directory/filename)
    rows,scores,initializer=partition(entries,labels)
    if dict(Counter(r['status'] for r in rows))!=report['candidate_status_counts']:
        raise ValueError('reported counts differ from verified raw labels')
    for row,entry in zip(rows,entries,strict=True):
        directory=(catalog_path.parent/entry['source_bank']/entry['snapshot']).resolve()
        for filename in ('identity.json','snapshot.pkl'):lock(directory/filename)
        row['snapshot']=str(directory)
    witnessed=[r for r in rows if r['witnessed']]
    if not witnessed:raise ValueError('no witnessed training support')
    draft=dict(schema='jit_complementary_support_v1',source_report_sha256=report['report_sha256'],
        role='train',candidate_status_counts=report['candidate_status_counts'],
        initializer=next(m for m in members if m['policy']['name']==initializer),
        initializer_selection='maximum successes on this TRAIN panel; ties by name',
        evaluator_success_counts=scores,inputs=inputs,entries=witnessed,
        no_witness_reset_admission=False,trainer_config_ready=False,training_transitions=0,
        required_before_ppo=['snapshot RSI adapter and fixed x=2.5 reset mixture',
            'stratum mixture, reward, initializer/normalizer, fresh critic/optimizer',
            'fixed PPO budget/seeds and matched control', 'small production training smoke'],
        final_test_used=False)
    draft['support_sha256']=canonical_sha256(draft)
    path=output/'support.json'
    if path.exists() and read(path)!=draft:raise ValueError('support changed; use a new output directory')
    write(path,draft)
    write(output/'unwitnessed_targets.json',{'training_admitted':False,
        'scope':'TRAIN development targets, not an untouched final test',
        'entries':[r for r in rows if not r['witnessed']]})
    result=dict(status='completed',witnessed_support_count=len(witnessed),
        unwitnessed_target_count=len(rows)-len(witnessed),initializer=initializer,
        evaluator_success_counts=scores,source_report_sha256=report['report_sha256'],
        support_sha256=draft['support_sha256'],training_transitions=0,trainer_config_ready=False)
    write(output/'summary.json',result)
    return result


def run(source,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    try:
        result=build(source,output)
    except Exception as exc:
        result=dict(status='engineering_error',error=str(exc),traceback=traceback.format_exc())
        write(output/'failure.json',result)
    print(f"[support] {result['status']}\nReturn this file: {bundle(output)}",flush=True)
    return result
