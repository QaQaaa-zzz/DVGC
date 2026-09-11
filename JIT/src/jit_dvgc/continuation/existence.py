"""Versioned first-success witnesses, explicit unknowns, and bounded subset jobs.

Full evaluator matrices remain a different artifact. A skipped evaluator is never
assigned a negative label, and a landing/failure conflict is quarantined here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from ..evidence_integrity import canonical_sha256, read_verified_protocol, validate_label_row
from ..jump_evidence_validation import read, write, file_sha, verify_hash

SCHEMA = 'jit_first_success_witness_plan_v1'


def evidence_status(row):
    if row is None:
        return 'untested'
    if row.get('label') not in (0, 1):
        return 'unknown'
    if row.get('valid_contact_seen') and row.get('physical_failure'):
        return 'conflict'
    return 'success' if row['label'] == 1 else 'failure'


def aggregate_candidate(order, evaluated):
    if not order or len(order) != len(set(order)) or set(evaluated)-set(order):
        raise ValueError('distinct declared evaluator order required')
    outcomes = {name: {'status': evidence_status(evaluated.get(name)),
                       'label': evaluated[name]['label'] if name in evaluated and evidence_status(evaluated[name]) in ('success','failure') else None}
                for name in order}
    winners = [name for name in order if outcomes[name]['status']=='success']
    failed = all(item['status']=='failure' for item in outcomes.values())
    return dict(label=1 if winners else 0 if failed else None,
                witness_status='observed_landing' if winners else 'no_success_witness_under_declared_bank' if failed else 'unknown',
                successful_policy_names=winners, per_policy_outcomes=outcomes)


def replay_first_success(matrices, order):
    """Offline scheduling analysis only; caller verifies the full input labels."""
    if not order or len(set(order))!=len(order) or set(matrices)!=set(order):
        raise ValueError('complete matrices and distinct evaluator order required')
    count=len(matrices[order[0]])
    if any(len(rows)!=count for rows in matrices.values()):
        raise ValueError('matrix count drift')
    entries=[]; cost=0; calls=0
    for index in range(count):
        selected={}
        for name in order:
            row=matrices[name][index]; selected[name]=row
            charge=row['environment_interactions']
            if type(charge) is not int or charge<0: raise ValueError('invalid label cost')
            cost+=charge; calls+=1
            if evidence_status(row)=='success': break
        entries.append(aggregate_candidate(order,selected))
    return dict(entries=entries,hypothetical_useful_interactions=cost,hypothetical_evaluator_calls=calls,
                new_environment_interactions=0,scope='offline scheduling; excludes padding/process/failed attempts; not measured wall-clock savings')


def prepare(bank_path, catalog_path, output, *, order, seed, budget, backend='serial', batch_size=1,
            indices=None, max_candidates_per_process=4096, timeout_seconds=600):
    from ..probe_bank import load_probe_bank, validate_probe_arrivals
    from ..unified_continuation_labels import validate_unified_boundary_catalog
    from ..unified_continuation_shards import validate_candidate_selection
    bank_path,catalog_path=Path(bank_path).resolve(),Path(catalog_path).resolve()
    bank=load_probe_bank(bank_path);catalog=read(catalog_path)
    protocol=read_verified_protocol(catalog_path.parent/'protocol.json')
    members={m['name']:m for m in bank['members']}
    evaluator_names={m['name'] for m in members.values() if 'evaluator' in m['roles']}
    if set(order)!=evaluator_names or len(order)!=len(evaluator_names):
        raise ValueError('order must include every declared evaluator exactly once')
    proposer=members[protocol['policy_name']]
    if (protocol.get('logical_role')!='train' or 'proposer' not in proposer['roles']
        or protocol.get('probe_bank_sha256')!=bank['bank_sha256']
        or catalog.get('status')!='completed' or protocol['protocol_sha256']!=catalog['protocol_sha256']):
        raise ValueError('completed TRAIN catalog from the declared bank required')
    rows=validate_unified_boundary_catalog(catalog,policy_record=proposer['policy'],frozen_manifest_sha256=proposer['frozen_file_sha256'])
    selected=validate_candidate_selection(0,len(rows),indices)
    validate_probe_arrivals(catalog_path,[rows[i] for i in selected],proposer['policy'])
    if backend not in ('serial','vectorized') or any(type(x) is not int or x<=0 for x in (budget,batch_size,max_candidates_per_process,timeout_seconds)) or type(seed) is not int or seed<0:
        raise ValueError('invalid bounded execution settings')
    maximum=len(selected)*len(order)*bank['max_ticks']
    if budget<maximum: raise ValueError('budget must reserve complete-bank worst case')
    plan=dict(schema=SCHEMA,bank=str(bank_path),bank_sha256=bank['bank_sha256'],catalog=str(catalog_path),
              proposer=proposer['name'],evaluator_order=list(order),candidate_indices=selected,seed=seed,
              horizon=bank['max_ticks'],budget=budget,maximum_first_attempt_interactions=maximum,
              backend=backend,batch_size=batch_size,max_candidates_per_process=max_candidates_per_process,
              timeout_seconds=timeout_seconds,role='train',final_test_used=False,
              conflict_policy='quarantine_contact_and_physical_failure_v1',
              aggregation='first_observed_nonconflicting_landing; all failures required for no-witness',
              input_files={str(p):file_sha(p) for p in (bank_path,catalog_path,catalog_path.parent/'protocol.json')},
              source_files={str(p):file_sha(p) for p in [
                  *Path(__file__).resolve().parents[1].rglob('*.py'),
                  Path(__file__).resolve().parents[3]/'cli/label_policy_family_first_landing.py']})
    plan['plan_sha256']=canonical_sha256(plan)
    with Path(output).open('x') as f:
        import json
        f.write(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    return plan


def load_plan(path):
    from ..probe_bank import load_probe_bank
    plan=read(path);verify_hash(plan,'plan_sha256')
    if plan.get('schema')!=SCHEMA: raise ValueError('existence plan schema drift')
    for field in ('input_files','source_files'):
        for p,sha in plan[field].items():
            if file_sha(p)!=sha: raise ValueError('existence plan input/source drift')
    bank=load_probe_bank(Path(plan['bank']))
    if bank['bank_sha256']!=plan['bank_sha256']: raise ValueError('existence bank drift')
    return plan,bank


def validate_subset(result_dir,plan,member,proposer,indices):
    from ..policy_family_landing import _requested_contract, _verify_cached_contract, _verify_cached_rows
    report=read(result_dir/'summary.json');execution=read(result_dir/'execution.json')
    if report.get('status')!='completed_subset' or execution.get('status')!='completed':
        raise ValueError('subset incomplete')
    for obj in (report,execution):
        if obj.get('selected_candidate_indices')!=indices or obj.get('execution_backend')!=plan['backend'] or obj.get('batch_size')!=plan['batch_size']:
            raise ValueError('subset execution identity drift')
    requested=_requested_contract(plan['catalog'],proposer['policy'],member['policy'],member['frozen_file_sha256'],plan['horizon'],plan['seed'])
    protocol=_verify_cached_contract(result_dir,requested)
    for obj in (report,execution):
        if obj.get('logical_protocol_sha256')!=protocol['protocol_sha256']:
            raise ValueError('subset protocol identity drift')
    if (report.get('candidate_count')!=len(indices)
        or execution.get('selected_candidate_count')!=len(indices)
        or execution.get('maximum_environment_interactions')!=len(indices)*plan['horizon']):
        raise ValueError('subset count/reservation drift')
    schedule={'serial':None,'vectorized':'vmap_checked_shared_warp_v2'}[plan['backend']]
    if execution.get('device_step_schedule')!=schedule:
        raise ValueError('subset execution schedule drift')
    rows=read(result_dir/'labels.json');catalog=read(plan['catalog'])
    _verify_cached_rows(rows,[catalog['entries'][i] for i in indices],protocol)
    if file_sha(result_dir/'labels.json')!=report.get('labels_file_sha256'):
        raise ValueError('subset label hash drift')
    if [r['candidate_index'] for r in rows]!=indices or any(r['policy_key_candidate_index']!=i for i,r in zip(indices,rows)):
        raise ValueError('subset global key drift')
    charge=report['environment_interactions']
    if type(charge) is not int or not 0<=charge<=len(indices)*plan['horizon']:
        raise ValueError('subset cost drift')
    padding=report.get('inactive_lane_interactions')
    useful=sum(r['environment_interactions'] for r in rows)
    if (type(padding) is not int or padding<0
        or report.get('useful_label_interactions')!=useful or charge!=useful+padding):
        raise ValueError('subset padding cost drift')
    return rows,charge


def run(plan_path,output,*,gpu='0',python=sys.executable):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(plan_path,output,gpu=gpu,python=python)


def _run(plan_path,output,*,gpu,python):
    plan,bank=load_plan(plan_path);members={m['name']:m for m in bank['members']};proposer=members[plan['proposer']]
    identity={'plan_sha256':plan['plan_sha256']}
    if (output/'identity.json').exists() and read(output/'identity.json')!=identity:
        raise ValueError('output plan changed')
    write(output/'identity.json',identity)
    evaluated={i:{} for i in plan['candidate_indices']};attempts=[];spent=0;budget_exhausted=False
    identity_error=None
    cli=Path(__file__).resolve().parents[3]/'cli/label_policy_family_first_landing.py'
    started=time.perf_counter()
    # Audit every prior reservation before considering any new launch. In
    # particular, later evaluators still cost money when an earlier retry wins.
    for attempt in sorted(output.glob('evaluator_*/chunk_*/attempt_*')):
        reservation=read(attempt/'reservation.json')
        name=reservation.get('evaluator');indices=reservation.get('candidate_indices')
        if (reservation.get('plan_sha256')!=plan['plan_sha256']
            or name not in plan['evaluator_order'] or not isinstance(indices,list)
            or not indices or any(type(i) is not int or i not in evaluated for i in indices)
            or indices!=sorted(set(indices))
            or reservation.get('maximum_interactions')!=len(indices)*plan['horizon']):
            raise ValueError('attempt reservation identity drift')
        completion=read(attempt/'completion.json') if (attempt/'completion.json').exists() else {}
        cost=reservation['maximum_interactions']
        if completion.get('status')=='completed':
            if read(attempt/'exit.json')['returncode']!=0:
                raise ValueError('completed attempt exit drift')
            rows,cost=validate_subset(attempt/'result',plan,members[name],proposer,indices)
            for i,row in zip(indices,rows):
                if name in evaluated[i] and 'engineering_error' not in evaluated[i][name] and evaluated[i][name]!=row:
                    raise ValueError('repeated completed candidate evidence drift')
                evaluated[i][name]=row
        else:
            for i in indices:
                evaluated[i].setdefault(name, {'label':None,
                    'engineering_error':completion.get('error') or completion.get('status','interrupted')})
        spent+=cost
        attempts.append(dict(path=str(attempt),charged_interactions=cost,
                             status=completion.get('status','interrupted')))
    write(output/'cost_ledger.json',dict(charged_interactions=spent,attempts=attempts))
    for position,name in enumerate(plan['evaluator_order']):
        unresolved=[i for i,v in evaluated.items()
                    if aggregate_candidate(plan['evaluator_order'],v)['label']!=1
                    and (name not in v or 'engineering_error' in v[name])]
        for offset in range(0,len(unresolved),plan['max_candidates_per_process']):
            indices=unresolved[offset:offset+plan['max_candidates_per_process']]
            # Resume can shrink a later evaluator's subset after an earlier
            # retry succeeds. Preserve old chunks and identify the new selection.
            stage=output/f'evaluator_{position:03d}'/f'chunk_{canonical_sha256(indices)}'
            reservation=dict(plan_sha256=plan['plan_sha256'],evaluator=name,candidate_indices=indices,
                             maximum_interactions=len(indices)*plan['horizon'])
            if spent+reservation['maximum_interactions']>plan['budget']:
                budget_exhausted=True;break
            attempt=stage/f'attempt_{len(list(stage.glob("attempt_*"))):04d}'
            write(attempt/'reservation.json',reservation);write(attempt/'indices.json',indices)
            command=[python,str(cli),'--catalog',plan['catalog'],'--acquisition-frozen-policy',proposer['frozen_policy'],
                     '--evaluator-frozen-policy',members[name]['frozen_policy'],'--output-dir',str(attempt/'result'),
                     '--shard-index','0','--shard-count','1','--candidate-indices',str(attempt/'indices.json'),
                     '--max-ticks',str(plan['horizon']),'--protocol-seed',str(plan['seed']),
                     '--execution-backend',plan['backend'],'--batch-size',str(plan['batch_size'])]
            write(attempt/'command.json',command)
            env=dict(os.environ,PYTHONPATH=str(cli.parents[1]/'src'),JAX_PLATFORMS='cuda,cpu',CUDA_VISIBLE_DEVICES=str(gpu),
                     XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1')
            begin=time.perf_counter();wall_start=datetime.now(timezone.utc).isoformat();error=None;code=None;rows=None
            print(f'[existence] {name} candidates={len(indices)} log={attempt/"process.log"}',flush=True)
            try:
                with (attempt/'process.log').open('w') as log:
                    child=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=plan['timeout_seconds'])
                code=child.returncode
                if code: raise RuntimeError(f'worker exited {code}')
                try:
                    current_plan,_=load_plan(plan_path)
                    if current_plan['plan_sha256']!=plan['plan_sha256']:
                        raise ValueError('existence plan changed during worker execution')
                except Exception as exc:
                    identity_error=f'{type(exc).__name__}: {exc}'
                    raise
                rows,cost=validate_subset(attempt/'result',plan,members[name],proposer,indices)
            except Exception as exc:
                cost=reservation['maximum_interactions'];error=f'{type(exc).__name__}: {exc}'
            write(attempt/'exit.json',dict(returncode=code,start_utc=wall_start,end_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-begin))
            write(attempt/'completion.json',dict(status='completed' if rows is not None else 'engineering_error',
                                                charged_interactions=cost,error=error))
            spent+=cost;attempts.append(dict(path=str(attempt),charged_interactions=cost,status='completed' if rows is not None else 'engineering_error'))
            if rows is not None:
                for i,row in zip(indices,rows): evaluated[i][name]=row
            else:
                for i in indices: evaluated[i][name]={'label':None,'engineering_error':error}
            write(output/'cost_ledger.json',dict(charged_interactions=spent,attempts=attempts))
            if identity_error: break
        if budget_exhausted or identity_error: break
    catalog=read(plan['catalog']);entries=[]
    for index,values in evaluated.items():
        entries.append({**catalog['entries'][index],'candidate_index':index,
                        **aggregate_candidate(plan['evaluator_order'],values)})
    report=dict(schema='jit_first_success_witness_index_v1',plan_sha256=plan['plan_sha256'],bank_sha256=bank['bank_sha256'],
                status='completed' if all(r['label'] is not None for r in entries) else 'partial_unknown',
                entries=entries,charged_interactions=spent,attempts=attempts,wall_seconds=time.perf_counter()-started,
                budget_exhausted=budget_exhausted,execution_identity_error=identity_error,complete_evaluator_matrix=False,
                positive_count=sum(r['label']==1 for r in entries),negative_count=sum(r['label']==0 for r in entries),
                unknown_count=sum(r['label'] is None for r in entries),final_test_used=False,training_transitions=0)
    report['index_sha256']=canonical_sha256(report)
    write(output/'witness_index.json',report);write(output/'summary.json',{k:v for k,v in report.items() if k!='entries'})
    write(output/'cost_ledger.json',dict(charged_interactions=spent,attempts=attempts))
    return report
