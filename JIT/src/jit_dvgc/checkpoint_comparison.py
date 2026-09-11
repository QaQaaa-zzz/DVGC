"""Bounded matched-schedule development exploration of frozen checkpoints."""
from pathlib import Path
import csv
import fcntl
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from .jump_evidence_validation import read, write, file_sha, verify_hash
from .evidence_integrity import canonical_sha256


def schedule_bounds(template, anchors, evaluator_count):
    from .acquisition.causal_jump import _variant_specs, VARIANT_PARTITION_MODULUS
    from .unified_boundary import action_sparse_directions
    for value in (template['max_forward_ticks'],template['max_candidates_per_attempt'],evaluator_count):
        if type(value) is not int or value<=0: raise ValueError('positive integer schedule limits required')
    for key in ('strengths','lookbacks_m'):
        if not template[key] or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in template[key]):
            raise ValueError('finite positive perturbation settings required')
    if any(type(a['proposal_family_index']) is not int or not 0<=a['proposal_family_index']<VARIANT_PARTITION_MODULUS for a in anchors):
        raise ValueError('invalid proposal family partition')
    directions=action_sparse_directions(action_names=template['action_names'],signs=template['signs'])
    variants=_variant_specs(lookbacks_m=template['lookbacks_m'],strengths=template['strengths'],directions=directions)
    trajectories=sum(sum(v['ordinal']%VARIANT_PARTITION_MODULUS==a['proposal_family_index'] for v in variants) for a in anchors)
    if not trajectories: raise ValueError('empty exploration schedule')
    maximum=len(anchors)*((len(variants)+VARIANT_PARTITION_MODULUS-1)//VARIANT_PARTITION_MODULUS)*template['max_forward_ticks']
    candidates=trajectories*template['max_candidates_per_attempt']
    return dict(trajectories=trajectories,acquisition_ceiling=maximum,candidate_ceiling=candidates,
                label_ceiling=candidates*evaluator_count*400)


def prepare(spec_path, output):
    from .probe_bank import load_probe_bank
    from .unified_policy_freeze import load_frozen_unified_manifest
    spec=read(spec_path);bank=load_probe_bank(Path(spec['bank']))
    members={m['name']:m for m in bank['members']}
    if bank['max_ticks']!=400: raise ValueError('comparison requires declared 400 tick endpoint')
    for m in members.values():
        if load_frozen_unified_manifest(Path(m['frozen_policy']))['policy']!=m['policy']:
            raise ValueError('bank member binding drift')
    names=spec['proposers'];order=spec['evaluator_order']
    if len(names)<2 or len(set(names))!=len(names) or any(n not in members or 'proposer' not in members[n]['roles'] for n in names):
        raise ValueError('distinct declared proposers required')
    if set(order)!={n for n,m in members.items() if 'evaluator' in m['roles']} or len(set(order))!=len(order):
        raise ValueError('complete distinct common evaluator order required')
    template=spec['acquisition']
    if template['role']!='train' or template['sampling_mode']!='trajectory_slices_v2' or template['slice_spacing_m']!=.05:
        raise ValueError('TRAIN real 5cm trajectory samples required')
    bounds=schedule_bounds(template,read(template['anchors']),len(order))
    for key in ('per_arm_budget','timeout_seconds'):
        if type(spec[key]) is not int or spec[key]<=0: raise ValueError('positive bounded settings required')
    if spec['per_arm_budget']<bounds['acquisition_ceiling']+bounds['label_ceiling']:
        raise ValueError('budget below full declared worst case')
    if set(spec['training_surcharges'])!=set(names) or any(type(v) is not int or v<0 for v in spec['training_surcharges'].values()):
        raise ValueError('explicit nonnegative incremental training costs required')
    inputs=[Path(spec_path),Path(spec['bank']),Path(spec['baseline_csv']),*(Path(template[k]) for k in ('anchors','start_contract','nominal_centerline'))]
    inputs += [Path(m['frozen_policy']) for m in members.values()]
    plan=dict(schema='jit_checkpoint_discovery_plan_v1',spec=spec,bounds=bounds,
              maximum_total_interactions=len(names)*spec['per_arm_budget'],role='train',final_test_used=False,
              comparison_scope='matched schedule; conservative catalog-order equal-cost replay; no automatic promotion',
              input_files={str(p.resolve()):file_sha(p) for p in inputs},
              source_files={str(p.resolve()):file_sha(p) for directory in ('src/jit_dvgc','cli') for p in (Path(__file__).resolve().parents[2]/directory).rglob('*.py')})
    plan['plan_sha256']=canonical_sha256(plan)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',plan)
    return plan


def verify_plan(path):
    plan=read(path);verify_hash(plan,'plan_sha256')
    for field in ('input_files','source_files'):
        for name,sha in plan[field].items():
            if file_sha(name)!=sha: raise ValueError('comparison input/source drift')
    return plan


def _worker(command, output, maximum, timeout):
    output.mkdir(parents=True,exist_ok=False)
    write(output/'reservation.json',dict(maximum_interactions=maximum))
    write(output/'command.json',command)
    start=time.perf_counter();begin=datetime.now(timezone.utc).isoformat();code=None;error=None
    env=dict(os.environ,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',JIT_AUTO_PUBLISH='0')
    child=None
    try:
        with (output/'process.log').open('w') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=child.wait(timeout=timeout)
        if code: error=f'worker exited {code}'
    except Exception as exc: error=f'{type(exc).__name__}: {exc}'
    finally:
        if child is not None:
            # The label supervisor launches evaluator grandchildren. Kill its
            # entire process group on timeout/interruption, preserving reservation.
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()
    write(output/'exit.json',dict(returncode=code,error=error,start_utc=begin,end_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start))
    return error


def project_catalog(catalog_path, member):
    from .probe_bank import validate_probe_arrivals
    from .unified_continuation_labels import validate_unified_boundary_catalog
    from .unified_envelope_snapshot import load_unified_envelope_snapshot,snapshot_context_sha256
    from .analysis.capability_tube import physical_coordinates_from_arrays
    from .analysis.policy_envelopes import project_row
    from .model import load_host_model
    from .config import load_config
    catalog=read(catalog_path)
    rows=validate_unified_boundary_catalog(catalog,policy_record=member['policy'],frozen_manifest_sha256=member['frozen_file_sha256'],allow_empty=True)
    validate_probe_arrivals(catalog_path,rows,member['policy'])
    config=read(member['policy']['formal_config']);bundle=load_host_model(load_config(Path(config['inputs']['up_config_path'])))
    if bundle.xml_sha256!=member['policy']['xml_sha256']: raise ValueError('projection model drift')
    result=[]
    for row in rows:
        snap=load_unified_envelope_snapshot(catalog_path.parent/row['source_bank']/row['snapshot'])
        coordinates=physical_coordinates_from_arrays(snap.qpos,snap.qvel,bundle=bundle)
        point=project_row(row,coordinates,snapshot_context_sha256(snap),x_slice_width_m=.05)
        result.append({**point,'trajectory_id':row['trajectory_id'],'trajectory_step':row['trajectory_step']})
    return result


def validate_witness_receipt(index, plan, catalog, members, proposer, output, budget):
    """Rebuild witness outcomes and costs from all identity-checked attempts."""
    from .continuation.existence import aggregate_candidate, validate_subset
    if (index.get('schema')!='jit_first_success_witness_index_v1'
        or index.get('plan_sha256')!=plan['plan_sha256'] or index.get('bank_sha256')!=plan['bank_sha256']
        or index.get('status') not in ('completed','partial_unknown')
        or index.get('execution_identity_error') is not None
        or index.get('final_test_used') is not False or index.get('training_transitions')!=0
        or type(index.get('charged_interactions')) is not int or not 0<=index['charged_interactions']<=budget):
        raise ValueError('witness receipt identity/accounting drift')
    evaluated={i:{} for i in plan['candidate_indices']};costs=[0]*len(catalog['entries']);charged=0
    paths=[Path(a['path']) for a in index['attempts']]
    inventory=set(Path(output).glob('evaluator_*/chunk_*/attempt_*'))
    if len(paths)!=len(set(paths)) or set(paths)!=inventory:
        raise ValueError('witness attempt inventory drift')
    for path,attempt in zip(paths,index['attempts'],strict=True):
        reservation=read(path/'reservation.json');name=reservation.get('evaluator');indices=reservation.get('candidate_indices')
        if (reservation.get('plan_sha256')!=plan['plan_sha256'] or name not in plan['evaluator_order']
            or not isinstance(indices,list) or not indices or any(type(i) is not int or i not in evaluated for i in indices)
            or indices!=sorted(set(indices)) or reservation.get('maximum_interactions')!=len(indices)*plan['horizon']):
            raise ValueError('witness attempt reservation drift')
        completion=read(path/'completion.json') if (path/'completion.json').exists() else {}
        status=completion.get('status','interrupted');cost=reservation['maximum_interactions']
        if status=='completed':
            if read(path/'exit.json').get('returncode')!=0: raise ValueError('witness attempt exit drift')
            rows,cost=validate_subset(path/'result',plan,members[name],proposer,indices)
            for i,row in zip(indices,rows,strict=True):
                if name in evaluated[i] and 'engineering_error' not in evaluated[i][name] and evaluated[i][name]!=row:
                    raise ValueError('witness repeated evidence drift')
                evaluated[i][name]=row;costs[i]+=row['environment_interactions']
        else:
            for i in indices:
                evaluated[i].setdefault(name,{'label':None,'engineering_error':completion.get('error') or status})
        if attempt.get('status')!=status or attempt.get('charged_interactions')!=cost:
            raise ValueError('witness attempt charge drift')
        charged+=cost
    expected=[{**catalog['entries'][i],'candidate_index':i,**aggregate_candidate(plan['evaluator_order'],values)} for i,values in evaluated.items()]
    expected_status='completed' if all(row['label'] is not None for row in expected) else 'partial_unknown'
    if index['entries']!=expected or index['status']!=expected_status or index['charged_interactions']!=charged:
        raise ValueError('witness outcomes/accounting drift')
    return costs


def run(output, *, gpu='0'):
    output=Path(output).resolve()
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(output,gpu)


def _run(output,gpu):
    from .probe_bank import load_probe_bank
    from .continuation.existence import prepare as prepare_existence
    from .analysis.checkpoint_discovery import summarize_arms, render
    plan=verify_plan(output/'plan.json');spec=plan['spec'];bank=load_probe_bank(Path(spec['bank']))
    members={m['name']:m for m in bank['members']};arms={};ledger=[];start=time.perf_counter()
    if (output/'started.json').exists(): raise ValueError('single attempt used; preserve evidence and predeclare a separate retry')
    write(output/'started.json',dict(plan_sha256=plan['plan_sha256'],start_utc=datetime.now(timezone.utc).isoformat()))
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    cli=Path(__file__).resolve().parents[2]/'cli/probe_bank.py'
    try:
        for name in spec['proposers']:
            verify_plan(output/'plan.json');arm=output/name;arm.mkdir()
            acq={**spec['acquisition'],'bank':spec['bank'],'proposer':name,'interaction_ceiling':plan['bounds']['acquisition_ceiling']}
            write(arm/'acquisition_spec.json',acq)
            command=[sys.executable,str(cli),'acquire','--spec',str(arm/'acquisition_spec.json'),'--output',str(arm/'acquire/result')]
            reserved=plan['bounds']['acquisition_ceiling'];entry=dict(proposer=name,stage='acquisition',charged_interactions=reserved,status='reserved');ledger.append(entry);write(output/'cost_ledger.json',ledger)
            print(f'[comparison] acquire {name}',flush=True)
            error=_worker(command,arm/'acquire',reserved,spec['timeout_seconds'])
            if error: raise RuntimeError(error)
            verify_plan(output/'plan.json')
            catalog_path=arm/'acquire/result/catalog.json';catalog=read(catalog_path)
            if catalog['status']!='completed' or type(catalog['environment_interactions']) is not int or not 0<=catalog['environment_interactions']<=reserved: raise ValueError('acquisition accounting drift')
            points=project_catalog(catalog_path,members[name]);write(arm/'projected.json',points)
            entry.update(charged_interactions=catalog['environment_interactions'],status='completed');write(output/'cost_ledger.json',ledger)
            witnesses=[];labelcost=[0]*len(points);label_charge=0
            if points:
                remaining=spec['per_arm_budget']-entry['charged_interactions']
                prepare_existence(spec['bank'],catalog_path,arm/'existence_plan.json',order=spec['evaluator_order'],seed=spec['label_seed'],budget=remaining,max_candidates_per_process=128,timeout_seconds=spec['timeout_seconds'])
                # Reserve all remaining work in case supervisor is interrupted before its final receipt.
                label_entry=dict(proposer=name,stage='labels',charged_interactions=remaining,status='reserved');ledger.append(label_entry);write(output/'cost_ledger.json',ledger)
                command=[sys.executable,str(cli),'run-existence','--plan',str(arm/'existence_plan.json'),'--output',str(arm/'labels/result'),'--gpu',str(gpu)]
                error=_worker(command,arm/'labels',remaining,spec['timeout_seconds']*len(spec['evaluator_order']))
                verify_plan(output/'plan.json')
                index_path=arm/'labels/result/witness_index.json'
                if not index_path.exists(): raise RuntimeError(error or 'missing witness index')
                index=read(index_path);verify_hash(index,'index_sha256')
                if error:
                    exit_receipt=read(arm/'labels/exit.json')
                    if not (error=='worker exited 2' and exit_receipt.get('returncode')==2 and index.get('status')=='partial_unknown'):
                        raise RuntimeError(error)
                from .continuation.existence import load_plan as load_existence_plan
                existence_plan,_=load_existence_plan(arm/'existence_plan.json')
                labelcost=validate_witness_receipt(index,existence_plan,catalog,members,members[name],arm/'labels/result',remaining)
                label_charge=index['charged_interactions']
                if not 0<=label_charge<=remaining:raise ValueError('label budget exceeded')
                label_entry.update(charged_interactions=label_charge,status=index['status']);write(output/'cost_ledger.json',ledger)
                witnesses=index['entries']
            arms[name]=dict(points=points,witnesses=witnesses,per_candidate_label_cost=labelcost,
                acquisition_interactions=entry['charged_interactions'],charged_interactions=entry['charged_interactions']+label_charge,
                training_surcharge=spec['training_surcharges'][name],trajectory_receipts=catalog['trajectory_receipts'])
            write(arm/'analysis_inputs.json',arms[name])
        # Compare exact scheduled perturbations, independent of policy-dependent trajectory length.
        schedules=[[{k:r[k] for k in ('trajectory_id','anchor_x_m','strength','direction')} for r in arm['trajectory_receipts']] for arm in arms.values()]
        if any(s!=schedules[0] for s in schedules[1:]) or len(schedules[0])!=plan['bounds']['trajectories']:raise ValueError('matched trajectory schedule drift')
        with Path(spec['baseline_csv']).open() as f:
            baseline={r['root_cell'] for r in csv.DictReader(f) if r['witnessed']=='True'}
        summary=summarize_arms(arms,baseline,baseline_name=spec['baseline_csv'])
        render(summary,output/'figures')
        result=dict(status='completed',charged_interactions=sum(e['charged_interactions'] for e in ledger),wall_seconds=time.perf_counter()-start,plan_sha256=plan['plan_sha256'],analysis=summary,training_transitions=0,final_test_used=False)
    except Exception as exc:
        result=dict(status='engineering_error',error=f'{type(exc).__name__}: {exc}',charged_interactions=sum(e['charged_interactions'] for e in ledger),wall_seconds=time.perf_counter()-start,plan_sha256=plan['plan_sha256'],training_transitions=0,final_test_used=False)
    write(output/'summary.json',result)
    return result
