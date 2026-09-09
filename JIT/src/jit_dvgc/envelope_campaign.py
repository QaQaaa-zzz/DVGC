"""Resumable exploration -> witnessed support -> PPO -> new frozen probe loop.

Stopping reports empirical search stagnation, never the complete physical limit.
"""
from collections import defaultdict,Counter
from pathlib import Path
import fcntl
import os
import subprocess
import sys
import traceback
from .jump_evidence_validation import read,write,file_sha,verify_hash
from .evidence_integrity import canonical_sha256
from .result_bundle import bundle

DEFAULT_OUTPUT='JIT/runs/campaign/empirical_envelope_v1'
DEFAULT_BASELINE='JIT/runs/policy_comparison/expanded_pi0_pi3_20260907'


def profile_for(round_index,extra_policies,source_files):
    return dict(version='iterative_discovery_v1',round_index=round_index,
        targets=[2.85+.025*(round_index%3),2.95+.025*(round_index%3)],
        strengths=[.1+.025*(round_index%3),.2+.025*(round_index%3)],
        max_trajectories=32,max_candidates=128,sampling_max_x_m=8.,acquisition_ceiling=16000,
        acquisition_seed=9850001+round_index*1000,label_seed=9850501+round_index*1000,
        serial_only=True,extra_policies=extra_policies,source_files=source_files)


def validate_profile(profile):
    r=profile['round_index']
    if type(r) is not int or not 0<=r<20:raise ValueError('invalid campaign round')
    if profile!=profile_for(r,profile['extra_policies'],profile['source_files']):raise ValueError('campaign profile drift')
    if len(profile['extra_policies'])!=r+1:raise ValueError('bank growth drift')
    for path,sha in profile['source_files'].items():
        if file_sha(path)!=sha:raise ValueError('campaign evidence changed')


def read_child(child):
    """Verify completed historical data without requiring old source-code hashes."""
    from .policy_comparison_runtime import complete_output
    from .unified_continuation_labels import validate_unified_boundary_catalog
    from .probe_bank import validate_probe_arrivals
    child=Path(child).resolve();plan=read(child/'plan.json');verify_hash(plan,'plan_sha256')
    summary=read(child/'summary.json')
    if summary.get('status') not in ('completed','completed_empty') or plan.get('role')!='train':raise ValueError('completed TRAIN source required')
    inputs={}
    def lock(p):inputs[str(Path(p).resolve())]=file_sha(p)
    for p in (child/'plan.json',child/'summary.json',child/'analysis_inputs.json'):lock(p)
    for p,sha in plan['input_files'].items():
        if file_sha(p)!=sha:raise ValueError('historical input changed')
        inputs[p]=sha
    manifest=read(child/'analysis_inputs.json');catalog_path=Path(manifest['catalog'])
    catalog=read(catalog_path);points=read(manifest['projected']);lock(catalog_path);lock(manifest['projected'])
    protocol=read(catalog_path.parent/'protocol.json');verify_hash(protocol,'protocol_sha256');lock(catalog_path.parent/'protocol.json')
    if (catalog.get('protocol_sha256')!=protocol['protocol_sha256'] or protocol.get('logical_role')!='train'
        or protocol.get('probe_bank_sha256')!=plan['bank_sha256']):
        raise ValueError('source role/bank mismatch')
    proposer=next(m for m in plan['members'] if m['policy']['name']==plan['proposer'])
    entries=validate_unified_boundary_catalog(catalog,policy_record=proposer['policy'],
        frozen_manifest_sha256=proposer['file_sha256'],allow_empty=True)
    validate_probe_arrivals(catalog_path,entries,proposer['policy'])
    labels={}
    for member in plan['members']:
        name=member['policy']['name']
        if not points:labels[name]=[];continue
        directory=Path(manifest['merged'][name])
        result=complete_output(directory,catalog_path,proposer,member,400,plan.get('label_seed',9841201))
        if result is None:raise ValueError('incomplete campaign source labels')
        labels[name]=result[1]
        for f in ('summary.json','protocol.json','labels.json'):lock(directory/f)
    rows=[]
    for i,(entry,point) in enumerate(zip(entries,points,strict=True)):
        if any(entry[k]!=point[k] for k in ('candidate_id','state_sha256','snapshot_context_sha256')):raise ValueError('projection/source mismatch')
        values={}
        for name,panel in labels.items():
            row=panel[i]
            if any(row[k]!=entry[k] for k in ('candidate_id','state_sha256','phase')) or type(row['label']) is not int or row['label'] not in (0,1):raise ValueError('label mismatch')
            values[name]=row['label']
        snapshot=(catalog_path.parent/entry['source_bank']/entry['snapshot']).resolve()
        for f in ('identity.json','snapshot.pkl'):lock(snapshot/f)
        rows.append(dict(key=canonical_sha256([inputs[str(catalog_path.resolve())],entry['candidate_id']]),
            state_sha256=entry['state_sha256'],snapshot_context_sha256=entry['snapshot_context_sha256'],
            phase=entry['phase'],trajectory_id=str(catalog_path)+'::'+entry['trajectory_id'],
            root_cell=point['root_cell'],snapshot=str(snapshot),witnessed=any(values.values()),
            labels=values,coordinates=point['coordinates'],source=str(child)))
    return rows,inputs,plan


def support_view(rows,inputs,recent_source):
    """Bounded, phase/group-balanced witnessed support; old witnesses are retained in ledger."""
    selected=[]
    for phase in ('upstream','downstream'):
        groups=defaultdict(list)
        for row in rows:
            if row['phase']==phase and row['witnessed']:groups[row['trajectory_id']].append(row)
        if not groups:raise ValueError('no witnessed support in one training phase')
        ordered=[sorted(g,key=lambda r:r['key']) for _,g in sorted(groups.items())]
        picked=[]
        # Round robin over groups prevents dense long trajectories filling the cap.
        for index in range(max(map(len,ordered))):
            for group in ordered:
                if index<len(group):picked.append(group[index])
            if len(picked)>=512:break
        picked=picked[:512];counts=Counter(r['trajectory_id'] for r in picked)
        for row in picked:
            selected.append({**row,'sampling_weight':(2. if row['source']==recent_source else 1.)/counts[row['trajectory_id']]})
    result=dict(schema='jit_iterative_witnessed_support_v1',status='completed',role='train',
        final_test_used=False,entries=selected,inputs=inputs,
        selection='phase 50/50, round-robin trajectory cap 512 per phase, recent source weight multiplier 2',
        training_guidance_only=True,unwitnessed_resets_used=False)
    result['support_sha256']=canonical_sha256(result)
    return result


def should_stop(gains,patience,min_gain):
    return len(gains)>=patience and all(g<min_gain for g in gains[-patience:])


def render_progress(output,rows,rounds,seed_cells):
    """Observed states only: no filled hull or invented interpolation between gaps."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);figdir=output/'figures';figdir.mkdir(exist_ok=True)
    columns=['round','policy','novel_root_cells','campaign_union_root_cells','charged_interactions','bank_size','candidate_count']
    with (figdir/'coverage_cost.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rounds)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    sources=list(dict.fromkeys(r['source'] for r in rows))
    for index,source in enumerate(sources):
        selected=[r for r in rows if r['source']==source and r['witnessed'] and r['coordinates']]
        if selected:
            axes[0].scatter([r['coordinates']['root_x_m'] for r in selected],
                [r['coordinates']['root_z_m'] for r in selected],s=5,alpha=.6,
                label=Path(source).parent.name if Path(source).name=='discovery' else 'seed '+Path(source).name)
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=7)
    axes[0].set(xlabel='x (m)',ylabel='Root z (m)',title='Witnessed arrivals: observed x-z projection')
    axes[1].plot([0]+[r['charged_interactions'] for r in rounds],
                 [seed_cells]+[r['campaign_union_root_cells'] for r in rounds],marker='o')
    axes[1].set(xlabel='New campaign interactions (PPO + discovery)',ylabel='Union root-state cells',
                title='Campaign seed panels + new witnessed cells')
    for ax in axes:ax.grid(alpha=.2)
    for ext in ('png','pdf','svg'):fig.savefig(figdir/f'campaign_progress.{ext}',dpi=180)
    plt.close(fig)


def run(repo,output,*,gpu='0',max_rounds=3,budget=40_000_000,ppo_steps=512_000,patience=2,min_gain=5):
    repo,output=Path(repo).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result={'status':'engineering_error','physical_boundary_proven':False}
        request=dict(max_rounds=max_rounds,budget=budget,ppo_steps=ppo_steps,patience=patience,min_gain=min_gain)
        try:
            if not (1<=max_rounds<=20 and patience>=1 and min_gain>=1 and ppo_steps>0 and ppo_steps%3200==0):raise ValueError('invalid campaign limits')
            if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('campaign limits changed; use new directory')
            write(output/'request.json',request)
            seed_paths=[repo/'JIT/runs/discovery/knee_boundary_v1_budgetfix/pi_1',repo/'JIT/runs/discovery/lower_boundary_v1/pi_2']
            rows=[];inputs={};seed_plan=None
            for path in seed_paths:
                part,locked,seed_plan=read_child(path);rows+=part;inputs.update(locked)
            baseline=Path(seed_plan['baseline']);members=seed_plan['members'][:4]
            initializer=Path(next(m for m in members if m['policy']['name']=='pi_2')['path'])
            bootstrap=Path(next(m for m in members if m['policy']['name']=='pi_2')['policy']['formal_config'])
            if not (output/'seed_inputs.json').exists():write(output/'seed_inputs.json',inputs)
            if read(output/'seed_inputs.json')!=inputs:raise ValueError('seed inputs changed')
            sources={str(p.relative_to(repo)):file_sha(p) for p in (repo/'JIT').rglob('*.py') if 'runs' not in p.parts}
            if (output/'sources.json').exists() and read(output/'sources.json')!=sources:raise ValueError('campaign code changed; use a new campaign directory')
            write(output/'sources.json',sources)
            initial_cells={r['root_cell'] for r in rows if r['witnessed']};cells=set(initial_cells)
            extra=[];gains=[];rounds=[];recent=str(seed_paths[-1].resolve())

            def cost():
                total=0
                for p in output.glob('round_*/training_attempt_*/reservation.json'):
                    reservation=read(p);done=p.parent/'completion.json'
                    total+=read(done)['charged_interactions'] if done.exists() else reservation['maximum_interactions']
                for p in output.glob('round_*/discovery/cost_ledger.json'):total+=read(p)['charged_interactions']
                return total

            for index in range(max_rounds):
                round_dir=output/f'round_{index:03d}';round_dir.mkdir(exist_ok=True)
                support=support_view(rows,inputs,recent)
                support_path=round_dir/'support.json'
                if support_path.exists() and read(support_path)!=support:raise ValueError('round support changed')
                write(support_path,support)
                iteration=4+index
                # All failed attempts are reserved; fresh retry directories retain evidence.
                training_maximum=ppo_steps+4*400
                completed=None
                for attempt in sorted(round_dir.glob('training_attempt_*')):
                    if (attempt/'completion.json').exists():
                        record=read(attempt/'completion.json')
                        for p,sha in record['artifacts'].items():
                            if file_sha(p)!=sha:raise ValueError('completed training artifact changed')
                        completed=attempt;break
                if completed is None:
                    if cost()+training_maximum>budget:result['status']='budget_exhausted';break
                    attempt=round_dir/f'training_attempt_{len(list(round_dir.glob("training_attempt_*"))):03d}'
                    attempt.mkdir()
                    run_id=f'pi_{iteration}_campaign_r{index}_attempt{attempt.name[-3:]}'
                    from .iterative_probe_training import make_config
                    config_path=attempt/'config.json'
                    make_config(support_path,initializer,bootstrap,config_path,run_id,iteration,ppo_steps,9860001+index)
                    write(attempt/'reservation.json',{'maximum_interactions':training_maximum,'config_sha256':file_sha(config_path)})
                    # GPU remains first/default; Brax debug callbacks require a CPU backend.
                    env=dict(os.environ,PYTHONPATH=str(repo/'JIT/src')+os.pathsep+os.environ.get('PYTHONPATH',''),
                        CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                        JIT_RUN_ROOT=str(attempt/'training'),JIT_AUTO_PUBLISH='0',PYTHONUNBUFFERED='1')
                    command=[sys.executable,str(repo/'JIT/cli/train_unified.py'),'--config',str(config_path),'--run-id',run_id]
                    with (attempt/'process.log').open('w') as log:
                        process=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                        try:
                            while True:
                                try:code=process.wait(timeout=30);break
                                except subprocess.TimeoutExpired:print(f'[campaign] training round {index}; log={attempt / "process.log"}',flush=True)
                        finally:
                            if process.poll() is None:
                                process.terminate()
                                try:process.wait(timeout=10)
                                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                    write(attempt/'exit.json',{'returncode':code})
                    if code:raise RuntimeError(f'training failed; see {attempt}')
                    report_path=attempt/'training'/run_id/'formal_report.json';report=read(report_path)
                    if report['status']!='completed' or report['completed_training_transitions']!=ppo_steps:raise ValueError('incomplete PPO')
                    checkpoint=attempt/'training'/run_id/'checkpoints'/f'transition_{ppo_steps}'
                    from .unified_policy_freeze import freeze_unified_policy
                    freeze_unified_policy(attempt/'frozen',config_path=config_path,checkpoint=checkpoint,iteration=iteration,formal_report=report_path)
                    policy_path=attempt/'frozen/frozen_unified_policy.json'
                    artifacts={str(p):file_sha(p) for p in [config_path,report_path,policy_path,checkpoint/'identity.json',checkpoint/'payload.pkl']}
                    panel_cost=report['train_panel_interactions']
                    if type(panel_cost) is not int or not 0<=panel_cost<=1600:raise ValueError('invalid final panel accounting')
                    write(attempt/'completion.json',{'charged_interactions':ppo_steps+panel_cost,
                        'policy':str(policy_path),'artifacts':artifacts})
                    completed=attempt
                initializer=Path(read(completed/'completion.json')['policy']);extra.append(str(initializer))
                profile=profile_for(index,list(extra),{str(support_path):file_sha(support_path),str(initializer):file_sha(initializer)})
                discovery=round_dir/'discovery'
                # Existing charged discovery work is counted once, not subtracted twice.
                used=read(discovery/'cost_ledger.json')['charged_interactions'] if (discovery/'cost_ledger.json').exists() else 0
                from .dense_tube import run as explore
                available=budget-cost()+used
                ceiling=16000+32*128*(4+len(extra))*400
                if not (discovery/'request.json').exists() and available<ceiling:
                    result['status']='budget_exhausted';break
                # Preserve the per-round request budget on resume.
                local_budget=read(discovery/'request.json')['budget'] if (discovery/'request.json').exists() else available
                outcome=explore(repo,discovery,baseline=baseline,gpu=gpu,budget=local_budget,
                                proposer=f'pi_{iteration}',profile=profile)
                if outcome['status'] not in ('completed','completed_empty'):raise RuntimeError('discovery failed; missing labels are not negative evidence')
                new,locked,_=read_child(discovery)
                new_cells={r['root_cell'] for r in new if r['witnessed']}
                gain=len(new_cells-cells);gains.append(gain);cells|=new_cells
                rows+=new;inputs.update(locked);recent=str(discovery.resolve())
                round_record=dict(round=index,policy=f'pi_{iteration}',novel_root_cells=gain,
                    campaign_union_root_cells=len(cells),charged_interactions=cost(),
                    bank_size=4+len(extra),candidate_count=len(new))
                if (round_dir/'summary.json').exists():
                    prior=read(round_dir/'summary.json')
                    if any(prior[k]!=v for k,v in round_record.items() if k!='charged_interactions'):
                        raise ValueError('completed round summary changed')
                    round_record=prior
                rounds.append(round_record)
                write(round_dir/'summary.json',rounds[-1])
                render_progress(output,rows,rounds,len(initial_cells))
                write(output/'progress.json',{'rounds':rounds,'physical_boundary_proven':False})
                write(output/'summary.json',{'status':'running','rounds':rounds,
                    'charged_interactions':cost(),'physical_boundary_proven':False,'final_test_used':False})
                bundle(output)
                if should_stop(gains,patience,min_gain):result['status']='empirical_stagnation';break
            else:result['status']='round_limit_reached'
            result.update(rounds=rounds,charged_interactions=cost(),seed_root_cells=len(initial_cells),
                campaign_union_root_cells=len(cells),baseline_scope='completed knee refinement and lower-boundary TRAIN panels only',
                physical_boundary_proven=False,final_test_used=False,
                completed_training_transitions=ppo_steps*len(extra),frozen_new_policies=extra)
        except Exception as exc:
            result.update(error=str(exc),traceback=traceback.format_exc())
            if 'rounds' in locals():
                result['rounds']=rounds
                try:result['charged_interactions']=cost()
                except Exception:result['accounting_available']=False
        finally:
            write(output/'summary.json',result)
            print(f"[campaign] {result['status']}\nReturn this file: {bundle(output)}",flush=True)
        return result
