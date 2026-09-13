"""Read-only recovery of completed experiment evidence after plotting failure."""
from pathlib import Path
import fcntl
import traceback
from .jump_evidence_validation import read,write,file_sha
from .envelope_campaign import read_child,render_progress
from .dense_tube_runtime import analyze
from .result_bundle import bundle


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('analysis output must be separate from the source run')
    output.mkdir(parents=True,exist_ok=True)
    with (source/'execution.lock').open('a') as lock, (output/'execution.lock').open('a') as out_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(out_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result={'status':'engineering_error','environment_interactions':0,'training_transitions':0,
                'source_run':str(source),'source_run_modified':False,'physical_boundary_proven':False}
        try:
            request={'source':str(source),'mode':'verified_report_recovery_v1'}
            if (output/'request.json').exists() and read(output/'request.json')!=request:
                raise ValueError('analysis request changed')
            write(output/'request.json',request)
            seed_inputs=read(source/'seed_inputs.json')
            for path,sha in seed_inputs.items():
                if file_sha(path)!=sha:raise ValueError('seed evidence changed')
            seed_children=[Path(p).parent for p in seed_inputs if Path(p).name=='analysis_inputs.json']
            rows=[];locked={};rounds=[]
            for path in sorted(set(seed_children)):
                part,inputs,_=read_child(path);rows+=part;locked.update(inputs)
            if not rows:raise ValueError('missing seed arrivals')
            cells={r['root_cell'] for r in rows if r['witnessed']};seed_count=len(cells)
            training_cost=0;discovery_cost=0;completed_steps=0
            for rd in sorted(source.glob('round_*')):
                discovery=rd/'discovery'
                if not discovery.exists():continue
                completions=list(rd.glob('training_attempt_*/completion.json'))
                if len(completions)!=1:raise ValueError('one completed frozen training source required')
                completion=read(completions[0])
                for path,sha in completion['artifacts'].items():
                    if file_sha(path)!=sha:raise ValueError('training artifact changed')
                    locked[path]=sha
                from .unified_policy_freeze import load_frozen_unified_manifest
                policy=load_frozen_unified_manifest(Path(completion['policy']))['policy']
                for reservation in rd.glob('training_attempt_*/reservation.json'):
                    done=reservation.parent/'completion.json'
                    training_cost+=read(done)['charged_interactions'] if done.exists() else read(reservation)['maximum_interactions']
                part,inputs,plan=read_child(discovery,recover_figures_failure=True)
                if next(m['path'] for m in plan['members'] if m['policy']['name']==plan['proposer'])!=completion['policy']:
                    raise ValueError('discovery proposer does not match trained policy')
                locked.update(inputs)
                figures=output/rd.name/'figures'
                analyze(plan,discovery,output/rd.name,figures_dir=figures)
                new_cells={r['root_cell'] for r in part if r['witnessed']}
                gain=len(new_cells-cells);cells|=new_cells;rows+=part
                cost=read(discovery/'cost_ledger.json')['charged_interactions'];discovery_cost+=cost
                completed_steps+=policy['source_training_transitions']
                rounds.append(dict(round=int(rd.name.split('_')[1]),policy=policy['name'],novel_root_cells=gain,
                    campaign_union_root_cells=len(cells),charged_interactions=training_cost+discovery_cost,
                    bank_size=len(plan['members']),candidate_count=len(part)))
            if not rounds:raise ValueError('no recoverable discovery round')
            render_progress(output,rows,rounds,seed_count)
            write(output/'verified_inputs.json',locked)
            result.update(status='analysis_completed',rounds=rounds,seed_root_cells=seed_count,
                campaign_union_root_cells=len(cells),source_completed_training_transitions=completed_steps,
                source_charged_interactions=training_cost+discovery_cost,final_test_used=False,
                baseline_scope='campaign seed panels plus these rounds; not all historical Tube versions',
                original_supervisor_status=read(source/'summary.json')['status'])
        except Exception as exc:result.update(error=str(exc),traceback=traceback.format_exc())
        finally:
            write(output/'summary.json',result)
            print(f"[analysis] {result['status']}\nReturn this file: {bundle(output)}",flush=True)
        return result
