"""One gated GPU lifetime for source checks, B1 search and independent B1 replay."""
from pathlib import Path
import json
import shutil
import time
from .production import ProductionRunner,read
from .contracts import file_sha,digest
from .protocol import atomic_json
from .teacher_runtime import TeacherEvaluationSession


class PersistentTeacherRunner(ProductionRunner):
    def __init__(self,manifest):
        super().__init__(manifest,_teacher_worker=True)
        self.persistent_teacher=True
        self.session=TeacherEvaluationSession()

    def child(self,name,argv,maximum,*,updates=0,extra_env=None):
        from ..pulse_exploration_runtime import evaluate
        if updates or argv[:3]!=['JIT/cli/run_pulse_exploration.py','--mode','evaluate']:
            raise ValueError('persistent teacher only executes physical evaluations')
        if sum(c['charged_interactions'] for c in self.costs)+maximum>self.spec['budgets']['max_physics']:
            raise ValueError('teacher worker reservation exhausted')
        config=Path(argv[argv.index('--spec')+1]);out=Path(argv[argv.index('--output')+1])
        cost=dict(stage=name,charged_interactions=maximum,charged_updates=0,phase='running',
            accounting='conservative reservation until actual receipt')
        self.costs.append(cost);atomic_json(self.root/'costs.json',self.costs)
        self.status('running',stage=name)
        start=time.monotonic()
        evaluate(read(config),out,session=self.session)
        cost.update(phase='completed',wall_seconds=time.monotonic()-start)
        return cost


def dispatch_teacher(parent,incumbent):
    if parent.spec.get('teacher_layout')!='source_control_in_candidate_batch' or parent.spec.get('teacher_colored_noise_candidates')!=0:
        raise ValueError('persistent B1 requires original 17-world candidate layout')
    validate_teacher_acceptance(parent.spec)
    root=parent.root/'teacher_worker';root.mkdir(exist_ok=False)
    maximum=len(parent.panels['new_roots'])*400*(1+17*2)
    spec={**parent.spec,'output':str(root),'teacher_execution':'worker_internal',
          'budgets':{**parent.spec['budgets'],'max_physics':maximum}}
    atomic_json(root/'production.json',spec);atomic_json(root/'panels.json',parent.panels)
    request=root/'request.json';atomic_json(request,dict(production=str(root/'production.json'),incumbent=incumbent,
        input_files={str(root/name):file_sha(root/name) for name in ('production.json','panels.json')}))
    request_sha=file_sha(request)
    cost=parent.child('teacher_worker',['JIT/cli/run_bridge_teacher_worker.py','--request',request],maximum)
    receipt=read(root/'completed.json')
    if (receipt['phase']!='completed' or file_sha(root/'results.json')!=receipt['results_sha256']
        or file_sha(request)!=request_sha or receipt.get('request_sha256')!=request_sha):
        raise ValueError('incomplete persistent teacher receipt')
    if receipt['charged_interactions']>maximum:raise ValueError('teacher actuals exceed reservation')
    teachers=read(root/'results.json')
    if set(teachers)!={r['root_id'] for r in parent.panels['new_roots']}:
        raise ValueError('persistent teacher omitted roots')
    # Preserve canonical round-level evidence paths while all heavy arrays stay
    # immutable in the worker directory. No original evaluation is overwritten.
    target=parent.root/'teachers';target.mkdir(exist_ok=False)
    for p in (root/'teachers').glob('*.json'):
        shutil.copyfile(p,target/p.name)
    cost.update(charged_interactions=receipt['charged_interactions'],
        active_interactions=receipt['active_interactions'],accounting='measured',
        worker_receipt=str(root/'completed.json'),worker_receipt_sha256=file_sha(root/'completed.json'))
    atomic_json(parent.root/'costs.json',parent.costs)
    return teachers


def run(request):
    request_path=Path(request);request_sha=file_sha(request_path);request=read(request_path)
    from .verified_files import LOCK_CACHE
    for path,sha in request['input_files'].items():LOCK_CACHE.verify(path,sha)
    spec=read(request['production']);root=Path(spec['output'])
    runner=PersistentTeacherRunner(spec)
    try:
        result=runner.teacher_search(request['incumbent'])
        if file_sha(request_path)!=request_sha:raise ValueError('teacher request changed during execution')
        for path,sha in request['input_files'].items():LOCK_CACHE.verify(path,sha)
        for p in (root/'teachers').glob('*_result.json'):
            row=read(p);proposal=p.with_name(p.name.replace('_result.json','_proposals.npz'))
            if proposal.exists():row['proposal_path']=str(proposal);atomic_json(p,row);result[row['root_id']]=row
        atomic_json(root/'results.json',result)
        runner.status('completed',stage='teacher_complete')
        atomic_json(root/'completed.json',dict(phase='completed',results_sha256=file_sha(root/'results.json'),
            charged_interactions=sum(c['charged_interactions'] for c in runner.costs),
            active_interactions=sum(c.get('active_interactions',0) for c in runner.costs),
            kernel_compile_count=runner.session.compile_count,roots=len(result),layout='B1_17_worlds',
            request_sha256=request_sha))
    except BaseException as exc:
        runner.status('failed',error=repr(exc));raise
    finally:runner.session.close()


def validate_teacher_acceptance(spec):
    """Experimental B1 cannot enter production on a CPU-only or failed audit."""
    receipt=spec.get('teacher_physical_acceptance')
    if not receipt or set(receipt)!={'path','sha256'}:
        raise ValueError('persistent B1 requires a passed physical acceptance receipt')
    if file_sha(receipt['path'])!=receipt['sha256']:raise ValueError('teacher acceptance changed')
    report=read(receipt['path'])
    if (report.get('schema')!='jit_b1_physical_acceptance_v1'
        or report.get('implementation_commit')!=spec['implementation_commit']
        or any(report.get(k) is not True for k in ('aba_passed','baseline_comparison_passed',
                  'full_search_replay_passed','rng_mapping_passed'))):
        raise ValueError('persistent B1 physical acceptance missing or failed')
