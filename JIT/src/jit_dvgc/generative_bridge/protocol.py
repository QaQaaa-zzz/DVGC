"""Stage commits for a bounded round. Callbacks own locked production receipts.

This engine never picks checkpoints by mtime and never launches a subprocess.
It can resume after G failure without repeating committed student computation.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from .contracts import digest,support_fingerprint,validate_pending_support_unchanged
from .outcomes import root_outcome


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        temporary=f.name
        json.dump(value,f,sort_keys=True,allow_nan=False,indent=2);f.write('\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def reserve_round(budgets,maximum):
    if set(budgets)!={'teacher','student','evaluation'} or any(type(v) is not int or v<0 for v in budgets.values()):
        raise ValueError('explicit integer teacher/student/evaluation budgets required')
    if budgets['student']<=0 or budgets['evaluation']<=0 or type(maximum) is not int or sum(budgets.values())>maximum:
        raise ValueError('budget must reserve student and evaluation before teacher')
    return dict(budgets)


def validate_teacher_results(roots,scheduled,results):
    if len(set(roots))!=len(roots) or len(set(scheduled))!=len(scheduled) or not set(scheduled)<=set(roots):
        raise ValueError('teacher subset must come from locked original roots')
    if set(results)!=set(scheduled):raise ValueError('incomplete teacher execution')
    out=deepcopy(results)
    for r in out.values():
        if r.get('teacher_status') not in ('searched_no_solution','verified_solution'):
            raise ValueError('invalid/incomplete teacher execution cannot fall back to PPO')
    for root in set(roots)-set(scheduled):out[root]={'teacher_status':'not_scheduled'}
    return out


class StageJournal:
    def __init__(self,path,contract):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.identity=digest(contract)
        lock=self.path/'round_contract.json'
        if lock.exists():
            if json.loads(lock.read_text())!={'sha256':self.identity,'contract':contract}:
                raise ValueError('resume input contract changed')
        else:atomic_json(lock,{'sha256':self.identity,'contract':contract})

    def stage(self,name,inputs,callback):
        identity=digest({'round':self.identity,'inputs':inputs})
        path=self.path/(name+'.json')
        if path.exists():
            r=json.loads(path.read_text())
            if r['input_sha256']!=identity or r['output_sha256']!=digest(r['result']):
                raise ValueError('committed stage identity changed: '+name)
            return r['result']
        running=self.path/(name+'.running.json')
        if running.exists():
            raise RuntimeError('uncommitted stage needs receipt reconciliation before retry: '+name)
        atomic_json(running,{'input_sha256':identity,'started':time.time()})
        try:
            result=callback()
            atomic_json(path,{'input_sha256':identity,'output_sha256':digest(result),'result':result})
        except BaseException as error:
            atomic_json(self.path/'errors'/f'{name}_{time.time_ns()}.json',
                {'stage':name,'input_sha256':identity,'error':repr(error),'completed':False})
            # Only post-acceptance offline G failure can be explicitly retried.
            # Other ambiguous partial stages need receipt reconciliation, never replay.
            if name=='generator_selection':running.unlink()
            raise
        running.unlink()
        return result


def run_round(path,*,support,roots,teacher_roots,solve,train,evaluate,accept,export,update,
              source,budgets,maximum_interactions):
    reservation=reserve_round(budgets,maximum_interactions)
    fingerprint=support_fingerprint(support)
    journal=StageJournal(path,{'support':fingerprint,'roots':roots,'teacher_roots':teacher_roots,
        'source':source,'budgets':reservation,'maximum_interactions':maximum_interactions})
    teachers=journal.stage('teacher_search',{'source':source},
        lambda:validate_teacher_results(roots,teacher_roots,solve()))
    teachers=validate_teacher_results(roots,teacher_roots,{k:teachers[k] for k in teacher_roots})
    demos=[r['demo'] for r in teachers.values() if r['teacher_status']=='verified_solution']
    validate_pending_support_unchanged(fingerprint,support)
    student=journal.stage('student_training',{'support':fingerprint,'demos':demos},
        lambda:train(deepcopy(support),demos or None))
    evaluation=journal.stage('student_evaluation',student,lambda:evaluate(student))
    if set(evaluation)!=set(roots):raise ValueError('student evaluation must cover every declared root')
    for r in evaluation.values():
        if any(k in r for k in ('teacher_status','teacher_found','demo','root_id')):
            raise ValueError('student results cannot rewrite teacher/root evidence')
    decision=journal.stage('actor_acceptance',{'student':student,'evaluation':evaluation},
        lambda:accept(student,evaluation))
    if type(decision.get('adopted')) is not bool:raise ValueError('final adoption decision required')
    if any(not student.get(k) or decision.get(k)!=student[k] for k in ('actor_sha256','normalizer_sha256')):
        raise ValueError('adoption decision must bind the exact evaluated student checkpoint')
    if demos and 'demo_samples_by_root' not in student:
        raise ValueError('nonempty demo needs actual training usage receipt')
    usage=student.get('demo_samples_by_root',{})
    if not set(usage)<=set(roots) or any(type(n) is not int or n<0 for n in usage.values()):
        raise ValueError('invalid per-root actual demo consumption')
    used_total=sum(usage.values())
    outcomes=[root_outcome({'root_id':r,**teachers[r],**evaluation[r],
        'student_adopted':decision['adopted'], 'student_checkpoint_sha256':student['actor_sha256'],
        'student_normalizer_sha256':student['normalizer_sha256'],
        'n_direct_demo_examples_available':teachers[r].get('demo',{}).get('count',0),
        'n_direct_demo_samples_used':usage.get(r,0),'round_has_teacher_demo':bool(demos)},
        round_demo_samples_used=used_total) for r in roots]
    corpus=journal.stage('generator_corpus_update',{'teachers':teachers,'student':student,
        'evaluation':evaluation,'decision':decision},lambda:export(teachers,student,evaluation,decision))
    selection=journal.stage('generator_selection',corpus,lambda:update(corpus))
    # Revalidate referenced bytes on every resume, even if the stage was cached.
    from .artifacts import load_corpus,validate_generator_receipt
    if 'path' in corpus and 'sha256' in corpus:load_corpus(corpus)
    if 'checkpoint_manifest' in selection:validate_generator_receipt(selection)
    result={'actor':student if decision['adopted'] else source['actor'],
        'generator':selection,'corpus_sha256':digest(corpus),'outcomes':outcomes,
        'actor_acceptance':decision,'source':source}
    completed=journal.stage('complete_round',result,lambda:result)
    # A single pointer references a fully committed Actor/G/corpus bundle.
    atomic_json(Path(path)/'current_source.json',{'complete_round_manifest':'complete_round.json',
        'complete_round_sha256':digest(completed)})
    return completed
