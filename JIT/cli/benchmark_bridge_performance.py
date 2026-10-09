"""Bounded isolated diagnostics; outputs are never production learner inputs."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('audit','evaluate','generator'),required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--persistent',action='store_true');args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    from jit_dvgc.generative_bridge.protocol import atomic_json
    if args.mode=='audit':
        from jit_dvgc.generative_bridge.performance import audit_round
        atomic_json(root/'audit.json',audit_round(args.input));return
    request=json.loads(Path(args.input).read_text());start=time.perf_counter()
    if args.mode=='evaluate':
        from jit_dvgc.pulse_exploration_runtime import evaluate
        session=None
        if args.persistent:
            from jit_dvgc.generative_bridge.teacher_runtime import TeacherEvaluationSession
            session=TeacherEvaluationSession()
        if not 1<=len(request['specs'])<=32:raise ValueError('bounded 1..32 evaluation requests required')
        charged=0;records=[]
        try:
            for i,path in enumerate(request['specs']):
                spec=json.loads(Path(path).read_text())
                rows=json.loads(Path(spec['candidates']).read_text())
                if len(rows)!=17 or spec.get('role') not in ('train','TRAIN'):
                    raise ValueError('diagnostic requires 17-lane TRAIN roots')
                if charged+spec['budget']>request['max_interactions']:raise ValueError('diagnostic reservation exhausted')
                t=time.perf_counter();out=root/f'request_{i:03d}'
                if session is None:
                    import subprocess,sys,jit_dvgc
                    cli=Path(jit_dvgc.__file__).resolve().parents[2]/'cli/run_pulse_exploration.py'
                    subprocess.run([sys.executable,str(cli),'--mode','evaluate','--spec',str(path),'--output',str(out)],check=True)
                else:evaluate(spec,out,session=session)
                status=json.loads((out/'status.json').read_text());charged+=status['charged_interactions']
                records.append(dict(status,request=i,spec=path,process_end_to_end_seconds=time.perf_counter()-t))
                atomic_json(root/'progress.json',dict(completed=i+1,total=len(request['specs']),charged_interactions=charged))
        finally:
            if session is not None:session.close()
        atomic_json(root/'benchmark.json',dict(mode='evaluate',persistent=args.persistent,
            end_to_end_seconds=time.perf_counter()-start,charged_interactions=charged,records=records,
            compile_count=None if session is None else session.compile_count,training_feedback=False))
    else:
        import jax
        import numpy as np
        from jit_dvgc.generative_bridge.worker import generator_template,generator_reference
        from jit_dvgc.generative_bridge.diffusion import restore_state,make_train_step,save_state
        config=json.loads(Path(request['config']).read_text())
        if request.get('updates')!=100:raise ValueError('fixed 100-update diagnostic required')
        net,template,identity=generator_template(generator_reference(config),config['seed'])
        state=restore_state(Path(config['incumbent']['checkpoint_manifest']).parent,template,identity)
        arrays=np.load(request['batches'],allow_pickle=False)
        if arrays['observations'].shape!=(100,256,76) or arrays['actions'].shape!=(100,256,16,4):raise ValueError('frozen batch mismatch')
        step=make_train_step(lambda params,x,o,k:net.apply(params,x,o,k),learning_rate=1e-5)
        losses=[];times=[]
        for i in range(100):
            t=time.perf_counter();state,loss=step(state,arrays['observations'][i],arrays['actions'][i]);times.append(time.perf_counter()-t);losses.append(loss)
        save_state(root/'final_state',state,identity)
        atomic_json(root/'benchmark.json',dict(mode='generator',end_to_end_seconds=time.perf_counter()-start,
            updates=100,cold_update_seconds=times[0],warm_updates_seconds=sum(times[1:]),
            losses=losses,update_seconds=times,training_feedback=False))
    atomic_json(root/'status.json',dict(phase='completed',training_feedback=False))


if __name__=='__main__':main()
