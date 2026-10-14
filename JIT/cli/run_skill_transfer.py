#!/usr/bin/env python3
"""Separate recovery acquisition, independent evaluation and student integration."""
import argparse
import json
from pathlib import Path
from jit_dvgc.skill_transfer import audit,acquire_bc,prepare,report


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('audit');a.add_argument('--config',required=True)
    a=s.add_parser('prepare');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('report');a.add_argument('--output',required=True)
    a=s.add_parser('acquire');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
    a.add_argument('--roots',nargs='+');a.add_argument('--updates',type=int,default=2000)
    a.add_argument('--seed',type=int,default=0);a.add_argument('--resume')
    a=s.add_parser('evaluate');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
    a.add_argument('--checkpoint',required=True);a.add_argument('--roots',nargs='+');a.add_argument('--repeats',type=int,default=3)
    a=s.add_parser('ppo');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('capture');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('smoke');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('consolidate');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
    a.add_argument('--recovery-checkpoint',required=True);a.add_argument('--updates',type=int,default=2000)
    a.add_argument('--visit-steps',type=int,default=4)
    x=p.parse_args()
    if x.command=='audit':result=audit(x.config)
    elif x.command=='prepare':result=prepare(x.config,x.output)
    elif x.command=='report':result=report(x.output)
    elif x.command=='acquire':result=acquire_bc(x.config,x.output,roots=x.roots,updates=x.updates,seed=x.seed,resume=x.resume)
    elif x.command in ('ppo','smoke'):
        from jit_dvgc.skill_transfer_ppo import run_recovery_ppo
        c=json.loads(Path(x.config).read_text())
        if x.command=='smoke':c['transitions']=3200
        result=run_recovery_ppo(c,x.output)
    elif x.command=='consolidate':
        from jit_dvgc.skill_transfer_student_runtime import run_student_block
        result=run_student_block(x.config,x.output,x.recovery_checkpoint,updates=x.updates,visit_steps=x.visit_steps)
    elif x.command=='capture':
        from jit_dvgc.unified_formal import build_unified_formal_environment
        from jit_dvgc.recovery_curriculum import build_recovery_curriculum
        c=json.loads(Path(x.config).read_text());audit(c)
        cfg,_,env=build_unified_formal_environment(c['runtime_config']['path'])
        if cfg.raw.get('success_criterion')!='stable_forward_recovery':raise ValueError('source endpoint differs')
        env._training_action_pulse=None
        env._reward_mode='original_all_phases'
        result=build_recovery_curriculum(c['teacher_manifest']['path'],x.output,env=env,source_rows=c['source_rows']['path'],
            replay_capacity=1,canonical_replay_required=False)
    else:
        from jit_dvgc.skill_transfer_evaluation import evaluate_recovery
        c=json.loads(Path(x.config).read_text());audit(c)
        result=evaluate_recovery(x.checkpoint,x.output,c['pi0_descriptor'],c['teacher_manifest']['path'],
            c['source_rows']['path'],c['repository'],roots=x.roots,repeats=x.repeats)
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()
