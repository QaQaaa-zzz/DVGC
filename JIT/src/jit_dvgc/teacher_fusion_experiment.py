"""Bounded single-network teacher fusion; isolated child stages and auditable costs."""
from pathlib import Path
import argparse,copy,json,os,subprocess,time,sys

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));q.replace(p)

def prepare(source,output):
 from .rsi_comparison import file_sha
 source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
 if (output/'spec.json').exists():raise FileExistsError('experiment already declared')
 base=read(source/'onset_00/spec.json');methods={m['key']:m for m in base['methods']};teachers=[dict(name=k,template=methods[k]['template']) for k in ('phase_u','fresh_rsi')]
 traces=[dict(source=k,path=str(source/f'onset_{o:02d}/batches/0000'/k/'prefixes.npz')) for k in ('phase_u','fresh_rsi') for o in (0,5,10,15,25)]
 ds=dict(teachers=teachers,source_traces=traces,states_per_teacher_stage=32,seed=9206101,horizon=400,reward_mode='phase_recovery',budget=204800,require_source_success=True)
 write(output/'initial_data_spec.json',ds)
 initializer=next(m['frozen_policy'] for m in read(methods['fresh_rsi']['template']['bank'])['members'] if m['name']==methods['fresh_rsi']['template']['proposer'])
 frozen=read(initializer);bootstrap=read(frozen['policy']['formal_config'])['bootstrap_formal_config']
 spec=dict(schema='jit_teacher_fusion_experiment_v1',source=str(source),output=str(output),initial_data_spec=str(output/'initial_data_spec.json'),teachers=teachers,initializer_template=initializer,bootstrap=bootstrap,source_bank=methods['fresh_rsi']['template']['bank'],evaluation_template=methods['fresh_rsi']['template'],training_steps=[640000,2560000],development_episodes=128,development_seed=9207000,refresh_seed=9217000,distillation_updates=4000,maximum_interactions=6500000,selection='highest minimum of five condition success rates; then mean; then earliest candidate',no_teacher_switching=True,source_spec_sha256=file_sha(source/'onset_00/spec.json'),input_files={str(Path(initializer)):file_sha(initializer)},stage_timeout_seconds=21600)
 write(output/'spec.json',spec);write(output/'status.json',dict(phase='prepared'));write(output/'ACTIVE_RUN.json',dict(name='Phase U + RSI 单网络融合',execution=str(output/'status.json')))
 (output/'INDEX.md').write_text('# Phase U + RSI 单网络融合\n\n[实时状态](status.json) · [配置与预算](spec.json) · [设计](DESIGN.md) · [运行日志](run.log) · [通知](notifications/notification_status.json)\n\n256个分阶段完整状态、同状态双老师评估、全新Actor监督训练、640000步PPO、学生状态补数据、2560000步PPO。总PPO320万步。教师仅在训练提供监督，部署只有一个学生Actor。\n\n五时刻新种子开发面板选型：优先最差条件成功率，再平均成功率。旧20万回合现在用于TRAIN，不是独立TEST。最大交互650万步，遇错停止，不自动扩预算。\n')
 return spec

def configure(spec,support,path,run_id,steps,checkpoints,initializer,seed,fresh):
 from .iterative_probe_training import make_config,load_config
 raw=make_config(support,initializer,spec['bootstrap'],path,run_id,0,steps,seed,checkpoints=checkpoints,reward_mode='phase_recovery',initialization_mode='fresh' if fresh else 'warm_start_frozen_unified')
 raw['reward_contract']='Phase U upstream reward + original recovery-phase downstream reward; stable_forward_recovery endpoint'
 raw['training_action_pulse']=dict(onsets=[0,5,10,15,25],steps=3,delta_limit=[.25]*4,probability=.5)
 raw['fusion_provenance']=dict(single_student=True,teacher_switching=False,source_data_role='train',initial_actor='fresh_then_supervised' if fresh else 'own_stage_a_actor',imitation_dataset_separate_from_ppo=True)
 write(path,raw);load_config(path)

def train_stage(args):
 from .unified_formal import run_unified_formal,checkpoint_identity,build_unified_formal_environment,load_unified_policy_formal_config
 from .policy_distillation import distill,wrap_trainer
 from brax.training.agents.ppo import train as ppo
 cfg=load_unified_policy_formal_config(Path(args['config']));identity=None;checkpoint=None
 if args.get('distill'):
  _,_,env=build_unified_formal_environment(Path(args['config']));identity=checkpoint_identity(cfg,env)
  checkpoint=Path(args['distill'])/'transition_0';distill(args['dataset'],checkpoint,identity=identity,privileged_observation_size=env.privileged_observation_size,seed=args['seed'],updates=args['updates'],batch_size=256)
  del env
 trainer=wrap_trainer(ppo.train,args['dataset'],coefficient=.2,batch_size=256,decay_transitions=cfg.ppo.requested_transitions,checkpoint_path=checkpoint,identity=identity)
 return run_unified_formal(Path(args['config']),args['run_id'],run_root=Path(args['run_root']),trainer=trainer)

def run(spec_path):
 from .rsi_comparison import file_sha
 from .unified_policy_freeze import freeze_development_checkpoint
 from .probe_bank import lock_probe_bank
 from .evidence_integrity import canonical_sha256
 import numpy as np
 spec=read(spec_path);root=Path(spec['output']);started=time.time();stage='preflight';done=[];charged=int(spec.get('validation_interactions',0));cli=Path(__file__).resolve().parents[2]/'cli'
 def status(phase,**extra):write(root/'status.json',dict(phase=phase,stage=stage,pid=os.getpid(),completed_stages=done,charged_interactions=charged,maximum_interactions=spec['maximum_interactions'],wall_seconds=time.time()-started,**extra))
 def child(label,module,command,args,output):
  nonlocal stage
  stage=label;output=Path(output);path=root/'stages'/f'{label}.json';write(path,args)
  cmd=[sys.executable,'-m',module,command,'--spec',str(path),'--output',str(output)]
  env=dict(os.environ,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',JIT_AUTO_PUBLISH='0')
  logpath=root/'logs'/f'{label}.log';logpath.parent.mkdir(exist_ok=True)
  with logpath.open('x') as log:
   p=subprocess.Popen(cmd,cwd='/home/qy/DVGC',env=env,stdout=log,stderr=subprocess.STDOUT);status('running',child_pid=p.pid,log=str(logpath),stage_output=str(output))
   try:rc=p.wait(timeout=spec['stage_timeout_seconds'])
   except BaseException:p.terminate();p.wait(timeout=30);raise
  if rc:raise RuntimeError(f'{label} exited {rc}: {logpath}')
  done.append(label);status('running')
 def collect(label,template,seed,onset,n):
  nonlocal charged
  output=root/'evaluation'/label
  ev=dict(template,seed=seed,num_envs=n,round_index=0,pulse_start_schedule=[onset],pulse_batch_mode='mixed',full_episode_rollout=True,controller_mode='fixed_random')
  child(label,'jit_dvgc.teacher_fusion_experiment','collect',ev,output)
  receipt=read(output/'status.json');charged+=receipt['charged_interactions'];assert charged<=spec['maximum_interactions'];return output/'prefixes.npz'
 def frozen(config,run_id,step,name):
  path=root/'frozen'/name;freeze_development_checkpoint(path,config_path=Path(config),checkpoint=root/'training'/run_id/'checkpoints'/f'transition_{step}',name=name)
  bp=root/'banks'/f'{name}.json';base=read(spec['source_bank']);lock_probe_bank(dict(version=name,task=base['task'],max_ticks=400,label_interaction_budget=400000,max_candidates_per_process=256,members=[dict(frozen_policy=str(path/'frozen_unified_policy.json'),roles=['proposer','evaluator'])]),bp)
  template={**spec['evaluation_template'],'bank':str(bp),'proposer':name,'reward_mode':'phase_recovery'};template.pop('phase_policy',None);return path/'frozen_unified_policy.json',template
 try:
  for p,sha in spec['input_files'].items():assert file_sha(p)==sha
  status('running');ds=read(spec['initial_data_spec']);child('initial_states','jit_dvgc.teacher_fusion_data','prepare',ds,root/'initial_states')
  ds['candidates']=str(root/'initial_states/candidates.json');child('initial_labels','jit_dvgc.teacher_fusion_data','evaluate',ds,root/'initial_labels');charged+=read(root/'initial_labels/status.json')['charged_interactions']
  config_a=root/'config_a.json';configure(spec,root/'initial_labels/support.json',config_a,'student_a',640000,[640000],spec['initializer_template'],9206201,True)
  args=dict(config=str(config_a),dataset=str(root/'initial_labels/imitation.npz'),distill=str(root/'distillation'),seed=9206200,updates=spec['distillation_updates'],run_id='student_a',run_root=str(root/'training'))
  child('distill_and_ppo_a','jit_dvgc.teacher_fusion_experiment','train',args,root/'training/student_a');charged+=640000+read(root/'training/student_a/formal_report.json')['train_panel_interactions']
  fa,ta=frozen(config_a,'student_a',640000,'student_640000')
  refresh_traces=[dict(source='student',template=ta,path=str(collect(f'refresh_onset_{o:02d}',ta,spec['refresh_seed']+o*100,o,128))) for o in [0,5,10,15,25]]
  rs=dict(ds,source_traces=refresh_traces,states_per_teacher_stage=16,require_source_success=False,seed=9206301,budget=51200);rs.pop('candidates',None)
  child('refresh_states','jit_dvgc.teacher_fusion_data','prepare',rs,root/'refresh_states');rs['candidates']=str(root/'refresh_states/candidates.json');child('refresh_labels','jit_dvgc.teacher_fusion_data','evaluate',rs,root/'refresh_labels');charged+=read(root/'refresh_labels/status.json')['charged_interactions']
  supports=[read(root/p/'support.json') for p in ('initial_labels','refresh_labels')];merged=copy.deepcopy(supports[0]);entries={e['key']:e for s in supports for e in s['entries']};merged['entries']=list(entries.values());merged['inputs']={k:v for s in supports for k,v in s['inputs'].items()};merged.pop('support_sha256');merged['selection']='initial teacher and refreshed student-state same-state teacher witnesses';merged['phase_counts']={p:sum(e['phase']==p for e in merged['entries']) for p in ('upstream','downstream')};merged['support_sha256']=canonical_sha256(merged);write(root/'merged_support.json',merged)
  arrays=[]
  for name in ('initial_labels','refresh_labels'):
   with np.load(root/name/'imitation.npz') as z:
    arrays.append({k:z[k] for k in ('observations','actions','weights')})
  for a in arrays:a['weights']=a['weights']/a['weights'].sum()*.5
  np.savez_compressed(root/'merged_imitation.npz',**{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
  config_b=root/'config_b.json';configure(spec,root/'merged_support.json',config_b,'student_b',2560000,[640000,1280000,1920000,2560000],fa,9206401,False)
  child('ppo_b','jit_dvgc.teacher_fusion_experiment','train',dict(config=str(config_b),dataset=str(root/'merged_imitation.npz'),run_id='student_b',run_root=str(root/'training')),root/'training/student_b');charged+=2560000+read(root/'training/student_b/formal_report.json')['train_panel_interactions']
  candidates=[dict(name='student_640000',template=ta,checkpoint=str(root/'training/student_a/checkpoints/transition_640000'),cumulative_steps=640000)]
  for step in [640000,1280000,1920000,2560000]:
   f,t=frozen(config_b,'student_b',step,f'student_{640000+step}');candidates.append(dict(name=f'student_{640000+step}',template=t,checkpoint=str(root/'training/student_b/checkpoints'/f'transition_{step}'),cumulative_steps=640000+step))
  from .batched_pulse_comparison import load_tape,verify_fixed_pulse_tape
  from .rsi_comparison import episode_results
  summaries={};paired={}
  for method in spec['teachers']+candidates:
   name=method['name'];summaries[name]={}
   for o in [0,5,10,15,25]:
    path=collect(f'dev_{name}_onset_{o:02d}',method['template'],spec['development_seed']+o*100,o,spec['development_episodes']);tape=load_tape(path);verify_fixed_pulse_tape(tape,o,3)
    if o not in paired:paired[o]=tape['delta']
    else:np.testing.assert_array_equal(paired[o],tape['delta'])
    rows=episode_results(tape,read(Path(spec['source'])/'onset_00/spec.json')['root_qpos_address']);summaries[name][str(o)]=dict(successes=sum(r['success'] for r in rows),episodes=len(rows),rate=sum(r['success'] for r in rows)/len(rows),trace=str(path));write(root/'development_results.json',summaries)
  best=max(candidates,key=lambda c:(min(x['rate'] for x in summaries[c['name']].values()),sum(x['rate'] for x in summaries[c['name']].values()),-c['cumulative_steps']))
  write(root/'best_model.json',dict(**best,criterion=spec['selection'],development=summaries[best['name']],scope='five saved student candidates; development only; no independent TEST'))
  with (root/'INDEX.md').open('a') as f:f.write('\n\n[开发面板结果](development_results.json) · [选定模型](best_model.json)\n')
  stage='complete';status('completed',best_model=str(root/'best_model.json'))
 except BaseException as exc:status('error',error=f'{type(exc).__name__}: {exc}',automatic_retry=False);raise

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','train','collect']);p.add_argument('--spec');p.add_argument('--source');p.add_argument('--output');a=p.parse_args()
 if a.mode=='prepare':prepare(a.source,a.output)
 elif a.mode=='run':run(a.spec)
 elif a.mode=='train':train_stage(read(a.spec))
 else:
  from .pulse_exploration_runtime import collect
  collect(read(a.spec),Path(a.output))
if __name__=='__main__':main()
