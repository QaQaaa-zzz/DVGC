"""E1 bounded launch and strict physical development selection (no import side effects)."""
from pathlib import Path
import hashlib,json,subprocess,sys

def resolve_learner_ppo(spec):
    from resolved_task_config import canonical_hash
    overrides=spec.get('learner_overrides',{})
    allowed={'learning_rate','num_updates_per_batch','clipping_epsilon','entropy_cost','requested_transitions','stability'}
    if set(overrides)-allowed:raise ValueError('Undeclared learner override axis')
    ppo={**spec['ppo'],**overrides}
    if 'effective_learner_ppo' in spec and ppo!=spec['effective_learner_ppo']:raise ValueError('Conflicting effective learner declaration')
    if 'effective_learner_ppo_sha256' in spec and canonical_hash(ppo)!=spec['effective_learner_ppo_sha256']:raise ValueError('Learner configuration hash mismatch')
    return ppo

def authorized_spec(config, gate):
    config,gate=Path(config),Path(gate)
    spec=json.loads(config.read_text());resolve_learner_ppo(spec);approval=json.loads(gate.read_text())
    expected=hashlib.sha256(config.read_bytes()).hexdigest()
    if approval.get('config_sha256')!=expected or not all(approval.get(k) is True for k in ['P0_complete','P1_complete','root_authorized_for_E1']):
        raise ValueError('E1 requires matching explicit P0/P1/root launch gate')
    if spec['requested_training_transitions']!=491520 or spec['ppo']['num_parallel_envs']!=384 or spec['ppo']['unroll_length']!=64:
        raise ValueError('E1 budget/rollout changed')
    spec['launch_ready']=True;spec['authorization_gate']=str(gate.resolve());spec['authorization_gate_sha256']=hashlib.sha256(gate.read_bytes()).hexdigest()
    return spec

def physical_rank(summary):
    """Lexicographic strict landing fraction, hold, takeoff; never reward."""
    n=len(summary)
    if not n:raise ValueError('Empty physical selection panel')
    return (sum(bool(s['stable_landing']) for s in summary)/n,
            sum(s['stable_hold_s'] for s in summary)/n,
            sum(s['takeoff_s'] is not None for s in summary)/n)

def evaluate_checkpoint(root, spec, checkpoint, folder):
    root,checkpoint,folder=Path(root),Path(checkpoint),Path(folder)
    folder.mkdir(parents=True,exist_ok=False);summary=[];controls=physics=0
    cfgpath=folder/'evaluation_config.json';cfgpath.write_text(json.dumps(spec,indent=2))
    for seed in spec['strict_development']['seeds']:
        case=folder/f'seed_{seed}'
        cmd=[sys.executable,str(root/'probe_physx_factors.py'),'--config',str(cfgpath),'--actor',str(checkpoint/'actor.npz'),'--output',str(case),'--n','1','--max-steps',str(spec['strict_development']['max_steps']),'--env-seed',str(seed),'--reset-mode','hard','--calibrate-before']
        with (folder/f'seed_{seed}.log').open('w') as log:subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        from physx_probe_summary import summarize_probe
        rows=summarize_probe(case);summary.extend(rows)
        budget=json.loads((case/'budget.json').read_text());controls+=budget['control_interactions'];physics+=budget['physics_interactions']
    result={'complete':True,'summary':summary,'rank':physical_rank(summary),'control_interactions':controls,'physics_transitions':physics,'selection':'strict natural stable landing fraction, mean stable hold, then takeoff fraction; earlier wins exact ties','checkpoint':str(checkpoint)}
    (folder/'result.json').write_text(json.dumps(result,indent=2));return result

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tensorboard-port',type=int,default=6028);a=p.parse_args()
    spec=authorized_spec(a.config,a.gate)
    if a.output.exists():raise ValueError('Refusing existing E1 output')
    launch=a.output.parent/(a.output.name+'_authorized_config.json')
    if launch.exists():raise ValueError('Refusing existing authorized config')
    launch.parent.mkdir(parents=True,exist_ok=True);launch.write_text(json.dumps(spec,indent=2))
    import time,urllib.request,socket
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',a.tensorboard_port))==0:raise ValueError('TensorBoard port already occupied; choose a separate port')
    monitor=a.output.parent/(a.output.name+'_monitoring');monitor.mkdir(exist_ok=False)
    with (monitor/'tensorboard.log').open('w') as log:
        tb=subprocess.Popen([sys.executable,'-m','tensorboard.main','--logdir',str(a.output/'tensorboard'),'--host','127.0.0.1','--port',str(a.tensorboard_port),'--reload_interval','5'],stdout=log,stderr=subprocess.STDOUT)
    url=f'http://127.0.0.1:{a.tensorboard_port}'
    (monitor/'tensorboard.json').write_text(json.dumps({'pid':tb.pid,'url':url,'logdir':str(a.output/'tensorboard'),'reward_scalar_verified':False},indent=2))
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve().parent/'train_physx_brax.py'),'--config',str(launch),'--output',str(a.output)])
    verified=False
    while child.poll() is None:
        if not verified:
            try:
                tags=json.load(urllib.request.urlopen(url+'/data/plugin/scalars/tags',timeout=2))
                verified=any('training/mean_step_reward' in values for values in tags.values())
                if verified:
                    (monitor/'reward_http_verified.json').write_text(json.dumps({'url':url,'tags':tags,'verified':True},indent=2));print('TENSORBOARD_REWARD_VERIFIED',url,flush=True)
            except Exception:pass
        time.sleep(2)
    if child.returncode:raise subprocess.CalledProcessError(child.returncode,child.args)
    if not verified:raise RuntimeError('Training completed but current reward scalar not verified via TensorBoard HTTP')
if __name__=='__main__':main()
