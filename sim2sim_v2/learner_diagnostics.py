"""Offline diagnostics only; trace observations are not on-policy PPO rollouts."""
from pathlib import Path
import json, pickle
import numpy as np
import jax
from brax_physx_learner import Learner
from policy_runtime import Actor

def capture_rollout(path, learner, transition):
    """Call before update with genuine sampled Transition; includes behavior state."""
    learner._validate_data(transition)
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    learner.save(path/'behavior', learner.training_transitions)
    with (path/'rollout.pkl').open('wb') as f: pickle.dump(jax.device_get(transition),f)
    (path/'capture.json').write_text(json.dumps({'role':'genuine_pre_update_rollout',
        'interactions':int(np.size(transition.reward)), 'sampling_complete':True},indent=2))

def decompose_capture(path):
    """A/B/C/D use identical genuine behavior batch; no fabricated logprobs."""
    import jax.numpy as jnp
    from brax.training.acme import running_statistics
    path=Path(path)
    learner=Learner.restore(path/'behavior')
    with (path/'rollout.pkl').open('rb') as f:data=pickle.load(f)
    learner._validate_data(data)
    data=jax.tree.map(jnp.asarray,data)
    baseline=learner.rollout_policy_metrics(data)
    # FP32 fused sample vs replay log_prob at magnitude293 differs by one ULP.
    # Keep strict analytic KL check alongside bounded likelihood roundoff.
    if baseline['ratio_max_abs_error_from_one']>1e-4 or abs(baseline['full_rollout_kl_new_old'])>1e-7:
        raise ValueError('Saved behavior does not reproduce sampled log_prob')
    newnorm=running_statistics.update(learner.normalizer,data.observation)
    mapping = None
    resolved_path=path.parent/'resolved_config.json'
    if resolved_path.exists():
        import mujoco
        model_path=Path(__file__).resolve().parent/'model/source.xml'
        model=mujoco.MjModel.from_xml_path(str(model_path))
        config=json.loads(resolved_path.read_text())['source_task']
        mapping=(model,config)
    def details(params,norm):
        z=(np.asarray(data.observation['state'])-np.asarray(norm.mean['state']))/np.asarray(norm.std['state'])
        logits=learner.network.policy_network.apply(norm,params.policy,data.observation)
        dist=learner.network.parametric_action_distribution.create_dist(logits)
        action=np.tanh(np.asarray(dist.loc))
        physical={}
        if mapping is not None:
            model,config=mapping
            target=np.empty_like(action)
            target[...,0]=action[...,0]*model.actuator_ctrlrange[0,1]
            target[...,1]=config['action']['base_rear_speed']+action[...,1]*config['action']['rear_speed_delta']
            for i in [2,3]:
                joint=model.actuator_trnid[i,0];initial=model.key_qpos[0,model.jnt_qposadr[joint]];lo,hi=model.actuator_ctrlrange[i]
                target[...,i]=initial+action[...,i]*np.where(action[...,i]>=0,hi-initial,initial-lo)
            lo,hi=model.actuator_ctrlrange.T
            physical={'physical_target_mean_per_axis':target.mean(axis=(0,1)).tolist(),
                      'physical_target_at_limit_fraction_per_axis':((target<=lo+1e-6)|(target>=hi-1e-6)).mean(axis=(0,1)).tolist(),
                      'physical_target_outside_limit_fraction_per_axis':((target<lo)|(target>hi)).mean(axis=(0,1)).tolist()}
        return {**physical,'normalized_obs_min':float(z.min()),'normalized_obs_max':float(z.max()),
                'normalized_obs_abs_quantiles':np.quantile(abs(z),[.5,.9,.99,1.]).tolist(),
                'action_mean_per_axis':action.mean(axis=(0,1)).tolist(),
                'action_std_per_axis':action.std(axis=(0,1)).tolist(),
                'pre_tanh_policy_std_mean_per_axis':np.asarray(dist.scale).mean(axis=(0,1)).tolist(),
                'normalized_action_near_bound_fraction_per_axis':(abs(action)>=.999).mean(axis=(0,1)).tolist()}
    rows={'A_no_change':{**baseline,**details(learner.params,learner.normalizer)},
          'B_normalizer_only':{**learner.rollout_policy_metrics(data,normalizer=newnorm),**details(learner.params,newnorm)}}
    for name,freeze in [('C_optimizer_only',True),('D_full',False)]:
        params,state,norm,metrics,finite=learner._update(learner.params,learner.optimizer_state,learner.normalizer,data,(learner._sgd_key if learner._sgd_key is not None else jax.random.PRNGKey(123)),freeze_normalizer=freeze)
        rows[name]={**learner.rollout_policy_metrics(data,params,norm),**details(params,norm),'finite':bool(finite)}
    np.savez(path/'normalizer_delta.npz',old_mean=np.asarray(learner.normalizer.mean['state']),old_std=np.asarray(learner.normalizer.std['state']),new_mean=np.asarray(newnorm.mean['state']),new_std=np.asarray(newnorm.std['state']))
    (path/'decomposition.json').write_text(json.dumps(rows,indent=2))
    import csv
    with (path/'kl_decomposition.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['case','full_rollout_KL_new_old','full_rollout_KL_old_new','legacy_definition_full_rollout_kl','ratio_mean','ratio_max_abs_error_from_one'])
        for name,row in rows.items():writer.writerow([name,row['full_rollout_kl_new_old'],row['full_rollout_kl_old_new'],row['legacy_definition_full_rollout_kl'],row['ratio_mean'],row['ratio_max_abs_error_from_one']])
    return rows

def main():
    root=Path(__file__).resolve().parent.parent
    evidence=root/'docs/evidence/sim2sim_ppo_20261009'
    out=root/'sim2sim_v2/runs/p1_learner';out.mkdir(parents=True,exist_ok=True)
    obs=np.concatenate([np.load(evidence/f'evaluation/{kind}/trace.npz')['obs'] for kind in ['warmstart_kd5','fresh_kd5']])
    ppo=pickle.load(open(evidence/'ppo/warmstart_kd5/checkpoint_final/learner_state.pkl','rb'))['ppo']
    results=[]
    for name in ['mjx_source_9904128','warmstart_kd5','fresh_kd5']:
        if name=='mjx_source_9904128': l=Learner(evidence/f'ppo/{name}/checkpoint',ppo)
        else:l=Learner.restore(evidence/f'ppo/{name}/checkpoint_final')
        export=out/name; l.save(export,l.training_transitions)
        actor=Actor(export/'actor.npz')
        mean=np.asarray(l.normalizer.mean['state']);std=np.asarray(l.normalizer.std['state'])
        for role,x in [('saved_evaluation',obs),('extreme_standardized',mean+np.random.default_rng(0).uniform(-100,100,(256,76)).astype(np.float32)*std),('reset_zero',np.zeros((4,76),np.float32))]:
            y=l.deterministic({'state':x,'privileged_state':np.zeros((len(x),106),np.float32)})
            z=actor(x);err=float(np.max(abs(y-z)))
            results.append({'checkpoint':name,'inputs':role,'n':len(x),'max_absolute_error':err,'threshold':3e-5,'passed':err<=3e-5})
    (out/'action_parity.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
