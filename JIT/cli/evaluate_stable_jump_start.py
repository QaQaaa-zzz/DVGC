#!/usr/bin/env python3
"""Evaluate one frozen Actor from fixed jump start under strict recovery semantics."""
import argparse
import csv
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--frozen-policy', type=Path)
    p.add_argument('--trained-checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (a.output/name).write_text(json.dumps(value, indent=2)+'\n')
    write('status.json', {'phase':'running','maximum_interactions':400})
    start=time.monotonic()
    try:
        import jax
        from jit_dvgc.iterative_probe_training import load_config, build_environment
        from jit_dvgc.unified_policy_freeze import load_frozen_unified_manifest, _load_policy_formal_config, _checkpoint_identity
        from jit_dvgc.checkpoint import load_checkpoint
        from jit_dvgc.ppo import make_checkpoint_policy
        from jit_dvgc.evaluation import capture_episode, save_episode_trace
        from jit_dvgc.video import render_trace
        from jit_dvgc.constants import END_REASONS
        config=load_config(a.config)
        if config.raw['success_criterion']!='stable_forward_recovery':
            raise ValueError('strict stable recovery config required')
        _,env=build_environment(config)
        if a.trained_checkpoint:
            from jit_dvgc.unified_training import checkpoint_identity
            from jit_dvgc.handoff_bank import pytree_sha256
            payload=load_checkpoint(a.trained_checkpoint,expected=checkpoint_identity(config,env))
            source={'actor_sha256':pytree_sha256(payload.actor_params)}
        elif a.frozen_policy:
            frozen=load_frozen_unified_manifest(a.frozen_policy)
            source=frozen['policy']; original=_load_policy_formal_config(Path(source['formal_config']))
            payload=load_checkpoint(Path(source['checkpoint']),expected=_checkpoint_identity(original))
        else:
            from jit_dvgc.iterative_probe_training import load_phase_initializer
            from jit_dvgc.handoff_bank import pytree_sha256
            payload=load_phase_initializer(config.raw['initialization'])
            source={'actor_sha256':pytree_sha256(payload.actor_params)}
        policy=make_checkpoint_policy(env,payload,deterministic=True)
        trace=capture_episode(env,jax.jit(lambda obs:policy(obs,jax.random.PRNGKey(9410001))),
            seed=9400001,horizon=400,reset_fn=jax.jit(env._reset_jump_start_unified),step_fn=jax.jit(env.step))
        save_episode_trace(trace,a.output/'trace')
        frames=trace.frames
        contact=next((i for i,f in enumerate(frames) if f.metrics.get('event/descent_valid_contact_seen',0)>0),None)
        best=max(f.metrics.get('event/descent_post_contact_ticks',0) for f in frames)
        summary=dict(policy=str(a.trained_checkpoint or a.frozen_policy or config.raw['initialization']['source_checkpoint']),config=str(a.config),success=bool(frames[-1].success),
            terminal=END_REASONS.get(frames[-1].end_code,str(frames[-1].end_code)),
            first_contact_step=contact,longest_stable_ticks=float(best),longest_stable_seconds=float(best)*.02,
            total_seconds=trace.environment_transitions*.02,
            post_contact_seconds=None if contact is None else (trace.environment_transitions-contact)*.02,
            charged_interactions=trace.environment_transitions,maximum_interactions=400,
            role='canonical_fixed_start_development',statistical_success_rate_available=False,
            final_test_used=False,source_actor_sha256=source['actor_sha256'])
        write('summary.json',summary)
        with (a.output/'steps.csv').open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['step','time_s','x_m','z_m','vx_mps','reward','valid_contact','stable_ticks','end_code'])
            for i,f in enumerate(frames):
                writer.writerow([i,i*.02,f.qpos[0],f.qpos[2],f.qvel[0],f.reward,
                    f.metrics.get('event/descent_valid_contact_seen',0),f.metrics.get('event/descent_post_contact_ticks',0),f.end_code])
        render_trace(env,trace,a.output/'replay.mp4',fps=50)
        write('status.json',dict(phase='completed',**summary,wall_seconds=time.monotonic()-start))
    except BaseException as exc:
        write('status.json',dict(phase='error',error=repr(exc),wall_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    main()
