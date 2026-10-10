"""Bounded per-stage runtime ownership; no caching of states or success labels."""
from pathlib import Path
from .contracts import digest
from .performance import measure


class TeacherEvaluationSession:
    def __init__(self):
        self.suffixes={};self.kernels={};self.compile_count=0

    def suffix(self,spec,names,output):
        from ..exploration_continuation import FrozenSuffixEvaluator
        key=(str(Path(spec['bank']).resolve()),tuple(names),spec['horizon'])
        if self.suffixes and key not in self.suffixes:
            raise ValueError('teacher session is pinned to one bank/horizon')
        if key not in self.suffixes:
            with measure('model_actor_generator_load',backend='gpu',component='suffix'):
                self.suffixes[key]=FrozenSuffixEvaluator(spec['bank'],names,spec['horizon'],output,spec['budget'])
        return self.suffixes[key]

    def execute(self,key,function,*args):
        import jax
        hit=key in self.kernels
        if not hit:
            if len(self.kernels)>=4:raise ValueError('bounded teacher kernel cache exhausted')
            with measure('jax_trace_lower_compile',backend='gpu'):
                self.kernels[key]=jax.jit(function).lower(*args).compile()
            self.compile_count+=1
        with measure('warm_rollout' if hit else 'first_rollout',backend='gpu',cache_hit=hit):
            return jax.device_get(self.kernels[key](*args))

    def close(self):
        self.kernels.clear();self.suffixes.clear()


def kernel_identity(spec,policy,count,record_preobs,prefix_name,bridge):
    # Pinned same-structure weights/normalizer may be closed over once; any change
    # has a distinct compiled kernel. State and all candidate actions are dynamic.
    return digest(dict(policy=policy,count=count,horizon=spec['horizon'],
        reward=spec.get('reward_mode'),success=spec.get('success_criterion'),
        record_preobs=record_preobs,prefix_name=prefix_name,bridge=bridge,
        warmup=spec.get('warmup_initializer'),
        preserve_context=spec.get('preserve_snapshot_episode_context',False),
        retention_diagnostics=spec.get('record_retention_diagnostics',False),
        rng_count=spec.get('suffix_rng_count',count),
        rng_indices=spec.get('suffix_rng_indices',list(range(count)))))
