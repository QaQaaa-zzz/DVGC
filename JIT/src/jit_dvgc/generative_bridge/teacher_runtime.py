"""Bounded per-stage runtime ownership; no caching of states or success labels."""
from pathlib import Path
from .contracts import digest
from .performance import measure


def _warp_ffi_backends():
    # MJX vendors Warp's FFI implementation; it is a different registry from
    # the separately installed warp package even though both expose one API.
    from mujoco.mjx.warp.ffi import ffi as mjx_ffi
    import warp._src.jax_experimental.ffi as external_ffi
    return list({id(module):module for module in (mjx_ffi,external_ffi)}.values())


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

    def memory_snapshot(self):
        """Current owned-process RSS and Warp graph roots, without GPU work."""
        import os
        import resource
        rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        backends={ffi.__name__:dict(callables=len(ffi._FFI_CALLABLE_REGISTRY),
            graph_captures=sum(len(c.captures) for c in ffi._FFI_CALLABLE_REGISTRY.values()))
            for ffi in _warp_ffi_backends()}
        return dict(rss_bytes=rss,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            warp_ffi_callables=sum(b['callables'] for b in backends.values()),warp_ffi_backends=backends,
            warp_graph_captures=sum(b['graph_captures'] for b in backends.values()),
            compiled_kernels=len(self.kernels),kernel_compile_count=self.compile_count)

    def memory_boundary(self,stage,directory,*,max_rss_bytes=8*1024**3):
        """Release transient graphs only after the evaluation is host-materialized.

        Eager MJX forward creates fresh Warp FFI wrappers per restored lane.
        Warp keeps those wrappers registered, so each wrapper's graph limit
        does not bound total retained native graphs. Its official cache release
        preserves callback registration and explicitly owned compiled kernels.
        A remaining RSS excess stops this worker for an explicit bounded restart.
        """
        import gc
        import time
        import jax
        from .protocol import atomic_json
        from .performance import record_event
        if type(max_rss_bytes) is not int or max_rss_bytes<=0:raise ValueError('positive teacher RSS limit required')
        before=self.memory_snapshot();started=time.time();clock=time.monotonic()
        for ffi in _warp_ffi_backends():ffi.clear_jax_callable_graph_cache()
        jax.clear_caches()
        gc.collect()
        after=self.memory_snapshot()
        state='host_rss_limit_exceeded' if after['rss_bytes']>max_rss_bytes else 'released'
        report=dict(schema='jit_teacher_memory_boundary_v1',stage=stage,state=state,
            before=before,after=after,max_rss_bytes=max_rss_bytes,
            explicit_kernels_preserved=True,automatic_restart=False,
            elapsed_seconds=time.monotonic()-clock)
        directory=Path(directory)
        atomic_json(directory/'memory_status.json',report)
        record_event(directory/'memory.jsonl','teacher_memory_boundary',started,report['elapsed_seconds'],
            stage=stage,before=before,after=after,max_rss_bytes=max_rss_bytes,status=state)
        if after['warp_graph_captures']:
            raise MemoryError('teacher transient Warp graphs remain after official cleanup')
        if state!='released':
            raise MemoryError(f'teacher host RSS {after["rss_bytes"]} exceeds declared {max_rss_bytes}; explicit restart required')
        return report


def kernel_identity(spec,policy,count,record_preobs,prefix_name,bridge):
    # Pinned same-structure weights/normalizer may be closed over once; any change
    # has a distinct compiled kernel. State and all candidate actions are dynamic.
    return digest(dict(policy=policy,count=count,horizon=spec['horizon'],
        reward=spec.get('reward_mode'),success=spec.get('success_criterion'),
        record_preobs=record_preobs,prefix_name=prefix_name,bridge=bridge,
        warmup=spec.get('warmup_initializer'),
        bridge_execution_horizon=spec.get('bridge_execution_horizon',16),
        capture_snapshot_steps=spec.get('capture_snapshot_steps',[]),
        capture_snapshot_lanes=spec.get('capture_snapshot_lanes',[]),
        preserve_context=spec.get('preserve_snapshot_episode_context',False),
        retention_diagnostics=spec.get('record_retention_diagnostics',False),
        rng_count=spec.get('suffix_rng_count',count),
        rng_indices=spec.get('suffix_rng_indices',list(range(count)))))
