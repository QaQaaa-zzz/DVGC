"""Memory-stable independent-process runner for frontier continuation shards."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax

from .phase_specific_frontier import (
    _frontier_label_context,
    _read_json,
    _runtime,
)
from .unified_continuation_shards import label_unified_continuation_shard


WARP_MEMORY_MAINTENANCE_STEP_INTERVAL = 64
WARP_MEMORY_TELEMETRY_STEP_INTERVAL = 4096


def _build_memory_stable_step(env: Any, *, device: str = "cuda:0"):
    try:
        import warp as wp
    except Exception as exc:
        raise RuntimeError(
            "Warp runtime is required for memory-stable frontier shard labeling"
        ) from exc

    compiled_step = jax.jit(env.step)
    mempool_supported = bool(wp.is_mempool_supported(device))
    mempool_enabled = (
        bool(wp.is_mempool_enabled(device)) if mempool_supported else False
    )
    if mempool_enabled:
        wp.set_mempool_release_threshold(device, 0)

    print(
        "[frontier-shard] warp_memory_maintenance=enabled "
        f"device={device} step_interval={WARP_MEMORY_MAINTENANCE_STEP_INTERVAL} "
        f"mempool_supported={mempool_supported} mempool_enabled={mempool_enabled}",
        flush=True,
    )
    step_count = 0

    def step_with_memory_maintenance(state, action):
        nonlocal step_count
        next_state = compiled_step(state, action)
        step_count += 1
        if step_count % WARP_MEMORY_MAINTENANCE_STEP_INTERVAL == 0:
            jax.block_until_ready(next_state)
            wp.synchronize_device(device)
        if step_count % WARP_MEMORY_TELEMETRY_STEP_INTERVAL == 0:
            fields = [f"steps={step_count}"]
            if mempool_enabled:
                fields.extend(
                    [
                        f"mempool_current={wp.get_mempool_used_mem_current(device)}",
                        f"mempool_high={wp.get_mempool_used_mem_high(device)}",
                    ]
                )
            fields.append(f"device_free={wp.get_device(device).free_memory}")
            print("[frontier-shard] warp_memory " + " ".join(fields), flush=True)
        return next_state

    return step_with_memory_maintenance


def run_memory_stable_frontier_label_shard(
    *,
    plan_path: Path,
    role_root: Path,
    role: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    plan, record, artifact, acquisition_dir, frozen_sha = _frontier_label_context(
        plan_path=plan_path,
        role_root=role_root,
        role=role,
    )
    output_dir = (
        Path(role_root)
        / "label_shards"
        / f"shard_{int(shard_index):03d}_of_{int(shard_count):03d}"
    )
    if (output_dir / "summary.json").is_file():
        summary = _read_json(output_dir / "summary.json")
        if summary.get("status") == "completed_shard":
            return summary
        raise RuntimeError(f"existing frontier label shard is not completed: {output_dir}")

    env, policy = _runtime(record=record, artifact=artifact)
    if jax.default_backend() != "gpu":
        raise RuntimeError("frontier continuation shard requires the visible JAX GPU")
    step_fn = _build_memory_stable_step(env)
    seeds = plan["seeds"][role]
    max_ticks = int(plan["fixed_probe_panel"]["max_label_ticks"])
    return label_unified_continuation_shard(
        acquisition_dir / "catalog.json",
        output_dir,
        env=env,
        policy=policy,
        policy_record=record,
        frozen_manifest_sha256=frozen_sha,
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        max_ticks=max_ticks,
        protocol_seed=int(seeds["labeling"]),
        compiled_step_fn=step_fn,
    )
