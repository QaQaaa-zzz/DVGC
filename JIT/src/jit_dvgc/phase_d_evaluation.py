"""Frozen, fixed-index evaluation of the Phase D recovery expert."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import load_config
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, END_REASONS
from .env import TwoPhaseBikeEnv
from .evaluation import capture_episode, save_episode_trace
from .handoff_bank import pytree_sha256
from .handoff_snapshot import compatibility_identity
from .phase_expert_init import ActorOnlyInitialization, make_actor_only_policy
from .snapshot_pool import SnapshotPool
from .phase_d_smoke import _source_bank_hashes


def _catalog_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_policy_key(eval_seed: int, source_checkpoint: str, source_tick: int) -> jax.Array:
    """Stable key derivation independent of Python's process-randomized hash."""
    digest = hashlib.sha256(f"{int(eval_seed)}|{source_checkpoint}|{int(source_tick)}".encode()).digest()
    words = np.frombuffer(digest[:8], dtype=np.uint32)
    return jax.random.fold_in(jax.random.PRNGKey(int(words[0])), int(words[1]))


def artifact_relative_path(artifact: Path, run_dir: Path) -> str:
    artifact_resolved = Path(artifact).resolve()
    run_resolved = Path(run_dir).resolve()
    try:
        return str(artifact_resolved.relative_to(run_resolved))
    except ValueError as exc:
        raise ValueError("artifact is outside run directory") from exc


def select_panel_entries(path: Path, *, eval_seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [row for row in payload.get("entries", []) if int(row["seed"]) in set(eval_seeds)]
    if set(eval_seeds) - {int(row["seed"]) for row in payload.get("entries", [])}:
        raise ValueError("unknown eval seed")
    # The catalog contains two physical 7987200 banks; select the bank that
    # contains the requested seed, yielding one source checkpoint panel.
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in entries:
        key = (row["source_bank"].split("_")[2], int(row["seed"]), row["parent_group_id"])
        grouped.setdefault(key, []).append(row)
    selected = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 7:
            continue
        selected.extend(sorted(rows, key=lambda row: (row["tick"], row["role"])))
    if len(selected) != 42:
        raise ValueError(f"panel must contain exactly 42 snapshots, found {len(selected)}")
    if len({(r["source_bank"], r["parent_group_id"], r["tick"]) for r in selected}) != 42:
        raise ValueError("panel snapshot identifiers are not unique")
    return selected


def validate_panel_budget(sample_count: int, max_ticks: int) -> int:
    if sample_count != 42:
        raise ValueError("panel sample count must be exactly 42")
    if max_ticks <= 0:
        raise ValueError("panel max-ticks must be positive")
    ceiling = sample_count * int(max_ticks)
    if ceiling > 4200:
        raise ValueError("panel interaction ceiling exceeded")
    return ceiling


def terminal_summary(info: dict[str, Any]) -> dict[str, Any]:
    code = int(np.asarray(info["end_code"]))
    return {"terminated": bool(np.asarray(info["terminated"])),
            "truncated": bool(np.asarray(info["truncated"])),
            "success": bool(np.asarray(info["success"])),
            "physical_failure": bool(np.asarray(info["physical_failure"])),
            "timeout": bool(np.asarray(info["timeout"])),
            "end_code": code, "reason": END_REASONS.get(code, f"unknown_{code}")}


def make_panel_compiled_fns(env):
    """Compile reusable reset/step callables once per panel."""
    return jax.jit(env.reset_descent_index), jax.jit(env.step)


def run_phase_d_panel(config_path: Path, checkpoint: Path, catalog: Path, *, eval_seeds: tuple[int, ...], max_ticks: int, run_id: str, run_root: Path | None = None) -> dict[str, Any]:
    config = load_config(Path(config_path))
    if config.phase != "descent_recovery" or config.formal is not None:
        raise ValueError("panel only supports non-formal descent_recovery")
    entries = select_panel_entries(catalog, eval_seeds=eval_seeds)
    ceiling = validate_panel_budget(len(entries), max_ticks)
    source_config = load_config(Path("JIT/configs/phase_u_continuation_10m.json"))
    source_env = TwoPhaseBikeEnv(source_config)
    base = Path(catalog).parent
    paths = [base / row["source_bank"] / row["snapshot"] for row in entries]
    pool = SnapshotPool.from_paths(paths, compatibility=compatibility_identity(source_env))
    env = TwoPhaseBikeEnv(config, snapshot_pool=pool)
    identity = CheckpointIdentity(config.config_sha256, env._bundle.xml_sha256, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, ACTION_ORDER)
    payload = load_checkpoint(Path(checkpoint), expected=identity)
    sidecar = json.loads((Path(checkpoint) / "identity.json").read_text())
    init = ActorOnlyInitialization(payload.observation_normalizer, payload.actor_params, payload.training_transitions, sidecar["payload_sha256"], pytree_sha256(payload.actor_params), {"actor_initialized": True, "critic_fresh": False, "optimizer_fresh": False})
    policy = make_actor_only_policy(env, init, deterministic=True)
    reset_index_fn, step_fn = make_panel_compiled_fns(env)
    root = Path(run_root) if run_root is not None else Path("JIT/runs/phase_d_evaluation")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "purpose": "descent_recovery_frozen_fixed_evaluation", "run_id": run_id, "config_sha256": config.config_sha256, "xml_sha256": env._bundle.xml_sha256, "checkpoint_payload_sha256": sidecar["payload_sha256"], "checkpoint_actor_sha256": init.actor_sha256, "catalog_sha256": _catalog_sha256(catalog), "source_bank_hashes": _source_bank_hashes(catalog, entries), "eval_seeds": list(eval_seeds), "sample_count": 42, "training_transitions": 0, "diagnostic_transition_ceiling": ceiling, "max_ticks": max_ticks, "producer_head": os.popen("git rev-parse HEAD").read().strip()}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summaries = []
    try:
        for index, row in enumerate(entries):
            policy_key = derive_policy_key(row["seed"], row["source_bank"].split("_")[2], row["tick"])
            sample_policy = lambda obs, key=policy_key: policy(obs, key)
            trace = capture_episode(env, sample_policy, seed=int(row["seed"]), horizon=max_ticks, reset_fn=lambda _key, i=index: reset_index_fn(jax.numpy.asarray(i, dtype=jax.numpy.int32)), step_fn=step_fn)
            trace_path = run_dir / "traces" / f"{row['source_bank']}_{row['parent_group_id']}_{row['role']}_{row['tick']}"
            artifact = save_episode_trace(trace, trace_path)
            terminal = trace.frames[-1]
            summaries.append({**{key: row.get(key) for key in ("source_bank", "seed", "parent_group_id", "role", "tick")}, "policy_key_derivation": "sha256(eval_seed|source_checkpoint|source_tick)", "transitions": trace.environment_transitions, "success": terminal.success, "physical_failure": terminal.physical_failure, "timeout": terminal.timeout, "end_code": terminal.end_code, "reason": END_REASONS.get(terminal.end_code, f"unknown_{terminal.end_code}"), "finite": bool(np.isfinite(np.asarray(trace.frames[-1].qpos)).all() and np.isfinite(np.asarray(trace.frames[-1].qvel)).all()), "trace_npz": artifact_relative_path(artifact.npz_path, run_dir)})
        report = {"status": "completed", "training_transitions": 0, "diagnostic_transitions": sum(x["transitions"] for x in summaries), "sample_count": len(summaries), "summaries": summaries}
        (run_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(report)
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return report
    except Exception as exc:
        manifest.update(status="engineering_error", reason=str(exc), diagnostic_transitions=sum(x["transitions"] for x in summaries), training_transitions=0)
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise
