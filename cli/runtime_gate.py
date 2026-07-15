"""Run or verify the complete local gate required before long DVGC PPO runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import (
    AUTHORITATIVE_XML_PATH,
    AUTHORITATIVE_XML_SHA256,
    ID_STAGE,
    config_hash,
    file_sha256,
    load_config,
)
from dvgc.composite import CompositeSession
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.experts import StageExpertRegistry
from dvgc.model import inspect_model
from dvgc.policy import load_bundle, save_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import (
    assert_brax_metric_contract,
    build_inference,
    make_ppo_train_fn,
    ppo_effective_timesteps,
    save_json,
    scalar,
)


GATE_VERSION = 4

# Restoring the serialized state itself must remain tight.  The subsequent
# contact step crosses the MJX Warp custom-call boundary, where repeated
# float32 executions can differ slightly even after sequential synchronization.
# These field-specific limits cover the independently observed replay jitter
# without weakening exact checks on reset state or discrete episode semantics.
SNAPSHOT_INITIAL_TOLERANCES = {
    "qpos": 1e-5,
    "qvel": 1e-5,
    "ctrl": 1e-5,
    "qacc_warmstart": 1e-5,
    "actor_obs": 1e-5,
    "critic_obs": 1e-5,
}
SNAPSHOT_STEP_TOLERANCES = {
    "qpos": 5e-5,
    "qvel": 2e-3,
    "reward": 1e-5,
    # Actor history contains an unnormalised accelerometer channel, so small
    # solver-order differences are amplified here relative to qpos/qvel.
    "actor_obs": 2e-2,
    "critic_obs": 1e-3,
}
SNAPSHOT_DISCRETE_FIELDS = (
    "done",
    "phase",
    "end_code",
    "recovery_success",
    "terminated",
    "truncated",
)


def source_fingerprint(root: Path) -> str:
    """Hash the actual PPO/reset/step dependency closure.

    Offline audit, report, controller, and test edits do not invalidate a
    previously executed dynamic runtime gate.  The resolved config and XML
    retain their separate hashes in the gate report.
    """
    relative_files = (
        "dvgc/action_mapping.py", "dvgc/bank.py", "dvgc/config.py",
        "dvgc/env.py", "dvgc/experts.py", "dvgc/composite.py", "dvgc/expert_training.py", "dvgc/curriculum.py", "dvgc/model.py", "dvgc/policy.py",
        "dvgc/rewards.py", "dvgc/rollout.py", "dvgc/runtime.py", "dvgc/wrappers.py",
        "dvgc/signals.py", "cli/train.py", "cli/train_expert.py", "cli/evaluate_composite.py", "cli/runtime_gate.py",
        "pyproject.toml", "requirements.txt",
    )
    files = [root / name for name in relative_files if (root / name).is_file()]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _host_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        array = np.asarray(jax.device_get(value))
        result[str(key)] = float(array) if array.ndim == 0 else array.tolist()
    return result


def _gate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep gate evidence compact while retaining all terminal causes."""
    host = _host_metrics(metrics)
    exact = {
        "eval/avg_episode_length",
        "eval/std_episode_length",
        "eval/episode_reward",
        "eval/episode_reward_std",
        "eval/episode_success",
        "eval/episode_success_std",
        "training/entropy_loss",
        "training/kl_mean",
        "training/policy_loss",
        "training/total_loss",
        "training/v_loss",
    }
    return {
        key: value
        for key, value in host.items()
        if key in exact or key.startswith("eval/episode_end/")
    }


def _assert_finite_state(state: Any) -> None:
    values = {
        "qpos": state.data.qpos,
        "qvel": state.data.qvel,
        "ctrl": state.data.ctrl,
        "reward": state.reward,
        **{f"obs/{key}": value for key, value in state.obs.items()},
    }
    for name, value in values.items():
        if not np.isfinite(np.asarray(jax.device_get(value))).all():
            raise RuntimeError(f"Non-finite value during raw rollout: {name}")


def _rollout_smoke(env: OrangeBikeDVGC, *, seed: int, random_actions: bool) -> dict[str, Any]:
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    terminal_counts = {"final_recovery": 0, "physical_failure": 0, "timeout": 0}
    end_reasons: dict[str, int] = {}
    completed_episodes = 0
    episode_steps = 0
    for _ in range(100):
        key, action_key, reset_key = jax.random.split(key, 3)
        action = (
            jax.random.uniform(action_key, (env.action_size,), minval=-1.0, maxval=1.0)
            if random_actions
            else jp.zeros((env.action_size,), jp.float32)
        )
        state = step_fn(state, action)
        episode_steps += 1
        _assert_finite_state(state)
        if scalar(state.done) > 0.5:
            final = bool(np.asarray(jax.device_get(state.info["recovery_success"])))
            terminated = bool(np.asarray(jax.device_get(state.info["terminated"])))
            truncated = bool(np.asarray(jax.device_get(state.info["truncated"])))
            if final:
                terminal_counts["final_recovery"] += 1
            elif terminated:
                terminal_counts["physical_failure"] += 1
            elif truncated:
                terminal_counts["timeout"] += 1
            else:
                raise RuntimeError("done was set without Recovery, physical termination, or timeout")
            code = int(np.asarray(jax.device_get(state.info["end_code"])))
            reason = END_REASON.get(code, f"unknown_{code}")
            end_reasons[reason] = end_reasons.get(reason, 0) + 1
            completed_episodes += 1
            state = env.reset(reset_key)
            episode_steps = 0
    return {
        "steps": 100,
        "completed_episodes": completed_episodes,
        "active_episode_steps": episode_steps,
        "terminal_counts": terminal_counts,
        "end_reasons": end_reasons,
    }


def _max_abs(left: Any, right: Any) -> float:
    a = np.asarray(jax.device_get(left), np.float64)
    b = np.asarray(jax.device_get(right), np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _snapshot_discrete_values(state: Any) -> dict[str, np.ndarray]:
    return {
        "done": np.array(jax.device_get(state.done), copy=True),
        **{
            key: np.array(jax.device_get(state.info[key]), copy=True)
            for key in SNAPSHOT_DISCRETE_FIELDS
            if key != "done"
        },
    }


def _snapshot_gate(env: OrangeBikeDVGC, work: Path) -> dict[str, Any]:
    state = env.reset(jax.random.PRNGKey(20))
    step_fn = jax.jit(env.step)
    action = jp.asarray([0.05, 0.10, -0.05, 0.05], jp.float32)
    for _ in range(5):
        state = step_fn(state, action)
    phase = ID_STAGE[int(np.asarray(jax.device_get(state.info["phase"])))]
    bank_path = work / "snapshot_roundtrip.pkl"
    SnapshotBank([env.snapshot_record(state, phase)]).save(bank_path)
    record = SnapshotBank.load(bank_path).records[0]
    restored = restore_snapshot(env, record, jax.random.PRNGKey(21))
    initial_errors = {
        "qpos": _max_abs(state.data.qpos, restored.data.qpos),
        "qvel": _max_abs(state.data.qvel, restored.data.qvel),
        "ctrl": _max_abs(state.data.ctrl, restored.data.ctrl),
        "qacc_warmstart": _max_abs(state.data.qacc_warmstart, restored.data.qacc_warmstart),
        "actor_obs": _max_abs(state.obs["state"], restored.obs["state"]),
        "critic_obs": _max_abs(state.obs["privileged_state"], restored.obs["privileged_state"]),
    }
    next_original = step_fn(state, action)
    # MJX Warp custom calls may use shared scratch storage.  Dispatching two
    # independent contact steps before either one is synchronized can race and
    # make even step_fn(state, action) disagree with itself.  Snapshot replay
    # is a sequential comparison, so finish the original branch first.
    jax.block_until_ready(next_original)
    original_step_values = {
        "qpos": np.array(jax.device_get(next_original.data.qpos), copy=True),
        "qvel": np.array(jax.device_get(next_original.data.qvel), copy=True),
        "reward": np.array(jax.device_get(next_original.reward), copy=True),
        "actor_obs": np.array(jax.device_get(next_original.obs["state"]), copy=True),
        "critic_obs": np.array(
            jax.device_get(next_original.obs["privileged_state"]), copy=True
        ),
    }
    original_discrete_values = _snapshot_discrete_values(next_original)
    next_restored = step_fn(restored, action)
    step_errors = {
        "qpos": _max_abs(original_step_values["qpos"], next_restored.data.qpos),
        "qvel": _max_abs(original_step_values["qvel"], next_restored.data.qvel),
        "reward": _max_abs(original_step_values["reward"], next_restored.reward),
        "actor_obs": _max_abs(original_step_values["actor_obs"], next_restored.obs["state"]),
        "critic_obs": _max_abs(
            original_step_values["critic_obs"], next_restored.obs["privileged_state"]
        ),
    }
    restored_discrete_values = _snapshot_discrete_values(next_restored)
    discrete_equal = {
        key: bool(np.array_equal(original_discrete_values[key], restored_discrete_values[key]))
        for key in SNAPSHOT_DISCRETE_FIELDS
    }
    discrete_mismatches = {
        key: {
            "original": original_discrete_values[key].tolist(),
            "restored": restored_discrete_values[key].tolist(),
        }
        for key, equal in discrete_equal.items()
        if not equal
    }
    initial_tolerances = dict(SNAPSHOT_INITIAL_TOLERANCES)
    step_tolerances = dict(SNAPSHOT_STEP_TOLERANCES)
    initial_exceeded = {
        key: value for key, value in initial_errors.items()
        if value > initial_tolerances[key]
    }
    step_exceeded = {
        key: value for key, value in step_errors.items()
        if value > step_tolerances[key]
    }
    if initial_exceeded or step_exceeded or discrete_mismatches:
        raise RuntimeError(
            "Snapshot round-trip exceeded field tolerances: "
            f"initial_exceeded={initial_exceeded}, step_exceeded={step_exceeded}, "
            f"discrete_mismatches={discrete_mismatches}, "
            f"initial={initial_errors}, step={step_errors}"
        )
    return {
        "initial_tolerances": initial_tolerances,
        "step_tolerances": step_tolerances,
        "initial_max_abs": initial_errors,
        "step_max_abs": step_errors,
        "step_discrete_equal": discrete_equal,
    }


def _ppo_gate(env: OrangeBikeDVGC, work: Path) -> tuple[Any, dict[str, Any]]:
    requested = 64
    layout = {"unroll_length": 4, "batch_size": 8, "num_minibatches": 1}
    progress: list[dict[str, Any]] = []

    def progress_fn(step: int, metrics: dict[str, Any]) -> None:
        progress.append({"step": int(step), "metrics": _gate_metrics(metrics)})

    train_fn = make_ppo_train_fn(
        timesteps=requested,
        episode_length=64,
        num_envs=8,
        num_eval_envs=2,
        num_evals=2,
        seed=30,
        learning_rate=1e-4,
        entropy_cost=1e-3,
        reward_scaling=0.1,
        checkpoint_dir=work / "ppo_initial",
        unroll_length=layout["unroll_length"],
        batch_size=layout["batch_size"],
        num_minibatches=layout["num_minibatches"],
        num_updates_per_batch=1,
        discounting=.995,
        gae_lambda=.97,
        clipping_epsilon=.10,
        max_grad_norm=.75,
    )
    _, params, final_metrics = train_fn(environment=env, progress_fn=progress_fn, eval_env=env)

    resume_progress: list[dict[str, Any]] = []

    def resume_progress_fn(step: int, metrics: dict[str, Any]) -> None:
        resume_progress.append({"step": int(step), "metrics": _gate_metrics(metrics)})

    resume_requested = 32
    resume_fn = make_ppo_train_fn(
        timesteps=resume_requested,
        episode_length=64,
        num_envs=8,
        num_eval_envs=2,
        num_evals=2,
        seed=31,
        learning_rate=1e-4,
        entropy_cost=1e-3,
        reward_scaling=0.1,
        checkpoint_dir=work / "ppo_resume",
        unroll_length=layout["unroll_length"],
        batch_size=layout["batch_size"],
        num_minibatches=layout["num_minibatches"],
        num_updates_per_batch=1,
        discounting=.995,
        gae_lambda=.97,
        clipping_epsilon=.10,
        max_grad_norm=.75,
        restore_params=params,
    )
    _, resumed_params, resumed_metrics = resume_fn(
        environment=env, progress_fn=resume_progress_fn, eval_env=env
    )
    return resumed_params, {
        "initial": {
            "requested_timesteps": requested,
            "effective_timesteps": ppo_effective_timesteps(requested, num_evals=2, **layout),
            "progress": progress,
            "final_metrics": _gate_metrics(final_metrics),
        },
        "resume": {
            "requested_timesteps": resume_requested,
            "effective_timesteps": ppo_effective_timesteps(
                resume_requested, num_evals=2, **layout
            ),
            "progress": resume_progress,
            "final_metrics": _gate_metrics(resumed_metrics),
        },
    }


def _policy_gate(env: OrangeBikeDVGC, cfg: Any, params: Any, work: Path) -> dict[str, Any]:
    candidate_bank = work / "empty_candidates.pkl"
    SnapshotBank().save(candidate_bank)
    bundle = save_bundle(
        work / "policy_bundle",
        params=params,
        config=cfg,
        xml_path=cfg.xml_path,
        candidate_bank=candidate_bank,
        downstream_bank=None,
        policy_version="runtime-gate",
        extra={"stage": "full"},
    )
    loaded_params, _, _ = load_bundle(bundle, verify_files=True)
    inference = build_inference(env, loaded_params, deterministic=True)
    state = env.reset(jax.random.PRNGKey(40))
    action_a, _ = inference(state.obs, jax.random.PRNGKey(41))
    action_b, _ = inference(state.obs, jax.random.PRNGKey(42))
    error = _max_abs(action_a, action_b)
    if error > 1e-7:
        raise RuntimeError(f"Deterministic inference depends on PRNG key: max_abs={error}")
    return {"bundle": str(bundle), "different_key_action_max_abs": error}


def _composite_gate(env: OrangeBikeDVGC, cfg: Any, params: Any, work: Path) -> dict[str, Any]:
    entry=work/"runtime_entry.pkl"; SnapshotBank().save(entry); policies={}
    for stage in ("landing","flight"):
        stage_cfg=load_config(overrides={**cfg.to_dict(),"training_stage":stage})
        policies[stage]=save_bundle(work/f"expert_{stage}",params=params,config=stage_cfg,xml_path=stage_cfg.xml_path,candidate_bank=None,downstream_bank=(entry if stage=="flight" else None),policy_version=f"runtime-{stage}",extra={"stage":stage})
    registry=StageExpertRegistry.build(policies,{"flight":entry},runtime_source_fingerprint=source_fingerprint(Path(__file__).resolve().parents[1])); registry.validate_files()
    inference={stage:build_inference(env,params,deterministic=True) for stage in policies}
    class Always:
        def match(self,state): return True,0.0
    key=jax.random.PRNGKey(70); initial=env.reset(key); action,_=inference["flight"](initial.obs,jax.random.PRNGKey(71)); step=jax.jit(env.step)
    direct=step(initial,action); jax.block_until_ready(direct)
    session=CompositeSession(env,("flight","landing"),inference,{"flight":Always()},initial,key); handed=session.step(step_fn=step); jax.block_until_ready(handed)
    errors={"qpos":_max_abs(direct.data.qpos,handed.data.qpos),"qvel":_max_abs(direct.data.qvel,handed.data.qvel),"actor_obs":_max_abs(direct.obs["state"],handed.obs["state"]),"last_action":_max_abs(direct.info["last_action"],handed.info["last_action"])}
    limits={"qpos":5e-5,"qvel":3e-3,"actor_obs":2e-2,"last_action":1e-6}
    exceeded={k:v for k,v in errors.items() if v>limits[k]}
    if exceeded or session.active_stage!="landing" or len(session.handoffs)!=1: raise RuntimeError(f"Composite handoff continuity failed: errors={errors}, exceeded={exceeded}")
    landing_hash=registry.specs["landing"].policy_hash; registry.validate_files()
    return {"registry_hash":registry.registry_hash,"controller_stack_hash":registry.specs["flight"].controller_stack_hash,"landing_policy_hash_before":landing_hash,"landing_policy_hash_after":registry.specs["landing"].policy_hash,"active_stage_after_handoff":session.active_stage,"handoffs":len(session.handoffs),"continuity_errors":errors,"continuity_tolerances":limits}


def _check_report(report_path: Path, root: Path, config_path: str) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = load_config(config_path)
    expected = {
        "gate_version": GATE_VERSION,
        "source_fingerprint": source_fingerprint(root),
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_hash": config_hash(cfg),
    }
    if report.get("status") != "PASS":
        raise SystemExit(f"Runtime gate is not PASS: {report.get('status')}")
    for key, value in expected.items():
        if report.get(key) != value:
            raise SystemExit(f"Runtime gate is stale for {key}")
    print(f"Runtime gate PASS and current: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="docs/RUNTIME_GATE.json")
    parser.add_argument("--work-dir", default="runs/runtime_gate")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if args.check_only:
        _check_report(output, root, args.config)
        return

    work = (root / args.work_dir).resolve()
    if work.exists():
        raise SystemExit(f"Runtime gate work directory already exists: {work}")
    work.mkdir(parents=True)
    cfg = load_config(
        args.config,
        {
            "training_stage": "full",
            "use_bank_resets": False,
            "domain_randomization": False,
            "obs_noise_enable": False,
        },
    )
    base_cfg = load_config(args.config)
    report: dict[str, Any] = {
        "gate_version": GATE_VERSION,
        "status": "RUNNING",
        "started_at": time.time(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "source_fingerprint": source_fingerprint(root),
        "xml_sha256": file_sha256(base_cfg.xml_path),
        "config_hash": config_hash(base_cfg),
        "gates": {},
    }
    save_json(output, report)
    try:
        model = inspect_model(AUTHORITATIVE_XML_PATH)
        if model["xml_sha256"] != AUTHORITATIVE_XML_SHA256:
            raise RuntimeError("Authoritative XML hash mismatch")
        if model["named_masses_kg"].get("load") != 4.0:
            raise RuntimeError("Authoritative XML payload is not 4 kg")
        actuator = {row["name"]: row for row in model["actuators"]}
        for name in ("cmd_hip_f", "cmd_knee_f"):
            if actuator[name]["forcerange"] != "-50 50":
                raise RuntimeError(f"{name} does not have +/-50 N m force limits")
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
        report["gates"]["model_load"] = {"status": "PASS", "nq": env.mj_model.nq, "nv": env.mj_model.nv, "nu": env.mj_model.nu}
        assert_brax_metric_contract(env)
        sample = env.reset(jax.random.PRNGKey(1))
        report["gates"]["observation_metrics"] = {
            "status": "PASS",
            "actor_shape": list(sample.obs["state"].shape),
            "critic_shape": list(sample.obs["privileged_state"].shape),
            "metric_keys": sorted(sample.metrics),
        }
        report["gates"]["zero_action_100"] = _rollout_smoke(env, seed=2, random_actions=False)
        report["gates"]["random_action_100"] = _rollout_smoke(env, seed=3, random_actions=True)
        report["gates"]["snapshot_roundtrip"] = {"status": "PASS", **_snapshot_gate(env, work)}
        params, ppo = _ppo_gate(env, work)
        report["gates"]["short_ppo"] = {"status": "PASS", **ppo}
        report["gates"]["policy_roundtrip_determinism"] = {
            "status": "PASS",
            **_policy_gate(env, cfg, params, work),
        }
        report["gates"]["expert_registry_composite_handoff"] = {
            "status": "PASS", **_composite_gate(env,cfg,params,work)
        }
        report["status"] = "PASS"
    except BaseException as exc:
        report.update({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        report["finished_at"] = time.time()
        report["elapsed_seconds"] = report["finished_at"] - report["started_at"]
        save_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
