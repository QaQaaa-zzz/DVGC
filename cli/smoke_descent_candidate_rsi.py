"""JIT/batch/reset smoke for a provisional Descent RSI candidate bank."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import config_hash, file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.provisional_descent import StratifiedRSISampler, validate_candidate
from dvgc.runtime import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampler-state", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seed", type=int, default=12450000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.sampler_state).exists():
        raise SystemExit("refusing to overwrite RSI smoke output")
    bank = SnapshotBank.load(args.bank)
    for row in bank.records:
        validate_candidate(row)
    sampler = StratifiedRSISampler(bank.records, seed=args.seed)
    draw_indices = sampler.sample_indices(240)
    strata = Counter(
        (bank.records[index]["provisional_label"], bank.records[index]["descent_layer"])
        for index in draw_indices
    )
    # Save at an actual interruption boundary, then prove exact continuation.
    sampler.sample_indices(17)
    saved_state = sampler.state_dict()
    expected = sampler.sample_indices(64)
    resumed = StratifiedRSISampler(bank.records, seed=args.seed + 99)
    resumed.load_state_dict(saved_state)
    actual = resumed.sample_indices(64)
    resume_exact = actual == expected
    save_json(args.sampler_state, saved_state)

    cfg = load_config(args.config, {
        "training_stage": "flight", "use_bank_resets": True,
        "natural_prob_flight": 0.0, "domain_randomization": False,
        "obs_noise_enable": False, "stage_reachability_objective": "",
        "expert_chain_termination": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=bank)
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(args.seed + 1000), args.batch_size)
    state = reset_batch(keys)
    initial = state
    finite = True
    done_during_smoke = np.zeros(args.batch_size, bool)
    actions = jp.asarray(initial.info["last_action"])
    for _ in range(args.steps):
        state = step_batch(state, actions)
        finite &= all(np.isfinite(np.asarray(value)).all() for value in (
            state.data.qpos, state.data.qvel, state.obs["state"],
            state.obs["privileged_state"], state.info["obs_history"],
        ))
        done_during_smoke |= np.asarray(state.done) > .5
    history_shape_ok = tuple(initial.info["obs_history"].shape[1:]) == (
        env._actor_history_len, env._actor_frame_dim,
    )
    observation_shape_ok = tuple(initial.obs["state"].shape) == (
        args.batch_size, env._actor_obs_dim,
    )
    delay_shape_ok = all(
        np.asarray(row["policy_state"]["delay_buffer"]).shape == (1, 4)
        for row in bank.records
    )
    phase_valid = bool(np.all(np.isin(np.asarray(initial.info["phase"]), (1, 2))))

    # Put the counterfactuals in one compiled batch invocation.  Comparing two
    # separate MJX-Warp batch-step calls would mix in their known cross-call
    # numerical variability and falsely attribute that to state cross-talk.
    quad_keys = jp.concatenate([keys, keys, keys, keys], axis=0)
    quad = reset_batch(quad_keys)
    base_actions = np.asarray(quad.info["last_action"], np.float32)[:args.batch_size]
    quad_actions = np.concatenate([base_actions] * 4, axis=0)
    changed_index = 2 * args.batch_size
    quad_actions[changed_index, 0] = (
        -1.0 if quad_actions[changed_index, 0] > 0.0 else 1.0
    )
    quad_result = jax.block_until_ready(step_batch(quad, jp.asarray(quad_actions)))
    numerical_envelope = {}
    independence = []
    for name in ("qpos", "qvel", "ctrl"):
        value = np.asarray(getattr(quad_result.data, name)).reshape(
            4, args.batch_size, -1
        )
        control_error = float(np.max(np.abs(value[0] - value[1])))
        changed_other_error = float(np.max(np.abs(value[2, 1:] - value[3, 1:])))
        numerical_envelope[name] = {
            "same_action_control_linf": control_error,
            "changed_world_other_rows_linf": changed_other_error,
        }
        independence.append(
            changed_other_error <= max(control_error * 1.05, 1e-7)
        )
    info_independent = all(np.array_equal(
        np.asarray(quad_result.info[name]).reshape(4, args.batch_size)[2, 1:],
        np.asarray(quad_result.info[name]).reshape(4, args.batch_size)[3, 1:],
    ) for name in ("phase", "end_code"))
    no_crosstalk = bool(all(independence) and info_independent)
    ctrl = np.asarray(quad_result.data.ctrl).reshape(4, args.batch_size, -1)
    changed_world_responded = bool(not np.array_equal(ctrl[2, 0], ctrl[3, 0]))
    strata_present = set(strata) == set(sampler.buckets)
    reloaded = SnapshotBank.load(args.bank)
    bank_reload_verified = (
        len(reloaded.records) == len(bank.records)
        and [row["id"] for row in reloaded.records] == [row["id"] for row in bank.records]
    )
    report = {
        "status": "PASS" if all((
            resume_exact, finite, history_shape_ok, observation_shape_ok,
            delay_shape_ok, phase_valid, no_crosstalk, changed_world_responded,
            strata_present, not done_during_smoke.any(), bank_reload_verified,
        )) else "FAIL",
        "artifact_role": "descent_provisional_rsi_interface_smoke",
        "candidate_bank_role": bank.metadata.get("artifact_role"),
        "formal_tube_or_jel": False,
        "ppo_authorization": False,
        "records": len(bank.records),
        "sampled_draws": len(draw_indices),
        "sampled_strata": {f"{label}:{layer}": count for (label, layer), count in sorted(strata.items())},
        "all_strata_sampled": strata_present,
        "sampler_resume_exact": resume_exact,
        "jit_batch_reset": True,
        "jit_batch_step": True,
        "batch_size": args.batch_size,
        "smoke_steps": args.steps,
        "finite": finite,
        "done_during_smoke": int(done_during_smoke.sum()),
        "history_shape_ok": history_shape_ok,
        "delay_buffer_shape_ok": delay_shape_ok,
        "observation_interface_shape_ok": observation_shape_ok,
        "phase_valid": phase_valid,
        "batch_state_crosstalk_absent": no_crosstalk,
        "crosstalk_test": "single_invocation_control_and_counterfactual_pairs",
        "batch_index_numerical_envelope": numerical_envelope,
        "unchanged_world_phase_and_end_code_equal": info_independent,
        "changed_world_responded": changed_world_responded,
        "bank_reload_verified": bank_reload_verified,
        "bank_sha256": file_sha256(args.bank),
        "xml_sha256": file_sha256(cfg.xml_path),
        "effective_config_hash": config_hash(cfg),
        "runtime_solver": env._effective_mjx_solver,
        "sampler_state_path": str(Path(args.sampler_state).resolve()),
        "training_interface_ready": True,
    }
    report["training_interface_ready"] = report["status"] == "PASS"
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(40)


if __name__ == "__main__":
    main()
