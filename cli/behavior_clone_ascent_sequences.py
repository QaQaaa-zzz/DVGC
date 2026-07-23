"""Behavior-clone successful bounded Ascent->Apex proposal sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import optax

from cli.acquire_ascent_apex_parents import _local_action
from dvgc.bank import SnapshotBank
from dvgc.config import load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import make_dvgc_ppo_networks, save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entry-bank", required=True)
    p.add_argument("--acquisition-report", required=True)
    p.add_argument("--initial-policy", required=True)
    p.add_argument("--output-policy", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=10_620_000)
    a = p.parse_args()
    bank = SnapshotBank.load(a.entry_bank)
    acquisition = json.loads(Path(a.acquisition_report).read_text())
    successful = {}
    for row in acquisition["search_outcomes"]:
        if row["success"]:
            successful.setdefault(row["trajectory_parent_id"], row)
    if len(successful) < 2:
        raise SystemExit("BC requires at least two independent successful parents")
    cfg = load_config(a.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "ascent_to_apex",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    observations, actions, lineage = [], [], []
    by_parent = {row["trajectory_parent_id"]: row for row in bank.records}
    for pi, (parent, outcome) in enumerate(sorted(successful.items())):
        row = by_parent[parent]
        state = restore_snapshot(env, row, jax.random.PRNGKey(a.seed + pi))
        for tick in range(int(outcome["entry_tick"])):
            action = _local_action(outcome["parameters"], tick)
            observations.append(np.asarray(state.obs["state"], np.float32))
            actions.append(np.asarray(action, np.float32))
            lineage.append({
                "trajectory_parent_id": parent, "tick": tick,
                "proposal_round": outcome["round"],
                "proposal_index": outcome["proposal_index"],
            })
            state = step(state, action)
    obs = jp.asarray(np.stack(observations))
    target = jp.asarray(np.stack(actions))
    params, _initial_cfg, initial_manifest = load_bundle(
        a.initial_policy, verify_files=True
    )
    try:
        from brax.training.acme import running_statistics
    except ImportError:
        from brax.training import running_statistics
    sample = env.reset(jax.random.PRNGKey(0)).obs
    networks = make_dvgc_ppo_networks(
        {key: tuple(value.shape) for key, value in sample.items()},
        env.action_size, running_statistics.normalize,
    )
    actor = params[1]

    def loss_fn(actor_params):
        logits = networks.policy_network.apply(
            params[0], actor_params, {"state": obs}
        )
        prediction = jp.tanh(logits[:, :env.action_size])
        return jp.mean(jp.square(prediction - target)), prediction

    optimizer = optax.adam(a.learning_rate)
    opt_state = optimizer.init(actor)

    @jax.jit
    def update(actor_params, state):
        (loss, prediction), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(actor_params)
        updates, state = optimizer.update(grads, state, actor_params)
        return optax.apply_updates(actor_params, updates), state, loss, prediction

    before_loss, before_prediction = loss_fn(actor)
    last_loss, prediction = before_loss, before_prediction
    for _ in range(a.epochs):
        actor, opt_state, last_loss, prediction = update(actor, opt_state)
    cloned = (params[0], actor, params[2])
    save_bundle(
        a.output_policy, params=cloned, config=cfg, xml_path=cfg.xml_path,
        candidate_bank=a.entry_bank, downstream_bank=None,
        policy_version="ascent-bc-independent-parents-v1",
        extra={
            "stage": "flight", "responsibility": "ascent_to_apex",
            "artifact_role": "proposal_controller_initialization",
            "initial_policy_version": initial_manifest["policy_version"],
            "independent_parent_count": len(successful),
        },
    )
    save_json(a.output_report, {
        "status": "PASS", "artifact_role": "ascent_behavior_cloning_initialization",
        "examples": len(observations), "independent_parent_count": len(successful),
        "parents": sorted(successful), "epochs": a.epochs,
        "learning_rate": a.learning_rate,
        "mse_before": float(before_loss), "mse_after": float(last_loss),
        "action_l2_after": float(jp.mean(jp.linalg.norm(prediction - target, axis=1))),
        "lineage": lineage, "not_a_tube": True,
    })
    print(json.dumps({
        "examples": len(observations), "parents": len(successful),
        "mse_before": float(before_loss), "mse_after": float(last_loss),
    }, indent=2))


if __name__ == "__main__":
    main()
