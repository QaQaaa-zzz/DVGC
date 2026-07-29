"""Build a bounded shared-actor initialization from phase-balanced teachers."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


def validate_dataset(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if (payload.get("schema") != "dvgc_phase_balanced_distillation_teacher_v1"
            or payload.get("artifact_role") != "phase_balanced_distillation_teacher_dataset"
            or payload.get("formal_tube_or_jel") is not False):
        raise ValueError("input is not a phase-balanced distillation teacher dataset")
    examples = list(payload.get("examples", []))
    observations = np.asarray([row["observation"] for row in examples], np.float32)
    actions = np.asarray([row["action"] for row in examples], np.float32)
    weights = np.asarray([row["training_weight"] for row in examples], np.float32)
    phases = [str(row["phase"]) for row in examples]
    if not examples or observations.ndim != 2 or observations.shape[1] != 140:
        raise ValueError("teacher observations must be a nonempty N x 140 array")
    if actions.shape != (len(examples), 4) or not np.isfinite(actions).all():
        raise ValueError("teacher actions must be finite N x 4")
    if np.max(np.abs(actions)) > 1.000001:
        raise ValueError("teacher action exceeds normalized action limits")
    if not np.isfinite(observations).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("teacher observations/weights must be finite and weights positive")
    masses = {phase: float(weights[np.asarray(phases) == phase].sum()) for phase in sorted(set(phases))}
    if set(masses) != {"takeoff", "ascent", "apex", "descent", "landing"}:
        raise ValueError(f"all five phases are required, got {sorted(masses)}")
    if any(not np.isclose(mass, .2, atol=1e-6) for mass in masses.values()):
        raise ValueError(f"teacher weights are not phase balanced: {masses}")
    return observations, actions, weights / weights.sum(), phases


def summarize_error(prediction: np.ndarray, target: np.ndarray, phases: list[str]) -> dict:
    error = np.linalg.norm(np.asarray(prediction) - np.asarray(target), axis=1)
    result = {}
    for phase in sorted(set(phases)):
        values = error[np.asarray(phases) == phase]
        result[phase] = {
            "count": len(values), "mean_action_l2": float(np.mean(values)),
            "p95_action_l2": float(np.percentile(values, 95)), "max_action_l2": float(np.max(values)),
        }
    result["all"] = {
        "count": len(error), "mean_action_l2": float(np.mean(error)),
        "p95_action_l2": float(np.percentile(error, 95)), "max_action_l2": float(np.max(error)),
    }
    return result


def summarize_fidelity(prediction: np.ndarray, target: np.ndarray,
                       phases: list[str]) -> dict:
    """Report coordinate and vector action error for each frozen expert phase."""
    delta = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    result = {}
    for phase in sorted(set(phases)):
        values = delta[np.asarray(phases) == phase]
        action_l2 = np.linalg.norm(values, axis=1)
        result[phase] = {
            "count": len(values),
            "rms": float(np.sqrt(np.mean(values * values))),
            "max": float(np.max(np.abs(values))),
            "mean_action_l2": float(np.mean(action_l2)),
            "p95_action_l2": float(np.percentile(action_l2, 95)),
            "max_action_l2": float(np.max(action_l2)),
        }
    return result


def downstream_fidelity_pass(fidelity: dict, *, rms_limit: float = .02,
                             max_limit: float = .05) -> bool:
    return all(
        fidelity[phase]["rms"] <= rms_limit and fidelity[phase]["max"] <= max_limit
        for phase in ("descent", "landing")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--base-policy", required=True, help="Frozen Landing policy used only as initialization")
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    args = parser.parse_args()
    output_policy, output_report = Path(args.output_policy), Path(args.output_report)
    if output_policy.exists() or output_report.exists():
        raise SystemExit("refusing to overwrite shared-actor distillation output")
    if args.steps <= 0 or not 0 < args.learning_rate <= 1e-3:
        raise SystemExit("invalid bounded distillation steps or learning rate")
    with Path(args.teacher_dataset).open("rb") as stream:
        payload = pickle.load(stream)
    observations, targets, weights, phases = validate_dataset(payload)

    import jax
    import jax.numpy as jnp
    import optax
    from dvgc.bank import SnapshotBank
    from dvgc.descent_supervised import build_actor_tools
    from dvgc.env import OrangeBikeDVGC
    from dvgc.policy import load_bundle, save_bundle

    params, policy_cfg, manifest = load_bundle(args.base_policy, verify_files=True)
    cfg = load_config(overrides={
        **policy_cfg, "use_bank_resets": False, "domain_randomization": False,
        "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    _, actor_action, _ = build_actor_tools(env, params)
    base_actor = params[1]
    frozen_scale = copy.deepcopy(base_actor["params"]["scale_parameter"])
    trainable = {
        "neutral_loc": copy.deepcopy(base_actor["params"]["neutral_loc"]),
        "trunk": copy.deepcopy(base_actor["params"]["trunk"]),
    }
    obs = jnp.asarray(observations); target = jnp.asarray(targets); weight = jnp.asarray(weights)

    def assemble(value):
        return {"params": {
            "neutral_loc": value["neutral_loc"], "trunk": value["trunk"],
            "scale_parameter": frozen_scale,
        }}

    def loss_fn(value):
        prediction = actor_action(assemble(value), obs)
        per_example = jnp.mean(jnp.square(prediction - target), axis=1)
        return jnp.sum(weight * per_example), prediction

    optimizer = optax.adam(float(args.learning_rate)); optimizer_state = optimizer.init(trainable)

    @jax.jit
    def update(value, state):
        (_, _), gradient = jax.value_and_grad(loss_fn, has_aux=True)(value)
        updates, state = optimizer.update(gradient, state, value)
        value = optax.apply_updates(value, updates)
        loss, prediction = loss_fn(value)
        return value, state, loss, prediction

    initial_loss, initial_prediction = loss_fn(trainable)
    history = []
    best = copy.deepcopy(trainable); best_loss = float(initial_loss)
    initial_fidelity = summarize_fidelity(np.asarray(initial_prediction), targets, phases)
    best_eligible = copy.deepcopy(trainable) if downstream_fidelity_pass(initial_fidelity) else None
    best_eligible_loss = float(initial_loss) if best_eligible is not None else float("inf")
    prediction = initial_prediction
    for step in range(1, int(args.steps) + 1):
        trainable, optimizer_state, loss, prediction = update(trainable, optimizer_state)
        value = float(loss)
        if value < best_loss:
            best, best_loss = copy.deepcopy(trainable), value
        step_fidelity = summarize_fidelity(np.asarray(prediction), targets, phases)
        if downstream_fidelity_pass(step_fidelity) and value < best_eligible_loss:
            best_eligible, best_eligible_loss = copy.deepcopy(trainable), value
        if step % 25 == 0 or step == args.steps:
            history.append({"step": step, "weighted_mse": value})
    selected = best_eligible if best_eligible is not None else best
    selected_loss = best_eligible_loss if best_eligible is not None else best_loss
    distilled_actor = assemble(selected)
    final_prediction = np.asarray(actor_action(distilled_actor, obs))
    fidelity = summarize_fidelity(final_prediction, targets, phases)
    fidelity_pass = downstream_fidelity_pass(fidelity)
    distilled = (params[0], distilled_actor, params[2])
    save_bundle(
        output_policy, params=distilled, config=cfg, xml_path=cfg.xml_path,
        candidate_bank=payload.get("phase_bank_path"), downstream_bank=None,
        policy_version="phase-balanced-distillation-seed0-v1",
        extra={
            "artifact_role": "final_shared_policy_initialization",
            "formal_tube_or_jel": False, "PPO_authorization": False,
            "teacher_dataset_sha256": file_sha256(args.teacher_dataset),
            "teacher_phase_bank_sha256": payload["phase_bank_sha256"],
            "expert_controller_identities": payload["expert_controller_identities"],
            "teacher_action_fidelity": fidelity,
            "downstream_teacher_fidelity_pass": fidelity_pass,
            "downstream_action_fidelity_limits": {"rms": .02, "max": .05},
            "base_policy_version": manifest["policy_version"],
            "frozen_assets": ["normalizer", "critic", "log_std"],
            "trainable_assets": ["actor_trunk", "actor_mean_head"],
        },
    )
    phase_history = defaultdict(list)
    for row, phase in zip(np.asarray(final_prediction), phases):
        phase_history[phase].append(row)
    report = {
        "status": "PASS" if fidelity_pass else "DOWNSTREAM_FIDELITY_BLOCKER",
        "artifact_role": "final_shared_policy_initialization",
        "formal_tube_or_jel": False, "PPO_authorization": False,
        "steps": int(args.steps), "learning_rate": float(args.learning_rate),
        "weighted_mse_before": float(initial_loss), "weighted_mse_after": selected_loss,
        "action_error_before": summarize_error(np.asarray(initial_prediction), targets, phases),
        "action_error_after": summarize_error(final_prediction, targets, phases),
        "teacher_action_fidelity": fidelity,
        "downstream_teacher_fidelity_pass": fidelity_pass,
        "downstream_action_fidelity_limits": {"rms": .02, "max": .05},
        "checkpoint_selection": ("lowest_loss_with_downstream_fidelity"
                                 if best_eligible is not None else "lowest_loss_diagnostic_only"),
        "history": history, "frozen_normalizer": True, "frozen_critic": True,
        "frozen_log_std": True, "output_policy": str(output_policy),
        "output_params_sha256": file_sha256(output_policy / "params.pkl"),
        "teacher_dataset_sha256": file_sha256(args.teacher_dataset),
        "next_gate": "fixed phase-wise physical retention preflight before joint Tube-RSI PPO",
    }
    save_json(output_report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
