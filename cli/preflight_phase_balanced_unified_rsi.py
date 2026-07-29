"""Engineering preflight for phase-balanced unified-policy Tube-RSI."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dvgc.config import file_sha256
from dvgc.runtime import save_json


STAGES = ("takeoff", "ascent", "apex", "descent", "landing")


def validate_static_contract(bank, report: dict) -> dict:
    masses = report.get("stage_weight_mass", {})
    flags = {
        "bank_role": bank.metadata.get("artifact_role") == "phase_balanced_tube_rsi_reset_bank",
        "not_formal_tube_or_jel": bank.metadata.get("formal_tube_or_jel") is False,
        "five_stages": set(masses) == set(STAGES),
        "equal_stage_mass": set(masses) == set(STAGES) and all(
            np.isclose(float(masses[stage]), .2, atol=1e-7) for stage in STAGES
        ),
        "positive_reset_weights": bool(bank.records) and all(
            np.isfinite(float(row.get("reset_weight", np.nan)))
            and float(row["reset_weight"]) > 0 for row in bank.records
        ),
        "training_only": bool(bank.records) and all(
            row.get("training_only") is True and row.get("artifact_role") == "proposal_support_bank"
            for row in bank.records
        ),
        "no_embedded_safe_claim": all(
            not row.get("certified_safe") and not row.get("safe_claim_allowed")
            for row in bank.records
        ),
    }
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-bank", required=True)
    parser.add_argument("--phase-bank-report", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reset-samples", type=int, default=256)
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_850_000)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite unified RSI preflight")
    if args.reset_samples < 100 or not 1 <= args.short_window <= 10:
        raise SystemExit("preflight requires >=100 reset samples and a 1--10 tick short window")

    import jax
    import jax.numpy as jnp
    from cli.runtime_gate import source_fingerprint
    from dvgc.bank import SnapshotBank
    from dvgc.config import load_config
    from dvgc.descent_supervised import build_actor_tools
    from dvgc.env import END_NONFINITE, OrangeBikeDVGC
    from dvgc.policy import load_bundle
    from dvgc.rollout import restore_snapshot

    bank = SnapshotBank.load(args.phase_bank)
    bank_report = json.loads(Path(args.phase_bank_report).read_text())
    static = validate_static_contract(bank, bank_report)
    if bank_report.get("output_bank_sha256") != file_sha256(args.phase_bank):
        static["bank_report_identity"] = False
    else:
        static["bank_report_identity"] = True
    params, policy_cfg, manifest = load_bundle(args.policy, verify_files=True)
    static["policy_is_distillation_initialization"] = (
        manifest.get("artifact_role") == "final_shared_policy_initialization"
        and manifest.get("formal_tube_or_jel") is False
        and manifest.get("PPO_authorization") is False
    )
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    runtime_current = gate.get("status") == "PASS" and gate.get("source_fingerprint") == source_fingerprint(Path.cwd())
    cfg = load_config(overrides={
        **policy_cfg, "training_stage": "flight", "use_bank_resets": True,
        "domain_randomization": False, "obs_noise_enable": False,
        "expert_chain_termination": False, "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=bank)
    _, actor_action, _ = build_actor_tools(env, params)
    reset = jax.jit(env.reset); step = jax.jit(env.step)

    reset_counts = Counter()
    finite_reset_step = True; action_limit = True; nonfinite_codes = 0
    parent_names = tuple(env._reset_parent_ids)
    for index in range(args.reset_samples):
        state = reset(jax.random.PRNGKey(args.seed + index))
        parent_index = int(np.asarray(state.info["reset_parent"]))
        if parent_index < 0:
            name = "natural"
        else:
            name = parent_names[parent_index].split(":", 1)[0]
        reset_counts[name] += 1
        action = np.asarray(actor_action(params[1], state.obs["state"][None, :]))[0]
        action_limit &= bool(np.isfinite(action).all() and np.max(np.abs(action)) <= 1.000001)
        next_state = step(state, jnp.asarray(action))
        arrays = (next_state.data.qpos, next_state.data.qvel, next_state.obs["state"])
        finite_reset_step &= all(bool(np.isfinite(np.asarray(value)).all()) for value in arrays)
        nonfinite_codes += int(int(np.asarray(next_state.info["end_code"])) == END_NONFINITE)

    roundtrips = []
    for stage in STAGES:
        record = next(row for row in bank.records if row["phase_rsi_stage"] == stage)
        key = jax.random.PRNGKey(args.seed + 1000 + STAGES.index(stage))
        first = restore_snapshot(env, record, key); second = restore_snapshot(env, record, key)
        action0 = np.asarray(actor_action(params[1], first.obs["state"][None, :]))[0]
        action1 = np.asarray(actor_action(params[1], second.obs["state"][None, :]))[0]
        qpos_exact = np.array_equal(np.asarray(first.data.qpos), np.asarray(second.data.qpos))
        qvel_exact = np.array_equal(np.asarray(first.data.qvel), np.asarray(second.data.qvel))
        action_exact = np.array_equal(action0, action1)
        phase_start = int(np.asarray(first.info["phase"])); phases = [phase_start]
        finite = True; first_qpos, first_qvel = None, None
        for tick in range(args.short_window):
            action = actor_action(params[1], first.obs["state"][None, :])[0]
            first = step(first, action)
            if tick == 0:
                first_qpos = np.asarray(first.data.qpos); first_qvel = np.asarray(first.data.qvel)
            phases.append(int(np.asarray(first.info["phase"])))
            finite &= bool(np.isfinite(np.asarray(first.data.qpos)).all()
                           and np.isfinite(np.asarray(first.data.qvel)).all()
                           and np.isfinite(np.asarray(first.obs["state"])).all())
        second = step(second, jnp.asarray(action1))
        first_step_exact = (np.array_equal(first_qpos, np.asarray(second.data.qpos))
                            and np.array_equal(first_qvel, np.asarray(second.data.qvel)))
        roundtrips.append({
            "stage": stage, "record_id": record["id"], "qpos_restore_exact": qpos_exact,
            "qvel_restore_exact": qvel_exact, "deterministic_action_exact": action_exact,
            "first_step_exact": first_step_exact, "short_window_finite": finite,
            "phase_sequence": phases, "phase_monotonic": phases == sorted(phases),
            "first_action": action0.tolist(),
        })
    observed_bank_stages = {stage for stage in STAGES if reset_counts[stage] > 0}
    checks = {
        **static, "runtime_gate_current": runtime_current,
        "all_five_bank_stages_sampled": observed_bank_stages == set(STAGES),
        "reset_and_step_finite": finite_reset_step, "actions_within_limits": action_limit,
        "no_nonfinite_terminal": nonfinite_codes == 0,
        "snapshot_roundtrip_exact": all(row["qpos_restore_exact"] and row["qvel_restore_exact"]
                                         for row in roundtrips),
        "deterministic_inference_exact": all(row["deterministic_action_exact"] for row in roundtrips),
        "deterministic_first_step_exact": all(row["first_step_exact"] for row in roundtrips),
        "short_window_finite": all(row["short_window_finite"] for row in roundtrips),
        "phase_monotonic": all(row["phase_monotonic"] for row in roundtrips),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "artifact_role": "phase_balanced_unified_rsi_engineering_preflight",
        "PPO_authorization": bool(all(checks.values())),
        "formal_tube_or_jel": False, "checks": checks,
        "reset_samples": args.reset_samples, "reset_source_counts": dict(reset_counts),
        "roundtrip_short_window": roundtrips,
        "phase_bank_sha256": file_sha256(args.phase_bank),
        "policy_params_sha256": file_sha256(Path(args.policy) / "params.pkl"),
        "runtime_source_fingerprint": source_fingerprint(Path.cwd()),
        "next_gate": "bounded 2--5% joint Tube-RSI PPO pilot with fixed phase-wise evaluation",
    }
    save_json(output, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
