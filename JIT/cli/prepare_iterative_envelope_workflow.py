#!/usr/bin/env python3
"""Generate one resumable automatic pi_k -> pi_(k+1) envelope workflow.

The generated workflow executes only the fixed, already-decided method.  It
never tunes hyperparameters or repairs a failed policy.  Any data-isolation,
continuation-calibration, Tube, training, freeze, or acceptance-gate failure
stops the workflow and requires a new scientific/engineering decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.soft_tube import load_soft_tube


PYTHON = "/home/qy/mujoco_playground/.venv/bin/python"
UP_CONFIG = "JIT/configs/phase_u_continuation_smoke.json"
DOWN_CONFIG = "JIT/configs/descent_recovery_smoke.json"


def read(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-policy", type=Path, required=True)
    parser.add_argument("--source-tube", type=Path, required=True)
    parser.add_argument("--tag", default="auto")
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.config_out.exists():
        raise FileExistsError(f"workflow config already exists: {args.config_out}")

    selected = read(args.selected_policy)
    if selected.get("schema") != "jit_selected_iteration_policy_v1" or selected.get("status") != "selected":
        raise ValueError("workflow requires selected pi_k artifact")
    k = int(selected["iteration"])
    if k < 1 or selected.get("policy_name") != f"pi_{k}":
        raise ValueError("automatic workflow requires selected pi_k with k>=1")
    tube = load_soft_tube(args.source_tube)
    if int(tube.manifest.get("iteration", -1)) != k:
        raise ValueError("source Tube iteration must equal selected policy iteration")

    next_k = k + 1
    repo_root = args.repo_root.resolve()
    work_root = (
        args.work_root
        if args.work_root is not None
        else Path(f"JIT/runs/iteration_auto/pi_{k}_to_pi_{next_k}_{args.tag}")
    )
    plan = work_root / "frontier_plan.json"
    train = work_root / "frontier_train"
    calibration = work_root / "frontier_calibration"
    acceptance = work_root / "frontier_acceptance"
    fields = work_root / f"continuation_C{str(k)}"
    tube_next = Path(f"JIT/runs/soft_tube/soft_tube_iter{next_k}_pi{k}_conditioned_{args.tag}")
    smoke = work_root / f"tube{next_k}_smoke"
    isolation = work_root / "role_isolation.json"
    baseline_lock = work_root / "acceptance_baseline_lock"
    training_config = Path(f"JIT/configs/pi_unified_iter{next_k}_auto_core75_natural10_{args.tag}.json")
    run_id = f"pi_{next_k}_tube{next_k}_auto_core75_natural10_10009600_seed821101_{args.tag}"
    run_dir = Path("JIT/runs/pi_unified") / run_id
    frozen = Path(f"JIT/runs/frozen_unified/pi_{next_k}_auto_10009600_{args.tag}")
    gate = work_root / f"pi_{k}_to_pi_{next_k}_gate"
    selected_next = Path(f"JIT/runs/iteration_selection/pi_{next_k}_auto_{args.tag}")

    def req(path, kind="file"):
        return {"path": str(path), "kind": kind}

    def assertion(pointer, op, value):
        return {"pointer": pointer, "op": op, "value": value}

    stages = [
        {
            "name": "prepare_frontier_plan",
            "command": [PYTHON, "JIT/cli/run_iterative_frontier_protocol.py", "prepare-plan", "--selected-policy", str(args.selected_policy), "--source-tube", str(args.source_tube), "--output", str(plan)],
            "cwd": str(repo_root),
            "requires": [req(args.selected_policy), req(Path(args.source_tube) / "manifest.json")],
            "completion": {"path": str(plan), "kind": "json", "assertions": [assertion("/status", "eq", "predeclared_before_frontier_outcomes"), assertion("/iteration", "eq", k)], "exports": {}},
        },
        *[
            {
                "name": f"frontier_{role_name}",
                "command": [PYTHON, "JIT/cli/run_iterative_frontier_protocol.py", "run-role", "--plan", str(plan), "--role", role_name, "--output-dir", str(root)],
                "cwd": str(repo_root),
                "requires": [req(plan)],
                "completion": {"path": str(root / "role_manifest.json"), "kind": "json", "assertions": [assertion("/status", "eq", "completed"), assertion("/role", "eq", role_name), assertion("/iteration", "eq", k)], "exports": {}},
            }
            for role_name, root in (("train", train), ("calibration", calibration), ("acceptance", acceptance))
        ],
        {
            "name": "fit_and_calibrate_Ck",
            "command": [PYTHON, "JIT/cli/fit_iterative_continuation_fields.py", "--train-root", str(train), "--calibration-root", str(calibration), "--output-dir", str(fields)],
            "cwd": str(repo_root),
            "requires": [req(train / "role_manifest.json"), req(calibration / "role_manifest.json")],
            "completion": {"path": str(fields / "summary.json"), "kind": "json", "assertions": [assertion("/status", "eq", "completed_calibrated"), assertion("/iteration", "eq", k), assertion("/next_tube_construction_authorized", "eq", True)], "exports": {}},
        },
        {
            "name": f"build_Tube{next_k}",
            "command": [PYTHON, "JIT/cli/build_iterative_tube.py", "--source-tube", str(args.source_tube), "--train-root", str(train), "--fields-root", str(fields), "--output-dir", str(tube_next)],
            "cwd": str(repo_root),
            "requires": [req(fields / "summary.json"), req(train / "role_manifest.json")],
            "completion": {"path": str(tube_next / "manifest.json"), "kind": "json", "assertions": [assertion("/status", "eq", "completed"), assertion("/iteration", "eq", next_k), assertion("/source_iteration", "eq", k), assertion("/core_retained_count", "eq", len(tube.entries)), assertion("/expansion_count", "gt", 0)], "exports": {}},
        },
        {
            "name": f"smoke_Tube{next_k}",
            "command": [PYTHON, "JIT/cli/smoke_tube_rsi.py", "--up-config", UP_CONFIG, "--down-config", DOWN_CONFIG, "--soft-tube", str(tube_next), "--output-dir", str(smoke), "--samples-per-phase", "8"],
            "cwd": str(repo_root),
            "requires": [req(tube_next / "manifest.json")],
            "completion": {"path": str(smoke / "report.json"), "kind": "json", "assertions": [assertion("/status", "eq", "completed"), assertion("/tube_rsi_smoke", "eq", "GO"), assertion("/test_data_used", "eq", False)], "exports": {}},
        },
        {
            "name": "audit_role_isolation",
            "command": [PYTHON, "JIT/cli/audit_iterative_role_isolation.py", "--train-root", str(train), "--calibration-root", str(calibration), "--acceptance-root", str(acceptance), "--target-tube", str(tube_next), "--output", str(isolation)],
            "cwd": str(repo_root),
            "requires": [req(tube_next / "manifest.json"), req(train / "role_manifest.json"), req(calibration / "role_manifest.json"), req(acceptance / "role_manifest.json")],
            "completion": {"path": str(isolation), "kind": "json", "assertions": [assertion("/status", "eq", "independent"), assertion("/acceptance_target_tube_overlap_count", "eq", 0), assertion("/calibration_target_tube_overlap_count", "eq", 0)], "exports": {}},
        },
        {
            "name": "lock_pi_k_acceptance_baseline",
            "command": [PYTHON, "JIT/cli/run_iterative_acceptance_gate.py", "lock-baseline", "--selected-policy", str(args.selected_policy), "--source-tube", str(args.source_tube), "--acceptance-root", str(acceptance), "--output-dir", str(baseline_lock)],
            "cwd": str(repo_root),
            "requires": [req(isolation), req(acceptance / "role_manifest.json")],
            "completion": {"path": str(baseline_lock / "baseline_lock.json"), "kind": "json", "assertions": [assertion("/status", "eq", "locked_before_candidate_training"), assertion("/source_iteration", "eq", k), assertion("/boundary_negative_parent_group_count", "ge", 2)], "exports": {}},
        },
        {
            "name": f"prepare_pi{next_k}_training",
            "command": [PYTHON, "JIT/cli/prepare_iterative_unified_training.py", "--soft-tube", str(tube_next), "--tube-rsi-smoke-report", str(smoke / "report.json"), "--output-config", str(training_config), "--run-id", run_id],
            "cwd": str(repo_root),
            "requires": [req(baseline_lock / "baseline_lock.json"), req(smoke / "report.json")],
            "completion": {"path": str(training_config) + ".prepared.json", "kind": "json", "assertions": [assertion("/status", "eq", "completed"), assertion("/iteration", "eq", next_k)], "exports": {}},
        },
        {
            "name": f"train_pi{next_k}",
            "command": [PYTHON, "JIT/cli/train_unified.py", "--config", str(training_config), "--run-id", run_id],
            "cwd": str(repo_root),
            "requires": [req(training_config), req(baseline_lock / "baseline_lock.json")],
            "completion": {"path": str(run_dir / "formal_report.json"), "kind": "json", "assertions": [assertion("/status", "eq", "completed"), assertion("/completed_training_transitions", "eq", 10009600), assertion("/checkpoint_restored", "eq", True), assertion("/test_data_used", "eq", False)], "exports": {}},
        },
        {
            "name": f"freeze_pi{next_k}",
            "command": [PYTHON, "JIT/cli/freeze_unified_policy.py", "--config", str(training_config), "--checkpoint", str(run_dir / "checkpoints/transition_10009600"), "--iteration", str(next_k), "--output-dir", str(frozen), "--formal-report", str(run_dir / "formal_report.json")],
            "cwd": str(repo_root),
            "requires": [req(run_dir / "formal_report.json"), req(run_dir / "checkpoints/transition_10009600", "directory")],
            "completion": {"path": str(frozen / "frozen_unified_policy.json"), "kind": "json", "assertions": [assertion("/status", "eq", "frozen"), assertion("/policy/iteration", "eq", next_k), assertion("/policy/name", "eq", f"pi_{next_k}")], "exports": {}},
        },
        {
            "name": f"gate_pi{k}_to_pi{next_k}",
            "command": [PYTHON, "JIT/cli/run_iterative_acceptance_gate.py", "run-candidate", "--baseline-lock", str(baseline_lock / "baseline_lock.json"), "--candidate-frozen-policy", str(frozen / "frozen_unified_policy.json"), "--output-dir", str(gate)],
            "cwd": str(repo_root),
            "requires": [req(frozen / "frozen_unified_policy.json"), req(baseline_lock / "baseline_lock.json")],
            "completion": {"path": str(gate / "summary.json"), "kind": "json", "assertions": [assertion("/status", "eq", "completed"), assertion("/iteration_accepted", "eq", True), assertion("/core_gate/regression_count", "eq", 0), assertion("/boundary_gate/baseline_reproduction_failure_count", "eq", 0), assertion("/boundary_gate/candidate_success_parent_group_count", "ge", 2)], "exports": {}},
        },
        {
            "name": f"select_pi{next_k}",
            "command": [PYTHON, "JIT/cli/select_iteration_policy.py", "--frozen-policy", str(frozen / "frozen_unified_policy.json"), "--gate-summary", str(gate / "summary.json"), "--output-dir", str(selected_next)],
            "cwd": str(repo_root),
            "requires": [req(gate / "summary.json"), req(frozen / "frozen_unified_policy.json")],
            "completion": {"path": str(selected_next / "selected_policy.json"), "kind": "json", "assertions": [assertion("/status", "eq", "selected"), assertion("/iteration", "eq", next_k), assertion("/formal_acceptance_claim", "eq", True)], "exports": {}},
        },
    ]

    config = {
        "schema": "jit_iteration_workflow_v1",
        "workflow_name": f"automatic_pi_{k}_to_pi_{next_k}_{args.tag}",
        "state_dir": str(work_root / "workflow_state"),
        "variables": {},
        "environment": {
            "PYTHONPATH": str(repo_root / "JIT/src"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        "stages": stages,
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "jit_iterative_envelope_workflow_preparation_v1",
        "status": "completed",
        "source_iteration": k,
        "candidate_iteration": next_k,
        "selected_policy": str(args.selected_policy),
        "source_tube": str(args.source_tube),
        "workflow_config": str(args.config_out),
        "work_root": str(work_root),
        "target_tube": str(tube_next),
        "candidate_training_config": str(training_config),
        "candidate_run_id": run_id,
        "candidate_frozen_policy": str(frozen / "frozen_unified_policy.json"),
        "candidate_gate": str(gate / "summary.json"),
        "candidate_selected_policy": str(selected_next / "selected_policy.json"),
        "automatic_repair_on_gate_failure": False,
        "test_data_used": False,
    }
    sidecar = args.config_out.with_suffix(args.config_out.suffix + ".prepared.json")
    sidecar.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
