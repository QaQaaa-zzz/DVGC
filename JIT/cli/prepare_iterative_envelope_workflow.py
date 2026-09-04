#!/usr/bin/env python3
"""Generate one resumable causal trajectory-centered pi_k -> pi_(k+1) workflow.

The active JIT method separates three things that older workflows mixed:

1. raw Soft-Tube replay support;
2. ground-connected forward reachability;
3. continuation viability after a reached state.

A locked nominal centerline is supplied as a method reference and is NOT
recomputed each iteration. Frontier roles probe every 0.1 m centerline slice by
rolling from the natural ground reset. RSI is used only after a candidate has
already been reached by env.step. Candidate policy training is authorized only
when the causal TRAIN evidence adds new resolution-aware Jump-Capability cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.analysis.nominal_jump_centerline import load_nominal_jump_centerline
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
    parser.add_argument(
        "--nominal-centerline",
        type=Path,
        required=True,
        help="locked jit_nominal_jump_centerline_v2 method reference; reused across iterations",
    )
    parser.add_argument(
        "--source-causal-summary",
        type=Path,
        help="optional previous causal capability summary; omitted for the first causal round",
    )
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
    centerline = load_nominal_jump_centerline(args.nominal_centerline)
    if centerline.get("centerline_recomputed_each_iteration") is not False:
        raise ValueError("workflow requires one locked non-drifting nominal centerline")
    if centerline.get("natural_start_connected") is not True:
        raise ValueError("workflow centerline is not ground-connected")
    if args.source_causal_summary is not None:
        source_causal = read(args.source_causal_summary)
        if source_causal.get("schema") != "jit_causal_jump_capability_evidence_v1":
            raise ValueError("source causal capability summary schema drift")
        if source_causal.get("ground_reachability_verified") is not True:
            raise ValueError("source causal capability summary lacks reachability proof")

    next_k = k + 1
    repo_root = args.repo_root.resolve()
    work_root = (
        args.work_root
        if args.work_root is not None
        else Path(f"JIT/runs/iteration_auto/pi_{k}_to_pi_{next_k}_{args.tag}")
    )
    source_geometry = work_root / "source_control_tube_geometry"
    source_jump_view = work_root / "source_semantic_jump_view"
    raw_plan = work_root / "frontier_plan_unrevised.json"
    plan = work_root / "frontier_plan_causal.json"
    train = work_root / "frontier_train"
    calibration = work_root / "frontier_calibration"
    acceptance = work_root / "frontier_acceptance"
    causal_capability = work_root / "causal_jump_capability"
    fields = work_root / f"continuation_C{k}"
    tube_next = Path(f"JIT/runs/soft_tube/soft_tube_iter{next_k}_pi{k}_conditioned_{args.tag}")
    target_geometry = work_root / f"tube{next_k}_control_tube_geometry"
    target_jump_view = work_root / f"tube{next_k}_semantic_jump_view"
    smoke = work_root / f"tube{next_k}_smoke"
    isolation = work_root / "role_isolation.json"
    baseline_lock = work_root / "acceptance_baseline_lock"
    training_config = Path(f"JIT/configs/pi_unified_iter{next_k}_auto_core75_natural10_{args.tag}.json")
    run_id = f"pi_{next_k}_tube{next_k}_auto_core75_natural10_10009600_seed821101_{args.tag}"
    run_dir = Path("JIT/runs/pi_unified") / run_id
    frozen = Path(f"JIT/runs/frozen_unified/pi_{next_k}_auto_10009600_{args.tag}")
    gate = work_root / f"pi_{k}_to_pi_{next_k}_gate"
    capability_decision = work_root / f"pi_{k}_to_pi_{next_k}_capability_progression.json"
    selected_next = Path(f"JIT/runs/iteration_selection/pi_{next_k}_auto_{args.tag}")

    def req(path, kind="file"):
        return {"path": str(path), "kind": kind}

    def assertion(pointer, op, value):
        return {"pointer": pointer, "op": op, "value": value}

    causal_command = [
        PYTHON,
        "JIT/cli/analyze_causal_jump_capability.py",
        "--nominal-centerline",
        str(args.nominal_centerline),
        "--train-root",
        str(train),
        "--calibration-root",
        str(calibration),
        "--acceptance-root",
        str(acceptance),
        "--output-dir",
        str(causal_capability),
    ]
    if args.source_causal_summary is not None:
        causal_command.extend(["--source-causal-summary", str(args.source_causal_summary)])

    stages = [
        {
            "name": "analyze_source_control_tube_geometry",
            "command": [
                PYTHON,
                "JIT/cli/analyze_capability_tube.py",
                "--tube",
                str(args.source_tube),
                "--output-dir",
                str(source_geometry),
            ],
            "cwd": str(repo_root),
            "requires": [req(Path(args.source_tube) / "manifest.json"), req(args.nominal_centerline)],
            "completion": {
                "path": str(source_geometry / "summary.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/resolution_contract/x_slice_width_m", "eq", 0.1),
                ],
                "exports": {},
            },
        },
        {
            "name": "build_source_semantic_jump_view",
            "command": [
                PYTHON,
                "JIT/cli/analyze_jump_tube_view.py",
                "--capability-geometry-summary",
                str(source_geometry / "summary.json"),
                "--nominal-centerline",
                str(args.nominal_centerline),
                "--output-dir",
                str(source_jump_view),
            ],
            "cwd": str(repo_root),
            "requires": [req(source_geometry / "summary.json"), req(args.nominal_centerline)],
            "completion": {
                "path": str(source_jump_view / "summary.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/semantic_filter/post_landing_recovery_included", "eq", False),
                ],
                "exports": {},
            },
        },
        {
            "name": "prepare_unrevised_frontier_plan",
            "command": [
                PYTHON,
                "JIT/cli/run_iterative_frontier_protocol.py",
                "prepare-plan",
                "--selected-policy",
                str(args.selected_policy),
                "--source-tube",
                str(args.source_tube),
                "--output",
                str(raw_plan),
            ],
            "cwd": str(repo_root),
            "requires": [req(args.selected_policy), req(Path(args.source_tube) / "manifest.json")],
            "completion": {
                "path": str(raw_plan),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "predeclared_before_frontier_outcomes"),
                    assertion("/iteration", "eq", k),
                ],
                "exports": {},
            },
        },
        {
            "name": "revise_frontier_plan_causal_trajectory_centered",
            "command": [
                PYTHON,
                "JIT/cli/prepare_resolution_aware_frontier_plan.py",
                "--source-plan",
                str(raw_plan),
                "--source-tube",
                str(args.source_tube),
                "--capability-geometry-summary",
                str(source_geometry / "summary.json"),
                "--nominal-centerline",
                str(args.nominal_centerline),
                "--output",
                str(plan),
            ],
            "cwd": str(repo_root),
            "requires": [req(raw_plan), req(source_geometry / "summary.json"), req(args.nominal_centerline)],
            "completion": {
                "path": str(plan),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "predeclared_before_frontier_outcomes"),
                    assertion("/iteration", "eq", k),
                    assertion("/protocol_revision/name", "eq", "causal_trajectory_centered_frontier_v2"),
                    assertion("/jump_tube_contract/x_step_m", "eq", 0.1),
                    assertion("/jump_tube_contract/all_centerline_slices_probed", "eq", True),
                    assertion("/jump_tube_contract/source_tube_states_used_as_physical_resets", "eq", False),
                    assertion("/jump_tube_contract/rsi_may_establish_forward_reachability", "eq", False),
                ],
                "exports": {},
            },
        },
        *[
            {
                "name": f"frontier_{role_name}",
                "command": [
                    PYTHON,
                    "JIT/cli/run_causal_jump_frontier_role.py",
                    "--plan",
                    str(plan),
                    "--role",
                    role_name,
                    "--output-dir",
                    str(root),
                ],
                "cwd": str(repo_root),
                "requires": [req(plan)],
                "completion": {
                    "path": str(root / "role_manifest.json"),
                    "kind": "json",
                    "assertions": [
                        assertion("/status", "eq", "completed"),
                        assertion("/role", "eq", role_name),
                        assertion("/iteration", "eq", k),
                        assertion("/acquisition_mode", "eq", "ground_connected_causal_rollout_v1"),
                        assertion("/ground_reachability_proven", "eq", True),
                        assertion("/rsi_used_for_reachability", "eq", False),
                    ],
                    "exports": {},
                },
            }
            for role_name, root in (
                ("train", train),
                ("calibration", calibration),
                ("acceptance", acceptance),
            )
        ],
        {
            "name": "analyze_causal_jump_capability",
            "command": causal_command,
            "cwd": str(repo_root),
            "requires": [
                req(train / "role_manifest.json"),
                req(calibration / "role_manifest.json"),
                req(acceptance / "role_manifest.json"),
                req(args.nominal_centerline),
            ],
            "completion": {
                "path": str(causal_capability / "summary.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/ground_reachability_verified", "eq", True),
                    assertion("/rsi_used_to_establish_reachability", "eq", False),
                    assertion(
                        "/curriculum_capability/new_train_root_geometry_cell_count_vs_source_or_centerline",
                        "gt",
                        0,
                    ),
                ],
                "exports": {},
            },
        },
        {
            "name": "fit_and_calibrate_Ck",
            "command": [
                PYTHON,
                "JIT/cli/fit_iterative_continuation_fields.py",
                "--train-root",
                str(train),
                "--calibration-root",
                str(calibration),
                "--output-dir",
                str(fields),
            ],
            "cwd": str(repo_root),
            "requires": [req(causal_capability / "summary.json"), req(train / "role_manifest.json"), req(calibration / "role_manifest.json")],
            "completion": {
                "path": str(fields / "summary.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed_calibrated"),
                    assertion("/iteration", "eq", k),
                    assertion("/next_tube_construction_authorized", "eq", True),
                ],
                "exports": {},
            },
        },
        {
            "name": f"build_Tube{next_k}",
            "command": [
                PYTHON,
                "JIT/cli/build_iterative_tube.py",
                "--source-tube",
                str(args.source_tube),
                "--train-root",
                str(train),
                "--fields-root",
                str(fields),
                "--output-dir",
                str(tube_next),
            ],
            "cwd": str(repo_root),
            "requires": [req(fields / "summary.json"), req(train / "role_manifest.json"), req(causal_capability / "summary.json")],
            "completion": {
                "path": str(tube_next / "manifest.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/iteration", "eq", next_k),
                    assertion("/source_iteration", "eq", k),
                    assertion("/core_retained_count", "eq", len(tube.entries)),
                    assertion("/expansion_count", "gt", 0),
                ],
                "exports": {},
            },
        },
        {
            "name": f"analyze_Tube{next_k}_control_geometry",
            "command": [
                PYTHON,
                "JIT/cli/analyze_capability_tube.py",
                "--tube",
                str(tube_next),
                "--source-tube",
                str(args.source_tube),
                "--output-dir",
                str(target_geometry),
            ],
            "cwd": str(repo_root),
            "requires": [req(tube_next / "manifest.json")],
            "completion": {
                "path": str(target_geometry / "summary.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/resolution_contract/x_slice_width_m", "eq", 0.1),
                ],
                "exports": {},
            },
        },
        {
            "name": f"build_Tube{next_k}_semantic_jump_view",
            "command": [
                PYTHON,
                "JIT/cli/analyze_jump_tube_view.py",
                "--capability-geometry-summary",
                str(target_geometry / "summary.json"),
                "--source-capability-geometry-summary",
                str(source_geometry / "summary.json"),
                "--nominal-centerline",
                str(args.nominal_centerline),
                "--output-dir",
                str(target_jump_view),
            ],
            "cwd": str(repo_root),
            "requires": [req(target_geometry / "summary.json"), req(source_geometry / "summary.json")],
            "completion": {
                "path": str(target_jump_view / "summary.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/semantic_filter/post_landing_recovery_included", "eq", False),
                ],
                "exports": {},
            },
        },
        {
            "name": f"smoke_Tube{next_k}",
            "command": [
                PYTHON,
                "JIT/cli/smoke_tube_rsi.py",
                "--up-config",
                UP_CONFIG,
                "--down-config",
                DOWN_CONFIG,
                "--soft-tube",
                str(tube_next),
                "--output-dir",
                str(smoke),
                "--samples-per-phase",
                "8",
            ],
            "cwd": str(repo_root),
            "requires": [req(tube_next / "manifest.json"), req(causal_capability / "summary.json")],
            "completion": {
                "path": str(smoke / "report.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/tube_rsi_smoke", "eq", "GO"),
                    assertion("/test_data_used", "eq", False),
                ],
                "exports": {},
            },
        },
        {
            "name": "audit_role_isolation",
            "command": [
                PYTHON,
                "JIT/cli/audit_iterative_role_isolation.py",
                "--train-root",
                str(train),
                "--calibration-root",
                str(calibration),
                "--acceptance-root",
                str(acceptance),
                "--target-tube",
                str(tube_next),
                "--output",
                str(isolation),
            ],
            "cwd": str(repo_root),
            "requires": [req(tube_next / "manifest.json"), req(train / "role_manifest.json"), req(calibration / "role_manifest.json"), req(acceptance / "role_manifest.json")],
            "completion": {
                "path": str(isolation),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "independent"),
                    assertion("/acceptance_target_tube_overlap_count", "eq", 0),
                    assertion("/calibration_target_tube_overlap_count", "eq", 0),
                ],
                "exports": {},
            },
        },
        {
            "name": "lock_pi_k_acceptance_baseline",
            "command": [
                PYTHON,
                "JIT/cli/run_iterative_acceptance_gate.py",
                "lock-baseline",
                "--selected-policy",
                str(args.selected_policy),
                "--source-tube",
                str(args.source_tube),
                "--acceptance-root",
                str(acceptance),
                "--output-dir",
                str(baseline_lock),
            ],
            "cwd": str(repo_root),
            "requires": [req(isolation), req(acceptance / "role_manifest.json")],
            "completion": {
                "path": str(baseline_lock / "baseline_lock.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "locked_before_candidate_training"),
                    assertion("/source_iteration", "eq", k),
                    assertion("/boundary_negative_parent_group_count", "ge", 2),
                ],
                "exports": {},
            },
        },
        {
            "name": f"prepare_pi{next_k}_training",
            "command": [
                PYTHON,
                "JIT/cli/prepare_iterative_unified_training.py",
                "--soft-tube",
                str(tube_next),
                "--tube-rsi-smoke-report",
                str(smoke / "report.json"),
                "--output-config",
                str(training_config),
                "--run-id",
                run_id,
            ],
            "cwd": str(repo_root),
            "requires": [req(baseline_lock / "baseline_lock.json"), req(smoke / "report.json"), req(causal_capability / "summary.json")],
            "completion": {
                "path": str(training_config) + ".prepared.json",
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/iteration", "eq", next_k),
                ],
                "exports": {},
            },
        },
        {
            "name": f"train_pi{next_k}",
            "command": [PYTHON, "JIT/cli/train_unified.py", "--config", str(training_config), "--run-id", run_id],
            "cwd": str(repo_root),
            "requires": [req(training_config), req(baseline_lock / "baseline_lock.json")],
            "completion": {
                "path": str(run_dir / "formal_report.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/completed_training_transitions", "eq", 10009600),
                    assertion("/checkpoint_restored", "eq", True),
                    assertion("/test_data_used", "eq", False),
                ],
                "exports": {},
            },
        },
        {
            "name": f"freeze_pi{next_k}",
            "command": [
                PYTHON,
                "JIT/cli/freeze_unified_policy.py",
                "--config",
                str(training_config),
                "--checkpoint",
                str(run_dir / "checkpoints/transition_10009600"),
                "--iteration",
                str(next_k),
                "--output-dir",
                str(frozen),
                "--formal-report",
                str(run_dir / "formal_report.json"),
            ],
            "cwd": str(repo_root),
            "requires": [req(run_dir / "formal_report.json"), req(run_dir / "checkpoints/transition_10009600", "directory")],
            "completion": {
                "path": str(frozen / "frozen_unified_policy.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "frozen"),
                    assertion("/policy/iteration", "eq", next_k),
                    assertion("/policy/name", "eq", f"pi_{next_k}"),
                ],
                "exports": {},
            },
        },
        {
            "name": f"evaluate_pi{k}_to_pi{next_k}_locked_panel",
            "command": [
                PYTHON,
                "JIT/cli/run_iterative_acceptance_gate.py",
                "run-candidate",
                "--baseline-lock",
                str(baseline_lock / "baseline_lock.json"),
                "--candidate-frozen-policy",
                str(frozen / "frozen_unified_policy.json"),
                "--output-dir",
                str(gate),
            ],
            "cwd": str(repo_root),
            "requires": [req(frozen / "frozen_unified_policy.json"), req(baseline_lock / "baseline_lock.json")],
            "completion": {
                "path": str(gate / "summary.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/boundary_gate/baseline_reproduction_failure_count", "eq", 0),
                ],
                "exports": {},
            },
        },
        {
            "name": f"analyze_pi{next_k}_capability_progression",
            "command": [
                PYTHON,
                "JIT/cli/analyze_capability_progression.py",
                "--gate-summary",
                str(gate / "summary.json"),
                "--output",
                str(capability_decision),
            ],
            "cwd": str(repo_root),
            "requires": [req(gate / "summary.json"), req(causal_capability / "summary.json")],
            "completion": {
                "path": str(capability_decision),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "completed"),
                    assertion("/retrospective_analysis", "eq", False),
                    assertion("/empirical_envelope_expansion_observed", "eq", True),
                    assertion("/candidate_policy_authority_eligible", "eq", True),
                ],
                "exports": {},
            },
        },
        {
            "name": f"select_pi{next_k}",
            "command": [
                PYTHON,
                "JIT/cli/select_iteration_policy.py",
                "--frozen-policy",
                str(frozen / "frozen_unified_policy.json"),
                "--gate-summary",
                str(gate / "summary.json"),
                "--capability-decision",
                str(capability_decision),
                "--output-dir",
                str(selected_next),
            ],
            "cwd": str(repo_root),
            "requires": [req(gate / "summary.json"), req(capability_decision), req(frozen / "frozen_unified_policy.json")],
            "completion": {
                "path": str(selected_next / "selected_policy.json"),
                "kind": "json",
                "assertions": [
                    assertion("/status", "eq", "selected"),
                    assertion("/iteration", "eq", next_k),
                    assertion("/selection_semantics", "eq", "prospective_capability_progression_v1"),
                    assertion("/formal_acceptance_claim", "eq", True),
                ],
                "exports": {},
            },
        },
    ]

    config = {
        "schema": "jit_iteration_workflow_v1",
        "workflow_name": f"causal_trajectory_centered_pi_{k}_to_pi_{next_k}_{args.tag}",
        "state_dir": str(work_root / "workflow_state"),
        "variables": {},
        "environment": {
            "PYTHONPATH": str(repo_root / "JIT/src"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        "method_contract": {
            "causal_trajectory_centered_jump_tube": True,
            "nominal_centerline": str(args.nominal_centerline),
            "nominal_centerline_sha256": str(centerline["centerline_sha256"]),
            "centerline_recomputed_each_iteration": False,
            "x_step_m": 0.1,
            "source_tube_is_forward_reachability_proof": False,
            "source_tube_states_used_as_frontier_resets": False,
            "forward_reachability": "natural_start_connected_env_step_only",
            "rsi_establishes_forward_reachability": False,
            "rsi_role": "continuation_evaluation_after_reachability",
            "post_landing_recovery_frontier_eligible": False,
            "causal_train_cell_growth_required_before_training": True,
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
        "nominal_centerline": str(args.nominal_centerline),
        "source_causal_summary": str(args.source_causal_summary) if args.source_causal_summary else None,
        "source_control_tube_geometry": str(source_geometry / "summary.json"),
        "source_semantic_jump_view": str(source_jump_view / "summary.json"),
        "causal_frontier_plan": str(plan),
        "causal_capability_summary": str(causal_capability / "summary.json"),
        "workflow_config": str(args.config_out),
        "work_root": str(work_root),
        "target_tube": str(tube_next),
        "target_control_tube_geometry": str(target_geometry / "summary.json"),
        "target_semantic_jump_view": str(target_jump_view / "summary.json"),
        "candidate_training_config": str(training_config),
        "candidate_run_id": run_id,
        "candidate_frozen_policy": str(frozen / "frozen_unified_policy.json"),
        "candidate_gate": str(gate / "summary.json"),
        "candidate_capability_decision": str(capability_decision),
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
