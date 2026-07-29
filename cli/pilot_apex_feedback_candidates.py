"""Label fixed Apex candidates with the frozen bounded feedback controller."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import mujoco
import numpy as np

from cli.discover_apex_feedback_bridge import (
    TERMINAL_FEATURES, TERMINAL_INDEX, _actions, _run_bridge,
)
from cli.runtime_gate import source_fingerprint
from cli.train_stage_reachability_model import parent_key
from dvgc.bank import SnapshotBank
from dvgc.certification import DYNAMICS_VARIANTS
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import build_inference, save_json
from dvgc.stage_reachability import reachability_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--support-bank", required=True)
    parser.add_argument("--terminal-bank", required=True)
    parser.add_argument("--descent-policy", required=True)
    parser.add_argument("--landing-policy", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--downstream-horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10800000)
    args = parser.parse_args()
    if args.branches != 4:
        raise SystemExit("this preregistered pilot requires four branches")
    root = Path(args.output_root)
    outputs = [root / "labels.json", root / "stable_entries.pkl", root / "report.json"]
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing overwrite Apex feedback pilot")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    candidates = SnapshotBank.load(args.candidate_bank)
    support = SnapshotBank.load(args.support_bank)
    support_metadata = dict(support.metadata)
    support_metadata["support_features"] = [row["physical_feature"] for row in support.records]
    terminal = SnapshotBank.load(args.terminal_bank)
    center = np.asarray(terminal.metadata["normalization_center"], float)
    scale = np.asarray(terminal.metadata["normalization_scale"], float)
    target = np.asarray([
        [(row["physical_feature"][TERMINAL_INDEX[name]] - center[i]) / scale[i]
         for i, name in enumerate(TERMINAL_FEATURES)] for row in terminal.records
    ], float)
    dparams, dcfg, _ = load_bundle(args.descent_policy, verify_files=True)
    lparams, lcfg, _ = load_bundle(args.landing_policy, verify_files=True)

    def runtime(variant):
        overrides = {key: value for key, value in variant.items() if key != "id"}
        cfg = load_config(args.config, {**dcfg, **overrides, "training_stage": "flight",
            "use_bank_resets": False, "domain_randomization": False,
            "obs_noise_enable": False, "stage_reachability_objective": ""})
        landing_cfg = load_config(args.config, {**lcfg, **overrides, "training_stage": "landing",
            "use_bank_resets": False, "domain_randomization": False, "obs_noise_enable": False})
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), stage_support_bank=support)
        landing_env = OrangeBikeDVGC(landing_cfg, snapshot_bank=SnapshotBank())
        return (env, jax.jit(env.step), build_inference(env, dparams, deterministic=True),
                landing_env, jax.jit(landing_env.step),
                build_inference(landing_env, lparams, deterministic=True))

    runtimes = {}
    model = mujoco.MjModel.from_xml_path(str(load_config(args.config).xml_path))
    labels, outcomes, stable_records = [], [], []
    for candidate_index, row in enumerate(candidates.records):
        branches = []
        for branch in range(args.branches):
            variant = DYNAMICS_VARIANTS[branch % len(DYNAMICS_VARIANTS)]
            if variant["id"] not in runtimes:
                runtimes[variant["id"]] = runtime(variant)
            seed = args.seed + candidate_index * 100_000 + branch
            result = _run_bridge(
                runtimes[variant["id"]], row, seed, support_metadata, model,
                target, center, scale, args.horizon, args.lookahead,
                args.downstream_horizon,
            )
            stable, formal = result.pop("stable_snapshot"), result.pop("formal_snapshot")
            success = bool(result["stable_physical_descent"])
            branch_row = {
                "branch_index": branch, "seed": seed, "dynamics_variant": variant["id"],
                "success": success, "time_to_next_stage": next((trace["tick"] for trace in result["trace"]
                    if trace["stable_physical_descent"]), None),
                "failure_reason": None if success else result["termination_reason"],
                "formal_descent_support_entry": bool(result["formal_descent_support_entry"]),
                "final_recovery": bool(result["final_landing_recovery"]),
            }
            branches.append(branch_row)
            outcomes.append({"candidate_id": row["id"], "root_parent_id": parent_key(row),
                             **branch_row, "termination_reason": result["termination_reason"]})
            if stable is not None:
                stable.update({"candidate_kind": "stable_physical_descent_active_learning_entry",
                    "trajectory_parent_id": parent_key(row), "upstream_candidate_id": row["id"],
                    "branch": branch, "seed": seed, "safe_claim_allowed": False})
                stable_records.append(stable)
        label = reachability_label(stage="apex", successes=sum(x["success"] for x in branches),
            branches=args.branches, branch_records=branches, controller_bank_exhausted=True)
        label.update({"candidate_id": row["id"], "candidate_kind": row.get("candidate_kind"),
            "trajectory_parent": parent_key(row),
            "controller_bank": ["receding_horizon_bounded_shooting_v1"],
            "local_success_semantics": "four consecutive stable physical Descent ticks",
            "formal_successes": sum(x["formal_descent_support_entry"] for x in branches),
            "final_successes": sum(x["final_recovery"] for x in branches)})
        labels.append(label)
    root.mkdir(parents=True, exist_ok=True)
    stable_bank = SnapshotBank(stable_records, {"artifact_role": "stable_physical_descent_active_learning_entries",
        "safe_claim_allowed": False, "certified_tube": False,
        "candidate_bank_sha256": file_sha256(args.candidate_bank)})
    stable_bank.save(root / "stable_entries.pkl")
    label_payload = {"status": "PASS", "artifact_role": "apex_feedback_construction_labels",
        "safe_claim_allowed": False, "not_a_tube": True, "labels": labels,
        "candidate_bank_sha256": file_sha256(args.candidate_bank), "seed_base": args.seed}
    save_json(root / "labels.json", label_payload)
    report = {"status": "PASS", "artifact_role": "apex_ood_boundary_feedback_pilot",
        "controller": {"type": "receding_horizon_bounded_shooting", "lookahead": args.lookahead,
                       "horizon": args.horizon, "actions": [np.asarray(x).tolist() for x in _actions()]},
        "states": len(candidates.records), "root_parents": len({parent_key(row) for row in candidates.records}),
        "branches": len(outcomes), "local_successes": sum(x["success"] for x in outcomes),
        "successful_states": sum(label["s"] > 0 for label in labels),
        "formal_successes": sum(x["formal_descent_support_entry"] for x in outcomes),
        "final_successes": sum(x["final_recovery"] for x in outcomes),
        "termination_reasons": dict(Counter(x["termination_reason"] for x in outcomes)),
        "candidate_bank_sha256": file_sha256(args.candidate_bank),
        "support_bank_sha256": file_sha256(args.support_bank),
        "terminal_bank_sha256": file_sha256(args.terminal_bank),
        "stable_entries": len(stable_records), "stable_bank_sha256": file_sha256(root / "stable_entries.pkl"),
        "labels_sha256": file_sha256(root / "labels.json"), "outcomes": outcomes}
    save_json(root / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
