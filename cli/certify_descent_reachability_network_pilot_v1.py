"""Branch-certify the preregistered Descent reachability-network pilot.

This is construction acquisition, not independent Tube audit.  The selected
state identities come exclusively from the immutable ranking artifact and are
never changed after observing rollout outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.run_backward_descent_nominal_pilot import (
    C_L,
    EXPECTED,
    PI_D,
    PI_L,
    PERTURBATIONS,
    _batched,
    _micro_states,
    _outcome,
)
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import active_prefix_exact, make_descent_landing_rollout
from dvgc.backward_tube import p0_decision, p1_decision
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.policy import load_bundle
from dvgc.runtime import save_json
from cli.train_descent_reachability_network_v3 import final_branch_success


DEFAULT_RANKING = Path("runs/descent_reachability_network_v3/final_semantics_recalibrated_20260729/ranked_proposals.json")
DEFAULT_RUN = Path("runs/descent_reachability_network_v3/final_semantics_ranked_pilot_remaining_20260729")
CONFIG = Path("configs/default.json")
HORIZON = 200


def final_safety_decision(branches: list[dict]) -> dict:
    """Construction P1: 3/4 legal Final outcomes; Chain is not a prerequisite."""
    if len(branches) != 4:
        return {"pass": False, "successes": 0, "branches": len(branches),
                "reasons": ["requires_exactly_four_branches"]}
    successes = sum(final_branch_success(row) for row in branches)
    reasons = [] if successes >= 3 else ["fewer_than_three_final_recovery_successes"]
    return {"pass": not reasons, "successes": successes, "branches": len(branches),
            "reasons": reasons}


def _atomic_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _verify_runtime() -> dict:
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    current = source_fingerprint(Path.cwd())
    checks = {
        "runtime_gate_pass": gate.get("status") == "PASS",
        "runtime_fingerprint_current": gate.get("source_fingerprint") == current,
        "xml_current": file_sha256(load_config(CONFIG).xml_path) == EXPECTED["xml"],
        "C_L_current": file_sha256(C_L) == EXPECTED["C_L"],
        "pi_D_current": file_sha256(PI_D / "params.pkl") == EXPECTED["pi_D"],
        "pi_L_current": file_sha256(PI_L / "params.pkl") == EXPECTED["pi_L"],
    }
    if not all(checks.values()):
        raise SystemExit(f"runtime/provenance gate failed: {checks}")
    return checks


def _setup():
    checks = _verify_runtime()
    dparams, policy_cfg, _ = load_bundle(PI_D, verify_files=True)
    lparams, _, _ = load_bundle(PI_L, verify_files=True)
    cfg = load_config(CONFIG, {**policy_cfg, "episode_length": 750, "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "expert_chain_termination": False, "training_stage": "flight"})
    entry = SnapshotBank.load(C_L)
    env = __import__("dvgc.env", fromlist=["OrangeBikeDVGC"]).OrangeBikeDVGC(
        cfg, snapshot_bank=SnapshotBank(), cert_bank=entry)
    rollout = make_descent_landing_rollout(env, dparams, lparams, horizon=HORIZON,
                                            residual_ticks=8, ticks_per_knot=1)
    return checks, env, rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", default=str(DEFAULT_RANKING))
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--seed", type=int, default=53_000_000)
    args = parser.parse_args()
    ranking_path, root = Path(args.ranking), Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    if "independent_audit" in str(ranking_path).lower():
        raise SystemExit("independent audit ranking input is forbidden")
    ranking = json.loads(ranking_path.read_text())
    if ranking.get("status") != "PASS" or ranking.get("formal_tube_or_matcher") is not False:
        raise SystemExit("ranking is not an eligible proposal-only artifact")
    selected = ranking["selected_pilot"]
    if not 1 <= len(selected) <= 4 or len({row["candidate_id"] for row in selected}) != len(selected):
        raise SystemExit("pilot must contain one to four parent-disjoint states")
    root.mkdir(parents=True)
    save_json(root / "cost_estimate.json", {
        "status": "BOUNDED", "states": len(selected), "P0_exact_replays_per_state": 2,
        "P1_branches_per_P0_state": 4, "maximum_rollouts": 6 * len(selected),
        "horizon": HORIZON, "estimated_wall_hours_upper_bound": 1.0,
        "PPO": False, "independent_audit": False,
    })
    save_json(root / "manifest.json", {
        "status": "FROZEN", "ranking": str(ranking_path),
        "ranking_sha256": file_sha256(ranking_path), "seed": args.seed,
        "selected": selected, "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
    })

    checks, env, rollout = _setup()
    banks: dict[str, list] = {}
    rows, admitted = [], []
    for position, proposal in enumerate(selected):
        artifact = proposal["source_artifact"]
        if "independent_audit" in artifact.lower():
            raise RuntimeError("independent audit source is forbidden")
        if artifact not in banks:
            banks[artifact] = pickle.loads(Path(artifact).read_bytes())
        source = banks[artifact][int(proposal["source_index"])]
        if source["physical_state_hash"] != proposal["physical_state_hash"]:
            raise RuntimeError("ranked proposal identity mismatch")
        if source["candidate_id"] != proposal["candidate_id"]:
            raise RuntimeError("ranked parent identity mismatch")
        record = source["snapshot_v4"]
        commands = jnp.asarray(np.asarray(source["controller_suffix"], np.float32)[None])
        seed = args.seed + position * 10_000
        state = _batched(env, record, 1, seed)
        first = jax.device_get(rollout(state, commands, jax.random.PRNGKey(seed)))
        second = jax.device_get(rollout(state, commands, jax.random.PRNGKey(seed)))
        exact, mismatch = active_prefix_exact(first, second)
        repeats = [_outcome(first, 0, exact, mismatch), _outcome(second, 0, exact, mismatch)]
        p0 = p0_decision(repeats)
        branches = []
        p1 = {"pass": False, "reasons": ["p0_not_passed"], "successes": 0, "branches": 0}
        chain_p1 = {"pass": False, "reasons": ["p0_not_passed"], "successes": 0, "branches": 0}
        if p0["pass"]:
            micro = _micro_states(env, record, seed + 1000)
            branch_commands = jnp.repeat(commands, 4, axis=0)
            raw = jax.device_get(rollout(micro, branch_commands, jax.random.PRNGKey(seed + 2000)))
            branches = [_outcome(raw, index) | {"perturbation_vx_vz": PERTURBATIONS[index].tolist()}
                        for index in range(4)]
            chain_p1 = p1_decision(p0, branches, repeats[0]["failure_type"])
            p1 = final_safety_decision(branches)
        result = {
            "proposal": {key: proposal[key] for key in proposal if key != "ranking_only"},
            "P0": p0, "final_safety_P1": p1, "legacy_chain_P1": chain_p1,
            "repeats": repeats, "branches": branches,
            "construction_label": "safe" if p1["pass"] else (
                "boundary" if p0["pass"] or p1.get("successes", 0) else "dead"),
        }
        rows.append(result)
        if p1["pass"]:
            admitted.append({
                "snapshot_v4": record, "physical_state_hash": source["physical_state_hash"],
                "candidate_id": source["candidate_id"], "source_node_id": source["source_node_id"],
                "controller_suffix": source["controller_suffix"], "P0": p0,
                "final_safety_P1": p1, "legacy_chain_P1": chain_p1,
                "artifact_role": "proposal_support_bank", "formal_tube_or_jel": False,
            })
        save_json(root / "certification.partial.json", {"completed": len(rows), "rows": rows})

    class_counts = Counter(row["construction_label"] for row in rows)
    status = "PASS" if admitted else "FAIL"
    report = {
        "status": status,
        "artifact_role": "reachability_ranked_construction_certification_pilot",
        "formal_tube_or_matcher": False,
        "states": len(rows), "unique_parents": len({row["proposal"]["candidate_id"] for row in rows}),
        "P0": sum(row["P0"]["pass"] for row in rows),
        "final_safety_P1": sum(row["final_safety_P1"]["pass"] for row in rows),
        "legacy_chain_P1": sum(row["legacy_chain_P1"]["pass"] for row in rows),
        "final_branch_successes": sum(
            final_branch_success(branch) for row in rows for branch in row["branches"]),
        "chain_branch_successes": sum(
            bool(branch.get("downstream_entry") and final_branch_success(branch))
            for row in rows for branch in row["branches"]),
        "class_counts": dict(class_counts),
        "termination_reasons": dict(Counter(
            branch["failure_type"] for row in rows for branch in row["branches"]
            if not branch.get("final_recovery", False)
        )),
        "runtime_provenance": checks,
        "rows": rows,
        "PPO_authorization": False,
        "next": "expand_ranked_construction_certification" if status == "PASS" else "network_prospective_transfer_failed",
    }
    _atomic_pickle(root / "p1_proposal_support.pkl", admitted)
    save_json(root / "DESCENT_REACHABILITY_RANKED_PILOT_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps({key: report[key] for key in (
        "status", "states", "unique_parents", "P0", "final_safety_P1", "legacy_chain_P1",
        "final_branch_successes", "chain_branch_successes", "class_counts",
        "termination_reasons", "PPO_authorization", "next",
    )}, indent=2))


if __name__ == "__main__":
    main()
