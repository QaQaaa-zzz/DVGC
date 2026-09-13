"""CPU coordinator and comparisons for a small, non-training GPU evidence audit.

Each runtime stage gets a fresh process. Results are diagnostic and never admit
Tube rows, select an Actor, alter historical runs, or authorize PPO.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

from .evidence_integrity import canonical_sha256

SCHEMA = "jit_jump_evidence_validation_v1"
FIXED_XML_SHA256 = "0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_hash(value, key):
    if canonical_sha256({k: v for k, v in value.items() if k != key}) != value.get(key):
        raise ValueError(f"{key} identity drift")


def select_sample_indices(frames, per_phase=3):
    """First/middle/last real frames per phase; downstream must be descending."""
    selected, counts = [], {}
    for phase in (0, 1):
        eligible = [i for i, frame in enumerate(frames)
                    if frame["phase_index"] == phase and not frame["done"]
                    and not frame["valid_contact_seen"] and (phase == 0 or frame["vz"] < 0)]
        counts[str(phase)] = len(eligible)
        n = min(len(eligible), per_phase)
        offsets = ([0] if n == 1 else [round(i * (len(eligible) - 1) / (n - 1)) for i in range(n)])
        selected.extend(eligible[offset] for offset in offsets)
    return sorted(set(selected)), counts


def compare_traces(left, right, *, atol, rtol, fields):
    """Compare aligned CPU arrays; missing fields, NaNs and unequal lengths fail."""
    import numpy as np
    differences = {}
    for field in fields:
        worst, first, passed = 0.0, None, True
        for tick, (a, b) in enumerate(zip(left, right)):
            if field not in a or field not in b:
                passed, first = False, tick
                break
            x, y = np.asarray(a[field]), np.asarray(b[field])
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                passed, first = False, tick
                break
            delta = float(np.max(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0
            worst = max(worst, delta)
            discrete = x.dtype.kind in "biu" and y.dtype.kind in "biu"
            equal = np.array_equal(x, y) if discrete else np.allclose(x, y, atol=atol, rtol=rtol)
            if not equal:
                passed = False
                if first is None:
                    first = tick
        if len(left) != len(right):
            passed = False
            if first is None:
                first = min(len(left), len(right))
        differences[field] = {"passed": passed, "max_abs_error": worst, "first_different_frame": first}
    return {"passed": bool(left) and bool(right) and all(d["passed"] for d in differences.values()),
            "left_frame_count": len(left), "right_frame_count": len(right), "fields": differences}


SEMANTIC_LABEL_FIELDS = (
    "candidate_id", "state_sha256", "snapshot_context_sha256", "phase", "phase_index",
    "parent_group_id", "parent_state_sha256", "evaluator_policy_name", "evaluator_actor_sha256",
    "evaluator_payload_sha256", "success_criterion", "label", "continuation_success",
    "outcome_class", "environment_interactions", "terminal_done", "terminal_success",
    "physical_failure", "timeout", "end_code", "end_reason", "apex_seen", "phase_transitioned",
    "valid_contact_seen", "recovery_success", "final_active_phase", "acquisition_protocol_sha256",
)


def compare_label_outputs(serial_dir, merged_dir, catalog_path, *, atol, rtol):
    from .evidence_integrity import read_verified_protocol
    serial, merged = read(Path(serial_dir) / "labels.json"), read(Path(merged_dir) / "labels.json")
    catalog = read(catalog_path)
    protocols = [read_verified_protocol(Path(d) / "protocol.json") for d in (serial_dir, merged_dir)]
    contract_fields = ("policy_actor_sha256", "policy_payload_sha256", "frozen_unified_manifest_sha256",
                       "candidate_catalog_file_sha256", "candidate_catalog_protocol_sha256",
                       "candidate_count", "protocol_seed", "max_ticks_per_candidate", "success_criterion",
                       "acquisition_policy_actor_sha256", "acquisition_policy_payload_sha256")
    for field in contract_fields:
        if field not in protocols[0] or protocols[0][field] != protocols[1].get(field):
            raise ValueError(f"serial/shard protocol {field} drift")
    if protocols[0]["candidate_catalog_file_sha256"] != file_sha(catalog_path):
        raise ValueError("serial/shard catalog file drift")
    if len(serial) != len(merged) or len(serial) != catalog["candidate_count"]:
        raise ValueError("serial/shard coverage mismatch")
    for directory, rows, protocol in zip((serial_dir, merged_dir), (serial, merged), protocols):
        summary = read(Path(directory) / "summary.json")
        if summary.get("status") != "completed" or summary.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("label output incomplete or protocol mismatched")
        if summary.get("labels_file_sha256") != file_sha(Path(directory) / "labels.json"):
            raise ValueError("label output file hash drift")
        if any(row.get("label_protocol_sha256") != protocol["protocol_sha256"] for row in rows):
            raise ValueError("label row logical protocol drift")
    mismatches = []
    for index, (a, b, candidate) in enumerate(zip(serial, merged, catalog["entries"], strict=True)):
        for field in ("candidate_id", "state_sha256", "snapshot_context_sha256"):
            if a.get(field) != candidate.get(field) or b.get(field) != candidate.get(field):
                raise ValueError(f"label/catalog {field} drift")
        if b.get("candidate_index") != index or b.get("policy_key_candidate_index") != index:
            raise ValueError("merged shard global index drift")
        fields = [field for field in SEMANTIC_LABEL_FIELDS if field not in a or field not in b or a[field] != b[field]]
        observation = compare_traces([a], [b], atol=atol, rtol=rtol, fields=("actor_observation",))
        if fields or not observation["passed"]:
            mismatches.append({"candidate_id": candidate["candidate_id"], "fields": fields,
                               "observation": observation})
    return {"passed": not mismatches, "candidate_count": len(serial), "mismatches": mismatches,
            "different_serial_and_shard_protocol_hashes_expected": True,
            "semantic_fields_compared": list(SEMANTIC_LABEL_FIELDS)}


def choose_proposer(repo, explicit=None):
    if explicit:
        return Path(explicit).resolve()
    candidates = []
    for path in (Path(repo) / "JIT/runs/frozen_unified").rglob("frozen_unified_policy.json"):
        record = read(path).get("policy", {})
        if record.get("name") == "pi_0" and "round1" in str(record.get("formal_config", "")).lower():
            candidates.append(path.resolve())
    if len(candidates) != 1:
        raise ValueError("cannot uniquely identify Round1 pi_0; supply --frozen-policy. Candidates: "
                         + json.dumps([str(p) for p in candidates]))
    return candidates[0]


def build_plan(repo, output, *, frozen_policy=None, evaluators=(), per_phase=3,
               shard_size=2, prefix_seed=9400001, label_seed=9521602,
               atol=1e-6, rtol=1e-5, interaction_budget=20000, gpu="0"):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if not 1 <= per_phase <= 8 or not 1 <= shard_size <= 16:
        raise ValueError("small audit requires per-phase 1..8 and shard-size 1..16")
    if not all(math.isfinite(x) and 0 <= x <= 1e-3 for x in (atol, rtol)):
        raise ValueError("numerical audit tolerances must be finite in [0, 1e-3]")
    proposer = choose_proposer(repo, frozen_policy)
    paths = [proposer, *[Path(p).resolve() for p in evaluators]]
    unique = list(dict.fromkeys(paths))
    if len(unique) > 4:
        raise ValueError("small audit supports at most four frozen policies")
    records, horizon = [], None
    for path in unique:
        frozen = read(path)
        verify_hash(frozen, "freeze_protocol_sha256")
        record = frozen["policy"]
        if frozen.get("status") != "frozen" or record["xml_sha256"] != FIXED_XML_SHA256:
            raise ValueError("frozen policy/task XML mismatch")
        config_path = Path(record["formal_config"])
        if not config_path.is_absolute():
            config_path = repo / config_path
        config = read(config_path)
        if canonical_sha256(config) != record["formal_config_sha256"]:
            raise ValueError("frozen formal config file drift")
        current = int(config["ppo"]["episode_horizon"])
        if not 0 < current <= 1000 or horizon not in (None, current):
            raise ValueError("probe audit needs identical frozen horizons in 1..1000")
        horizon = current
        checkpoint = Path(record["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = repo / checkpoint
        missing = [str(p) for p in (checkpoint / "payload.pkl", checkpoint / "identity.json",
                                    checkpoint.parent.parent / "formal_report.json") if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing production artifacts: " + json.dumps(missing))
        records.append({"path": str(path), "file_sha256": file_sha(path), "policy": record})
    # One captured prefix; per evaluator/candidate: prefix replay + four suffixes;
    # then one serial and one sharded evaluation of the same candidates.
    samples = 2 * per_phase
    maximum = horizon + len(records) * samples * horizon * 7
    if maximum > interaction_budget:
        raise ValueError(f"audit ceiling {maximum} exceeds --interaction-budget {interaction_budget}")
    sources = sorted((repo / "JIT/src/jit_dvgc").rglob("*.py")) + sorted((repo / "JIT/cli").glob("*.py"))
    plan = {"schema": SCHEMA, "status": "locked_before_rollout", "repo": str(repo), "output": str(output),
            "proposer": records[0], "evaluators": records, "horizon": horizon, "per_phase": per_phase,
            "shard_size": shard_size, "prefix_seed": int(prefix_seed), "label_seed": int(label_seed),
            "atol": float(atol), "rtol": float(rtol), "gpu": str(gpu),
            "maximum_environment_interactions": maximum, "requested_budget": interaction_budget,
            "sources": {str(p.relative_to(repo)): file_sha(p) for p in sources},
            "purpose": "engineering_validation_only", "training_transitions": 0,
            "test_data_used": False, "tube_admission_authorized": False,
            "selection_rule": "first_middle_last_real_prelanding_frame_in_each_phase; descending_only_downstream",
            "coverage_requirement": "full_requested_phase_counts_and_observed_apex_transition",
            "formal_envelope_claim_authorized": False}
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def load_plan(path):
    plan = read(path)
    verify_hash(plan, "plan_sha256")
    for item in [plan["proposer"], *plan["evaluators"]]:
        if file_sha(item["path"]) != item["file_sha256"]:
            raise ValueError("audit frozen manifest changed after lock")
    for path, sha in plan["sources"].items():
        if file_sha(Path(plan["repo"]) / path) != sha:
            raise ValueError(f"audit source changed after lock: {path}")
    return plan


def package_results(output):
    """Whitelist small evidence/logs; never include checkpoints or pickle snapshots."""
    output = Path(output)
    paths = [p for p in output.rglob("*") if p.is_file() and not p.is_symlink()
             and p.suffix in {".json", ".log", ".md"} and p.stat().st_size <= 20_000_000]
    bundle = output / "results_to_send.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, str(path.relative_to(output)))
    return bundle


def run_validation(repo, output, **options):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": SCHEMA, "status": "preflight", "training_transitions": 0,
               "test_data_used": False, "formal_envelope_claim_authorized": False,
               "stages": [], "production_validation_completed": False}
    plan = None
    try:
        plan = build_plan(repo, output, **options)
        write(output / "plan.json", plan)
        summary["plan_sha256"] = plan["plan_sha256"]
        summary["maximum_environment_interactions"] = plan["maximum_environment_interactions"]
        print(f"[audit] at most {plan['maximum_environment_interactions']} env.step calls; no PPO", flush=True)
        cli = Path(repo) / "JIT/cli/validate_jump_evidence.py"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan["gpu"], XLA_PYTHON_CLIENT_PREALLOCATE="false")
        env["PYTHONPATH"] = str(Path(repo) / "JIT/src") + os.pathsep + env.get("PYTHONPATH", "")

        def stage(name, kind, evaluator=0, sample=0, shard=0, count=1):
            directory = output / name
            directory.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, str(cli), "--worker", kind, "--plan", str(output / "plan.json"),
                       "--worker-output", str(directory), "--evaluator-index", str(evaluator),
                       "--sample-index", str(sample), "--shard-index", str(shard), "--shard-count", str(count)]
            print(f"[audit] {name}", flush=True)
            stage_row = {"name": name, "status": "running"}
            summary["stages"].append(stage_row)
            write(output / "summary.json", summary)
            with (directory / "process.log").open("w") as log:
                process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            print(f"[audit] {name} running; log: {directory / 'process.log'}", flush=True)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
            stage_row.update(returncode=code, status="completed" if code == 0 else "engineering_error")
            report = read(directory / "report.json") if (directory / "report.json").exists() else {}
            stage_row["environment_interactions"] = report.get("environment_interactions")
            if code:
                raise RuntimeError(f"stage {name} failed; inspect process.log and report.json")
            return report

        capture = stage("capture", "capture")
        count = capture["candidate_count"]
        if count == 0:
            raise RuntimeError("no nonterminal reached states; return capture diagnostics")
        shards = math.ceil(count / plan["shard_size"])
        replay, labels = [], []
        for evaluator in range(len(plan["evaluators"])):
            for sample in range(count):
                replay.append(stage(f"evaluator_{evaluator}/replay_{sample:03d}", "replay", evaluator, sample))
            stage(f"evaluator_{evaluator}/serial", "serial", evaluator)
            for shard in range(shards):
                stage(f"evaluator_{evaluator}/shard_{shard:03d}", "shard", evaluator, shard=shard, count=shards)
            stage(f"evaluator_{evaluator}/merge", "merge", evaluator, count=shards)
            comparison = compare_label_outputs(output / f"evaluator_{evaluator}/serial/result",
                output / f"evaluator_{evaluator}/merge/result", output / "capture/catalog.json",
                atol=plan["atol"], rtol=plan["rtol"])
            labels.append(comparison)
        gates = {"seed_prefix_first_valid_landing": capture["prefix_valid_landing_observed"],
                 "phase_and_apex_coverage": capture["coverage_complete"],
                 "prefix_replay_context": all(r["prefix_replay"]["passed"] for r in replay),
                 "preserved_context_restore": all(r["preserved_restore"]["passed"] for r in replay),
                 "fresh_context_restore": all(r["fresh_restore"]["passed"] for r in replay),
                 "counter_reset_behavior_equivalence": all(r["counter_effect"]["passed"] for r in replay),
                 "serial_shard_equivalence": all(r["passed"] for r in labels)}
        summary.update(status="passed_on_sampled_states" if all(gates.values()) else "diagnostic_mismatch",
                       gates=gates, label_comparisons=labels, production_validation_completed=True,
                       scope="only this frozen-policy set, prefix, sampled states and tolerances; not a global proof")
    except BaseException as exc:
        summary.update(status="engineering_error", error=f"{type(exc).__name__}: {exc}")
        write(output / "failure.json", {"error": summary["error"], "traceback": traceback.format_exc()})
    finally:
        costs = [s.get("environment_interactions") for s in summary["stages"]]
        summary["known_environment_interactions"] = sum(c for c in costs if type(c) is int)
        summary["interaction_accounting_complete"] = all(type(c) is int for c in costs)
        write(output / "summary.json", summary)
        bundle = package_results(output)
        print(f"[audit] {summary['status']}\nReturn this file: {bundle}", flush=True)
    return summary
