"""Versioned complementary probes and bounded, process-isolated suffix experiments.

This path never selects one successor Actor. Legacy three-policy runs remain on
policy_family_landing. Hash locks identify content, not external chronology.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from .evidence_integrity import canonical_sha256

BANK_SCHEMA = "jit_complementary_probe_bank_v1"
PLAN_SCHEMA = "jit_probe_bank_label_plan_v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _file_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _verify(value, field):
    if canonical_sha256({k: v for k, v in value.items() if k != field}) != value.get(field):
        raise ValueError(f"{field} self-hash drift")


def lock_probe_bank(spec: dict, output: Path) -> dict:
    """Lock frozen identities and a declared task; no checkpoint admission claim."""
    from .unified_policy_freeze import load_frozen_unified_manifest
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(spec.get("version", ""))):
        raise ValueError("bank version must be a nonempty safe identifier")
    task = spec.get("task", {})
    for field in ("xml_sha256", "start_contract_sha256", "centerline_sha256", "resolution_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(task.get(field, ""))):
            raise ValueError(f"bank requires declared task {field}")
    if task.get("success_criterion") != "first_valid_landing":
        raise ValueError("probe bank endpoint must be first_valid_landing")
    if task.get("continuation_start_semantics") != "fresh_continuation_v1":
        raise ValueError("only explicitly declared fresh_continuation_v1 is implemented; replay equivalence is pending")
    for key in ("max_ticks", "label_interaction_budget", "max_candidates_per_process"):
        if type(spec.get(key)) is not int or spec[key] <= 0:
            raise ValueError(f"bank requires a positive integer {key}")
    members = []
    for entry in spec.get("members", []):
        path = Path(entry["frozen_policy"]).resolve()
        frozen = load_frozen_unified_manifest(path)
        record = frozen["policy"]
        if record["xml_sha256"] != task["xml_sha256"]:
            raise ValueError("bank member XML drift")
        roles = sorted(set(entry.get("roles", [])))
        if not roles or not set(roles) <= {"proposer", "evaluator"}:
            raise ValueError("bank roles must be proposer and/or evaluator")
        if record["name"] in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", record["name"]):
            raise ValueError("unsafe probe name")
        members.append({"name": record["name"], "roles": roles,
                        "frozen_policy": str(path), "frozen_file_sha256": _file_sha(path),
                        "policy": record})
    names = [m["name"] for m in members]
    if len(set(names)) != len(names):
        raise ValueError("duplicate bank member name")
    for role in ("proposer", "evaluator"):
        if not any(role in m["roles"] for m in members):
            raise ValueError(f"bank needs at least one {role}")
    bank = {"schema": BANK_SCHEMA, "status": "identity_locked_runtime_unverified",
            "version": spec["version"], "task": task, "members": members,
            **{key: spec[key] for key in ("max_ticks", "label_interaction_budget", "max_candidates_per_process")},
            "whole_tube_actor_retention_required": False,
            "technical_gpu_smoke_verified": False,
            "snapshot_replay_equivalence_verified": False,
            "formal_envelope_claim_authorized": False}
    bank["bank_sha256"] = canonical_sha256(bank)
    _write_new(output, bank)
    return bank


def load_probe_bank(path: Path) -> dict:
    bank = _read(path)
    _verify(bank, "bank_sha256")
    if bank.get("schema") != BANK_SCHEMA:
        raise ValueError("probe bank schema drift")
    for member in bank["members"]:
        if _file_sha(member["frozen_policy"]) != member["frozen_file_sha256"]:
            raise ValueError("probe frozen file identity drift")
        if _read(member["frozen_policy"])["policy"] != member["policy"]:
            raise ValueError("probe policy identity drift")
    return bank


def validate_probe_arrivals(path: Path, entries: list[dict], proposer: dict) -> None:
    from .acquisition.causal_jump import validate_jump_start_reachability_payload
    from .unified_continuation_labels import validate_candidate_snapshot
    from .unified_envelope_snapshot import load_unified_envelope_snapshot
    seen = set()
    for row in entries:
        key = row["candidate_id"]
        if key in seen:
            raise ValueError("duplicate arrival candidate ID")
        seen.add(key)
        provenance = row.get("jump_start_reachability", {})
        validate_jump_start_reachability_payload(provenance)
        _verify(provenance, "reachability_sha256")
        snapshot = load_unified_envelope_snapshot(path.parent / row["source_bank"] / row["snapshot"])
        validate_candidate_snapshot(snapshot, row, policy_record=proposer)


def prepare_probe_labels(bank_path: Path, catalogs: list[Path], output: Path,
                         *, role: str, seed: int) -> dict:
    """Lock all proposer catalogs and every evaluator shard before suffix outcomes."""
    from .evidence_integrity import read_verified_protocol
    from .unified_continuation_shards import contiguous_shard_bounds
    bank = load_probe_bank(bank_path)
    if role not in {"train", "calibration", "acceptance"}:
        raise ValueError("final TEST is not a development probe role")
    members = {m["name"]: m for m in bank["members"]}
    inputs, jobs = [], []
    seen_catalogs = set()
    for index, path in enumerate(catalogs):
        path = Path(path).resolve()
        catalog = _read(path)
        protocol = read_verified_protocol(path.parent / "protocol.json")
        if catalog.get("status") != "completed" or catalog.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("probe catalog/protocol incomplete or mismatched")
        if protocol.get("probe_bank_sha256") != bank["bank_sha256"]:
            raise ValueError("legacy/different-bank acquisition cannot enter a new bank run")
        if protocol.get("evidence_mode") != "probe_bank_arrivals_v1":
            raise ValueError("catalog did not use arrival evidence deduplication")
        proposer = members[protocol["policy_name"]]
        if "proposer" not in proposer["roles"] or protocol["policy_actor_sha256"] != proposer["policy"]["actor_sha256"]:
            raise ValueError("catalog proposer identity drift")
        if protocol.get("policy_payload_sha256") != proposer["policy"]["payload_sha256"]:
            raise ValueError("catalog proposer payload drift")
        if protocol.get("nominal_centerline_sha256") != bank["task"]["centerline_sha256"]:
            raise ValueError("catalog centerline drift")
        if protocol.get("logical_role") != role or protocol.get("start_contract_sha256") != bank["task"]["start_contract_sha256"]:
            raise ValueError("catalog role/start contract drift")
        entries = catalog.get("entries", [])
        if not entries or catalog.get("candidate_count") != len(entries):
            raise ValueError("catalog candidate count drift")
        if any(not row.get("snapshot_context_sha256") for row in entries):
            raise ValueError("exact stored context identity missing")
        validate_probe_arrivals(path, entries, proposer["policy"])
        sha = _file_sha(path)
        if sha in seen_catalogs:
            raise ValueError("duplicate proposer catalog")
        seen_catalogs.add(sha)
        count = math.ceil(len(entries) / bank["max_candidates_per_process"])
        item = {"path": str(path), "file_sha256": sha, "proposer": proposer["name"],
                "candidate_count": len(entries), "shard_count": count}
        inputs.append(item)
        for evaluator in bank["members"]:
            if "evaluator" not in evaluator["roles"]:
                continue
            for shard in range(count):
                start, stop = contiguous_shard_bounds(len(entries), shard, count)
                jobs.append({"job_id": f"catalog_{index:03d}/{evaluator['name']}/shard_{shard:03d}",
                             "catalog_index": index, "evaluator": evaluator["name"],
                             "shard_index": shard, "shard_count": count,
                             "maximum_interactions": (stop - start) * bank["max_ticks"]})
    if not inputs:
        raise ValueError("probe plan needs catalogs")
    maximum = sum(job["maximum_interactions"] for job in jobs)
    if maximum > bank["label_interaction_budget"]:
        raise ValueError("label plan exceeds declared budget before execution")
    plan = {"schema": PLAN_SCHEMA, "bank": str(Path(bank_path).resolve()),
            "bank_sha256": bank["bank_sha256"], "role": role, "seed": int(seed),
            "catalogs": inputs, "jobs": jobs, "maximum_label_interactions": maximum,
            "budget_scope": "this_label_plan_including_retries",
            "bootstrap_acquisition_training_costs_included": False,
            "snapshot_replay_equivalence_verified": False}
    plan["plan_sha256"] = canonical_sha256(plan)
    _write_new(output, plan)
    return plan


def run_probe_labels(plan_path: Path, output: Path, *, python: str = sys.executable) -> dict:
    """Allow only one supervisor for an output, including across interrupted runs."""
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another probe supervisor owns this output") from exc
        return _run_probe_labels(plan_path, output, python=python)


def _run_probe_labels(plan_path: Path, output: Path, *, python: str) -> dict:
    """Run each suffix shard in a fresh process, preserve every attempt and cost.

    An interrupted/unknown-cost attempt consumes its full reservation; it cannot
    be silently retried for free. Valid existing attempts are reused only through
    the same contract checks as a new merge.
    """
    from .policy_family_landing import merge_policy_family_evaluator_shards, merge_any_policy_landing_labels
    plan = _read(plan_path)
    _verify(plan, "plan_sha256")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("probe plan schema drift")
    bank = load_probe_bank(Path(plan["bank"]))
    if plan["bank_sha256"] != bank["bank_sha256"]:
        raise ValueError("plan bank identity drift")
    for catalog in plan["catalogs"]:
        if _file_sha(catalog["path"]) != catalog["file_sha256"]:
            raise ValueError("plan catalog file drift")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "plan_identity.json"
    if lock.exists():
        if _read(lock) != {"plan_sha256": plan["plan_sha256"]}:
            raise ValueError("output belongs to a different label plan")
    else:
        _write_new(lock, {"plan_sha256": plan["plan_sha256"]})
    members = {m["name"]: m for m in bank["members"]}
    cli = Path(__file__).resolve().parents[2] / "cli" / "label_policy_family_first_landing.py"
    chosen = {}
    spent = 0
    for job in plan["jobs"]:
        attempts = output / job["job_id"]
        for receipt_path in sorted(attempts.glob("attempt_*/reservation.json")):
            receipt = _read(receipt_path)
            if receipt != {"plan_sha256": plan["plan_sha256"], "job": job}:
                raise ValueError("attempt reservation drift")
            result_path = receipt_path.parent / "result" / "summary.json"
            result = _read(result_path) if result_path.exists() else {}
            cost = result.get("environment_interactions")
            if type(cost) is not int or cost < 0 or cost > job["maximum_interactions"]:
                cost = job["maximum_interactions"]
            spent += cost
            if result.get("status") == "completed_shard":
                chosen[job["job_id"]] = receipt_path.parent / "result"
    for job in plan["jobs"]:
        if job["job_id"] in chosen:
            continue
        if spent + job["maximum_interactions"] > bank["label_interaction_budget"]:
            raise RuntimeError("label budget exhausted including failed/unknown attempts")
        attempts = output / job["job_id"]
        attempt = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):04d}"
        _write_new(attempt / "reservation.json", {"plan_sha256": plan["plan_sha256"], "job": job})
        catalog = plan["catalogs"][job["catalog_index"]]
        command = [python, str(cli), "--catalog", catalog["path"],
                   "--acquisition-frozen-policy", members[catalog["proposer"]]["frozen_policy"],
                   "--evaluator-frozen-policy", members[job["evaluator"]]["frozen_policy"],
                   "--output-dir", str(attempt / "result"), "--shard-index", str(job["shard_index"]),
                   "--shard-count", str(job["shard_count"]), "--max-ticks", str(bank["max_ticks"]),
                   "--protocol-seed", str(plan["seed"])]
        with (attempt / "process.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        _write_new(attempt / "process_exit.json", {"returncode": result.returncode})
        report_path = attempt / "result" / "summary.json"
        report = _read(report_path) if report_path.exists() else {}
        cost = report.get("environment_interactions")
        spent += cost if type(cost) is int and 0 <= cost <= job["maximum_interactions"] else job["maximum_interactions"]
        if result.returncode or report.get("status") != "completed_shard":
            raise RuntimeError(f"probe shard failed; preserved attempt: {attempt}")
        chosen[job["job_id"]] = attempt / "result"
    witnessed, contributions = [], {name: 0 for name in members}
    successful_contexts = {name: set() for name in members}
    for index, catalog in enumerate(plan["catalogs"]):
        labels = {}
        for member in bank["members"]:
            if "evaluator" not in member["roles"]:
                continue
            evaluator = member["name"]
            directories = [chosen[j["job_id"]] for j in plan["jobs"]
                           if j["catalog_index"] == index and j["evaluator"] == evaluator]
            merged_dir = output / "merged" / f"catalog_{index:03d}" / evaluator
            merge_policy_family_evaluator_shards(catalog_path=Path(catalog["path"]), shard_dirs=directories,
                output_dir=merged_dir, evaluator_name=evaluator,
                acquisition_frozen_policy=Path(members[catalog["proposer"]]["frozen_policy"]),
                evaluator_frozen_policy=Path(member["frozen_policy"]),
                max_ticks=bank["max_ticks"], protocol_seed=plan["seed"])
            labels[evaluator] = _read(merged_dir / "labels.json")
        rows, family = merge_any_policy_landing_labels(labels)
        for row in rows:
            row.update(catalog_file_sha256=catalog["file_sha256"], proposer=catalog["proposer"],
                       bank_sha256=bank["bank_sha256"], logical_role=plan["role"])
            row["evidence_id"] = canonical_sha256({"catalog": catalog["file_sha256"], "candidate": row["candidate_id"]})
            row["witness_status"] = "observed_landing" if row["label"] else "no_success_witness_under_declared_bank"
            for name in row["successful_policy_names"]:
                contributions[name] += 1
                successful_contexts[name].add(row["snapshot_context_sha256"])
            witnessed.append(row)
    report = {"schema": "jit_probe_bank_landing_witness_index_v1", "status": "completed",
              "bank_sha256": bank["bank_sha256"], "plan_sha256": plan["plan_sha256"],
              "entries": witnessed, "evaluator_success_counts": contributions,
              "evaluator_unique_context_contributions": {name: len(contexts - set().union(
                  *(other for other_name, other in successful_contexts.items() if other_name != name)))
                  for name, contexts in successful_contexts.items()},
              "observed_exact_context_count": len({r["snapshot_context_sha256"] for r in witnessed if r["label"]}),
              "charged_label_interactions_including_retries": spent,
              "cost_is_upper_bound_when_attempt_telemetry_missing": True,
              "end_to_end_interaction_total": None,
              "physical_cell_count": None, "single_actor_coverage_required": False,
              "snapshot_replay_equivalence_verified": False, "formal_envelope_claim_authorized": False}
    report["index_sha256"] = canonical_sha256(report)
    result_path = output / "witness_index.json"
    if result_path.exists():
        if _read(result_path) != report:
            raise ValueError("existing witness index differs; preserve and use a new experiment")
    else:
        _write_new(result_path, report)
    return report


def acquire_probe_catalog(spec_path: Path, output: Path) -> dict:
    """Collect one declared proposer/role; repeat with other members for a bank round."""
    from .acquisition.causal_jump import collect_jump_start_connected_candidates
    from .analysis.nominal_jump_centerline import load_nominal_jump_centerline
    from .checkpoint import load_checkpoint
    from .ppo import make_checkpoint_policy
    from .unified_formal import build_unified_formal_environment
    from .unified_training import checkpoint_identity
    spec = _read(spec_path)
    bank = load_probe_bank(Path(spec["bank"]))
    member = next(m for m in bank["members"] if m["name"] == spec["proposer"])
    if "proposer" not in member["roles"]:
        raise ValueError("probe is not a declared proposer")
    if spec["role"] not in {"train", "calibration", "acceptance"}:
        raise ValueError("invalid acquisition role")
    if _file_sha(spec["start_contract"]) != bank["task"]["start_contract_sha256"]:
        raise ValueError("declared start contract file drift")
    centerline = load_nominal_jump_centerline(Path(spec["nominal_centerline"]))
    if centerline["centerline_sha256"] != bank["task"]["centerline_sha256"]:
        raise ValueError("bank centerline drift")
    anchors = _read(spec["anchors"])
    record = member["policy"]
    config, _, env = build_unified_formal_environment(Path(record["formal_config"]))
    payload = load_checkpoint(Path(record["checkpoint"]), expected=checkpoint_identity(config, env))
    policy = make_checkpoint_policy(env, payload, deterministic=True)
    return collect_jump_start_connected_candidates(anchors, Path(output), env=env, policy=policy,
        policy_record=record, frozen_manifest_sha256=member["frozen_file_sha256"],
        nominal_centerline=Path(spec["nominal_centerline"]), protocol_seed=int(spec["seed"]),
        strengths=spec["strengths"], action_names=spec["action_names"], signs=spec["signs"],
        lookbacks_m=spec["lookbacks_m"], max_forward_ticks=int(spec["max_forward_ticks"]),
        evidence_mode="probe_bank_arrivals_v1", probe_bank_sha256=bank["bank_sha256"],
        logical_role=spec["role"], start_contract_sha256=bank["task"]["start_contract_sha256"],
        acquisition_interaction_ceiling=int(spec["interaction_ceiling"]))
