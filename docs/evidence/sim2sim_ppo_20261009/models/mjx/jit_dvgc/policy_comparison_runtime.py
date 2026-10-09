"""Preflight/projection, bounded label workers and common-panel figure assembly."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import traceback

from .jump_evidence_validation import read, write, file_sha, verify_hash, FIXED_XML_SHA256
from .evidence_integrity import canonical_sha256
from .policy_comparison import verify_plan

ROLES = ("train", "calibration", "acceptance")
NAMES = ("pi_0", "pi_1", "pi_2", "pi_3")


def complete_output(directory, catalog, acquisition, evaluator, horizon, seed):
    from .policy_family_landing import _load_completed_evaluator, _requested_contract
    expected = _requested_contract(Path(catalog), acquisition["policy"], evaluator["policy"],
                                   evaluator["file_sha256"], horizon, seed)
    return _load_completed_evaluator(Path(directory), evaluator=evaluator["policy"],
                                    catalog_path=Path(catalog), expected_contract=expected)


def paths_for_shard(plan, output, job_index, shard_index):
    job = plan["jobs"][job_index]
    root = Path(output) / "jobs" / job["job_id"] / f"shard_{shard_index:03d}"
    return [p / "result" for p in sorted(root.glob("attempt_*")) if (p / "result/summary.json").exists()
            and read(p / "result/summary.json").get("status") == "completed_shard"]


def shard_kwargs(plan, job_index, shard_index, result):
    job = plan["jobs"][job_index]
    return dict(catalog_path=Path(job["catalog"]), acquisition_frozen_policy=Path(plan["members"][0]["path"]),
                evaluator_frozen_policy=Path(plan["members"][job["member_index"]]["path"]), output_dir=Path(result),
                shard_index=shard_index, shard_count=len(job["shard_maximum_interactions"]),
                max_ticks=plan["horizon"], protocol_seed=job["seed"])


def validate_resume(plan, output):
    from .policy_family_landing import run_policy_family_evaluator_shard
    from .unified_policy_freeze import load_frozen_unified_manifest
    for member in plan["members"]:
        if load_frozen_unified_manifest(Path(member["path"]))["policy"] != member["policy"]:
            raise ValueError("frozen checkpoint identity changed")
    for j, job in enumerate(plan["jobs"]):
        if job["reuse_directory"]:
            if complete_output(job["reuse_directory"], job["catalog"], plan["members"][0],
                               plan["members"][job["member_index"]], plan["horizon"], job["seed"]) is None:
                raise ValueError("locked reused evaluator is no longer completed")
        for i in range(len(job["shard_maximum_interactions"])):
            for path in paths_for_shard(plan, output, j, i):
                # Existing completed output takes the cache-only path, never builds a GPU env.
                result = run_policy_family_evaluator_shard(**shard_kwargs(plan, j, i, path))
                if result["environment_interactions"] != sum(r["environment_interactions"] for r in read(path / "labels.json")):
                    raise ValueError("cached shard cost telemetry differs from label rows")


def prepare(output):
    from .unified_policy_freeze import load_frozen_unified_manifest
    from .unified_continuation_labels import validate_unified_boundary_catalog, validate_candidate_snapshot
    from .unified_continuation_shards import contiguous_shard_bounds
    from .unified_envelope_snapshot import load_unified_envelope_snapshot, snapshot_context_sha256
    from .acquisition.causal_jump import ACQUISITION_MODE, validate_jump_start_reachability_payload
    from .analysis.capability_tube import physical_coordinates_from_arrays, resolution_contract
    from .analysis.policy_envelopes import project_row
    from .analysis.nominal_jump_centerline import load_nominal_jump_centerline
    from .config import load_config
    from .model import load_host_model
    import matplotlib  # Fail before spending GPU interactions if plotting dependency is absent.
    output = Path(output)
    request = read(output / "request.json")
    if (output / "plan.json").exists():
        plan = verify_plan(output / "plan.json")
        if plan["request"] != request:
            raise ValueError("request/plan mismatch")
        validate_resume(plan, output)
        return {"status": "validated_resume", "plan_sha256": plan["plan_sha256"]}
    if not 1 <= request["shard_size"] <= 600 or request["interaction_budget"] <= 0:
        raise ValueError("shard size must be 1..600 and budget positive")
    repo, scan = Path(request["repo"]), Path(request["scan_root"])
    inputs = {}
    def lock(path):
        path = Path(path).resolve()
        inputs[str(path)] = file_sha(path)
        return path
    source = read(lock(scan / "frontier_plan_causal_expanded.json"))
    verify_hash(source, "plan_sha256")
    contract = source["jump_tube_contract"]
    if contract["continuation_success_criterion"] != "first_valid_landing_before_physical_failure":
        raise ValueError("source scan endpoint changed")
    legacy_paths = [repo / p for p in contract["continuation_frozen_policies"]]
    if request["frozen_policies"]:
        paths = [Path(p) for p in request["frozen_policies"]]
    else:
        matches = [p for p in (repo / "JIT/runs/frozen_unified").rglob("frozen_unified_policy.json")
                   if read(p).get("policy", {}).get("name") == "pi_3"]
        if len(matches) != 1:
            raise ValueError(f"expected one frozen pi_3, found {[str(p) for p in matches]}; supply all four --frozen-policy paths")
        paths = [*legacy_paths, matches[0]]
    members = []
    horizon = int(source["fixed_probe_panel"]["max_label_ticks"])
    if not 1 <= horizon <= 1000:
        raise ValueError("invalid locked labeling horizon")
    for path in paths:
        path = lock(path)
        record = load_frozen_unified_manifest(path)["policy"]
        config = read(lock(repo / record["formal_config"]))
        if record["xml_sha256"] != FIXED_XML_SHA256 or config["ppo"]["episode_horizon"] != horizon:
            raise ValueError("common task XML/horizon mismatch")
        members.append({"path": str(path), "file_sha256": inputs[str(path)], "policy": record})
    members.sort(key=lambda m: m["policy"]["name"])
    if [m["policy"]["name"] for m in members] != list(NAMES):
        raise ValueError("comparison requires exactly pi_0, pi_1, pi_2, pi_3")
    for member, legacy in zip(members[:3], legacy_paths, strict=True):
        if member["file_sha256"] != file_sha(legacy):
            raise ValueError("cannot replace frozen members of the old locked scan")
    if members[0]["file_sha256"] != file_sha(repo / contract["proposal_frozen_policy"]):
        raise ValueError("source proposer is not the declared pi_0")
    center_path = lock(repo / contract["nominal_centerline"])
    centerline = load_nominal_jump_centerline(center_path)
    if centerline["centerline_sha256"] != contract["nominal_centerline_sha256"]:
        raise ValueError("locked centerline drift")
    config = read(repo / members[0]["policy"]["formal_config"])
    bundle = load_host_model(load_config(lock(repo / config["inputs"]["up_config_path"])))
    if bundle.xml_sha256 != FIXED_XML_SHA256:
        raise ValueError("physical projection XML mismatch")
    panels, jobs, historical = {}, [], []
    for role in ROLES:
        role_root = scan / f"frontier_{role}"
        catalog_path = lock(role_root / "acquisition/catalog.json")
        catalog = read(catalog_path)
        acquisition = read(lock(role_root / "acquisition/summary.json"))
        protocol = read(lock(role_root / "acquisition/protocol.json"))
        verify_hash(protocol, "protocol_sha256")
        if protocol["protocol_sha256"] != catalog["protocol_sha256"] or acquisition["protocol_sha256"] != catalog["protocol_sha256"]:
            raise ValueError("acquisition protocol/catalog identity drift")
        if acquisition.get("status") != "completed" or acquisition.get("candidate_count") != catalog.get("candidate_count"):
            raise ValueError("acquisition report incomplete or count mismatch")
        if catalog.get("acquisition_mode") != ACQUISITION_MODE or catalog.get("jump_start_reachability_proven") is not True or catalog.get("rsi_used_for_reachability") is not False:
            raise ValueError("shared comparison requires real forward-arrival catalogs")
        rows = validate_unified_boundary_catalog(catalog, policy_record=members[0]["policy"],
                                                 frozen_manifest_sha256=members[0]["file_sha256"])
        role_groups = {a["parent_group_id"] for a in source["anchors"] if a["role"] == role}
        projected = []
        for row in rows:
            if row["parent_group_id"] not in role_groups:
                raise ValueError("candidate parent is outside locked logical role")
            provenance = row["jump_start_reachability"]
            verify_hash(provenance, "reachability_sha256")
            validate_jump_start_reachability_payload(provenance)
            if provenance["proposal_parent_group_id"] != row["parent_group_id"] or provenance["perturbation_start_state_sha256"] != row["parent_state_sha256"]:
                raise ValueError("candidate ancestry identity mismatch")
            if provenance["jump_start_state_sha256"] != contract["jump_start_state_sha256"]:
                raise ValueError("candidate jump-start identity mismatch")
            snapshot = load_unified_envelope_snapshot(catalog_path.parent / row["source_bank"] / row["snapshot"])
            validate_candidate_snapshot(snapshot, row, policy_record=members[0]["policy"])
            coords = physical_coordinates_from_arrays(snapshot.qpos, snapshot.qvel, bundle=bundle)
            if snapshot.down_events["valid_contact_seen"] or (row["phase"] == "downstream" and coords["root_vz_mps"] >= 0):
                raise ValueError("candidate phase/prelanding semantics mismatch")
            if row["phase"] == "upstream" and snapshot.up_events["apex_seen"]:
                raise ValueError("upstream candidate already passed Apex")
            projected.append(project_row(row, coords, snapshot_context_sha256(snapshot)))
        project_path = output / "panels" / f"{role}.json"
        write(project_path, projected)
        lock(project_path)
        panels[role] = {"catalog": str(catalog_path), "catalog_sha256": file_sha(catalog_path),
                        "projected": str(project_path), "candidate_count": len(rows),
                        "proposer": "pi_0", "seed": source["seeds"][role]["labeling"]}
        historical.append({"path": str(role_root / "acquisition/summary.json"), "kind": "acquisition",
                           "environment_interactions": acquisition.get("environment_interactions")})
        for member_index, member in enumerate(members):
            name = member["policy"]["name"]
            old = role_root / "labels/per_policy" / name
            seed = panels[role]["seed"]
            # Only original three members can reuse the old experiment's labels.
            completed = complete_output(old, catalog_path, members[0], member, horizon, seed) if member_index < 3 else None
            reuse = str(old.resolve()) if completed else None
            if completed:
                for filename in ("summary.json", "labels.json", "protocol.json"):
                    lock(old / filename)
            if (old / "summary.json").exists() and member_index < 3:
                old_report = read(old / "summary.json")
                historical.append({"path": str(old / "summary.json"), "kind": "historical_evaluator",
                                   "status": old_report.get("status"), "environment_interactions": old_report.get("environment_interactions"),
                                   "completed_candidates": old_report.get("label_count", old_report.get("completed_candidate_count"))})
            count = math.ceil(len(rows) / request["shard_size"])
            maxima = [(lambda b: (b[1]-b[0])*horizon)(contiguous_shard_bounds(len(rows), i, count)) for i in range(count)]
            jobs.append({"job_id": f"{role}_{name}", "role": role, "evaluator": name, "member_index": member_index,
                         "catalog": str(catalog_path), "seed": seed, "reuse_directory": reuse,
                         "shard_maximum_interactions": maxima if not reuse else []})
    maximum = sum(sum(j["shard_maximum_interactions"]) for j in jobs)
    if maximum > request["interaction_budget"]:
        raise ValueError(f"first-attempt ceiling {maximum} exceeds declared budget {request['interaction_budget']}")
    write(output / "historical_cost_records.json", {"records": historical,
          "complete_end_to_end_ledger": False, "partial_counts_are_lower_bounds": True,
          "note": "Original records preserved. No bootstrap/PPO costs or undocumented retries inferred."})
    contexts = {role: {r["snapshot_context_sha256"] for r in read(panels[role]["projected"])} for role in ROLES}
    sources = sorted((repo / "JIT/src/jit_dvgc").rglob("*.py")) + sorted((repo / "JIT/cli").glob("*.py"))
    plan = {"schema": "jit_policy_envelope_comparison_plan_v1", "status": "locked_before_new_labels",
            "created_utc": datetime.now(timezone.utc).isoformat(), "request": request, "repo": str(repo),
            "members": members, "horizon": horizon, "panels": panels, "jobs": jobs,
            "input_files": inputs, "sources": {str(p.relative_to(repo)): file_sha(p) for p in sources},
            "source_plan_sha256": source["plan_sha256"], "centerline": str(center_path),
            "first_attempt_maximum_interactions": maximum, "resolution": resolution_contract(),
            "legacy_family": list(NAMES[:3]), "comparison_family": list(NAMES),
            "cross_role_exact_context_overlap": {a + "/" + b: len(contexts[a] & contexts[b]) for i,a in enumerate(ROLES) for b in ROLES[i+1:]},
            "roles_pooled": False, "final_test_used": False, "training_transitions": 0,
            "panel_scope": "legacy pi_0 frontier arrivals; acquisition excluded existing training-support states; not exhaustive capability",
            "continuation_start_semantics": "fresh_continuation_v1",
            "numerical_replay_status": "known_mismatch; user accepted and declined further validation on 2026-09-07",
            "start_description": "x=2.5 near-ground reset; initial wheel clearance about 0.031 m accepted by user",
            "snapshot_replay_equivalence_verified": False, "formal_envelope_claim_authorized": False}
    plan["plan_sha256"] = canonical_sha256(plan)
    write(output / "plan.json", plan)
    return {"status": "prepared", "first_attempt_maximum_interactions": maximum,
            "reused_evaluators": sum(bool(j["reuse_directory"]) for j in jobs)}


def analyze(plan, output, role):
    from .analysis.policy_envelopes import summarize, render_comparison
    from .policy_family_landing import merge_any_policy_landing_labels
    panel = plan["panels"][role]
    projected = read(panel["projected"])
    labels, sources = {}, {}
    for job in [j for j in plan["jobs"] if j["role"] == role]:
        path = Path(job["reuse_directory"] or Path(output) / "jobs" / job["job_id"] / "merged")
        result = complete_output(path, job["catalog"], plan["members"][0],
                                 plan["members"][job["member_index"]], plan["horizon"], job["seed"])
        if result is None:
            raise ValueError("cannot plot incomplete policy panel")
        labels[job["evaluator"]] = result[1]
        sources[job["evaluator"]] = {"path": str(path), "labels_sha256": file_sha(path / "labels.json")}
    report = summarize(projected, labels, role=role)
    report.update(plan_sha256=plan["plan_sha256"], label_sources=sources, panel=panel,
                  numerical_replay_status=plan["numerical_replay_status"], policy_identities=plan["members"])
    destination = Path(output) / "figures" / role
    # Close a derived view of the original three-member family; never add pi_3 to old raw outputs.
    rows, family = merge_any_policy_landing_labels({n: labels[n] for n in NAMES[:3]})
    write(destination / "legacy_family_labels.json", {"role": role, "family": family, "entries": rows,
          "source_plan_sha256": plan["source_plan_sha256"], "raw_historical_files_modified": False})
    render_comparison(projected, report, destination, centerline=read(plan["centerline"])["points"])
    return {"status": "completed", "role": role, "metrics": report["metrics"]}


def run_worker(output, destination, *, kind, job_index=0, shard_index=0, role="train"):
    output, destination = Path(output), Path(destination)
    try:
        if kind == "prepare":
            report = prepare(output)
        else:
            plan = verify_plan(output / "plan.json")
            if kind == "analyze":
                report = analyze(plan, output, role)
            elif kind == "label":
                import jax
                if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
                    raise RuntimeError("label worker requires exactly one visible JAX GPU")
                from .policy_family_landing import run_policy_family_evaluator_shard
                report = run_policy_family_evaluator_shard(**shard_kwargs(plan, job_index, shard_index, destination / "result"))
                if report["environment_interactions"] != sum(r["environment_interactions"] for r in read(destination / "result/labels.json")):
                    raise ValueError("shard cost telemetry differs from label rows")
            elif kind == "merge":
                from .policy_family_landing import merge_policy_family_evaluator_shards
                job = plan["jobs"][job_index]
                paths = [paths_for_shard(plan, output, job_index, i) for i in range(len(job["shard_maximum_interactions"]))]
                if any(not p for p in paths):
                    raise ValueError("missing completed shard")
                report = merge_policy_family_evaluator_shards(catalog_path=Path(job["catalog"]),
                    shard_dirs=[p[-1] for p in paths], output_dir=destination / "merged", evaluator_name=job["evaluator"],
                    acquisition_frozen_policy=Path(plan["members"][0]["path"]),
                    evaluator_frozen_policy=Path(plan["members"][job["member_index"]]["path"]),
                    max_ticks=plan["horizon"], protocol_seed=job["seed"])
            else:
                raise ValueError("unknown comparison worker")
        report = {**report, "worker_runtime_versions": {name: importlib.metadata.version(name)
                  for name in ("jax", "mujoco", "brax", "numpy", "matplotlib")}}
        write(destination / "worker_report.json", report)
        return 0
    except BaseException as exc:
        print(traceback.format_exc(), flush=True)
        write(destination / "worker_failure.json", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return 1
