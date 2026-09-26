#!/usr/bin/env python3
"""Rebuild compact paper evidence without JAX, MuJoCo, models, or run directories.

Default: rebuild derived tables from sources/. --check: verify byte equality.
--capture: refresh compact source snapshots from the local immutable run artifacts.
No simulator or Git command is called. Capture is explicit; normal reproduction is offline.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNS = "JIT/runs/experiments/"
REUSE = RUNS + "delayed_exploration_reuse_20260912/queue/stage_0_result/"
COMPLETE = RUNS + "delayed_completion_start_20260912/"
CAMPAIGN = "JIT/runs/campaign/all_proposers_v1/"


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def scalar_fields(value):
    return {key: item for key, item in value.items() if not isinstance(item, (dict, list))}


def pool_fields(pool):
    return {"pool_sha256": pool["pool_sha256"], "entries": [
        {"key": key, "root_cell": row["root_cell"], "status": row["status"],
         "phase": row["phase"], "first_discovery_round": row["first_discovery_round"],
         "observation_labels": [obs["label"] for obs in row["observations"]]}
        for key, row in sorted(pool["entries"].items())]}


def capture():
    sources = HERE / "sources"
    sources.mkdir(exist_ok=True)
    records = {}

    def take(source_id, path, projection=None, description="Complete small source JSON"):
        raw = (REPO / path).read_bytes()
        if path.endswith(".csv"):
            value = list(csv.DictReader(io.StringIO(raw.decode())))
        else:
            value = json.loads(raw)
        value = projection(value) if projection else value
        snapshot = encoded(value)
        name = source_id + ".json"
        (sources / name).write_bytes(snapshot)
        records[source_id] = {"original_path": path, "original_sha256": sha(raw),
            "original_bytes": len(raw), "snapshot_path": "sources/" + name,
            "snapshot_sha256": sha(snapshot), "snapshot_bytes": len(snapshot),
            "projection": description}

    take("historical_summary", CAMPAIGN + "summary.json")
    take("historical_geometry_manifest", CAMPAIGN + "round_001/figures/figure_manifest.json")
    geometry_path = CAMPAIGN + "round_001/figures/all_points.csv"
    geometry_raw = (REPO / geometry_path).read_bytes()
    geometry_rows = list(csv.DictReader(io.StringIO(geometry_raw.decode())))
    geometry_columns = ["key", "state_sha256", "snapshot_context_sha256", "proposer", "phase",
        "witnessed", "root_cell", "root_x_m", "root_y_m", "root_z_m", "root_vx_mps",
        "root_vz_mps", "roll_deg", "pitch_deg"]
    geometry_stream = io.StringIO(newline="")
    geometry_writer = csv.DictWriter(geometry_stream, fieldnames=geometry_columns, lineterminator="\n")
    geometry_writer.writeheader()
    geometry_writer.writerows({key: row[key] for key in geometry_columns} for row in geometry_rows)
    geometry_snapshot = gzip.compress(geometry_stream.getvalue().encode(), mtime=0)
    geometry_name = "historical_geometry.csv.gz"
    (sources / geometry_name).write_bytes(geometry_snapshot)
    records["historical_geometry"] = {"original_path": geometry_path,
        "original_sha256": sha(geometry_raw), "original_bytes": len(geometry_raw),
        "snapshot_path": "sources/" + geometry_name, "snapshot_sha256": sha(geometry_snapshot),
        "snapshot_bytes": len(geometry_snapshot), "format": "csv_gzip", "rows": len(geometry_rows),
        "column_mapping": {key: key for key in geometry_columns},
        "projection": "All original rows in original order; listed columns only; no sample or label filtering"}
    for index in range(2):
        root = CAMPAIGN + f"round_{index:03d}/"
        take(f"historical_r{index}_proposers", root + "figures/proposer_metrics.csv",
             description="All proposer rows from source CSV")
        report = next((REPO / root / "training_attempt_000/training").glob("*/formal_report.json"))
        take(f"historical_r{index}_training", str(report.relative_to(REPO)), scalar_fields,
             "All scalar fields of formal training report; no model payload")
        ledgers = sorted((REPO / root).rglob("cost_ledger.json"))
        for ordinal, ledger in enumerate(ledgers):
            take(f"historical_r{index}_ledger{ordinal}", str(ledger.relative_to(REPO)),
                lambda value: {"attempts": [
                    {"stage": Path(a["attempt"]).parent.name,
                     "attempt_name": Path(a["attempt"]).name,
                     "charged": a["charged"], "measured": a["measured"]}
                    for a in value["attempts"]]},
                "Every attempt retained: stage, attempt ordinal, charged interactions, measured flag")

    take("delayed_reuse_status", REUSE + "pipeline_status.json")
    take("delayed_completion_summary", COMPLETE + "summary.json")
    take("delayed_completion_status", COMPLETE + "pipeline_status.json")
    take("delayed_completion_declaration", COMPLETE + "declaration.json")
    for index in range(2):
        root = REUSE + f"round_{index:03d}/"
        take(f"delayed_r{index}_training_config", root + "policy_training.json",
             lambda value: {key: value[key] for key in
                ("ppo", "initialization", "jump_start_probability", "pending_fraction", "schema")},
             "PPO configuration, initialization, reset probabilities, schema")
        take(f"delayed_r{index}_support", root + "training_support.json",
             lambda value: {"support_sha256": value["support_sha256"],
                 "realized_pending_fraction_by_phase": value["realized_pending_fraction_by_phase"],
                 "entries": [{key: row.get(key) for key in ("phase", "evidence_status", "sampling_weight")}
                             for row in value["entries"]]},
             "All reset rows retained as phase/status/weight plus actual pending fractions")
        report_path = (root + "training/stage_0_result_round_000/formal_report.json"
                       if index == 0 else COMPLETE + "training/stage_0_result_round_001/formal_report.json")
        take(f"delayed_r{index}_training", report_path, scalar_fields,
             "All scalar fields from completed formal training report")
        for arm in ("learned_residual", "fixed_random"):
            tag = f"delayed_r{index}_{arm}"
            take(tag + "_config", root + arm + "/explorer.json", scalar_fields,
                 "All scalar explorer configuration fields; no source lock maps")
            acquisition = (RUNS + "delayed_exploration_20260912/queue/stage_0_result/round_000/"
                           if index == 0 and arm == "learned_residual" else root) + arm + "/arrivals/"
            take(tag + "_acquisition", acquisition + "status.json")
            take(tag + "_training_metrics", acquisition + "training_metrics.json")
            for endpoint in ("initial", "final"):
                take(tag + "_" + endpoint, acquisition + endpoint + "_episodes.json",
                     lambda value: {**scalar_fields(value), "episodes": [
                         scalar_fields(row) for row in value["episodes"]]},
                     "All episodes and scalar timing/cost fields; per-step cell lists omitted")
            take(tag + "_before_pool", root + arm + "/before_learning/pool.json", pool_fields,
                 "All candidate keys, physical cells, status, phase, discovery round and label histories")
            after = root + arm + "/after_learning/pool.json" if index == 0 else COMPLETE + arm + "/pool.json"
            take(tag + "_after_pool", after, pool_fields,
                 "All candidate keys, physical cells, status, phase, discovery round and label histories")
    for name, training, exploration in (
        ("early", RUNS + "multicheckpoint_pi6_20260911/", RUNS + "checkpoint_discovery_20260911/"),
        ("late", RUNS + "multicheckpoint_pi6_256k_20260911/", RUNS + "multicheckpoint_pi6_256k_20260911/exploration/")):
        take(name + "_training_summary", training + "summary.json")
        take(name + "_exploration_summary", exploration + "summary.json",
             lambda value: {**scalar_fields(value), "analysis": {
                 key: value["analysis"][key] for key in ("arm_metrics", "equal_budget", "common_budgets",
                    "baseline_name", "baseline_root_cells", "complete_evaluator_matrix", "interpretation")}},
             "Complete arm and equal-budget tables; baseline and scope; point/curve arrays omitted")
    take("off_target_engineering", RUNS + "per_policy_coverage_ppo_20260911/analysis_summary.json")
    take("corrected_residual_engineering", "JIT/runs/engineering/residual_suffix_ppo_20260911/analysis_summary.json")
    # Source semantics are retained verbatim in small line excerpts, not imported/executed.
    path = "JIT/src/jit_dvgc/exploration_training.py"
    raw = (REPO / path).read_bytes()
    lines = raw.decode().splitlines()
    matches = [{"line": i + 1, "text": line} for i, line in enumerate(lines)
               if "rollout('final',not random_control)" in line or "rollout('initial'" in line]
    assert any("not random_control" in item["text"] for item in matches)
    value = encoded(matches)
    (sources / "diagnostic_semantics.json").write_bytes(value)
    records["diagnostic_semantics"] = {"original_path": path, "original_sha256": sha(raw),
        "original_bytes": len(raw), "snapshot_path": "sources/diagnostic_semantics.json",
        "snapshot_sha256": sha(value), "snapshot_bytes": len(value),
        "projection": "Verbatim initial/final rollout calls and source line numbers"}
    head = (REPO / ".git/HEAD").read_text().strip()
    ref = head[5:] if head.startswith("ref: ") else None
    if ref:
        loose = REPO / ".git" / ref
        if loose.exists():
            head = loose.read_text().strip()
        else:
            head = next(line.split()[0] for line in (REPO / ".git/packed-refs").read_text().splitlines()
                        if line.endswith(" " + ref))
    manifest = {"schema": "jit_paper_source_snapshots_v1", "captured_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": head, "source_ref": ref, "repository": "/home/qy/DVGC",
        "original_artifacts_modified": False, "snapshot_scope": "Minimal exact field projections; no model or trajectory payloads",
        "sources": records}
    (sources / "manifest.json").write_bytes(encoded(manifest))


def rebuild(check=False):
    manifest = json.loads((HERE / "sources/manifest.json").read_text())
    data = {}
    for source_id, meta in manifest["sources"].items():
        raw = (HERE / meta["snapshot_path"]).read_bytes()
        assert sha(raw) == meta["snapshot_sha256"], ("snapshot hash mismatch", source_id)
        data[source_id] = (list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode())))
                           if meta.get("format") == "csv_gzip" else json.loads(raw))
    delayed, costs, historical, checkpoints, proposer_rows = [], [], [], [], []

    def cost(scope, stage, charged, source_ids, maximum=None, arm="", round_index="", active=None,
             padding=None, accounting="measured_completed", note=""):
        costs.append(dict(scope=scope, stage=stage, round_index=round_index, arm=arm,
            charged_interactions=charged, maximum_interactions=maximum,
            active_interactions=active, padding_interactions=padding, accounting=accounting,
            source_ids=";".join(source_ids), limitation=note))

    summary = data["historical_summary"]
    geometry = data["historical_geometry"]
    assert len(geometry) == manifest["sources"]["historical_geometry"]["rows"]
    assert len({row["root_cell"] for row in geometry if row["witnessed"] == "True"}) == summary["campaign_union_root_cells"]
    assert {row["witnessed"] for row in geometry}.issubset({"True", "False"})
    previous_cost, previous_cells = 0, summary["seed_root_cells"]
    historical.append(dict(round_index=-1, new_policy="inherited_seed", new_candidates=0,
        new_root_cells=0, cumulative_root_cells=previous_cells, new_interactions=0,
        cumulative_new_interactions=0, ppo_interactions=0, source_ids="historical_summary"))
    for row in summary["rounds"]:
        index = row["round"]
        report_id = f"historical_r{index}_training"
        report = data[report_id]
        assert report["status"] == "completed"
        ppo, panel = report["completed_training_transitions"], report["train_panel_interactions"]
        ledger_ids = [key for key in data if key.startswith(f"historical_r{index}_ledger")]
        groups = Counter()
        for key in ledger_ids:
            for attempt in data[key]["attempts"]:
                assert attempt["measured"]
                stage = ("acquisition" if attempt["stage"] == "acquire" else
                         "suffix_labels" if attempt["stage"].startswith("label_") else "other")
                groups[stage] += attempt["charged"]
        assert groups["other"] == 0
        assert ppo + panel + sum(groups.values()) == row["charged_interactions"] - previous_cost
        assert row["campaign_union_root_cells"] - previous_cells == row["novel_root_cells"]
        historical.append(dict(round_index=index, new_policy=row["policy"], new_candidates=row["candidate_count"],
            new_root_cells=row["novel_root_cells"], cumulative_root_cells=row["campaign_union_root_cells"],
            new_interactions=row["charged_interactions"] - previous_cost,
            cumulative_new_interactions=row["charged_interactions"], ppo_interactions=ppo,
            source_ids="historical_summary;" + report_id))
        for stage, count in (("policy_ppo", ppo), ("fixed_train_panel", panel)):
            cost("historical_campaign_new", stage, count, [report_id], round_index=index,
                 note="Historical landing semantics; shared ancestor and differing proposer schedules")
        for stage in ("acquisition", "suffix_labels"):
            cost("historical_campaign_new", stage, groups[stage], ledger_ids, round_index=index)
        for p in data[f"historical_r{index}_proposers"]:
            proposer_rows.append({"round_index": index, **{k: int(v) if k != "proposer" else v for k, v in p.items()},
                "source_ids": f"historical_r{index}_proposers"})
        previous_cost, previous_cells = row["charged_interactions"], row["campaign_union_root_cells"]
    assert previous_cost == summary["charged_interactions"]
    assert summary["inherited_recorded_charge"] + previous_cost == summary["total_including_inherited_recorded_charge"]
    cost("historical_inherited_recorded", "inherited_recorded", summary["inherited_recorded_charge"],
         ["historical_summary"], accounting="historical_recorded_inherited",
         note="Not complete bootstrap lifecycle cost; do not treat as new campaign cost")

    for index in range(2):
        training = data[f"delayed_r{index}_training"]
        training_config = data[f"delayed_r{index}_training_config"]
        assert training["status"] == "completed"
        for stage, field in (("policy_ppo", "completed_training_transitions"), ("fixed_train_panel", "train_panel_interactions")):
            cost("delayed_pilot", stage, training[field], [f"delayed_r{index}_training"], round_index=index,
                 note="Shared next policy trained from learned-arm pending only; not independent arm training")
        for arm in ("learned_residual", "fixed_random"):
            tag = f"delayed_r{index}_{arm}"
            acquired = data[tag + "_acquisition"]
            config = data[tag + "_config"]
            before = {row["key"]: row for row in data[tag + "_before_pool"]["entries"]}
            after = {row["key"]: row for row in data[tag + "_after_pool"]["entries"]}
            assert before.keys() == after.keys()
            assert acquired["phase"] == "completed"
            assert acquired["active_interactions"] + acquired["padding_interactions"] == acquired["forward_interactions"]
            counts = Counter()
            for key, old in before.items():
                new = after[key]
                assert old["root_cell"] == new["root_cell"]
                if old["status"] == "witnessed": counts["previously_witnessed"] += 1
                elif new["status"] != "witnessed": counts["still_pending"] += 1
                elif old["observation_labels"][-1] == 0: counts["previous_bank_no_witness_to_witness"] += 1
                else: counts["unknown_to_witness"] += 1
            witnessed = [row for row in after.values() if row["status"] == "witnessed"]
            pending = [row for row in after.values() if row["status"] == "pending"]
            metrics = data[tag + "_training_metrics"]
            final = data[tag + "_final"]
            successes = sum(row["success"] and not row["physical_failure"] and row["completed"] and row["finite"] for row in final["episodes"])
            assert successes == acquired["final_successes"]
            delayed.append(dict(round_index=index, arm=arm, source_proposer=config["proposer"],
                exploration_seed=config["seed"], shared_policy_training_seed=training_config["ppo"]["seed"],
                candidate_contexts=len(after), newly_acquired_contexts=sum(row["first_discovery_round"] == index for row in after.values()),
                witnessed_contexts=len(witnessed), verified_root_cells=len({row["root_cell"] for row in witnessed}),
                pending_contexts=len(pending), pending_root_cells=len({row["root_cell"] for row in pending}),
                previous_bank_no_witness_to_witness=counts["previous_bank_no_witness_to_witness"],
                unknown_to_witness=counts["unknown_to_witness"],
                forward_scheduled_interactions=acquired["forward_interactions"],
                forward_active_interactions=acquired["active_interactions"],
                forward_padding_interactions=acquired["padding_interactions"],
                training_arrival_credit=acquired["training_new_cells"],
                final_diagnostic_arrival_credit=acquired["final_diagnostic_new_cells"],
                initial_baseline_cells=acquired["initial_baseline_cells"],
                residual_optimizer_updates=sum(row["optimizer_updates"] for row in metrics),
                training_episodes=sum(row["episodes"] for row in metrics),
                training_landing_successes=sum(row["successes"] for row in metrics),
                final_diagnostic_episodes=len(final["episodes"]), final_diagnostic_successes=successes,
                final_diagnostic_deterministic=arm == "learned_residual", independent_repetitions=False,
                source_ids=";".join(tag + suffix for suffix in ("_before_pool", "_after_pool", "_acquisition", "_training_metrics", "_config", "_final"))))
            reused = index == 0 and arm == "learned_residual"
            cost("delayed_pilot", "forward", acquired["forward_interactions"], [tag + "_acquisition"],
                 maximum=acquired["forward_interactions"], round_index=index, arm=arm,
                 active=acquired["active_interactions"], padding=acquired["padding_interactions"],
                 accounting="inherited_reused_once" if reused else "measured_completed",
                 note="Includes initial/final diagnostics and inactive padding; inherited arrival charged once")
    for row in data["delayed_reuse_status"]["costs"]:
        if row["stage"].endswith(("/before_learning", "/after_learning")):
            rnd, arm, when = row["stage"].split("/")
            cost("delayed_pilot", "suffix_" + when, row["charged_interactions"], ["delayed_reuse_status"],
                 maximum=row["maximum_interactions"], round_index=int(rnd[-3:]), arm=arm)
    for row in data["delayed_completion_summary"]["costs"]:
        if row["stage"].endswith("_delayed_suffix"):
            arm = row["stage"].removesuffix("_delayed_suffix")
            cost("delayed_pilot", "suffix_after_learning", row["charged_interactions"],
                 ["delayed_completion_summary"], round_index=1, arm=arm)
    latest = data["delayed_completion_summary"]
    assert latest["completed_second_round"] and data["delayed_completion_status"]["phase"] == "completed"
    assert sum(row["charged_interactions"] for row in latest["costs"]) == latest["new_interactions"]
    for row in latest["metrics"]:
        derived = next(item for item in delayed if item["round_index"] == 1 and item["arm"] == row["arm"])
        for field in ("verified_root_cells", "pending_root_cells", "previous_bank_no_witness_to_witness", "unknown_to_witness"):
            assert derived[field] == row[field], field

    for name in ("early", "late"):
        training = data[name + "_training_summary"]
        exploration = data[name + "_exploration_summary"]
        assert training["status"] == exploration["status"] == "completed"
        analysis = exploration["analysis"]
        for row in analysis["arm_metrics"]:
            common = next(eq for eq in analysis["equal_budget"] if eq["arm"] == row["arm"] and eq["scenario"] == "exploration_only")
            inclusive = next(eq for eq in analysis["equal_budget"] if eq["arm"] == row["arm"] and eq["scenario"] == "training_inclusive")
            checkpoints.append({"experiment": name, **row,
                "common_exploration_budget": common["common_budget"],
                "common_budget_spent": common["interactions"],
                "common_budget_novel_root_cells": common["novel_root_cells"],
                "common_training_inclusive_budget": inclusive["common_budget"],
                "common_training_inclusive_novel_cells": inclusive["novel_root_cells"],
                "common_budget_below_training_surcharge": inclusive["budget_below_training_surcharge"],
                "baseline_root_cells": analysis["baseline_root_cells"],
                "source_ids": name + "_exploration_summary"})
        assert sum(row["charged_interactions"] for row in analysis["arm_metrics"]) == exploration["charged_interactions"]
        cost("checkpoint_" + name, "training_and_panels", training["charged_interactions"],
             [name + "_training_summary"], maximum=training["maximum_total_interactions"],
             note="One shared training lineage; never sum checkpoint-prefix surcharges")
        for row in analysis["arm_metrics"]:
            cost("checkpoint_" + name, "exploration", row["charged_interactions"],
                 [name + "_exploration_summary"], arm=row["arm"], note="Exploration acquisition plus suffix cost")
    off = data["off_target_engineering"]
    for stage, field in (("failed_preflight", "failed_preflight_reservation"), ("passed_preflight", "passed_preflight_scheduled"), ("pilot", "pilot_scheduled")):
        cost("off_target_full_action_engineering", stage, off["cost"][field], ["off_target_engineering"],
             active=off["status"]["active_interactions"] if stage == "pilot" else None,
             padding=off["status"]["padding_interactions"] if stage == "pilot" else None,
             accounting="failed_reservation" if stage == "failed_preflight" else "scheduled_slots",
             note="Off-target full-action policy, not corrected residual; deterministic success collapsed")
    corrected = data["corrected_residual_engineering"]["status"]
    cost("corrected_residual_engineering", "forward", corrected["forward_interactions"], ["corrected_residual_engineering"],
         active=corrected["active_interactions"], padding=corrected["padding_interactions"],
         note="One optimizer update; engineering wiring only")
    cost("corrected_residual_engineering", "suffix", corrected["suffix_interactions"], ["corrected_residual_engineering"])

    totals = {scope: sum(row["charged_interactions"] for row in costs if row["scope"] == scope)
              for scope in sorted({row["scope"] for row in costs})}
    inherited = sum(row.get("inherited_interactions", 0) for row in data["delayed_reuse_status"]["costs"])
    assert totals["delayed_pilot"] == data["delayed_reuse_status"]["charged_interactions"] + latest["new_interactions"] + inherited
    assert totals["off_target_full_action_engineering"] == off["cost"]["total_charged"]
    assert totals["corrected_residual_engineering"] == corrected["charged_interactions"]
    tables = {"delayed_rounds.csv": delayed, "cost_breakdown.csv": costs,
              "historical_campaign.csv": historical, "historical_proposers.csv": proposer_rows,
              "checkpoint_comparison.csv": checkpoints, "geometry_points.csv": geometry}
    limitations = {
        "all_data": ["TRAIN development; final TEST unused", "Correlated states and shared policy ancestry; not independent replicates",
            "Root cells are discrete high-dimensional quantization, not continuous reachable volume or a safety certificate",
            "Recorded source HEAD identifies audit checkout, not every historical experiment revision; original artifact hashes retained"],
        "delayed_rounds.csv": ["Cumulative arm pools; candidate contexts differ from deduplicated physical cells",
            "Only learned-arm pending trains shared next pi; random is conditional exploration control, not independent lifecycle baseline",
            "Learned final diagnostic deterministic; random final diagnostic stochastic; success counts are not matched-mode estimates",
            "Arrival reward is provisional and almost saturates selected-candidate ceiling; not verified coverage improvement",
            "No bank-failure-to-witness or unknown-to-witness conversion in either round",
            "Old reuse error is immutable gate timeout before second PPO; fresh completion artifact completes remaining work"],
        "historical_campaign.csv": ["Campaign-scoped inherited baseline, not global history deduplication",
            "Historical landing semantics retain four conflicting forward receipts; no silent quarantine relabel",
            "Different proposer panels and accumulated schedules; total cell gain not attributable solely to newest PPO"],
        "geometry_points.csv": ["Every original row retained, including unwitnessed candidates; no example selection",
            "Historical witnessed boolean preserved; no silent application of later conflict-quarantine semantics",
            "Projected physical coordinates omit closed-loop context; use identifiers for provenance, not state equality claims",
            "Observed points and gaps only; do not fill hulls or claim a continuous reachable envelope"],
        "checkpoint_comparison.csv": ["Matched perturbation schedules and common evaluator bank within each experiment",
            "Common budget results are conservative offline candidate-order truncation, not new real-time matched-budget runs",
            "Unknowns retained; first-success does not produce a full evaluator matrix",
            "Training-inclusive cost not amortized; early and late training are fresh separate runs"],
        "cost_breakdown.csv": ["Rows are additive within scope; do not add checkpoint prefix surcharges from comparison CSV",
            "Inherited delayed forward work appears once, as inherited_reused_once",
            "Group totals do not constitute complete project lifecycle accounting; older bootstrap and other engineering runs excluded",
            "Blank maximum/active/padding fields mean unavailable or not separately assigned, never zero",
            "Full-action engineering is off-target; corrected residual is one-update wiring evidence"]}
    evidence = {"schema": "jit_paper_evidence_v1", "captured_utc": manifest["captured_utc"],
        "source_head": manifest["source_head"], "source_ref": manifest["source_ref"],
        "sources": manifest["sources"], "table_rows": {name: len(rows) for name, rows in tables.items()},
        "scope_interaction_totals": totals, "limitations": limitations,
        "grain": {"delayed_rounds.csv": "One row per round and arm; pool counts cumulative, forward costs incremental",
            "cost_breakdown.csv": "One additive charged component per scope/stage/round/arm",
            "historical_campaign.csv": "Inherited starting point then one row per completed production round",
            "historical_proposers.csv": "One cumulative proposer row per production round",
            "geometry_points.csv": "One original historical candidate context per row; repeated physical cells retained",
            "checkpoint_comparison.csv": "One proposer/checkpoint arm per early or late matched schedule"},
        "verification": {"source_snapshots_hash_checked": True, "cost_sums_asserted": True,
            "candidate_key_and_cell_continuity_asserted": True, "latest_summary_cross_checked": True,
            "original_artifacts_changed": False, "new_simulation_interactions": 0,
            "method": "Read JSON/CSV artifacts and source excerpt; no model loading, GPU, or Git commands"}}
    outputs = {"evidence.json": encoded(evidence)}
    for name, rows in tables.items():
        buf = io.StringIO(newline="")
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        outputs[name] = buf.getvalue().encode()
    for name, contents in outputs.items():
        if check:
            assert (HERE / name).read_bytes() == contents, "Derived artifact differs: " + name
        else:
            (HERE / name).write_bytes(contents)
    print(json.dumps({"mode": "check" if check else "rebuild", "source_snapshots": len(data),
                      "tables": evidence["table_rows"], "scope_interaction_totals": totals}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.capture:
        assert not args.check, "Capture and check are separate operations"
        capture()
    rebuild(args.check)
