"""Replay formal descent branches and classify first-contact handoff failures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import DYNAMICS_VARIANTS
from dvgc.composite import CanonicalEntryMatcher, CompositeSession
from dvgc.config import STAGE_ID, file_sha256, load_config
from dvgc.entry import ENTRY_FEATURE_NAMES, entry_feature_from_physical
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.research_semantics import classify_handoff, summarize_handoff
from dvgc.rollout import frozen_rollout, restore_snapshot
from dvgc.runtime import build_inference, save_json


def _events(report, comparison=None):
    previous = {}
    if comparison:
        for row in comparison["rows"]:
            for branch in row["branch_evidence"]:
                previous[(row["id"], branch["branch_index"])] = branch["terminal_cause"]
    events = []
    for row in report["rows"]:
        for branch in row["branch_evidence"]:
            key = (row["id"], branch["branch_index"])
            cause = branch["terminal_cause"]
            converted = previous.get(key) == "physical_failure" and cause == "handoff_missed_final"
            if cause == "handoff_missed_final" or converted:
                events.append({"candidate_id": row["id"], "candidate_index": row["candidate_index"],
                               "parent_id": row.get("parent"), "layer": row.get("layer"),
                               "branch": branch, "physical_to_handoff": converted,
                               "previous_terminal_cause": previous.get(key)})
    return events


def _label(branches, cfg):
    successes = sum(bool(row["final_recovery"]) for row in branches)
    failures = len(branches) - successes
    posterior = beta_posterior(successes, failures, alpha0=cfg.beta_alpha0,
                               beta0=cfg.beta_beta0, q_low=cfg.posterior_q_low,
                               q_high=cfg.posterior_q_high)
    label = posterior_label(posterior, len(branches), min_branches=cfg.min_branches,
                            safe_threshold=cfg.safe_threshold,
                            dead_threshold=cfg.dead_threshold,
                            boundary_max_width=cfg.boundary_max_width)
    return label, posterior


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-label", required=True)
    p.add_argument("--policy", required=True); p.add_argument("--stable-report", required=True)
    p.add_argument("--comparison-report", default="")
    p.add_argument("--candidate-bank", required=True); p.add_argument("--landing-policy", required=True)
    p.add_argument("--entry-bank", required=True); p.add_argument("--start-event", type=int, required=True)
    p.add_argument("--end-event", type=int, required=True); p.add_argument("--landing-seed", type=int, required=True)
    p.add_argument("--output", required=True); p.add_argument("--proposal-bank", required=True)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args(); output, proposal_path = Path(a.output), Path(a.proposal_bank)
    if output.exists() or proposal_path.exists(): raise SystemExit("Handoff shard output already exists")
    report = json.loads(Path(a.stable_report).read_text())
    comparison = json.loads(Path(a.comparison_report).read_text()) if a.comparison_report else None
    events = _events(report, comparison)
    if not (0 <= a.start_event < a.end_event <= len(events)): raise SystemExit("Invalid event slice")
    params, cfg_dict, manifest = load_bundle(a.policy, verify_files=True)
    landing_params, landing_cfg_dict, landing_manifest = load_bundle(a.landing_policy, verify_files=True)
    bank = SnapshotBank.load(a.candidate_bank); candidates = {r["id"]: r for r in bank.records_for_phase("flight", include_training_only=False)}
    entry = SnapshotBank.load(a.entry_bank); safe = entry.records_for_phase("landing", final_labels=["safe"], include_training_only=False)
    base = load_config(a.config, {**cfg_dict, "training_stage": "flight", "expert_chain_termination": False,
                                 "domain_randomization": False, "obs_noise_enable": False, "use_bank_resets": False})
    lbase = load_config(a.config, {**landing_cfg_dict, "training_stage": "landing",
                                  "domain_randomization": False, "obs_noise_enable": False, "use_bank_resets": False})
    flight_variants = {}; landing_variants = []
    for spec in DYNAMICS_VARIANTS:
        override = {k: v for k, v in spec.items() if k != "id"}
        cfg = load_config(a.config, {**base.to_dict(), **override})
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=entry)
        flight_variants[spec["id"]] = (env, jax.jit(env.step), {
            "flight": build_inference(env, params, deterministic=True),
            "landing": build_inference(env, landing_params, deterministic=True),
        }, CanonicalEntryMatcher(env, "flight", a.entry_bank))
        lcfg = load_config(a.config, {**lbase.to_dict(), **override})
        lenv = OrangeBikeDVGC(lcfg, snapshot_bank=SnapshotBank())
        landing_variants.append((spec["id"], lenv, jax.jit(lenv.step),
                                 build_inference(lenv, landing_params, deterministic=True)))
    results, proposals = [], []
    for event_index in range(a.start_event, a.end_event):
        event = events[event_index]; branch = event["branch"]; record = candidates[event["candidate_id"]]
        env, step_fn, inference, matcher = flight_variants[branch["dynamics_variant"]]
        key = jax.random.PRNGKey(int(branch["branch_seed"])); state = restore_snapshot(env, record, key)
        session = CompositeSession(env, ("flight", "landing"), inference, {"flight": matcher}, state, key)
        first_contact = None; contact_step = None; trace = []; early_match = False
        for step in range(int(base.branch_horizon)):
            state = session.step(step_fn=step_fn, action_noise_std=float(base.action_noise_std))
            feature = np.asarray(jax.device_get(env._landing_entry_feature(
                state.data, state.info["had_valid_landing"] > 0, state.info["contact_age"] > 0,
                state.info["landing_entry_age"])), np.float64)
            z = (feature - matcher.center) / matcher.scale
            delta = matcher.features - z[None, :]; sq = delta * delta
            nearest = int(np.argmin(np.sum(sq, axis=1))); distance = float(np.sqrt(np.sum(sq[nearest])))
            valid_contact = bool(np.asarray(jax.device_get(state.metrics["event/landing"])))
            matched_now = bool(session.handoffs and session.handoffs[-1]["step"] == step + 1)
            if matched_now and first_contact is None: early_match = True
            if first_contact is None and valid_contact:
                first_contact = env.snapshot_record(state, "landing"); contact_step = step + 1
            trace.append({"step": step + 1, "valid_contact": valid_contact, "matched": matched_now,
                          "distance": distance, "nearest_entry_id": safe[nearest]["id"],
                          "squared_distance_contribution": sq[nearest].tolist()})
            if bool(np.asarray(jax.device_get(state.done))): break
        replay_final = bool(np.asarray(jax.device_get(session.state.info.get("recovery_success", 0))))
        replay_chain = bool(session.handoffs)
        landing_evidence = []
        if first_contact is not None:
            for b in range(int(lbase.max_branches)):
                variant_id, lenv, lstep, linfer = landing_variants[b % len(landing_variants)]
                seed = int(a.landing_seed) + event_index * 10_000 + b
                lkey = jax.random.PRNGKey(seed)
                _, outcome = frozen_rollout(lenv, linfer, restore_snapshot(lenv, first_contact, lkey), lkey,
                                            horizon=int(lbase.branch_horizon), step_fn=lstep,
                                            action_noise_std=float(lbase.action_noise_std))
                landing_evidence.append({"branch_index": b, "branch_seed": seed,
                    "seed_namespace": f"handoff_decomposition:{a.policy_label}:landing_recovery",
                    "dynamics_variant": variant_id, "final_recovery": bool(outcome["final"]),
                    "termination_reason": END_REASON.get(int(outcome["end_code"]), "unknown")})
                label, posterior = _label(landing_evidence, lbase)
                if b + 1 >= int(lbase.min_branches) and label in ("dead", "boundary"): break
                if b + 1 >= int(lbase.max_branches): break
        else:
            label, posterior = "unknown", None
        if landing_evidence: label, posterior = _label(landing_evidence, lbase)
        phase_ok = first_contact is not None and int(first_contact["oracle_phase"]) == STAGE_ID["landing"]
        replay_ok = replay_final and not replay_chain
        event_order_valid = phase_ok and not early_match and replay_ok
        handoff_class = classify_handoff(has_contact=first_contact is not None,
                                         event_order_valid=event_order_valid,
                                         matched_c_l=replay_chain, landing_label=label)
        contact_trace = [row for row in trace if contact_step is not None and
                         contact_step - 2 <= row["step"] <= contact_step + int(base.landing_entry_window_steps)]
        result = {"policy_label": a.policy_label, "policy_version": manifest["policy_version"],
                  "event_index": event_index, **{k: event[k] for k in ("candidate_id", "candidate_index", "parent_id", "layer", "physical_to_handoff", "previous_terminal_cause")},
                  "branch_index": branch["branch_index"], "formal_branch_seed": branch["branch_seed"],
                  "dynamics_variant": branch["dynamics_variant"], "original_terminal_cause": branch["terminal_cause"],
                  "first_valid_contact_step": contact_step, "oracle_phase": None if first_contact is None else next((name for name, value in STAGE_ID.items() if value == int(first_contact["oracle_phase"])), "unknown"),
                  "matched_c_l": replay_chain, "first_match_step": session.handoffs[0]["step"] if session.handoffs else None,
                  "replay_final": replay_final, "replay_chain": replay_chain, "event_order_valid": event_order_valid,
                  "landing_policy_label": label, "landing_policy_posterior": posterior,
                  "landing_policy_branches": landing_evidence, "handoff_class": handoff_class,
                  "minimum_distance": min((x["distance"] for x in trace), default=None),
                  "contact_window": contact_trace, "termination_reason": END_REASON.get(int(np.asarray(jax.device_get(session.state.info["end_code"]))), "unknown")}
        results.append(result)
        if handoff_class == "H1":
            proposal = first_contact
            entry_feature = entry_feature_from_physical(first_contact["physical_feature"], valid_landing=True,
                                                       support=True, contact_age=first_contact["contact_age"], cfg=base)
            proposal.update({"id": hashlib.sha256(f"handoff:{a.policy_label}:{event_index}".encode()).hexdigest()[:32],
                             "candidate_kind": "pending_landing_entry_proposal", "entry_feature": entry_feature.astype(np.float32),
                             "entry_source_id": event["candidate_id"], "entry_source_parent": event.get("parent_id"),
                             "entry_source_policy": manifest["policy_version"], "entry_capture_mode": "formal_branch_first_valid_contact",
                             "entry_construction_seed": int(branch["branch_seed"]), "bootstrap_eligible": True})
            proposals.append(proposal)
        print(f"[handoff {a.policy_label}] {event_index + 1}/{len(events)} {handoff_class}", flush=True)
    metadata = {"artifact_role": "pending_entry_proposals", "active_for_matching": False,
                "policy_label": a.policy_label, "policy_hash": file_sha256(Path(a.policy) / "params.pkl"),
                "landing_policy_hash": file_sha256(Path(a.landing_policy) / "params.pkl"),
                "entry_bank_sha256": file_sha256(a.entry_bank), "entry_matcher_radius": entry.metadata["entry_matcher"]["radius"],
                "candidate_bank_sha256": file_sha256(a.candidate_bank), "xml_sha256": entry.metadata["xml_sha256"],
                "landing_recovery_seed": int(a.landing_seed), "entry_feature_names": ENTRY_FEATURE_NAMES}
    SnapshotBank(proposals, metadata).save(proposal_path)
    payload = {"status": "PASS", "artifact_role": "handoff_decomposition_shard",
               "policy_label": a.policy_label, "policy_hash": metadata["policy_hash"],
               "stable_report_sha256": file_sha256(a.stable_report), "candidate_bank_sha256": metadata["candidate_bank_sha256"],
               "entry_bank_sha256": metadata["entry_bank_sha256"], "matcher_radius": metadata["entry_matcher_radius"],
               "start_event": a.start_event, "end_event": a.end_event, "total_events": len(events),
               "summary": summarize_handoff(results), "proposal_count": len(proposals), "rows": results}
    save_json(output, payload); print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
