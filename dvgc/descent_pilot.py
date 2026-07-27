"""Evaluation and sidecar helpers for the bounded unified Descent RSI pilot."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.provisional_descent import FEATURE_NAMES, state_identity
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference


REWARD_KEYS = (
    "reward/descent_rsi_pilot_shaping", "reward/descent_rsi_pilot_survival",
    "reward/descent_rsi_pilot_roll", "reward/descent_rsi_pilot_pitch",
    "reward/descent_rsi_pilot_angular", "reward/descent_rsi_pilot_vz",
    "reward/descent_rsi_pilot_action_smooth_penalty",
    "reward/descent_rsi_pilot_action_magnitude_penalty",
    "reward/descent_rsi_pilot_failure_penalty",
)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"states": 0}
    survival = {str(h): sum(int(row["survived_ticks"]) >= h for row in rows)
                for h in (8, 16, 24)}
    ticks = np.asarray([row["survived_ticks"] for row in rows], np.float64)
    actions = np.concatenate([np.asarray(row["actions"], np.float64) for row in rows], axis=0)
    reward = defaultdict(float)
    failures = Counter(row["termination_reason"] for row in rows)
    for row in rows:
        for key, value in row["reward_components"].items():
            reward[key] += float(value)
    return {
        "states": len(rows), "survival_counts": survival,
        "survival_rates": {key: value / len(rows) for key, value in survival.items()},
        "time_to_failure": {
            "median": float(np.median(ticks)), "lower_quartile": float(np.quantile(ticks, .25)),
            "min": int(ticks.min()), "max": int(ticks.max()),
        },
        "minimum_margins": {
            key: float(min(row["minimum_margins"][key] for row in rows))
            for key in rows[0]["minimum_margins"]
        },
        "failure_reasons": dict(sorted(failures.items())),
        "action": {
            "mean": actions.mean(axis=0).tolist(), "std": actions.std(axis=0).tolist(),
            "saturation_fraction": float(np.mean(np.abs(actions) >= .95)),
            "max_abs": float(np.max(np.abs(actions))),
        },
        "reward_component_sum": dict(sorted(reward.items())),
    }


def evaluate(
    env: OrangeBikeDVGC, records: Sequence[Mapping[str, Any]], *, params: Any | None,
    seed: int, horizon: int = 24, policy_name: str = "policy",
) -> dict[str, Any]:
    """Deterministic pointwise evaluation without certification semantics."""
    step = jax.jit(env.step)
    inference: Callable | None = build_inference(env, params, deterministic=True) if params is not None else None
    geometry = GroundSupportSolver(env._config.xml_path)
    rows = []
    for index, record in enumerate(records):
        state = restore_snapshot(env, record, jax.random.PRNGKey(seed + index * 1000))
        actions, reward_terms = [], defaultdict(float)
        min_margin = {
            "roll_rad": float("inf"), "pitch_rad": float("inf"),
            "angular_rate": float("inf"), "body_clearance_m": float("inf"),
        }
        survived, reason = 0, "pilot_horizon_reached"
        for tick in range(1, int(horizon) + 1):
            if inference is None:
                action = np.zeros(env.action_size, np.float32)
            else:
                action, _ = inference(state.obs, jax.random.PRNGKey(seed + index * 1000 + tick))
                action = np.asarray(action, np.float32)
            actions.append(action)
            state = step(state, action)
            for key in REWARD_KEYS:
                reward_terms[key] += float(state.metrics[key])
            feature = np.asarray(env._physical_feature(state.data), np.float64)
            contact = geometry.measure(
                np.asarray(state.data.qpos), np.asarray(state.data.qvel), np.asarray(state.data.ctrl)
            )
            min_margin["roll_rad"] = min(min_margin["roll_rad"], np.deg2rad(float(env._config.max_roll_deg)) - abs(feature[3]))
            min_margin["pitch_rad"] = min(min_margin["pitch_rad"], np.deg2rad(float(env._config.max_pitch_deg)) - abs(feature[4]))
            min_margin["angular_rate"] = min(min_margin["angular_rate"], float(env._config.recovery_max_angvel) - np.linalg.norm(feature[9:12]))
            min_margin["body_clearance_m"] = min(min_margin["body_clearance_m"], float(contact["nonwheel"]))
            if not np.isfinite(feature).all():
                reason = "nonfinite"; break
            if float(state.done):
                reason = END_REASON.get(int(state.info["end_code"]), "unknown"); break
            survived = tick
        rows.append({
            "candidate_id": record["id"], "policy": policy_name,
            "provisional_label": record.get("provisional_label"),
            "descent_layer": record.get("descent_layer"),
            "candidate_cluster": record.get("candidate_cluster"),
            "survived_ticks": survived, "termination_reason": reason,
            "minimum_margins": min_margin, "actions": np.asarray(actions).tolist(),
            "reward_components": dict(reward_terms),
        })
    groups = {
        "overall": _summary(rows),
        "labels": {name: _summary([row for row in rows if row["provisional_label"] == name])
                   for name in ("provisional_core", "provisional_frontier")},
        "layers": {name: _summary([row for row in rows if row["descent_layer"] == name])
                   for name in ("early", "middle", "late")},
    }
    return {"policy": policy_name, "horizon": horizon, "rows": rows, "summary": groups}


def build_heldout(
    env: OrangeBikeDVGC, records: Sequence[Mapping[str, Any]], *, seed: int,
    feature_scale: Sequence[float], dedup_distance: float = .12, maximum: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create coherent held-out states by bounded short MJX continuations."""
    step = jax.jit(env.step); geometry = GroundSupportSolver(env._config.xml_path)
    scale = np.maximum(np.asarray(feature_scale, np.float64), 1e-8)
    base = np.asarray([row["physical_feature"] for row in records], np.float64)
    selected, rejected = [], Counter()
    parents = []
    for row in records:
        if row.get("candidate_cluster") not in {p.get("candidate_cluster") for p in parents}:
            parents.append(row)
    deltas = (
        np.asarray([0, 0, 0, .06], np.float32),
        np.asarray([0, 0, .04, 0], np.float32),
        np.asarray([.025, 0, 0, 0], np.float32),
        np.asarray([-.025, 0, 0, 0], np.float32),
    )
    for parent_index, parent in enumerate(parents):
        for delta_index, delta in enumerate(deltas):
            state = restore_snapshot(env, parent, jax.random.PRNGKey(seed + parent_index * 100 + delta_index))
            action = np.clip(np.asarray(parent["policy_state"]["last_action"], np.float32) + delta, -1, 1)
            failed = False
            for _ in range(2 + (delta_index % 2)):
                state = step(state, action)
                if float(state.done): failed = True; break
            if failed:
                rejected["short_continuation_failed"] += 1; continue
            row = env.snapshot_record(state, "flight")
            feature = np.asarray(row["physical_feature"], np.float64)
            contact = geometry.measure(row["qpos"], row["qvel"], row["ctrl"])
            if feature[8] >= 0 or int(row["oracle_phase"]) != 2:
                rejected["not_physical_descent"] += 1; continue
            if contact["body_contacts"] or contact["wheel_contacts"] or contact["nonwheel"] < .002:
                rejected["contact_or_clearance"] += 1; continue
            existing = np.concatenate([base, np.asarray([r["physical_feature"] for r in selected], np.float64)], axis=0) if selected else base
            nearest = float(np.min(np.linalg.norm((existing - feature) / scale, axis=1)))
            if nearest < float(dedup_distance):
                rejected["dedup"] += 1; continue
            row.update({
                "id": state_identity({**row, "candidate_schema": "descent_provisional_candidate_v1",
                    "provisional_label": "provisional_frontier", "descent_layer": parent["descent_layer"],
                    "candidate_source": "local_rsi_perturbation", "artifact_role": "proposal_support_bank",
                    "formal_tube_member": False, "formal_jel_member": False})[:32],
                "artifact_role": "descent_rsi_heldout_evaluation_sidecar",
                "heldout_only": True, "training_eligible": False,
                "source_parent_id": parent["id"], "candidate_cluster": parent.get("candidate_cluster"),
                "provisional_label": parent["provisional_label"], "descent_layer": parent["descent_layer"],
                "nearest_training_distance": nearest,
                "perturbation": {"action_delta": delta.tolist(), "ticks": 2 + (delta_index % 2)},
            })
            selected.append(row)
            if len(selected) >= maximum: break
        if len(selected) >= maximum: break
    report = {
        "status": "PASS" if selected else "INSUFFICIENT",
        "states": len(selected), "clusters": len({row["candidate_cluster"] for row in selected}),
        "parents": len({row["source_parent_id"] for row in selected}),
        "rejected": dict(rejected), "training_eligible": False,
    }
    return selected, report


def expansion_proposals(
    env: OrangeBikeDVGC, records: Sequence[Mapping[str, Any]], params: Any, *,
    seed: int, feature_scale: Sequence[float], dedup_distance: float = .12,
    maximum: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect legal final-policy states as a non-promoting analysis sidecar."""
    step=jax.jit(env.step); inference=build_inference(env,params,deterministic=True)
    geometry=GroundSupportSolver(env._config.xml_path);scale=np.maximum(np.asarray(feature_scale),1e-8)
    base=np.asarray([row["physical_feature"] for row in records]);selected=[];rejected=Counter()
    for index,parent in enumerate(records):
        state=restore_snapshot(env,parent,jax.random.PRNGKey(seed+index*1000))
        for tick in range(1,25):
            action,_=inference(state.obs,jax.random.PRNGKey(seed+index*1000+tick));state=step(state,action)
            if float(state.done): break
            if tick not in (8,16,24): continue
            row=env.snapshot_record(state,"flight");feature=np.asarray(row["physical_feature"],np.float64)
            contact=geometry.measure(row["qpos"],row["qvel"],row["ctrl"])
            if feature[8]>=0 or contact["body_contacts"] or contact["wheel_contacts"] or contact["nonwheel"]<.002:
                rejected["physical_legality"]+=1;continue
            existing=np.concatenate([base,np.asarray([r["physical_feature"] for r in selected])],axis=0) if selected else base
            nearest=float(np.min(np.linalg.norm((existing-feature)/scale,axis=1)))
            if nearest<dedup_distance:rejected["dedup"]+=1;continue
            row.update({"id":state_identity({**row,"candidate_schema":"descent_provisional_candidate_v1","provisional_label":"provisional_frontier","descent_layer":parent["descent_layer"],"candidate_source":"local_rsi_perturbation","artifact_role":"proposal_support_bank","formal_tube_member":False,"formal_jel_member":False})[:32],"artifact_role":"descent_candidate_expansion_proposals_v1","proposal_only":True,"formal_tube_member":False,"formal_jel_member":False,"source_parent_id":parent["id"],"proposal_tick":tick,"nearest_training_distance":nearest,"descent_layer":parent["descent_layer"],"provisional_label":"provisional_frontier"})
            selected.append(row)
            if len(selected)>=maximum:break
        if len(selected)>=maximum:break
    return selected,{"status":"PASS","proposal_states":len(selected),"clusters":len({r.get("source_parent_id") for r in selected}),"rejected":dict(rejected),"automatic_promotion":False}
