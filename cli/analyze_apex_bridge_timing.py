"""Offline, event-aligned diagnosis of the exhausted Apex bridge search.

This consumes existing traces only.  The diagnostic classifiers are deliberately
small parent-held-out linear rankers and are not Tube or viability estimators.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


STATE_FEATURES = (
    "roll", "pitch", "vx", "vz", "wx", "wy", "wz", "hip", "knee",
)
FEATURE_INDEX = {
    "roll": 3, "pitch": 4, "vx": 6, "vz": 8, "wx": 9, "wy": 10,
    "wz": 11, "hip": 13, "knee": 14,
}


def _percentiles(values):
    if not values:
        return None
    a = np.asarray(values, float)
    return {
        "min": float(np.min(a)), "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)), "mean": float(np.mean(a)),
    }


def _first_action_tick(trace):
    for frame in trace:
        if np.max(np.abs(np.asarray(frame["action"], float))) > 1e-8:
            return int(frame["tick"])
    return None


def _first_divergence_tick(trace, *, max_roll, max_pitch, max_rate):
    """First clear loss of pose margin or excessive angular rate."""
    for frame in trace:
        roll_margin = max_roll - abs(float(frame["roll"]))
        pitch_margin = max_pitch - abs(float(frame["pitch"]))
        rates = np.asarray(frame["angular_velocity"], float)
        if (
            roll_margin <= .25 * max_roll
            or pitch_margin <= .25 * max_pitch
            or abs(rates[0]) > max_rate
            or abs(rates[1]) > max_rate
        ):
            return int(frame["tick"])
    return None


def _auc(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if not len(pos) or not len(neg):
        return None
    comparisons = score[pos, None] - score[None, neg]
    return float(np.mean((comparisons > 0) + .5 * (comparisons == 0)))


def _lopo_linear_ranker(x, labels, parents):
    x = np.asarray(x, float); parents = np.asarray(parents, object)
    result = {}
    for name, y_values in labels.items():
        y = np.asarray(y_values, float)
        predictions = np.zeros(len(y), float)
        folds = []
        for parent in sorted(set(parents.tolist())):
            test = parents == parent; train = ~test
            center = np.mean(x[train], axis=0)
            scale = np.std(x[train], axis=0)
            scale = np.where(scale < 1e-4, 1e-4, scale)
            xt = (x[train] - center) / scale
            xv = (x[test] - center) / scale
            xt = np.column_stack([np.ones(len(xt)), xt])
            xv = np.column_stack([np.ones(len(xv)), xv])
            regularizer = np.eye(xt.shape[1]) * .1
            regularizer[0, 0] = 0.
            weights = np.linalg.solve(xt.T @ xt + regularizer, xt.T @ y[train])
            pred = np.clip(xv @ weights, 0., 1.)
            predictions[test] = pred
            folds.append({
                "held_out_parent": str(parent), "branches": int(np.sum(test)),
                "prevalence": float(np.mean(y[test])),
                "brier": float(np.mean(np.square(pred - y[test]))),
            })
        result[name] = {
            "branches": len(y), "prevalence": float(np.mean(y)),
            "lopo_brier": float(np.mean(np.square(predictions - y))),
            "lopo_auc": _auc(y, predictions),
            "folds": folds,
        }
    return result


def _action_family(row):
    p = row["parameters"]
    def sign(value):
        return "positive" if value > .05 else "negative" if value < -.05 else "neutral"
    return f"coast_{p['coast']}__hip_{sign(p['hip'])}__knee_{sign(p['knee'])}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search-report", required=True)
    p.add_argument("--apex-bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stable-window", type=int, default=4)
    p.add_argument("--max-roll-deg", type=float, default=35.)
    p.add_argument("--max-pitch-deg", type=float, default=75.)
    p.add_argument("--max-angular-rate", type=float, default=4.)
    a = p.parse_args()

    search = json.loads(Path(a.search_report).read_text())
    bank = SnapshotBank.load(a.apex_bank)
    records = {row["id"]: row for row in bank.records}
    rows = [row for row in search["outcomes"] if row["dynamic_evidence"]]
    max_roll = math.radians(a.max_roll_deg)
    max_pitch = math.radians(a.max_pitch_deg)
    timing = []
    x, parents = [], []
    labels = {
        "survive_to_physical_descent": [],
        "roll_failure_before_descent": [],
        "pitch_failure_before_descent": [],
    }
    by_parent = defaultdict(list); by_action = defaultdict(list)
    for row in rows:
        trace = row["trace"]
        apex = row["vertical_velocity_zero_crossing_tick"]
        failure = row["physical_failure_tick"]
        first_negative = row["first_negative_vertical_velocity_tick"]
        stable = bool(
            apex is not None and first_negative is not None
            and (failure is None or failure - first_negative >= a.stable_window)
        )
        divergence = _first_divergence_tick(
            trace, max_roll=max_roll, max_pitch=max_pitch,
            max_rate=a.max_angular_rate,
        )
        action_start = _first_action_tick(trace)
        item = {
            "candidate_id": row["candidate_id"],
            "parent": row["independent_parent"],
            "failure_reason": row["failure_reason"],
            "t_apex_physical": apex, "t_first_negative_vz": first_negative,
            "t_failure": failure, "t_obvious_pose_divergence": divergence,
            "t_first_nonzero_action": action_start,
            "failure_minus_apex": (
                None if apex is None or failure is None else failure - apex
            ),
            "divergence_before_nonzero_action": bool(
                divergence is not None
                and (action_start is None or divergence < action_start)
            ),
            "stable_physical_descent_diagnostic": stable,
            "action_family": _action_family(row),
            "minimum_support_distance": row["minimum_support_distance"],
        }
        timing.append(item); by_parent[item["parent"]].append(item)
        by_action[item["action_family"]].append(item)
        feature = np.asarray(records[row["candidate_id"]]["physical_feature"], float)
        x.append([feature[FEATURE_INDEX[name]] for name in STATE_FEATURES])
        parents.append(item["parent"])
        labels["survive_to_physical_descent"].append(int(stable))
        labels["roll_failure_before_descent"].append(int(
            row["failure_reason"] == "roll_limit" and not stable
        ))
        labels["pitch_failure_before_descent"].append(int(
            row["failure_reason"] == "pitch_limit" and not stable
        ))

    def summary(group):
        latency = [r["failure_minus_apex"] for r in group
                   if r["failure_minus_apex"] is not None]
        return {
            "branches": len(group),
            "apex_crossed": sum(r["t_apex_physical"] is not None for r in group),
            "stable_physical_descent_diagnostic": sum(
                r["stable_physical_descent_diagnostic"] for r in group
            ),
            "stable_rate": float(np.mean([
                r["stable_physical_descent_diagnostic"] for r in group
            ])),
            "divergence_before_action_rate": float(np.mean([
                r["divergence_before_nonzero_action"] for r in group
            ])),
            "failure_minus_apex": _percentiles(latency),
            "failure_reasons": dict(Counter(r["failure_reason"] for r in group)),
        }

    action_summary = {
        name: summary(group) for name, group in by_action.items() if len(group) >= 10
    }
    action_rank = sorted(
        ({"action_family": name, **values} for name, values in action_summary.items()),
        key=lambda row: (-row["stable_rate"], -row["branches"], row["action_family"]),
    )
    parent_summary = {str(name): summary(group) for name, group in by_parent.items()}
    stable_y = np.asarray(labels["survive_to_physical_descent"], float)
    candidate_rates = defaultdict(list)
    for item, value in zip(timing, stable_y):
        candidate_rates[item["candidate_id"]].append(value)
    action_rates = defaultdict(list)
    for item, value in zip(timing, stable_y):
        action_rates[item["action_family"]].append(value)
    overall = float(np.mean(stable_y))
    candidate_between = float(np.mean([
        (np.mean(values) - overall) ** 2 for values in candidate_rates.values()
    ]))
    action_between = float(np.mean([
        (np.mean(values) - overall) ** 2 for values in action_rates.values()
    ]))
    payload = {
        "status": "PASS",
        "artifact_role": "offline_apex_bridge_failure_timing_diagnostic",
        "not_tube_or_viability_model": True,
        "inputs": {
            "search_report_sha256": file_sha256(a.search_report),
            "apex_bank_sha256": file_sha256(a.apex_bank),
        },
        "definitions": {
            "stable_physical_descent_diagnostic": (
                f"physical apex and negative vz observed, then survives at least "
                f"{a.stable_window} control ticks before pose failure"
            ),
            "obvious_pose_divergence": (
                "remaining roll/pitch gate margin <=25% or abs roll/pitch rate "
                f"> {a.max_angular_rate} rad/s"
            ),
        },
        "dynamic_branches": len(rows),
        "timing": {
            "apex_crossed": sum(r["t_apex_physical"] is not None for r in timing),
            "stable_physical_descent_diagnostic": int(np.sum(stable_y)),
            "divergence_before_nonzero_action": sum(
                r["divergence_before_nonzero_action"] for r in timing
            ),
            "failure_minus_apex": _percentiles([
                r["failure_minus_apex"] for r in timing
                if r["failure_minus_apex"] is not None
            ]),
        },
        "parent_summary": parent_summary,
        "action_family_ranking": action_rank,
        "initial_state_vs_action_effect": {
            "candidate_group_between_rate_variance": candidate_between,
            "action_family_between_rate_variance": action_between,
            "larger_observed_group_effect": (
                "initial_candidate" if candidate_between > action_between
                else "action_family"
            ),
        },
        "lopo_survival_rankers": _lopo_linear_ranker(x, labels, parents),
        "state_features": list(STATE_FEATURES),
        "best_candidate_ranking": sorted([
            {
                "candidate_id": candidate_id, "branches": len(values),
                "stable_rate": float(np.mean(values)),
                "parent": next(r["parent"] for r in timing
                               if r["candidate_id"] == candidate_id),
            }
            for candidate_id, values in candidate_rates.items()
        ], key=lambda row: (-row["stable_rate"], row["candidate_id"])),
        "trace_field_limitations": [
            "existing trace does not contain hip/knee velocity",
            "existing trace does not contain contact/wheel-unloading measurements",
            "those quantities require the separately bounded control-authority replay",
        ],
    }
    save_json(a.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
