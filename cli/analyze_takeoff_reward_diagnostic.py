"""Check that Takeoff success dominates missed-liftoff reward before PPO."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dvgc.runtime import save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    payload = json.loads(Path(a.evaluation).read_text())
    controllers = {}
    passed = True
    for name in ("old_takeoff", "new_takeoff"):
        rows = [row for row in payload["outcomes"] if row["controller"] == name]
        success = [row for row in rows if row["success"]]
        missed = [row for row in rows if row["reason"] == "takeoff_missed_liftoff_deadline"]
        terms = sorted({key for row in rows for key in row["reward_breakdown"]})
        def stats(selected, term):
            values = [row["reward_breakdown"].get(term, 0.) for row in selected]
            return {"mean": float(np.mean(values)) if values else None,
                    "p95": float(np.quantile(values, .95)) if values else None,
                    "max": float(np.max(values)) if values else None}
        total_term = ("reward/stage_entry_total" if "reward/stage_entry_total" in terms
                      else "reward/total" if "reward/total" in terms else "reward")
        success_mean = stats(success, total_term)["mean"]
        missed_mean = stats(missed, total_term)["mean"]
        dominates = bool(success and missed and success_mean > missed_mean + 1.)
        event_values = [row["reward_breakdown"].get("reward/stage_entry_event", 0.)
                        for row in success]
        event_latched_once = bool(success and all(value > 0 for value in event_values))
        controller_pass = dominates and event_latched_once
        passed &= controller_pass
        controllers[name] = {
            "success_episodes": len(success), "missed_liftoff_episodes": len(missed),
            "success_return": stats(success, total_term),
            "missed_liftoff_return": stats(missed, total_term),
            "success_return_dominates": dominates,
            "success_event_present": event_latched_once,
            "term_statistics": {
                term: {"success": stats(success, term), "missed_liftoff": stats(missed, term)}
                for term in terms
            },
        }
    save_json(a.output, {
        "status": "PASS" if passed else "FAIL",
        "artifact_role": "takeoff_reward_pretraining_diagnostic",
        "controllers": controllers,
        "checks": {
            "success_dominates_missed_liftoff": passed,
            "compressed_holding_not_selected_by_return": passed,
            "success_event_is_present": passed,
        },
    })
    print(json.dumps({"status": "PASS" if passed else "FAIL",
                      "controllers": {k: {x: v[x] for x in (
                          "success_episodes", "missed_liftoff_episodes",
                          "success_return", "missed_liftoff_return",
                          "success_return_dominates", "success_event_present")}
                                      for k, v in controllers.items()}}, indent=2))


if __name__ == "__main__":
    main()
