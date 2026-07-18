"""Invalidate the stale self-comparison gate and resume fresh post-PPO Cycle 5."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dvgc.config import file_sha256
from dvgc.runtime import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);a=p.parse_args();run=a.run
    state_path=run/"controller_state.json";state=json.loads(state_path.read_text())
    if state.get("current_stage")!="gate_pause" or state.get("active_worker_unit"):
        raise SystemExit("Cycle-5 resume requires an inactive gate-pause controller")
    before=run/"cycle_4/stable/report.json";stale_gate=run/"cycle_4/roll_targeted_block_gate.json"
    train=run/"roll_targeted/train/report.json";policy=run/"roll_targeted/train/policy";checkpoint=run/"roll_targeted/train/orbax/000000102400"
    candidate=run/"trajectory_mining_corrected/candidate_pool.pkl";required=(before,stale_gate,train,policy/"params.pkl",checkpoint,candidate)
    missing=[str(path) for path in required if not path.exists()]
    if missing:raise SystemExit(f"Cycle-5 resume input missing: {missing}")
    before_data=json.loads(before.read_text());gate=json.loads(stale_gate.read_text());training=json.loads(train.read_text())
    new_hash=file_sha256(policy/"params.pkl");candidate_hash=file_sha256(candidate)
    checks={"before_policy_is_frozen":before_data.get("policy_hash")=="52721668eed0cc78b41a45ad7c319e687f43add8977f2b4bdfcad8208c4353f2",
        "new_policy_matches_training":training.get("policy_hash")==new_hash,
        "new_policy_expected":new_hash=="da9bd4865dfc197c59fd5091a32bc6d99f6d07e9bdb18641463dc8e14224fdce",
        "candidate_expected":candidate_hash=="d031e9677827e1c9d3e8cdc8addcd38cfc57834e4d452d128969dbf538e20e55",
        "ppo_healthy":training.get("status")=="PASS" and training.get("cumulative_effective_steps")==102400
            and not training.get("health",{}).get("oom") and not training.get("health",{}).get("timeout")
            and not training.get("health",{}).get("nonfinite_metric_keys"),
        "stale_gate_self_compared":gate.get("before_report_sha256")==gate.get("after_report_sha256"),
        "cycle5_absent":not (run/"cycle_5").exists()}
    if not all(checks.values()):raise SystemExit(f"Cycle-5 resume preflight failed: {checks}")
    lifecycle=run/"cycle_4/roll_targeted_block_gate.lifecycle.json"
    payload={"status":"INVALID_ENGINEERING_STALE_POSTTRAIN_REPORT","eligible_for_research_conclusion":False,
        "reason":"Gate compared the Cycle-4 pre-PPO stable report with itself; the post-PPO policy was never certified.",
        "checks":checks,"stale_gate":str(stale_gate),"stale_gate_sha256":file_sha256(stale_gate),
        "before_report":str(before),"before_report_sha256":file_sha256(before),"before_policy_hash":before_data["policy_hash"],
        "posttrain_policy_hash":new_hash,"posttrain_checkpoint":str(checkpoint),"marked_at":time.time()}
    if lifecycle.exists() and json.loads(lifecycle.read_text())!=payload:raise SystemExit("Existing stale-gate lifecycle marker changed")
    if not lifecycle.exists():save_json(lifecycle,payload)
    history=list(state.get("history",[]));history.append({"action":"invalidate_stale_posttrain_gate","completed_at":time.time(),"outputs":[str(lifecycle)]})
    state.update({"controller_version":3,"current_stage":"stable_stage_a","current_cycle":5,"route_phase":"roll_targeted",
        "current_candidate":str(candidate),"current_policy":str(policy),"current_checkpoint":str(checkpoint),
        "current_cumulative_steps":102400,"pre_roll_cycle":4,"pre_roll_policy_hash":before_data["policy_hash"],
        "pre_roll_stable_report":str(before),"last_completed_action":"single_roll_targeted_ppo",
        "in_progress_action":None,"expected_outputs":[],"next_decision":"stable_stage_b","stop_reason":None,
        "active_worker_unit":None,"retry_count":0,"failure_signature":None,"consecutive_failure_count":0,
        "research_gate_valid":False,"invalid_stale_gate_marker":str(lifecycle),"heartbeat":time.time(),"history":history})
    state.setdefault("provenance",{}).update({"current_policy_hash":new_hash,"candidate_bank_sha256":candidate_hash})
    save_json(state_path,state);print(json.dumps({"status":"PASS","cycle":5,"policy_hash":new_hash,
        "candidate_bank_sha256":candidate_hash,"invalid_gate_marker":str(lifecycle),"next_stage":"stable_stage_a"},indent=2))


if __name__=="__main__":main()
