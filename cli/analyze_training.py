"""Create a compact structured health and learning report for one PPO run."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.runtime import save_json


def _nonfinite_paths(value, path=""):
    found=[]
    if isinstance(value,dict):
        for key,item in value.items(): found.extend(_nonfinite_paths(item,f"{path}/{key}"))
    elif isinstance(value,list):
        for index,item in enumerate(value): found.extend(_nonfinite_paths(item,f"{path}/{index}"))
    elif isinstance(value,float) and not math.isfinite(value): found.append(path)
    return found


def _series(rows,key):
    return [{"step":int(row["step"]),"value":float(row[key])} for row in rows if key in row]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True); parser.add_argument("--output",default="")
    parser.add_argument("--console-log",default="")
    args=parser.parse_args(); run=Path(args.run); metrics_path=run/"training_metrics.json"
    payload=json.loads(metrics_path.read_text(encoding="utf-8")); rows=payload.get("progress",[])
    eval_rows=[row for row in rows if "eval/episode_reward" in row]; last=eval_rows[-1] if eval_rows else {}
    stage=str(payload.get("stage","")); end_prefix="eval/episode_end/"
    reasons={key[len(end_prefix):]:float(value) for key,value in last.items() if key.startswith(end_prefix) and not key.endswith("_std")}
    final_rate=float(reasons.get("recovery",0.0)); timeout_rate=float(reasons.get("stage_timeout",0.0))
    physical_rate=sum(value for key,value in reasons.items() if key not in ("recovery","stage_timeout"))
    reward_terms={key.removeprefix("eval/episode_reward/"):float(value) for key,value in last.items() if key.startswith("eval/episode_reward/") and not key.endswith("_std")}
    bank_path=payload.get("reset_protocol",{}).get("bank"); bank_count=0
    if bank_path and Path(bank_path).is_file(): bank_count=len(SnapshotBank.load(bank_path).records_for_phase(stage))
    cfg=json.loads((run/"config.json").read_text(encoding="utf-8")); natural=float(cfg.get(f"natural_prob_{stage}",1.0))
    console=Path(args.console_log).read_text(encoding="utf-8",errors="replace") if args.console_log and Path(args.console_log).is_file() else ""
    broadphase_requirements=[int(value) for value in re.findall(r"naconmax to (\d+)",console)]
    ccd_requirements=[int(value) for value in re.findall(r"naccdmax to (\d+)",console)]
    narrowphase_overflow="narrowphase overflow" in console.lower()
    nonfinite=_nonfinite_paths(payload); training_sps=[item["value"] for item in _series(rows,"training/sps")]
    eval_sps=[item["value"] for item in _series(rows,"eval/sps")]
    episode_length=float(last.get("eval/avg_episode_length",0.0))
    phase_visitation={
        phase:(float(last.get(f"eval/episode_reward/phase/{phase}",0.0))/episode_length if episode_length>0 else 0.0)
        for phase in ("approach","takeoff","flight","landing")
    }
    report={
        "status":payload.get("status"),"stage":stage,"seed":payload.get("seed"),
        "requested_timesteps":payload.get("requested_timesteps"),"effective_timesteps":payload.get("effective_timesteps"),
        "last_progress_step":max((int(row.get("step",0)) for row in rows),default=0),
        "last_evaluation_step":int(last.get("step",0)) if last else None,
        "outcomes":{"final_recovery_rate":final_rate,"chain_success_rate":final_rate if stage=="landing" else float(last.get("eval/episode_event/chain",0.0)),"physical_failure_rate":physical_rate,"timeout_rate":timeout_rate,"termination_reason_distribution":reasons},
        "reward":{"episode_total":last.get("eval/episode_reward"),"components":reward_terms},
        "phase_visitation":phase_visitation,
        "reset_source_distribution":{"candidate_bank":1.0-natural,"natural_start":natural,"candidate_records":bank_count,"protocol":payload.get("reset_protocol")},
        "optimization":{"entropy_loss":_series(rows,"training/entropy_loss"),"kl_mean":_series(rows,"training/kl_mean"),"value_loss":_series(rows,"training/v_loss"),"policy_distribution":{"mean_std":_series(rows,"training/policy_dist_mean_std"),"min_std":_series(rows,"training/policy_dist_min_std"),"max_std":_series(rows,"training/policy_dist_max_std"),"mean_loc":_series(rows,"training/policy_dist_mean_loc")}},
        "throughput":{"training_sps":_series(rows,"training/sps"),"eval_sps":_series(rows,"eval/sps"),"training_sps_range":None if not training_sps else [min(training_sps),max(training_sps)],"eval_sps_range":None if not eval_sps else [min(eval_sps),max(eval_sps)]},
        "health":{"nonfinite_count":len(nonfinite),"nonfinite_paths":nonfinite,"oom":("out of memory" in console.lower() or "oom" in console.lower()),"broadphase_overflow":"broadphase overflow" in console.lower(),"narrowphase_overflow":narrowphase_overflow,"peak_reported_naconmax_requirement":max(broadphase_requirements,default=None),"ccd_overflow":"ccd overflow" in console.lower(),"peak_reported_naccdmax_requirement":max(ccd_requirements,default=None),"compile_restart":("compile restart" in console.lower()),"error_type":payload.get("error_type"),"error":payload.get("error")},
        "evaluation_series":[{"step":int(row["step"]),"final_recovery_rate":float(row.get("eval/episode_end/recovery",0.0)),"physical_failure_rate":sum(float(value) for key,value in row.items() if key.startswith(end_prefix) and not key.endswith("_std") and key not in (end_prefix+"recovery",end_prefix+"stage_timeout")),"timeout_rate":float(row.get(end_prefix+"stage_timeout",0.0)),"episode_reward":float(row["eval/episode_reward"]),"avg_episode_length":float(row.get("eval/avg_episode_length",0.0))} for row in eval_rows],
    }
    runtime_invalid=(report["health"]["oom"] or report["health"]["broadphase_overflow"] or report["health"]["narrowphase_overflow"] or report["health"]["ccd_overflow"] or report["health"]["nonfinite_count"]>0)
    report["analysis_status"]=("INVALID_RUNTIME" if runtime_invalid else "COMPLETED_HEALTHY" if payload.get("status")=="completed" else "INCOMPLETE")
    output=Path(args.output) if args.output else run/"analysis.json"
    if output.exists(): raise SystemExit(f"Analysis output already exists: {output}")
    save_json(output,report); print(json.dumps({"status":report["status"],"outcomes":report["outcomes"],"health":report["health"],"throughput":report["throughput"]},indent=2))


if __name__=="__main__": main()
