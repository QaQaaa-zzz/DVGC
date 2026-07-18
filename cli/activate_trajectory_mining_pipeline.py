"""Atomically activate a new successful-trajectory mining run."""
from __future__ import annotations

import argparse,json,os,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True);p.add_argument("--output",default="runs/ACTIVE_PIPELINE.json")
    p.add_argument("--prepared-resume",action="store_true");a=p.parse_args()
    run=Path(a.run)
    if a.prepared_resume:
        required=(run/"controller_state.json",run/"trajectory_mining_corrected/report.json",run/"trajectory_mining_corrected/physical_audit.json")
        if not run.is_dir() or any(not path.exists() for path in required):raise SystemExit("Prepared trajectory-mining resume is incomplete")
        state=json.loads((run/"controller_state.json").read_text());audit=json.loads(required[-1].read_text())
        if state.get("current_stage")!="stable_stage_a" or audit.get("status")!="PASS":raise SystemExit("Prepared trajectory-mining resume preflight is not PASS")
    else:
        if run.exists():raise SystemExit(f"New trajectory-mining run already exists: {run}")
        run.mkdir(parents=True)
    payload={"status":"ACTIVE","activated_at":time.time(),"run_path":str(run),
        "controller_unit":"dvgc-trajectory-mining-controller.service","start_script":"/home/qy/DVGC/scripts/start_trajectory_mining_controller.sh"}
    output=Path(a.output);temporary=output.with_suffix(output.suffix+".tmp");temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(temporary,output);print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
