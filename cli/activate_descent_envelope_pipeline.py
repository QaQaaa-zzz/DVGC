"""Atomically point the read-only watchdog at an authorized envelope run."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--output",default="runs/ACTIVE_PIPELINE.json")
    args=parser.parse_args();run=Path(args.run)
    if run.exists():raise SystemExit(f"New envelope run already exists: {run}")
    run.mkdir(parents=True)
    payload={"status":"ACTIVE","activated_at":time.time(),"run_path":str(run),
             "controller_unit":"dvgc-descent-envelope-controller.service",
             "start_script":"/home/qy/DVGC/scripts/start_descent_envelope_controller.sh"}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(temporary,output)
    print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
