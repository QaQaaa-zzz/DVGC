"""Check or write one hash-bound remaining-pipeline completion marker."""
from __future__ import annotations

import argparse
import json

from dvgc.pipeline import marker_is_current, write_marker


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("check","record")); parser.add_argument("--marker",required=True)
    parser.add_argument("--step",required=True); parser.add_argument("--token",action="append",default=[])
    parser.add_argument("--input",action="append",default=[]); parser.add_argument("--output",action="append",default=[])
    parser.add_argument("--exit-status",type=int,default=0); parser.add_argument("--log",default="")
    args=parser.parse_args()
    if args.mode=="check":
        current,reason=marker_is_current(args.marker,tokens=args.token,inputs=args.input,outputs=args.output)
        print(json.dumps({"step":args.step,"current":current,"reason":reason}))
        raise SystemExit(0 if current else 1)
    payload=write_marker(args.marker,step=args.step,tokens=args.token,inputs=args.input,outputs=args.output,exit_status=args.exit_status,log_path=args.log or None)
    print(json.dumps({"step":args.step,"status":payload["status"],"exit_status":payload["exit_status"]}))


if __name__=="__main__": main()
