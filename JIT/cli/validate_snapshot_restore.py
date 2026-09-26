"""Run separately budgeted GPU restoration engineering validation."""
import argparse
import json
from pathlib import Path
from jit_dvgc.batch_snapshot_restore_validation import validate_restore

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--candidates',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--timing-counts',type=int,nargs='*',default=[])
    args=parser.parse_args()
    try:
        print(validate_restore(args.spec,args.candidates,args.output,args.count,args.timing_counts))
    except Exception as exc:
        status_path=Path(args.output)/'status.json'
        if status_path.parent.exists():
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(phase='error',error=str(exc))
            status_path.write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
        raise
