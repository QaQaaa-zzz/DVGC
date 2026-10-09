"""Sequential, finite frozen-policy jobs in fresh Isaac processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--jobs',required=True,type=Path);p.add_argument('--protocol',required=True,type=Path);p.add_argument('--root',required=True,type=Path)
a=p.parse_args();jobs=json.loads(a.jobs.read_text());results=[]
for job in jobs:
    out=a.root/job['name']
    if out.exists():raise FileExistsError(out)
    cmd=[sys.executable,'fast_path_probe.py','--engine','physx','--protocol',str(a.protocol),'--target-config',job['config'],'--output',str(out),'--cases',*job['cases']]
    with (a.root/(job['name']+'.log')).open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    results.append(dict(name=job['name'],returncode=result.returncode))
    a.jobs.with_suffix('.status.json').write_text(json.dumps(results,indent=2)+'\n')
    print('JOB_DONE',job['name'],result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
