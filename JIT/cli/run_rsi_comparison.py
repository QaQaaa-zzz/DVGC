#!/usr/bin/env python3
"""Prepare, run or report a fresh RSI paired comparison."""
import argparse
from jit_dvgc.rsi_comparison import prepare, run, report, launch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode', choices=['prepare','run','report','launch'])
p.add_argument('--alignment');p.add_argument('--source-spec');p.add_argument('--output');p.add_argument('--spec')
p.add_argument('--steps',type=int,default=1_000_000);p.add_argument('--episodes',type=int,default=100)
p.add_argument('--seed',type=int,default=9182601)
p.add_argument('--repository');p.add_argument('--snapshot')
p.add_argument('--report-output',help='New derived report directory; preserve an earlier report')
a=p.parse_args()
if a.mode=='prepare':
    if not all((a.alignment,a.source_spec,a.output)):p.error('prepare requires alignment, source-spec and output')
    print(prepare(a.alignment,a.source_spec,a.output,steps=a.steps,episodes=a.episodes,seed=a.seed))
elif a.mode=='run':
    if not a.spec:p.error('run requires spec')
    run(a.spec)
elif a.mode=='report':
    if not a.output:p.error('report requires output')
    report(a.output,a.report_output)
else:
    if not all((a.spec,a.repository,a.snapshot)):p.error('launch requires spec, repository and snapshot')
    print(launch(a.spec,a.repository,a.snapshot))
