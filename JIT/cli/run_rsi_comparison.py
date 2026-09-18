#!/usr/bin/env python3
"""Prepare, run or report a fresh RSI paired comparison."""
import argparse
from jit_dvgc.rsi_comparison import prepare, prepare_phase_comparison, run, report, launch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode', choices=['prepare','prepare-phase','prepare-batched','run','report','launch'])
p.add_argument('--batch-size',type=int,default=256)
p.add_argument('--previous');p.add_argument('--checkpoint')
p.add_argument('--alignment');p.add_argument('--source-spec');p.add_argument('--output');p.add_argument('--spec')
p.add_argument('--steps',type=int,default=1_000_000);p.add_argument('--episodes',type=int,default=100)
p.add_argument('--seed',type=int,default=9182601)
p.add_argument('--pulse-start-schedule',type=int,nargs='+')
p.add_argument('--additional-methods',help='JSON manifest of extra frozen policies for a batched comparison')
p.add_argument('--resource-wait-timeout-seconds',type=int,default=1800)
p.add_argument('--repository');p.add_argument('--snapshot')
p.add_argument('--report-output',help='New derived report directory; preserve an earlier report')
a=p.parse_args()
if a.mode=='prepare-batched':
    if not all((a.previous,a.output)):p.error('prepare-batched requires previous and output')
    from jit_dvgc.batched_pulse_comparison import prepare as prepare_batched
    print(prepare_batched(a.previous,a.output,episodes=a.episodes,batch_size=a.batch_size,seed=a.seed,
                          pulse_start_schedule=a.pulse_start_schedule,additional_methods=a.additional_methods,
                          resource_wait_timeout_seconds=a.resource_wait_timeout_seconds))
elif a.mode=='prepare-phase':
    if not all((a.previous,a.checkpoint,a.output)):p.error('prepare-phase requires previous, checkpoint and output')
    print(prepare_phase_comparison(a.previous,a.checkpoint,a.output))
elif a.mode=='prepare':
    if not all((a.alignment,a.source_spec,a.output)):p.error('prepare requires alignment, source-spec and output')
    print(prepare(a.alignment,a.source_spec,a.output,steps=a.steps,episodes=a.episodes,seed=a.seed))
elif a.mode=='run':
    if not a.spec:p.error('run requires spec')
    from jit_dvgc.rsi_comparison import read
    if read(a.spec).get('schema')=='jit_batched_paired_pulse_comparison_v1':
        from jit_dvgc.batched_pulse_comparison import run as run_batched
        run_batched(a.spec)
    else:run(a.spec)
elif a.mode=='report':
    if not a.output:p.error('report requires output')
    from pathlib import Path
    from jit_dvgc.rsi_comparison import read
    if read(Path(a.output)/'spec.json').get('schema')=='jit_batched_paired_pulse_comparison_v1':
        from jit_dvgc.analysis.batched_pulse_report import report as report_batched
        report_batched(a.output,a.report_output)
    else:report(a.output,a.report_output)
else:
    if not all((a.spec,a.repository,a.snapshot)):p.error('launch requires spec, repository and snapshot')
    print(launch(a.spec,a.repository,a.snapshot))
