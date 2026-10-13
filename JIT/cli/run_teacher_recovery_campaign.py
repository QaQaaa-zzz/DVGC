"""Prepare or execute the explicitly approved fresh teacher-recovery100 campaign."""
import argparse
from jit_dvgc.generative_bridge.recovery_campaign import prepare,run_campaign,run_round

p=argparse.ArgumentParser(description=__doc__)
sub=p.add_subparsers(dest='mode',required=True)
q=sub.add_parser('prepare')
for key in ('parent','output','repository','monitor-panel','monitor-initial'):
    q.add_argument('--'+key,required=True)
q.add_argument('--port',type=int,default=6038)
q=sub.add_parser('run');q.add_argument('--declaration',required=True)
q=sub.add_parser('round');q.add_argument('--spec',required=True)
a=p.parse_args()
if a.mode=='prepare':
    print(prepare(a.parent,a.output,a.repository,monitor_panel=a.monitor_panel,
        monitor_initial=a.monitor_initial,port=a.port)['campaign_output'])
elif a.mode=='run':run_campaign(a.declaration)
else:run_round(a.spec)
