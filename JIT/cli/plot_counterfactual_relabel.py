#!/usr/bin/env python3
"""Plot an explicit hypothetical re-labeling of completed paired rollouts."""
import argparse
from jit_dvgc.analysis.counterfactual_relabel import report

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',required=True);p.add_argument('--base-report',required=True);p.add_argument('--output',required=True)
p.add_argument('--target-method',default='baseline');p.add_argument('--x-min',type=float,default=4.)
p.add_argument('--x-max',type=float,default=4.5);p.add_argument('--partial-min',type=int,default=1)
p.add_argument('--partial-max',type=int,default=12);p.add_argument('--zero-fraction',type=float,default=.5)
p.add_argument('--seed',type=int,default=9182801)
a=p.parse_args()
print(report(a.source,a.base_report,a.output,target_method=a.target_method,x_min=a.x_min,x_max=a.x_max,
             partial_min=a.partial_min,partial_max=a.partial_max,zero_fraction=a.zero_fraction,seed=a.seed))
