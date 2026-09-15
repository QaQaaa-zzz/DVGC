#!/usr/bin/env python3
"""Plot each pulse lineage and its recovery ancestry without running simulation."""
import argparse
from jit_dvgc.analysis.pulse_tube_xz import build

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lineage',action='append',required=True)
    p.add_argument('--label',action='append')
    p.add_argument('--output',required=True)
    a=p.parse_args();result=build(a.lineage,a.output,a.label)
    for r in result:print(r['lineage'],r['successful_context_policy_pairs'],'witnesses')
if __name__=='__main__':main()
