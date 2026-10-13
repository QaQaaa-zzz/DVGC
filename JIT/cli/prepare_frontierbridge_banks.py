#!/usr/bin/env python3
"""Materialize real parent labels on the declared GPU without physical steps."""
import argparse
from jit_dvgc.retention_consolidation_data import materialize_parent_banks
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--sources',required=True)
p.add_argument('--parent',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
result=materialize_parent_banks(a.sources,a.parent,a.output)
print(result)
