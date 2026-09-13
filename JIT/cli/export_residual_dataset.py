#!/usr/bin/env python3
"""Export sparse causal action/observation pairs from completed TRAIN evidence."""
import argparse
import json
from pathlib import Path
from jit_dvgc.residual_dataset import export_dataset

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(export_dataset(json.loads(a.spec.read_text()),a.output),indent=2))

if __name__=='__main__':main()
