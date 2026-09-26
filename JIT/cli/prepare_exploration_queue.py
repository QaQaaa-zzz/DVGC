#!/usr/bin/env python3
"""Lock a finite sequence of delayed exploration loops without GPU execution."""
import argparse
from pathlib import Path
from jit_dvgc.exploration_loop_declaration import prepare_queue


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--templates',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=prepare_queue(args.templates,args.output)
    print('Prepared maximum interactions:',plan['max_interactions'])


if __name__=='__main__':main()
