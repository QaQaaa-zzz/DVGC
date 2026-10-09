"""Dedicated bounded GPU teacher worker; launched only through resource gating."""
import argparse
import faulthandler
faulthandler.enable(all_threads=True)
from jit_dvgc.generative_bridge.teacher_worker import run

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    run(parser.parse_args().request)
