"""Finish saved A2 reports without rerunning training, feedback, or physics."""
import argparse
from jit_dvgc.generative_bridge.campaign import finalize_saved_campaign

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(finalize_saved_campaign(args.previous,args.output))
