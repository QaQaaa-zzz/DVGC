"""Write the mandatory cost estimate before stage reachability rollouts."""
from __future__ import annotations
import argparse
from pathlib import Path
from dvgc.runtime import save_json

def estimate(*,states:int,branches:int,horizon:int,pilot_fraction:float,throughput:float|None,hypothesis:str):
    rollouts=states*branches;steps=rollouts*horizon;pilot_states=max(1,round(states*pilot_fraction));pilot_rollouts=pilot_states*branches
    seconds=(steps/throughput) if throughput and throughput>0 else None
    return {"status":"READY","artifact_role":"cost_estimate","unique_states":states,"branches_per_state":branches,"horizon":horizon,"total_rollouts":rollouts,"total_environment_steps":steps,"estimated_compilations":1,"pilot_fraction":pilot_fraction,"pilot_unique_states":pilot_states,"pilot_rollouts":pilot_rollouts,"pilot_environment_steps":pilot_rollouts*horizon,"pilot_throughput_steps_per_second":throughput,"estimated_total_seconds":seconds,"estimated_total_hours":seconds/3600 if seconds is not None else None,"hypothesis":hypothesis,"review_required_over_two_hours":bool(seconds is not None and seconds>7200)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--unique-states',type=int,required=True);p.add_argument('--branches',type=int,default=4);p.add_argument('--horizon',type=int,required=True);p.add_argument('--pilot-fraction',type=float,default=.04);p.add_argument('--throughput',type=float);p.add_argument('--hypothesis',required=True);a=p.parse_args()
 if not .02<=a.pilot_fraction<=.05:raise SystemExit('Pilot fraction must be 2--5%')
 save_json(Path(a.output),estimate(states=a.unique_states,branches=a.branches,horizon=a.horizon,pilot_fraction=a.pilot_fraction,throughput=a.throughput,hypothesis=a.hypothesis))
if __name__=='__main__':main()
