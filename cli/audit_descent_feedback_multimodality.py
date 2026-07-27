"""Audit multimodality of already discovered local Descent corrections."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cli.run_unified_descent_feedback_probe import _assets, _exact
from dvgc.descent_probe import batched_base_state, make_residual_rollout
from dvgc.env import END_REASON
from dvgc.runtime import save_json


def _cosine(a,b):
    a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1);den=np.linalg.norm(a)*np.linalg.norm(b)
    return 0.0 if den<1e-12 else float(np.dot(a,b)/den)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--authority",required=True);parser.add_argument("--snapshots",required=True);parser.add_argument("--output",required=True);args=parser.parse_args();output=Path(args.output)
    if output.exists():raise SystemExit(f"refusing overwrite {output}")
    authority=json.loads(Path(args.authority).read_text())
    with Path(args.snapshots).open("rb") as handle:snapshots=pickle.load(handle)
    _,_,env,params=_assets();rollout=make_residual_rollout(env,params,horizon=24,ticks_per_knot=4,residual_ticks=8)
    rows=[]
    for index,(snapshot,row) in enumerate(zip(snapshots,authority["rows"],strict=True)):
        if not row["authoritative_correction"]:continue
        successful=[item for item in row["top5"] if item["exact_replay"]["repeat1_repeat2_exact"] and item["gain"]>=2 and item["no_new_failure_type"]]
        residual=np.asarray([item["residual_knots"] for item in successful]);actions=np.asarray([item["exact_replay"]["replay"]["actions"] for item in successful])
        distance=np.linalg.norm(residual[:,None]-residual[None,:],axis=(2,3));medoid_index=int(np.argmin(distance.sum(1)));medoid=successful[medoid_index]
        cosine=np.asarray([[_cosine(a,b) for b in residual] for a in residual]);opposite=bool(np.any(cosine<-.25))
        factory=lambda count,index=index,snapshot=snapshot:batched_base_state(env,snapshot["snapshot"],9100000+index,count)
        aggregate={}
        for name,knots in (("mean",residual.mean(0)),("median",np.median(residual,axis=0))):
            probe={"residual_knots":np.asarray(knots,np.float32).tolist(),"survival":0,"minimum_margin":0,"terminal_margin":0,"end_code":0,"actions":[],"features":[]}
            exact=_exact(rollout,factory,probe,9700000+index);replay=exact["replay"];failure=END_REASON.get(replay["end_code"],"horizon")
            aggregate[name]={"repeat1_repeat2_exact":exact["repeat1_repeat2_exact"],"survival":replay["survival"],"gain":replay["survival"]-row["baseline"]["survival"],"failure":failure,"physically_valid":exact["repeat1_repeat2_exact"] and replay["survival"]-row["baseline"]["survival"]>=2 and failure in {row["baseline"]["failure"],"horizon"}}
        variance={str(h):{"per_action":np.var(actions[:,:h],axis=(0,1)).tolist(),"mean":float(np.mean(np.var(actions[:,:h],axis=(0,1))))} for h in (1,2,4)}
        agreement={}
        for channel,name in enumerate(("steer","drive","hip","knee")):
            signs=np.sign(residual[:,:,channel].reshape(-1));nz=signs[signs!=0];agreement[name]=1.0 if not len(nz) else float(max(np.mean(nz>0),np.mean(nz<0)))
        original=np.asarray(snapshot["original_cem_residual"]);local=np.asarray(medoid["residual_knots"])[0]
        rows.append({"snapshot_index":index,"candidate_id":row["candidate_id"],"tick":row["tick"],"successful_sequences":len(successful),"action_variance":variance,"residual_direction_agreement":agreement,"opposite_successful_clusters":opposite,"successful_medoid":{"rank":medoid["rank"],"exact":medoid["exact_replay"]["repeat1_repeat2_exact"],"gain":medoid["gain"]},"mean_replay":aggregate["mean"],"median_replay":aggregate["median"],"original_vs_local_medoid":{"cosine":_cosine(original,local),"original_magnitude":float(np.linalg.norm(original)),"local_first_knot_magnitude":float(np.linalg.norm(local))},"stable_point_target":bool(medoid["exact_replay"]["repeat1_repeat2_exact"])})
    report={"status":"PASS","audited_authoritative_snapshots":len(rows),"snapshots_with_opposite_successful_clusters":sum(row["opposite_successful_clusters"] for row in rows),"mean_physically_valid":sum(row["mean_replay"]["physically_valid"] for row in rows),"median_physically_valid":sum(row["median_replay"]["physically_valid"] for row in rows),"stable_medoid_targets":sum(row["stable_point_target"] for row in rows),"set_valued_or_non_identifiable":sum(not row["stable_point_target"] for row in rows)>len(rows)/2,"rows":rows,"heldout_used":False,"ppo_authorization":False}
    output.mkdir(parents=True);save_json(output/"successful_action_multimodality_audit.json",report);print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
