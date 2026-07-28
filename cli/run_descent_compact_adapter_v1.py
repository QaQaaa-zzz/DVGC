"""Freeze and certify a compact-support residual adapter for one Descent gap."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L, _micro_states
from cli.run_backward_descent_rsi_pilot import certify_policy
from cli.run_descent_localized_consolidation_v1 import (
    PRIOR, SOURCE, TARGET_NODE, _cert_summary, _collect_prefix, _node_record,
    verified_assets_allowing_runtime_gate_refresh,
)
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import (compact_observation_command_adapter,
                                  compact_observation_residual_adapter)
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_predecessor import expand_residual_knots
from dvgc.descent_supervised import build_actor_tools
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.runtime import save_json


DEFAULT_RUN=Path("runs/descent_diverse_p1_predecessor_recovery_v3_compact_adapter")


def choose_compact_radius(teacher, preservation, micro, mean, std):
    normalize=lambda value:(np.asarray(value)-np.asarray(mean))/np.asarray(std)
    teacher_n=normalize(teacher); preservation_n=normalize(preservation); micro_n=normalize(micro)
    cross=np.sqrt(np.mean((teacher_n[:,None]-preservation_n[None,:])**2,axis=-1))
    micro_distance=np.sqrt(np.mean((micro_n[:,None]-teacher_n[None,:])**2,axis=-1)).min(axis=1)
    exclusion_cap=.45*float(cross.min()); coverage_floor=1.25*float(micro_distance.max())
    return {"radius":coverage_floor,"exclusion_cap":exclusion_cap,"micro_distance_max":float(micro_distance.max()),
            "teacher_preservation_distance_min":float(cross.min()),"geometry_pass":coverage_floor < exclusion_cap}


def _identity_hash(base_hash,prototypes,targets,radius):
    digest=hashlib.sha256(base_hash.encode())
    for value in (prototypes,targets,np.asarray(radius,np.float32)):
        array=np.asarray(value);digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_pickle(path,value):
    temporary=path.with_suffix(path.suffix+".partial")
    with temporary.open("wb") as handle:
        pickle.dump(value,handle,pickle.HIGHEST_PROTOCOL);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def _collect_recorded_commands(env,actor,policy,record,seed,commands):
    from cli.run_backward_descent_nominal_pilot import _restore
    state=_restore(env,record,jax.random.PRNGKey(seed));observations=[]
    for command in np.asarray(commands):
        observations.append(np.asarray(state.obs["state"],np.float32))
        state=env.step(state,jnp.asarray(command))
    return np.asarray(observations)


def _micro_command_observations(env,micro,commands):
    state=micro;observations=[];step=jax.jit(jax.vmap(env.step))
    for command in np.asarray(commands):
        observations.append(np.asarray(state.obs["state"],np.float32))
        state=step(state,jnp.broadcast_to(jnp.asarray(command),(state.data.qpos.shape[0],len(command))))
    return np.concatenate(observations,axis=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",default=str(DEFAULT_RUN))
    parser.add_argument("--command-source",choices=("residual","recorded"),default="residual");args=parser.parse_args()
    root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    valid,failed,raw_failed=verified_assets_allowing_runtime_gate_refresh()
    if not valid:raise SystemExit(f"frozen scientific asset mismatch: {failed}; raw={raw_failed}")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text());fingerprint=source_fingerprint(Path.cwd())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=fingerprint:raise SystemExit("runtime gate stale")
    cfg=load_config("configs/backward_descent_rsi_pilot_v1.json",{"use_bank_resets":False,"expert_chain_termination":False,
        "domain_randomization":False,"obs_noise_enable":False})
    if file_sha256(cfg.xml_path)!=EXPECTED["xml"] or cfg.action_mapping_version!=ACTION_MAPPING_VERSION:raise SystemExit("runtime model mismatch")
    root.mkdir(parents=True)
    save_json(root/"cost_estimate.json",{"estimated_seconds":900,"certified_states":18,"branches_per_state":"2 exact + 4 micro",
        "PPO_steps":0,"new_CEM":False,"heldout_used":False})
    save_json(root/"preregistration.json",{"experiment":"descent_compact_observation_adapter_v1","target_node":TARGET_NODE,
        "prototype_ticks":5,"preservation_ticks":8,"distance":"RMS Euclidean in frozen-normalizer actor space",
        "radius":"1.25 * max target-micro nearest-prototype distance","hard_exclusion_cap":"0.45 * minimum teacher-to-preservation distance",
        "kernel":"max(1-d/r,0)^2","command_source":args.command_source,
        "physical_gate":"18/18 P0 and P1, no new failure types","PPO_authorization":False,
        "artifact_role":"expert_conditioned_provisional_envelope_controller"})
    dparams,_,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L));_,actor,_=build_actor_tools(env,dparams)
    balanced=json.loads((SOURCE/"balanced_p1_launch_subset_v1_frozen.json").read_text())["nodes"]
    full=json.loads((SOURCE/"full_p1_bank_v1.json").read_text())["nodes"]
    baseline_ids=json.loads((SOURCE/"descent_balanced_rsi_pilot_v1/prefit_physical_gate.json").read_text())["baseline_balanced"]["P1_ids"]
    harvested=pickle.loads((SOURCE/"trajectory_harvested_snapshots.pkl").read_bytes());by_id={node["node_id"]:node for node in balanced}
    target_node=by_id[TARGET_NODE];prior=json.loads(PRIOR.read_text())
    row=next(value for value in prior["rows"] if value["proposal"]["physical_state_sha256"]==target_node["source_state_hash"] and value["controller"]==target_node["controller_type"])
    residuals=expand_residual_knots(row["residual_knots"]);record=_node_record(target_node,harvested)
    if args.command_source=="recorded":
        commands=np.asarray(row["search_best"]["actions"][:5],np.float32)
        prototypes=_collect_recorded_commands(env,actor,dparams[1],record,85_000_000,commands)
    else:
        prototypes,commands=_collect_prefix(env,actor,dparams[1],record,85_000_000,targets=residuals,ticks=5)
        prototypes=np.asarray(prototypes);commands=np.asarray(commands)
    preservation=[]
    for index,node_id in enumerate(sorted(baseline_ids)):
        obs,_=_collect_prefix(env,actor,dparams[1],_node_record(by_id[node_id],harvested),85_100_000+index,ticks=8);preservation.extend(obs)
    micro=_micro_states(env,record,85_200_000)
    micro_obs=(_micro_command_observations(env,micro,commands) if args.command_source=="recorded" else np.asarray(micro.obs["state"]))
    mean=np.asarray(dparams[0].mean["state"]);std=np.asarray(dparams[0].std["state"])
    geometry=choose_compact_radius(prototypes,np.asarray(preservation),micro_obs,mean,std)
    save_json(root/"adapter_geometry_preflight.json",geometry|{"teacher_prototypes":len(prototypes),"preservation_samples":len(preservation),
        "target_micro_states":len(micro_obs),"selected_without_physical_results":True})
    if not geometry["geometry_pass"]:raise SystemExit(f"compact support is not geometrically separable: {geometry}")
    adapter=(compact_observation_command_adapter(jnp.asarray(prototypes),jnp.asarray(commands),jnp.asarray(mean),jnp.asarray(std),geometry["radius"])
             if args.command_source=="recorded" else
             compact_observation_residual_adapter(jnp.asarray(prototypes),jnp.asarray(residuals[:len(prototypes)]),jnp.asarray(mean),jnp.asarray(std),geometry["radius"]))
    preservation_action=np.asarray(adapter(jnp.asarray(preservation),jnp.zeros((len(preservation),4),jnp.float32)))
    if np.max(np.abs(preservation_action))!=0:raise SystemExit("adapter leaks into preservation prefixes")
    targets=commands if args.command_source=="recorded" else residuals[:len(prototypes)]
    identity=_identity_hash(EXPECTED["pi_D"],prototypes,targets,geometry["radius"])
    artifact={"schema":"compact-observation-residual-adapter-v1","base_policy_sha256":EXPECTED["pi_D"],"policy_identity_hash":identity,
        "prototypes":prototypes,"targets":targets,"command_source":args.command_source,
        "normalizer_mean":mean,"normalizer_std":std,"radius":geometry["radius"],
        "target_node":TARGET_NODE,"action_order":["steer","drive","hip","knee"]}
    _atomic_pickle(root/"adapter.pkl",artifact)
    save_bundle(root/"checkpoint_expert",params=dparams,config=cfg,xml_path=cfg.xml_path,
        candidate_bank=SOURCE/"balanced_p1_launch_subset_v1_frozen.json",downstream_bank=C_L,
        policy_version="descent-compact-observation-adapter-v1",extra={"artifact_role":"expert_conditioned_provisional_envelope_controller",
        "adapter_sha256":file_sha256(root/"adapter.pkl"),"policy_identity_hash":identity,"base_policy_weights_unchanged":True})
    loader=lambda node:_node_record(node,harvested);cert=certify_policy(env,dparams,lparams,full,86_000_000,record_loader=loader,
        descent_action_adapter=adapter,policy_identity_hash=identity)
    ids=[node["node_id"] for node in full];summary=_cert_summary(cert,ids)
    save_json(root/"construction_certification.json",cert)
    accepted=summary["P0"]==len(full) and summary["P1"]==len(full)
    report={"status":"PASS" if accepted else "FAIL","head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "geometry":geometry,"construction":summary,"policy_identity_hash":identity,"adapter_sha256":file_sha256(root/"adapter.pkl"),
        "base_policy_weights_unchanged":True,"frozen_scientific_assets_unchanged":verified_assets_allowing_runtime_gate_refresh()[0],
        "formal_tube_or_jel":False,"artifact_role":"expert_conditioned_provisional_envelope","PPO_authorization":False,"heldout_used":False}
    save_json(root/"DESCENT_COMPACT_ADAPTER_V1_REPORT.json",report);save_json(root/"completed.json",{"status":report["status"],
        "next":"independent_pointwise_audit" if accepted else "compact_adapter_support_gap"})
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
