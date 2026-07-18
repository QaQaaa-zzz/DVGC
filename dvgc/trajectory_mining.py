"""Pure selection rules for successful-rollout descent snapshot mining."""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping,Sequence

import numpy as np


LAYERS=("middle","late","early")
POLICY_STATE_KEYS=("last_action","obs_history","actor_observation","filter_phase","phase_probs",
                   "contact_probs","phase_progress","phase_confidence","estimator_hidden",
                   "delay_buffer","prev_acc_z","prev_vz")


def trajectory_parent_id(policy_hash:str,candidate_id:str,branch_seed:int)->str:
    payload=f"trajectory-parent:{policy_hash}:{candidate_id}:{int(branch_seed)}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def policy_state_complete(record:Mapping)->bool:
    state=record.get("policy_state",{})
    return all(key in state for key in POLICY_STATE_KEYS)


def representative_indices(length:int,max_children:int=4)->list[tuple[str,int]]:
    if length<=0:return []
    proposals=[("earliest",0),("middle",length//2),("late",length-1)]
    # The fourth state is selected later by physical roll margin.
    result=[]
    for name,index in proposals:
        if index not in {value for _,value in result}:result.append((name,index))
    return result[:int(max_children)]


def select_trace_records(trace:Sequence[Mapping],*,max_children:int=4,min_step_gap:int=2)->list[tuple[str,Mapping]]:
    if not trace:return []
    selected=representative_indices(len(trace),max_children)
    used={index for _,index in selected}
    margin_order=sorted(range(len(trace)),key=lambda i:(abs(float(trace[i]["record"]["physical_feature"][3]))
                         +.25*abs(float(trace[i]["record"]["physical_feature"][9])),i))
    for index in margin_order:
        if all(abs(index-old)>=int(min_step_gap) for old in used):
            selected.append(("best_roll_margin",index));break
    selected=selected[:int(max_children)]
    return [(role,trace[index]) for role,index in sorted(selected,key=lambda item:item[1])]


def layer_targets(target:int,masses:Mapping[str,float])->dict[str,int]:
    raw={name:int(target)*float(masses[name]) for name in LAYERS}
    result={name:int(np.floor(raw[name])) for name in LAYERS}
    order=sorted(LAYERS,key=lambda name:(-(raw[name]-result[name]),LAYERS.index(name)))
    for name in order[:int(target)-sum(result.values())]:result[name]+=1
    return result


def canonical_state_byte_hash(row:Mapping)->str:
    """Hash the complete simulator/PolicyState bytes, ignoring record wrappers."""
    digest=hashlib.sha256()
    for key in ("qpos","qvel","ctrl","qacc_warmstart"):
        value=np.ascontiguousarray(row[key]);digest.update(key.encode());digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode());digest.update(value.tobytes())
    for key in sorted(row.get("policy_state",{})):
        value=np.ascontiguousarray(row["policy_state"][key]);digest.update(key.encode());digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode());digest.update(value.tobytes())
    for key in ("oracle_phase","had_airborne","had_valid_landing","contact_age","airborne_count","recovery_count"):
        digest.update(f"{key}:{row.get(key)}".encode())
    return digest.hexdigest()


def declared_snapshot_hash(row:Mapping)->str:
    """Return the declared snapshot identity, or a canonical identity for raw rows."""
    return str(row.get("snapshot_identity_sha256") or canonical_state_byte_hash(row))


def selection_is_eligible(row:Mapping)->bool:
    if not bool(row.get("bootstrap_eligible",True)) or bool(row.get("training_only",False)):
        return False
    return "oracle_phase" not in row or int(row["oracle_phase"])==2


def selection_uniqueness(rows:Sequence[Mapping])->dict:
    ids=[str(row["id"]) for row in rows]
    snapshots=[declared_snapshot_hash(row) for row in rows]
    byte_hashes=[canonical_state_byte_hash(row) for row in rows]
    checks={"candidate_ids_unique":len(ids)==len(set(ids)),
            "snapshot_hashes_unique":len(snapshots)==len(set(snapshots)),
            "state_byte_hashes_unique":len(byte_hashes)==len(set(byte_hashes))}
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
            "selected_records":len(rows),"unique_candidate_ids":len(set(ids)),
            "unique_snapshot_hashes":len(set(snapshots)),"unique_state_byte_hashes":len(set(byte_hashes))}


def select_parent_balanced_with_report(rows:Sequence[Mapping],*,target:int,masses:Mapping[str,float],parent_cap:int=4)->tuple[list[dict],dict]:
    """Select globally unique states across all quota and fallback rounds."""
    selected=[];counts=Counter();layers=Counter();rejected=Counter()
    ordered=sorted(rows,key=lambda row:(-float(row.get("mining_rank_score",0.0)),str(row["id"])))
    unique=[];seen_ids=set();seen_snapshots=set();seen_bytes=set()
    for row in ordered:
        if not selection_is_eligible(row):rejected["ineligible"]+=1;continue
        identifier=str(row["id"]);snapshot=declared_snapshot_hash(row);byte_hash=canonical_state_byte_hash(row)
        reasons=[]
        if identifier in seen_ids:reasons.append("candidate_id")
        if snapshot in seen_snapshots:reasons.append("snapshot_hash")
        if byte_hash in seen_bytes:reasons.append("state_byte_hash")
        if reasons:
            for reason in reasons:rejected[reason]+=1
            rejected["duplicate_records"]+=1
            continue
        seen_ids.add(identifier);seen_snapshots.add(snapshot);seen_bytes.add(byte_hash);unique.append(row)
    parent_capacity=len({str(row["trajectory_parent_id"]) for row in unique})*int(parent_cap)
    quota_target=min(int(target),parent_capacity)
    targets=layer_targets(quota_target,masses)
    selected_ids=set();selected_snapshots=set();selected_bytes=set()

    def add(row):
        identifier=str(row["id"]);snapshot=declared_snapshot_hash(row);byte_hash=canonical_state_byte_hash(row)
        parent=str(row["trajectory_parent_id"]);layer=str(row["descent_layer"])
        if identifier in selected_ids or snapshot in selected_snapshots or byte_hash in selected_bytes:return False
        if counts[parent]>=int(parent_cap):return False
        selected.append(dict(row));selected_ids.add(identifier);selected_snapshots.add(snapshot);selected_bytes.add(byte_hash)
        counts[parent]+=1;layers[layer]+=1
        return True

    # Pass one gives every new trajectory parent at most one state while filling layer targets.
    for pass_index in range(int(parent_cap)):
        for row in unique:
            parent=str(row["trajectory_parent_id"]);layer=str(row["descent_layer"])
            if counts[parent]!=pass_index or counts[parent]>=int(parent_cap):continue
            if layers[layer]>=targets[layer] and any(layers[k]<targets[k] for k in LAYERS):continue
            add(row)
            if len(selected)>=quota_target:break
        if len(selected)>=quota_target:break
    # Layer masses are proposal targets, not acceptance gates. Fill any deficit
    # from other validated layers without relaxing trajectory-parent caps.
    if len(selected)<quota_target:
        for pass_index in range(int(parent_cap)):
            for row in unique:
                parent=str(row["trajectory_parent_id"])
                if counts[parent]!=pass_index or counts[parent]>=int(parent_cap):continue
                add(row)
                if len(selected)>=quota_target:break
            if len(selected)>=quota_target:break
    uniqueness=selection_uniqueness(selected)
    if uniqueness["status"]!="PASS":raise AssertionError(f"Trajectory selection uniqueness failure: {uniqueness['checks']}")
    shortfall=quota_target-len(selected)
    report={**uniqueness,"configured_target":int(target),"quota_target":quota_target,"quota_shortfall":shortfall,
            "exhausted_unique_support":bool(shortfall),"eligible_unique_records":len(unique),
            "unique_parents":len(counts),"maximum_children_per_parent":max(counts.values(),default=0),
            "duplicate_rejected":dict(rejected),"selected_layers":dict(layers),
            "selected_order":[{"selection_index":i,"id":str(row["id"]),
                "snapshot_hash":declared_snapshot_hash(row),"state_byte_hash":canonical_state_byte_hash(row),
                "trajectory_parent_id":str(row["trajectory_parent_id"]),"descent_layer":str(row["descent_layer"])}
                for i,row in enumerate(selected)]}
    return selected,report


def select_parent_balanced(rows:Sequence[Mapping],*,target:int,masses:Mapping[str,float],parent_cap:int=4)->list[dict]:
    return select_parent_balanced_with_report(rows,target=target,masses=masses,parent_cap=parent_cap)[0]
