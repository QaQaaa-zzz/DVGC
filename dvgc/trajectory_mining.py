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


def select_parent_balanced(rows:Sequence[Mapping],*,target:int,masses:Mapping[str,float],parent_cap:int=4)->list[dict]:
    targets=layer_targets(target,masses);selected=[];counts=Counter();layers=Counter()
    ordered=sorted(rows,key=lambda row:(-float(row.get("mining_rank_score",0.0)),str(row["id"])))
    # Pass one gives every new trajectory parent at most one state while filling layer targets.
    for pass_index in range(int(parent_cap)):
        for row in ordered:
            parent=str(row["trajectory_parent_id"]);layer=str(row["descent_layer"])
            if counts[parent]!=pass_index or counts[parent]>=int(parent_cap):continue
            if layers[layer]>=targets[layer] and any(layers[k]<targets[k] for k in LAYERS):continue
            selected.append(dict(row));counts[parent]+=1;layers[layer]+=1
            if len(selected)>=int(target):return selected
    # Layer masses are proposal targets, not acceptance gates. Fill any deficit
    # from other validated layers without relaxing trajectory-parent caps.
    selected_ids={str(row["id"]) for row in selected}
    for pass_index in range(int(parent_cap)):
        for row in ordered:
            parent=str(row["trajectory_parent_id"])
            if str(row["id"]) in selected_ids or counts[parent]!=pass_index or counts[parent]>=int(parent_cap):continue
            selected.append(dict(row));selected_ids.add(str(row["id"]));counts[parent]+=1
            if len(selected)>=int(target):return selected
    return selected
