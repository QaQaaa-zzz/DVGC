"""Fit a parent-split state-level viability ensemble for acquisition only."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.config import file_sha256
from dvgc.runtime import save_json
from dvgc.viability import ViabilityEnsemble


def parent_key(row):
    return str(row.get("entry_source_id", row.get("parent_candidate_id", row["id"])))


def parent_split(rows, seed):
    groups=defaultdict(list)
    for row in rows:groups[parent_key(row)].append(row)
    parents=sorted(groups,key=lambda key:hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    if len(parents)<3:raise ValueError("Need at least three parent groups for train/validation/acquisition isolation")
    ntrain=max(1,int(round(.7*len(parents))));nval=max(1,int(round(.15*len(parents))))
    if ntrain+nval>=len(parents):ntrain=len(parents)-2;nval=1
    assignment={key:("train" if i<ntrain else "validation" if i<ntrain+nval else "acquisition") for i,key in enumerate(parents)}
    split={name:[] for name in ("train","validation","acquisition")}
    for key,items in groups.items():split[assignment[key]].extend(items)
    return split,{name:sorted({parent_key(row) for row in values}) for name,values in split.items()}


def _rank_metrics(y,score):
    y=np.asarray(y,bool);score=np.asarray(score,float);pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return {"auroc":float("nan"),"auprc":float("nan")}
    order=np.argsort(-score);ys=y[order];tp=np.cumsum(ys);fp=np.cumsum(~ys)
    tpr=np.r_[0,tp/pos];fpr=np.r_[0,fp/neg];auroc=float(np.trapezoid(tpr,fpr))
    precision=tp/np.arange(1,len(y)+1);auprc=float(np.sum(precision[ys])/pos)
    return {"auroc":auroc,"auprc":auprc}


def _metrics(rows,mean,std,support,threshold=.7):
    target=np.asarray([float(row["final"]["posterior"]["mean"]) for row in rows]);label=np.asarray([row["final"]["label"]=="safe" for row in rows]);pred=mean>=threshold
    tp=int(np.sum(pred&label));fp=int(np.sum(pred&~label));fn=int(np.sum(~pred&label))
    result={**_rank_metrics(label,mean),"brier":float(np.mean((mean-target)**2)),"ece":float(np.mean(np.abs(mean-target))),
            "safe_precision":tp/(tp+fp) if tp+fp else 1.,"safe_recall":tp/(tp+fn) if tp+fn else 0.,
            "mean_disagreement":float(np.mean(std)),"mean_support":float(np.mean(support)),"states":len(rows),
            "labels":dict(Counter(row["final"]["label"] for row in rows))}
    boundary=[i for i,row in enumerate(rows) if row["final"]["label"]=="boundary"]
    result["boundary_mean_score"]=float(np.mean(mean[boundary])) if boundary else float("nan")
    return result


def development_rows(bank, reports, cfg):
    rows=[dict(row) for row in bank.records if row["final"]["branches"]>0 and not row.get("training_only",False)]
    by_id={row["id"]:row for row in rows};extra=defaultdict(list);sources=[]
    for path in reports:
        payload=json.loads(Path(path).read_text());sources.append({"path":str(path),"sha256":file_sha256(path),"role":payload.get("evidence_role",payload.get("artifact_role"))})
        policy=payload.get("descent_policy_hash",payload.get("policy_hash"))
        expected=bank.metadata.get("policy_hash",bank.metadata.get("stable_construction_policy_hash"))
        if policy and expected and policy!=expected:raise ValueError("Development evidence policy mismatch")
        for result in payload.get("rows",[]):
            if result.get("id") in by_id:extra[result["id"]].extend(result.get("branch_evidence",[]))
    for row in rows:
        evidence=list(row.get("certification_branches",[]))+extra[row["id"]]
        successes=sum(bool(item.get("final_recovery")) for item in evidence);branches=len(evidence)
        posterior=beta_posterior(successes,branches-successes,alpha0=cfg.beta_alpha0,beta0=cfg.beta_beta0,q_low=cfg.posterior_q_low,q_high=cfg.posterior_q_high)
        label=posterior_label(posterior,branches,min_branches=cfg.min_branches,safe_threshold=cfg.safe_threshold,dead_threshold=cfg.dead_threshold,boundary_max_width=cfg.boundary_max_width)
        row["final"]={"successes":successes,"failures":branches-successes,"branches":branches,"posterior":posterior,"label":label}
        row["development_evidence_sources"]=len(extra[row["id"]])
    return rows,sources


def leave_one_parent_out(rows, seed):
    groups=defaultdict(list)
    for row in rows:groups[parent_key(row)].append(row)
    predictions=[]
    for fold,parent in enumerate(sorted(groups)):
        train=[row for key,values in groups.items() if key!=parent for row in values]
        test=groups[parent]
        if len(train)<8:continue
        model=ViabilityEnsemble(3,64,seed+fold);model.fit(train,epochs=120)
        mean,std,support=model.predict_records(test)
        predictions.extend((row,float(p),float(s),float(q)) for row,p,s,q in zip(test,mean,std,support))
    if not predictions:return {"parents":0,"states":0}
    held=[row for row,_,_,_ in predictions];mean=np.asarray([p for _,p,_,_ in predictions]);std=np.asarray([s for _,_,s,_ in predictions]);support=np.asarray([q for _,_,_,q in predictions])
    return {"parents":len(groups),**_metrics(held,mean,std,support)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--bank",required=True);parser.add_argument("--output",required=True);parser.add_argument("--report",required=True);parser.add_argument("--members",type=int,default=5);parser.add_argument("--seed",type=int,default=0);parser.add_argument("--development-report",action="append",default=[]);parser.add_argument("--config",default="configs/default.json");args=parser.parse_args()
    from dvgc.config import load_config
    bank=SnapshotBank.load(args.bank);cfg=load_config(args.config);rows,sources=development_rows(bank,args.development_report,cfg)
    split,parents=parent_split(rows,args.seed);counts=Counter(row["final"]["label"] for row in split["train"])
    weights=np.asarray([1/max(1,counts[row["final"]["label"]]) for row in split["train"]],np.float64)
    model=ViabilityEnsemble(args.members,64,args.seed);train_report=model.fit(split["train"],sample_weights=weights);model.save(args.output)
    metrics={}
    for name,values in split.items():
        mean,std,support=model.predict_records(values);metrics[name]=_metrics(values,mean,std,support)
    if set(parents["train"])&set(parents["validation"]) or set(parents["train"])&set(parents["acquisition"]) or set(parents["validation"])&set(parents["acquisition"]):raise RuntimeError("Parent split leakage")
    mean,std,support=model.predict_records(rows)
    by_layer={}
    for layer in ("early","middle","late"):
        ids=[i for i,row in enumerate(rows) if row.get("descent_layer")==layer]
        if ids:by_layer[layer]=_metrics([rows[i] for i in ids],mean[ids],std[ids],support[ids])
    report={"status":"PASS","role":"acquisition_only_viability","bank_sha256":file_sha256(args.bank),"seed":args.seed,
            "sample_unit":"snapshot","branches_are_independent_samples":False,"audit_data_used":bool(args.development_report),
            "development_evidence_role":"CONSUMED_DEVELOPMENT_EVIDENCE","development_sources":sources,
            "prediction_can_promote_empirical_safe":False,"split_parents":parents,"train":train_report,"metrics":metrics,
            "leave_one_parent_out":leave_one_parent_out(rows,args.seed+1000),"by_layer":by_layer,
            "safe_tail_auprc":_rank_metrics([row["final"]["label"]=="safe" for row in rows],mean)["auprc"],
            "ensemble_disagreement":{"mean":float(np.mean(std)),"p95":float(np.quantile(std,.95)),"max":float(np.max(std))}}
    save_json(args.report,report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
