"""Small, fixed diagnostics for candidate-grouped feedback support."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def binary_metrics(y_true, y_pred):
    y=np.asarray(y_true,bool);p=np.asarray(y_pred,bool)
    tp=int(np.sum(y&p));tn=int(np.sum(~y&~p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p))
    recall=tp/max(tp+fn,1);specificity=tn/max(tn+fp,1);precision=tp/max(tp+fp,1)
    return {"balanced_accuracy":float((recall+specificity)/2),"positive_precision":float(precision),
            "positive_recall":float(recall),"tp":tp,"tn":tn,"fp":fp,"fn":fn}


def grouped_predictions(x, labels, groups, *, model: str):
    x=np.asarray(x,float);labels=np.asarray(labels,bool);groups=np.asarray(groups)
    prediction=np.zeros(len(labels),bool);score=np.zeros(len(labels),float)
    for group in sorted(set(groups.tolist())):
        test=groups==group;train=~test;mean=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-3)
        train_x=(x[train]-mean)/scale;test_x=(x[test]-mean)/scale
        if model=="linear":
            design=np.concatenate([train_x,np.ones((train_x.shape[0],1))],axis=1);target=labels[train].astype(float)*2-1
            alpha=1.0;weight=design.T@np.linalg.solve(design@design.T+alpha*np.eye(len(design)),target)
            score[test]=np.concatenate([test_x,np.ones((test_x.shape[0],1))],axis=1)@weight;prediction[test]=score[test]>=0
        elif model=="knn":
            distance=np.linalg.norm(test_x[:,None]-train_x[None,:],axis=-1)/np.sqrt(train_x.shape[1]);nearest=np.argsort(distance,axis=1)[:,:3]
            score[test]=labels[train][nearest].mean(1);prediction[test]=score[test]>=.5
        else:raise ValueError(model)
    return prediction,score


def candidate_grouped_diagnostic(x, labels, groups, *, permutations: int = 256, seed: int = 20260728):
    labels=np.asarray(labels,bool);groups=np.asarray(groups);output={}
    for model in ("linear","knn"):
        pred,score=grouped_predictions(x,labels,groups,model=model);metrics=binary_metrics(labels,pred)
        rng=np.random.default_rng(seed+(0 if model=="linear" else 1));null=[]
        for _ in range(permutations):
            shuffled=labels[rng.permutation(len(labels))];null_pred,_=grouped_predictions(x,shuffled,groups,model=model)
            null.append(binary_metrics(shuffled,null_pred)["balanced_accuracy"])
        metrics.update({"permutation_balanced_accuracy_mean":float(np.mean(null)),"permutation_balanced_accuracy_p95":float(np.quantile(null,.95)),"predictions":pred.tolist(),"scores":score.tolist()})
        output[model]=metrics
    other=np.asarray([[j for j in range(len(labels)) if groups[j]!=groups[i]] for i in range(len(labels))])
    normalized=(np.asarray(x)-np.asarray(x).mean(0))/np.maximum(np.asarray(x).std(0),1e-3)
    distance=np.linalg.norm(normalized[:,None]-normalized[None,:],axis=-1)/np.sqrt(normalized.shape[1])
    nearest=[indices[np.argmin(distance[i,indices])] for i,indices in enumerate(other)]
    output["cross_candidate_nearest_neighbor_label_consistency"]=float(np.mean(labels==labels[nearest]))
    return output


def weak_components(nodes: Sequence[str], directed_edges: Sequence[tuple[str,str]]):
    adjacency={node:set() for node in nodes}
    for left,right in directed_edges:adjacency[left].add(right);adjacency[right].add(left)
    components=[];remaining=set(nodes)
    while remaining:
        stack=[min(remaining)];component=set()
        while stack:
            node=stack.pop()
            if node in component:continue
            component.add(node);stack.extend(adjacency[node]-component)
        remaining-=component;components.append(sorted(component))
    return sorted(components,key=lambda value:(-len(value),value))
